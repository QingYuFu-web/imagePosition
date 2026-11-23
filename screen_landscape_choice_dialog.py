"""Verify rounded translucent landscape choice panes and their active rows."""

import cv2
import numpy as np

from inventory_choice_glyphs import _tick
from screen_caption_transcript import read_caption
from screen_component_captions import _bands, _tight


def _edge(gray, axis, nominal, start, stop, reach, thickness, entering):
    field = np.moveaxis(gray, axis, 0)
    found = []
    for ridge in range(max(thickness, nominal - reach), min(len(field) - thickness, nominal + reach + 1)):
        before = field[ridge - thickness:ridge, start:stop].mean(axis=0)
        after = field[ridge:ridge + thickness, start:stop].mean(axis=0)
        outside, inside = (before, after) if entering else (after, before)
        diff = outside - inside
        density = float((diff >= 11).mean())
        shade = float(np.median(diff))
        if density < .69 or shade < 12 or outside.std() < 8 or not .31 <= inside.std() / outside.std() <= .88:
            continue
        correlation = float(np.corrcoef(outside, inside)[0, 1])
        if correlation >= .85:
            found.append((shade * density * correlation, ridge))
    return max(found)[1] if found else None


def _corner(gray, x, y, dx, dy, size):
    shifts = np.arange(-2, max(9, round(.095 * size)))
    pads = np.arange(1, 4)
    offsets = [max(2, round(scale * size)) for scale in (.010, .029, .070)]
    positions = []
    for offset in offsets:
        samples = y + dy * shifts
        before = samples[:, None] - dy * pads
        after = samples[:, None] + dy * (pads - 1)
        if min(before.min(), after.min()) < 0 or max(before.max(), after.max()) >= len(gray):
            return False
        edge = np.median(gray[before, x + dx * offset], axis=1) - np.median(gray[after, x + dx * offset], axis=1)
        k = int(edge.argmax())
        if edge[k] < 19:
            return False
        positions.append(int(shifts[k]))
    near, middle, far = positions
    return (max(3, round(.013 * size)) <= near <= .080 * size and
            -1 <= far <= max(2, round(.01 * size)) and far - 1 <= middle <= near - 1)


def _measured_panel(gray, proposal):
    a, t, r, b = map(int, proposal)
    h, w = gray.shape
    width, height = r - a, b - t
    if (not 0 <= a < r <= w or not 0 <= t < b <= h or not .17 * w <= width <= .44 * w
            or not .175 * h <= height <= .54 * h or not .79 <= width / height <= 2.63):
        return None
    reach = max(8, round(.105 * height))
    thickness = max(2, round(.009 * height))
    x0, x1 = a + round(.14 * width), r - round(.14 * width)
    y0, y1 = t + round(.12 * height), b - round(.12 * height)
    left = _edge(gray, 1, a, y0, y1, reach, thickness, True)
    right = _edge(gray, 1, r, y0, y1, reach, thickness, False)
    top = _edge(gray, 0, t, x0, x1, reach, thickness, True)
    bottom = _edge(gray, 0, b, x0, x1, reach, thickness, False)
    if any(edge is None for edge in (left, top, right, bottom)):
        return None
    if not .79 * width <= right - left <= 1.17 * width or not .79 * height <= bottom - top <= 1.17 * height:
        return None
    corners = [(left, top, 1, 1), (right - 1, top, -1, 1),
               (left, bottom - 1, 1, -1), (right - 1, bottom - 1, -1, -1)]
    size = min(right - left, bottom - top)
    if sum(_corner(gray, x, y, dx, dy, size) for x, y, dx, dy in corners) < 3:
        return None
    return left, top, right, bottom


def _caption_rows(gray, box):
    l, t, r, b = box
    width, height = r - l, b - t
    a, z = l + round(.245 * width), r - round(.105 * width)
    y0, y1 = t + round(.088 * height), b - round(.085 * height)
    field = gray[y0:y1, a:z]
    captions = {}
    for level in (160, 188, 211):
        lines = []
        groups = _bands((field >= level).sum(axis=1) >= max(2, round(.008 * width)))
        for first, last in groups:
            line = _tight(field[first:last] >= level, a, y0 + first)
            if line is None:
                continue
            x, y, rr, bb = line
            if not .061 * height <= bb - y <= .19 * height or not 1.58 <= (rr - x) / (bb - y) <= 9.3:
                continue
            lines.append(line)
        if not 2 <= len(lines) <= 5:
            continue
        centers = [(y + bb) / 2 for _, y, _, bb in lines]
        pitch = height / len(lines)
        if (max(abs(c - t - (i + .5) * pitch) for i, c in enumerate(centers)) > .14 * pitch or
                np.ptp([x for x, _, _, _ in lines]) > .042 * width or
                not .247 * width <= min(x - l for x, _, _, _ in lines) <= .39 * width):
            continue
        captions[level] = lines
    if len(captions) < 2:
        return None
    readings = list(captions.values())
    if any(len(lines) != len(readings[0]) or
           max(abs(a[1] - b[1]) for a, b in zip(lines, readings[0], strict=True)) > .025 * height
           for lines in readings[1:]):
        return None
    return readings[0]


def _selected_row(gray, panel, lines):
    l, _, r, _ = panel
    width = r - l
    first, last = l + round(.043 * width), l + round(.236 * width)
    states = []
    for _, top, _, bottom in lines:
        h = bottom - top
        y0, y1 = top - round(.40 * h), bottom + round(.31 * h)
        field = gray[y0:y1, first:last]
        hits, occupied = [], []
        for level in (146, 169, 195, 218):
            ink = field >= level
            x, y, w, depth = cv2.boundingRect(np.uint8(ink))
            occupied.append(int(ink.sum()))
            if (not .70 <= depth / h <= 1.25 or not 1.18 <= w / depth <= 1.87
                    or abs(y0 + y + depth / 2 - (top + bottom) / 2) > .43 * h):
                continue
            if _tick(ink[y:y + depth, x:x + w], field[y:y + depth, x:x + w], minimum_fill=.13):
                hits.append((x, y, w, depth))
        if len(hits) >= 2 and max(max(abs(a - b) for a, b in zip(hits[0], seen, strict=True)) for seen in hits) <= 3:
            states.append(True)
        elif max(occupied) <= max(3, round(.012 * h * h)):
            states.append(False)
        else:
            return None
    return states.index(True) if states.count(True) == 1 else None


def landscape_choice_dialog_actions(frame, regions, index):
    """Return the owned pane and literal inactive choices, when visually supported."""
    h, w = frame.shape[:2]
    if regions.get("layer") not in {"page", "layer_uncertain"} or h < 380 or not 1.55 * h <= w <= 2.9 * h:
        return None
    candidates = [row["box"] for row in regions.get("candidates", ())
                  if .17 * w <= row["box"][2] - row["box"][0] <= .44 * w
                  and .175 * h <= row["box"][3] - row["box"][1] <= .54 * h]
    if not candidates:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    matches = {}
    for candidate in candidates[:10]:
        panel = _measured_panel(gray, candidate)
        if panel is None or panel in matches:
            continue
        lines = _caption_rows(gray, panel)
        if lines is not None and (picked := _selected_row(gray, panel, lines)) is not None:
            matches[panel] = (lines, picked)
    if len(matches) != 1:
        return None
    panel, (lines, picked) = next(iter(matches.items()))
    left, top, right, bottom = panel
    actions = []
    if index is not None:
        cache = {}
        for i, (a, t, r, b) in enumerate(lines):
            if i == picked:
                continue
            pad = max(5, round(.29 * (b - t)))
            reading = read_caption(frame, (a - pad, t - pad, r + pad, b + pad), index, cache=cache, choice=True)
            if reading is None:
                continue
            row_top = round(top + i * (bottom - top) / len(lines))
            row_bottom = round(top + (i + 1) * (bottom - top) / len(lines))
            label = reading["text"]
            actions.append({"role": "radio", "function": label, "preview_text": label,
                            "box": [left, row_top, right, row_bottom], "click_point": [(a + r) // 2, (t + b) // 2]})
    return panel, actions
