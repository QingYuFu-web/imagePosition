"""Verify a painted vertical selector on the front surface of a wide overlay."""

import cv2
import numpy as np

from screen_rail_soft_caption import read_soft_rail_caption, shared_caption_font
from screen_caption_transcript import read_caption
from screen_wide_panel_checks import _magnified_label, _read_label


def _runs(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def _rail_edge(chroma, pane_width):
    a, b = round(.061 * pane_width), round(.185 * pane_width)
    gap = max(2, round(.0016 * pane_width))
    height = chroma.shape[0]
    field = chroma[round(.063 * height):round(.943 * height):2]
    seam = field[:, a:b] - field[:, a + gap:b + gap]
    runs = _runs((seam >= 8).mean(axis=0) >= .80)
    if len(runs) != 1 or runs[0][1] - runs[0][0] > gap + 3:
        return None
    return a + int(runs[0][1])


def _row_edges(gray, chroma, rail):
    gap = max(2, round(.018 * rail))
    x0, x1 = round(.076 * rail), round(.939 * rail)
    support = np.maximum.reduce([
        plane[gap:-gap, x0:x1] - np.maximum(plane[:-2 * gap, x0:x1], plane[2 * gap:, x0:x1])
        for plane in (gray, chroma)
    ])
    density = (support >= 3).mean(axis=1)
    rows = [int(a + density[a:b].argmax()) + gap for a, b in _runs(density >= .64)
            if b - a <= 2 * gap]
    # The verified overlay rim can also be the first boundary of the rail.
    if not rows or rows[0] > .10 * rail:
        rows.insert(0, 0)
    return rows


def _lattice(edges, rail, height):
    if len(edges) < 4:
        return None
    lengths = np.diff(edges)
    typical = lengths[(.35 * rail <= lengths) & (lengths <= .70 * rail)]
    if len(typical) < 2:
        return None
    step = float(np.median(typical))
    if (abs(typical - step) <= .045 * step).mean() < .80:
        return None
    viable = []
    for anchor in edges:
        slots = np.rint((np.array(edges) - anchor) / step).astype(int)
        valid = abs(np.array(edges) - anchor - slots * step) <= .055 * step
        occupied = np.unique(slots[valid])
        if len(occupied) < 4 or occupied[-1] - occupied[0] + 1 - len(occupied) > 1:
            continue
        first, last = round(anchor + occupied[0] * step), round(anchor + occupied[-1] * step)
        if first < -.05 * step or last > height - .025 * step:
            continue
        viable.append((len(occupied), -abs(first), max(0, first), last))
    if not viable:
        return None
    _, _, first, last = max(viable)
    count = round((last - first) / step)
    return [(round(first + j * step), round(first + (j + 1) * step)) for j in range(count)]


def _selected_row(chroma, rail, rows):
    scores = []
    for a, b in rows:
        trim = round(.22 * (b - a))
        band = chroma[a + trim:b - trim, rail - round(.09 * rail):rail - round(.035 * rail)]
        scores.append(float(np.median(band)))
    order = np.argsort(scores)
    first, second = order[-1], order[-2]
    return int(first) if scores[first] >= 67 and scores[first] - scores[second] >= 22 else None


def _caption_box(gray, rail, row):
    top, bottom = row
    step = bottom - top
    inset = round(.105 * rail)
    y0, y1 = top + round(.14 * step), bottom - round(.14 * step)
    patch = gray[y0:y1, inset:rail - inset]
    tone = float(np.median(patch))
    ink = np.uint8(patch >= max(76, tone + 35))
    x, y, w, h = cv2.boundingRect(ink)
    if (not .18 * step <= h <= .38 * step or not 1.24 * h <= w <= .76 * rail
            or abs(inset + x + w / 2 - rail / 2) > .068 * rail
            or abs(y0 + y + h / 2 - (top + bottom) / 2) > .105 * step):
        return None
    if not .13 <= ink[y:y + h, x:x + w].mean() <= .61:
        return None
    pad = max(2, round(.24 * h))
    return max(0, inset + x - pad), y0 + y - pad, inset + x + w + pad, y0 + y + h + pad


def wide_panel_actions(frame, regions, index):
    """Name fully read inactive rail cells while leaving the inner preview alone."""
    if regions["layer"] != "wide_panel" or len(regions["regions"]) < 2 or index is None:
        return []
    left, top, right, bottom = regions["regions"][0]
    width, height = right - left, bottom - top
    if width < 1.42 * height or height < 180:
        return []
    pixels = frame[top:bottom, left:left + round(.20 * width)]
    blue = pixels[..., 0].astype(np.int16) - pixels[..., 2]
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY).astype(np.int16)
    rail = _rail_edge(blue, width)
    grid = _lattice(_row_edges(gray, blue, rail), rail, height) if rail is not None else None
    selected = _selected_row(blue, rail, grid) if grid and len(grid) >= 4 else None
    labels = [_caption_box(gray, rail, row) for row in grid] if selected is not None else []
    if sum(box is not None for box in labels) < 4:
        from screen_wide_rail_geometry import rail_caption_boxes, rail_grid

        if (measured := rail_grid(gray, blue, width)) is None:
            return []
        rail, grid, selected = measured
        labels = rail_caption_boxes(gray, rail, grid)
    cache = {}
    actions, named, missing = [], [], []
    for i, (row, caption) in enumerate(zip(grid, labels, strict=True)):
        if i == selected or caption is None:
            continue
        x0, y0, x1, y1 = caption
        reading = read_caption(frame, (x0 + left, y0 + top, x1 + left, y1 + top),
                               index, cache=cache, choice=True)
        label = reading["text"] if reading else None
        if label is None and min(x0, y0, rail - x1, height - y1) >= 4:
            # Compare the antialiased strokes only inside a proven rail cell.
            label = _read_label(gray, caption, index) or _magnified_label(gray, caption, index)
        tile = frame[y0 + top:y1 + top, x0 + left:x1 + left]
        if label is None:
            missing.append((tile, row))
            continue
        named.append((tile, label))
        a, b = row
        actions.append({"role": "switch_tab", "function": label, "preview_text": label,
                        "box": [left, top + a, left + rail, top + b],
                        "click_point": [left + rail // 2, top + (a + b) // 2]})
    if 1 <= len(missing) <= 2 and len(named) >= 4:
        paths = shared_caption_font(named, index)
        if paths:
            peers = [label for _, label in named]
            confirmed = [(label, row) for tile, row in missing
                         if (label := read_soft_rail_caption(tile, index, paths, peers)) is not None]
            for label, (a, b) in confirmed:
                if sum(label == candidate for candidate, _ in confirmed) == 1:
                    actions.append({"role": "switch_tab", "function": label, "preview_text": label,
                                    "box": [left, top + a, left + rail, top + b],
                                    "click_point": [left + rail // 2, top + (a + b) // 2]})
    texts = [item["function"] for item in actions]
    return [item for item in actions if texts.count(item["function"]) == 1]
