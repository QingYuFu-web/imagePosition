"""Read available chips and footer actions inside a rounded settings dialog."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption


def _runs(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2).tolist()


def _surface(frame):
    height, width = frame.shape[:2]
    if height < 1.46 * width or width < 280:
        return None
    scale = min(1., 490 / width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    pixels = small.astype(np.int16)
    paper = np.median(pixels[h // 3:2 * h // 3:5, w // 3:2 * w // 3:5], axis=(0, 1))
    if paper.min() < 164:
        return None
    off = np.max(abs(pixels - paper), axis=2)
    contours, _ = cv2.findContours(np.uint8(off <= 13), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels = []
    for contour in contours:
        x, y, pw, ph = cv2.boundingRect(contour)
        if (not .65 * w <= pw <= .93 * w or not 1.02 * pw <= ph <= 1.91 * pw
                or not .10 * h <= y <= .38 * h or y + ph > .90 * h
                or abs(2 * x + pw - w) > .10 * w or cv2.contourArea(contour) < .96 * pw * ph):
            continue
        margin = max(3, round(.04 * pw))
        mid = slice(y + round(.15 * ph), y + round(.84 * ph), 3)
        if min(x, w - x - pw) < margin:
            continue
        sides = [pixels[mid, a:b:2].mean(axis=2) for a, b in ((x - margin, x), (x + pw, x + pw + margin))]
        if any((side <= paper.mean() - 37).mean() < .72 for side in sides):
            continue
        corner = max(2, round(.02 * pw))
        corners = [off[a:a + corner, b:b + corner]
                   for a in (y, y + ph - corner) for b in (x, x + pw - corner)]
        if min((cut > 23).mean() for cut in corners) < .45:
            continue
        panels.append((x, y, x + pw, y + ph))
    if len(panels) == 1:
        return small, off, panels[0]
    return None


def _grouped_chips(distance, panel):
    x, y, right, bottom = panel
    w, h = right - x, bottom - y
    a, b = x + round(.044 * w), y + round(.21 * h)
    c, d = right - round(.042 * w), y + round(.79 * h)
    tile = distance[b:d, a:c]
    contours, _ = cv2.findContours(np.uint8(tile >= 13), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        left, top, width, height = cv2.boundingRect(contour)
        if (not .067 * w <= height <= .137 * w or not 1.21 * height <= width <= 3.8 * height
                or not .83 <= cv2.contourArea(contour) / (width * height) <= .991):
            continue
        center = tile[top + round(.22 * height):top + round(.78 * height),
                      left + round(.19 * width):left + round(.81 * width)]
        selected = float(np.median(center)) >= 11
        if not selected and (center >= 11).mean() > .49:
            continue
        candidates.append(((a + left, b + top, a + left + width, b + top + height), selected))
    candidates.sort(key=lambda item: item[0][1])
    rows = []
    for chip in candidates:
        cell, _ = chip
        if not rows or cell[1] - rows[-1][0][0][1] > .20 * (cell[3] - cell[1]):
            rows.append([])
        rows[-1].append(chip)
    if not 2 <= len(rows) <= 5 or any(not 2 <= len(row) <= 6 for row in rows):
        return None
    heights = [box[3] - box[1] for box, _ in candidates]
    typical = float(np.median(heights))
    if max(heights) > 1.21 * min(heights) or not .25 * h <= rows[-1][0][0][1] - rows[0][0][0][1] <= .55 * h:
        return None
    if sum(sum(selected for _, selected in row) == 1 for row in rows) < len(rows) - 1:
        return None
    if any(sum(selected for _, selected in row) > 1 for row in rows):
        return None
    starts = []
    for row in rows:
        row.sort()
        starts.append(row[0][0][0])
        if any(not .09 * typical <= second[0] - first[2] <= .58 * typical
               for (first, _), (second, _) in zip(row, row[1:])):
            return None
    if np.ptp(starts) > .22 * typical:
        return None
    return rows, typical


def _footer(distance, panel, rows, typical):
    x, _, right, bottom = panel
    w = right - x
    a, y = x + round(.39 * w), max(cell[3] for cell, _ in rows[-1]) + round(1.20 * typical)
    crop = distance[y:bottom - round(.63 * typical), a:right - round(.07 * w)]
    if min(crop.shape) < .5 * typical:
        return None
    ink = np.uint8(crop >= 48)
    vertical = _runs(ink.sum(axis=1) >= 2)
    if len(vertical) != 1:
        return None
    t, b = vertical[0]
    height = b - t
    if not .25 * typical <= height <= .61 * typical:
        return None
    row = ink[t:b].sum(axis=0) >= 2
    joined = cv2.morphologyEx(np.uint8(row.reshape(1, -1)), cv2.MORPH_CLOSE,
                              np.ones((1, max(3, round(.36 * height))), np.uint8))
    horizontal = _runs(joined[0] != 0)
    if (len(horizontal) != 2 or any(not 1.35 * height <= end - left <= 8.0 * height for left, end in horizontal)
            or not 1.1 * height <= horizontal[1][0] - horizontal[0][1] <= 6 * height):
        return None
    return [(a + left, y + t, a + end, y + b) for left, end in horizontal]


def _title(frame, box, index, cache):
    found = read_caption(frame, box, index, cache=cache, choice=True, join_radicals=True)
    if found and 2 <= len(found["text"]) <= 10 and all("\u3400" <= c <= "\u9fff" for c in found["text"]):
        return found["text"]
    return None


def grouped_option_dialog(frame, regions, index):
    """Return only proved actions for equally spaced chips on a dimmed foreground."""
    if index is None or regions["layer"] not in {"page", "dialog", "layer_uncertain"}:
        return None
    measured = _surface(frame)
    if measured is None:
        return None
    small, distance, panel = measured
    grouped = _grouped_chips(distance, panel)
    if grouped is None:
        return None
    rows, typical = grouped
    footer = _footer(distance, panel, rows, typical)
    if footer is None:
        return None
    sy, sx = np.array(frame.shape[:2]) / small.shape[:2]

    def original(box):
        return [round(coordinate * (sy if i % 2 else sx)) for i, coordinate in enumerate(box)]

    def add(role, name, box):
        action = original(box)
        point = [(action[0] + action[2]) // 2, (action[1] + action[3]) // 2]
        task = "选择" + name if role == "select_option" else name
        return {"role": role, "function": task, "preview_text": name, "box": action, "click_point": point}

    cache = {}
    actions = []
    for row in rows:
        for (a, b, c, d), selected in row:
            if selected:
                continue
            inset = max(4, round(.15 * (d - b)))
            if name := _title(frame, original((a + inset, b + inset, c - inset, d - inset)), index, cache):
                actions.append(add("select_option", name, (a, b, c, d)))
    for a, b, c, d in footer:
        inset = max(3, round(.22 * (d - b)))
        name = _title(frame, original((a - inset, b - inset, c + inset, d + inset)), index, cache)
        if name:
            role = "confirm" if name in {"确认", "确定"} else "reset" if name.startswith(("恢复", "重置")) else "modal_footer_action"
            actions.append(add(role, name, (a, b, c, d)))
    return (tuple(original(panel)), actions) if actions else None
