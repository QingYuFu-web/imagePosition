"""Read an outlined, numbered equipment plan selector next to a loadout slot row."""

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_wide_panel_checks import _read_glyph


def _slot_row(gray):
    height, width = gray.shape
    left, top = round(.257 * width), round(.825 * height)
    footer = gray[top:round(.992 * height), left:round(.86 * width)]
    small = cv2.GaussianBlur(footer, (3, 3), .6)
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, 1, round(.112 * height),
                               param1=32, param2=24, minRadius=round(.047 * height), maxRadius=round(.073 * height))
    if circles is None or not 4 <= len(circles[0]) <= 7:
        return None
    slots = sorted(circles[0].tolist())
    x, y, radius = np.asarray(slots).T
    step = np.diff(x)
    size = float(np.median(radius))
    if (np.ptp(y) > .18 * size or np.ptp(radius) > .21 * size
            or not 2.32 * size <= min(step) <= max(step) <= 3.22 * size or np.ptp(step) > .18 * size):
        return None
    for cx, cy, r in slots:
        dx, dy = int(round(cx)), int(round(cy))
        inner = round(.34 * r)
        core = footer[dy - inner:dy + inner + 1, dx - inner:dx + inner + 1]
        if not core.size or np.percentile(core, 87) - np.percentile(core, 13) > 17:
            return None
    return float(x[0] + left), float(np.median(y) + top), size


def _outlined(gray, slots):
    x, y, size = slots
    left, right = round(x - 7.05 * size), round(x - 1.33 * size)
    top, bottom = round(y - 1.38 * size), round(y + .98 * size)
    if min(left, top) < 0 or right >= gray.shape[1] or bottom >= len(gray):
        return None
    patch = gray[top:bottom, left:right]
    horizontal = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3) * .25
    vertical = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3) * .25

    def coherent(plane, axis):
        rising, falling = np.percentile(plane, (33, 67), axis=axis)
        return np.maximum(rising, -falling)

    line = coherent(horizontal[:, round(1.49 * size):round(4.91 * size)], 1)
    t0, t1 = max(1, round(.21 * size)), round(1.15 * size)
    b0, b1 = round(1.63 * size), len(line) - 1
    if min(t1 - t0, b1 - b0) < 4:
        return None
    edges = [(upper, lower) for upper in range(t0, t1) for lower in range(b0, b1)
             if 1.10 * size <= lower - upper <= 1.60 * size and min(line[upper], line[lower]) >= 17]
    if not edges:
        return None
    upper, lower = max(edges, key=lambda pair: sum(line[y] for y in pair))
    depth = lower - upper
    side = coherent(vertical[upper + round(.23 * depth):lower - round(.19 * depth)], 0)
    l0, l1, r0, r1 = 1, round(1.70 * size), round(4.77 * size), len(side) - 1
    if min(l1 - l0, r1 - r0) < 4:
        return None
    a, b = l0 + int(side[l0:l1].argmax()), r0 + int(side[r0:r1].argmax())
    if min(side[a], side[b]) < 17 or not 3.75 * size <= b - a <= 5.20 * size:
        return None
    inside = patch[upper + round(.20 * depth):lower - round(.19 * depth), a + 2:b - 1]
    if not inside.size or np.percentile(inside, 55) > 91:
        return None
    return left + a, top + upper, left + b + 1, top + lower + 1


def _list_icon(gray, button):
    left, top, right, bottom = button
    height, width = bottom - top, right - left
    a, b = left + round(.09 * width), left + round(.31 * width)
    t, d = top + round(.21 * height), bottom - round(.20 * height)
    chip = gray[t:d, a:b].astype(np.int16)
    peak = np.percentile(chip, 96)
    base = np.median(chip)
    if peak < 104 or peak - base < 51:
        return None
    boxes = []
    for cut in (.36, .49):
        ink = (chip >= base + cut * (peak - base))
        rows = _bands(ink.sum(axis=1) >= max(2, round(.17 * height)))
        if len(rows) != 3:
            continue
        runs = [_bands(ink[y:z].mean(axis=0) >= .58) for y, z in rows]
        if any(len(parts) != 2 for parts in runs):
            continue
        if not all(max(1, .007 * width) <= b0 - a0 <= .049 * width and .067 * width <= b1 - a1 <= .124 * width
                   and max(1, .005 * width) <= a1 - b0 <= .039 * width for ((a0, b0), (a1, b1)) in runs):
            continue
        if (max(np.ptp(np.asarray(runs), axis=0).ravel()) > max(2, .016 * width)
                or not .083 * height <= rows[1][0] - rows[0][0] <= .183 * height
                or abs(rows[2][0] - 2 * rows[1][0] + rows[0][0]) > max(1, .034 * height)):
            continue
        boxes.append(a + max(parts[-1][1] for parts in runs))
    return min(boxes) if len(boxes) == 2 else None


def _fang(glyph, peak):
    """Verify the rendered 方 strokes that are confusable in binary font matching."""
    evidence = 0
    for fraction in (.25, .35, .47):
        captured = np.uint8(glyph >= fraction * peak)
        x, y, width, height = cv2.boundingRect(captured)
        if min(width, height) < 9 or not .77 <= width / height <= 1.26:
            continue
        ink = captured[y:y + height, x:x + width] != 0
        rows = ink.sum(axis=1) / width
        head = np.flatnonzero(rows >= .77)
        if not head.size or not max(1, round(.080 * height)) <= head[0] <= .23 * height:
            continue
        band = head[-1]
        if band >= .30 * height:
            continue
        dot = np.where(ink[:head[0]])
        if not dot[0].size or not (.27 * width <= dot[1].mean() <= .70 * width
                                and np.ptp(dot[1]) <= .28 * width):
            continue
        body = np.flatnonzero((rows >= .54) & (np.arange(height) >= .34 * height) &
                             (np.arange(height) <= .58 * height))
        if not len(body):
            continue
        witness = []
        for proportion in (.64, .75, .84):
            row = np.flatnonzero(ink[round(proportion * (height - 1))])
            witness.append(bool(len(row) and row[0] <= .33 * width and row[-1] >= .68 * width and
                                (np.diff(row).max(initial=0) >= .28 * width)))
        if sum(witness) < 2 or (ink[-max(1, round(.15 * height)):, round(.55 * width):].sum() < .08 * width):
            continue
        evidence += 1
    return evidence >= 2


def _preset(gray, button, icon_right, index):
    left, top, right, bottom = button
    depth = bottom - top
    a = icon_right + round(.24 * depth)
    patch = gray[top + round(.21 * depth):bottom - round(.20 * depth), a:right - round(.14 * depth)]
    if min(patch.shape) < 10:
        return None
    signal = np.maximum(patch.astype(np.float32) - np.median(patch), 0)
    peak = float(np.percentile(signal, 98))
    if peak < 66 or np.percentile(patch, 98) < 110:
        return None
    x, y, width, height = cv2.boundingRect(np.uint8(signal >= .31 * peak))
    if not (.27 * depth <= height <= .46 * depth and 4.10 <= width / height <= 5.75
            and x >= 1 and x + width <= patch.shape[1] - 1):
        return None
    crop = signal[y:y + height, x:x + width]
    spacing = (width - .36 * height) / 4
    columns = (crop >= .33 * peak).sum(axis=0)
    stops = [0]
    for i in range(1, 5):
        center = i * spacing
        interval = range(max(1, round(center - .16 * height)), min(width - 1, round(center + .17 * height)) + 1)
        split = min(interval, key=lambda j: (columns[j], abs(j - center)))
        if columns[split] > .14 * height or split - stops[-1] < .57 * height:
            return None
        stops.append(split)
    stops.append(width)
    glyphs = [crop[:, a:b] for a, b in zip(stops[:-1], stops[1:], strict=True)]
    if not _fang(glyphs[2], peak):
        return None
    first = _read_glyph(glyphs[0], peak, index)
    second = _read_glyph(glyphs[1], peak, index)
    last = _read_glyph(glyphs[3], peak, index)
    if (first, second, last) not in {("装", "备", "案"), ("裝", "備", "案")}:
        return None
    digits = set()
    support = 0
    for level in (.26, .37, .49):
        options = [rank for rank in index.rank(glyphs[-1] >= level * peak, alphabet="code", limit=7)
                   if rank["text"].isdigit()]
        if (len(options) >= 2 and options[0]["distance"] <= .17
                and options[1]["distance"] - options[0]["distance"] >= .05):
            digits.add(options[0]["text"])
            support += 1
    return first + second + "方案" + next(iter(digits)) if len(digits) == 1 and support >= 2 else None


def equipment_plan_actions(frame, regions, index):
    """Emit one owned selector only for an independently named and enabled footer."""
    if index is None or regions.get("layer") != "wide_panel" or len(regions.get("regions", ())) < 2:
        return []
    left, top, right, bottom = regions["regions"][0]
    if not (0 <= left < right <= frame.shape[1] and 0 <= top < bottom <= len(frame)
            and bottom - top >= 295 and right - left >= 1.42 * (bottom - top)):
        return []
    gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    slots = _slot_row(gray)
    if slots is None or (button := _outlined(gray, slots)) is None:
        return []
    icon = _list_icon(gray, button)
    if icon is None or (name := _preset(gray, button, icon, index)) is None:
        return []
    a, b, c, d = button
    box = [left + a, top + b, left + c, top + d]
    return [{"role": "game_loadout", "function": f"打开{name[:-1]}", "preview_text": name,
             "box": box, "click_point": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]}]
