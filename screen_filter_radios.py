"""Keep visible alternatives on a measured radio row inside a trusted panel."""

import cv2
import numpy as np

from inventory_choice_glyphs import _outline, find_choice_glyphs
from screen_landscape_radio_rows import _carrier_tone, _label_box


def _selected_mark(frame, area, size):
    left, top, right, bottom = area
    blue, green, red = frame[top:bottom, left:right].astype(np.int16).transpose(2, 0, 1)
    bright = np.uint8((red >= 115) & (red - blue >= 55) & (green - blue >= 29) & (red - green >= 20))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = []
    for contour in contours:
        x, y, width, height = bounds = cv2.boundingRect(contour)
        if not (.76 * size <= min(width, height) <= max(width, height) <= 1.35 * size):
            continue
        geometry = _outline(contour, bounds)
        if geometry is not None and geometry[0] == "radio":
            marks.append((left + x + width / 2, top + y + height / 2))
    return marks[0] if len(marks) == 1 else None


def filter_radio_candidates(frame, panel, reference_row):
    """Require a filled mark and two directly visible alternatives with captions."""
    left, top, right, _ = panel
    x, y, _ = reference_row[0]
    size = float(np.median([mark[2] for mark in reference_row]))
    area = (max(left, round(x - 1.2 * size)), round(top + .55 * size),
            round(right - 1.35 * size), round(y - 2.1 * size))
    if area[2] <= area[0] or area[3] <= area[1]:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rings = [glyph for glyph in find_choice_glyphs(gray, area, (round(.99 * size), round(1.71 * size)))
             if glyph.kind == "radio" and glyph.confidence >= .68 and _carrier_tone(frame, glyph) == "off"]
    if len(rings) != 2 or (selected := _selected_mark(frame, area, size)) is None:
        return []
    rings.sort(key=lambda glyph: glyph.center[0])
    points = sorted([selected, *(glyph.center for glyph in rings)])
    xs, ys = np.asarray(points).T
    stride = np.diff(xs)
    if (np.ptp(ys) > .26 * size or not 4.5 * size <= min(stride) <= max(stride) <= 12 * size
            or np.ptp(stride) > .16 * min(stride)):
        return []
    options = []
    for glyph in rings:
        gx, gy = glyph.center
        next_mark = next((px for px in xs if px > gx + size), right)
        limit = min(right, round(next_mark - .7 * size))
        if (caption := _label_box(gray, glyph, limit)) is None:
            return []
        a, t, _, b = glyph.box
        x0, y0, x1, y1 = caption
        box = [int(a), int(min(t, y0)), int(x1), int(max(b, y1))]
        click = [round(gx), round(gy)]
        ambiguous = np.percentile(gray[y0:y1, x0:x1], 95) < 145
        item = {"box": box, "click_point": click, "fixed_center": True}
        if ambiguous:
            item["review_only"] = True
        # One neighboring selected choice provides state and label context.
        if 0 < gx - selected[0] <= 1.10 * stride.mean():
            context_left = max(left, round(selected[0] - .72 * stride.mean()))
            item["context_box"] = [context_left, *box[1:]]
        options.append(item)
    return options
