"""Find the search field and its neighboring action in a visible media tray."""

import cv2
import numpy as np

from inventory_toolbar_actions import _holes, _is_search
from screen_component_captions import _bands
from screen_dark_barrage_input import _round_fields


def _tile_rows(gray, field_top, paper):
    width = gray.shape[1]
    start, stop = round(.11 * width), field_top - round(.09 * width)
    left, right = round(.06 * width), round(.94 * width)
    separators = (gray[start:stop, left:right] >= paper - 7).mean(axis=1)
    seams = [(start + a, start + b) for a, b in _bands(separators >= .86)
             if 1 <= b - a <= .021 * width]
    for top, bottom in seams:
        inset = round(.045 * width)
        upper = gray[inset:top - inset] >= paper - 7
        lower = gray[bottom + inset:field_top - round(.06 * width)] >= paper - 7
        if min(len(upper), len(lower)) < .075 * width:
            continue
        vertical = np.minimum(upper.mean(axis=0), lower.mean(axis=0))
        columns = [(a + b) / 2 for a, b in _bands(vertical >= .75)
                   if .10 * width <= a < b <= .94 * width and b - a <= .023 * width]
        if len(columns) >= 3:
            spacing = np.diff(columns)
            if .13 * width <= spacing.min() <= spacing.max() <= .28 * width and np.ptp(spacing) <= .035 * width:
                return True
    return False


def _search_ink(gray, box):
    left, top, right, bottom = box
    height = bottom - top
    start, stop = top + round(.18 * height), bottom - round(.18 * height)
    lens = gray[start:stop, left + round(.18 * height):left + round(.85 * height)]
    background = float(np.median(lens))
    if background - np.percentile(lens, 8) < 42:
        return False
    for level in (25, 15, 10, 35):
        mask = np.uint8(lens.astype(np.int16) <= background - level)
        x, y, w, h = cv2.boundingRect(mask)
        if .32 * height <= min(w, h) <= max(w, h) <= .59 * height:
            glyph = mask[y:y + h, x:x + w]
            if _is_search(glyph, _holes(glyph)):
                break
    else:
        return False
    caption = gray[start:stop, left + round(1.10 * height):min(right, left + round(4.8 * height))]
    marks = np.uint8(caption.astype(np.int16) <= background - 49)
    _, _, w, h = cv2.boundingRect(marks)
    return .73 * height <= w <= 3.5 * height and .22 * height <= h <= .63 * height and .031 <= marks.mean() <= .36


def _right_action(gray, field, paper):
    _, top, pill_right, bottom = field
    tray_height, width = gray.shape
    height = bottom - top
    left, right = pill_right + round(.009 * width), round(.979 * width)
    if right <= left:
        return None
    crop = gray[top:bottom, left:right]
    mask = np.uint8(crop.astype(np.int16) <= paper - 52)
    x, y, w, h = cv2.boundingRect(mask)
    a, t, b, z = left + x, top + y, left + x + w, top + y + h
    if (not .062 * width <= w <= .129 * width or not .24 * height <= h <= .57 * height
            or abs(t + h / 2 - (top + bottom) / 2) > .13 * height or not .047 <= mask.mean() <= .27
            or not .015 * width <= a - pill_right <= .052 * width or not .010 * width <= width - b <= .057 * width
            or len(_bands(mask.any(axis=0))) < 2):
        return None
    px, py = max(3, round(.19 * h)), max(3, round(.39 * h))
    return {"box": [max(pill_right, a - px), max(0, t - py), min(width, b + px), min(tray_height, z + py)],
            "caption_box": [a, t, b, z], "click_point": [(a + b) // 2, (t + z) // 2]}


def media_picker_footer(frame, tray):
    """Return evidence for visible footer controls without assigning their purposes."""
    if tray is None or len(tray) != 4:
        return []
    left, top, right, bottom = map(round, tray)
    width, height = right - left, bottom - top
    image_height, image_width = frame.shape[:2]
    if (not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height)
            or width < max(230, .72 * image_width) or not .34 * width <= height <= .76 * width
            or image_height - bottom > .12 * width):
        return []
    gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    start = height - round(.13 * width)
    footer = gray[start:]
    paper = float(np.percentile(footer, 90))
    if paper < 228:
        return []
    mask = (footer >= max(203, paper - 31)) & (footer <= paper - 6)
    fields = [(a, start + t, b, start + z) for a, t, b, z in _round_fields(mask, width, 9., 14.)
              if .008 * width <= a <= .075 * width and .73 * width <= b <= .90 * width
              and .010 * width <= len(footer) - z <= .055 * width and z - t <= .091 * width]
    if len(fields) != 1:
        return []
    field, = fields
    if not _tile_rows(gray, field[1], paper) or not _search_ink(gray, field):
        return []
    a, t, b, z = field
    controls = [{"box": list(field), "click_point": [a + round(2 * (z - t)), (t + z) // 2], "fixed_center": True}]
    if (action := _right_action(gray, field, paper)) is not None:
        controls.append(action)
    for item in controls:
        point = item["click_point"]
        item["click_point"] = [point[0] + left, point[1] + top]
        for key in ("box", "caption_box"):
            if key in item:
                item[key] = [axis + offset for axis, offset in zip(item[key], (left, top, left, top), strict=True)]
    return controls
