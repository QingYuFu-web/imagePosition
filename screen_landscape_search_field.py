"""Verify a native landscape search field through its fill, underline, and contents."""

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_landscape_header import _magnifier


def _gesture_surface(frame):
    """Find the system gesture handle on a quiet strip below the page."""
    height, width = frame.shape[:2]
    reach = round(.17 * width)
    center = width // 2
    strip = cv2.cvtColor(frame[-round(.045 * height):, center - reach:center + reach], cv2.COLOR_BGR2GRAY)
    paint = float(np.median(strip))
    contrast = abs(strip.astype(np.int16) - paint)
    count, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(contrast >= 45), connectivity=8)
    if count != 2:
        return False
    x, y, w, h, area = stats[1]
    return (.058 * width <= w <= .205 * width and .0026 * height <= h <= .021 * height
            and abs(x + w / 2 - reach) <= .020 * width and w / h >= 10
            and 0 < y < y + h < len(strip) and area >= .69 * w * h)


def _body_families(boxes, field, height):
    left, _, right, bottom = field
    width, depth = right - left, field[3] - field[1]
    media = [box for box in boxes if box[0] >= left - .11 * depth and box[2] <= right + .08 * width
             and bottom + .28 * depth <= box[1] <= min(.70 * height, bottom + 2.0 * depth)
             and .088 * width <= box[2] - box[0] <= .29 * width and box[3] - box[1] >= .82 * depth]
    for base in media:
        size = base[2] - base[0]
        family = sorted(box for box in media if abs(base[1] - box[1]) <= .10 * depth
                        and .84 * size <= box[2] - box[0] <= 1.22 * size
                        and abs((base[3] - base[1]) - (box[3] - box[1])) <= .19 * depth)
        for first, second, third in zip(family, family[1:], family[2:]):
            a, b = second[0] - first[0], third[0] - second[0]
            if .81 * size <= min(a, b) <= max(a, b) <= 1.42 * size and abs(a - b) <= .18 * size:
                return True
    rail = [box for box in boxes if box[2] < left - .12 * depth and box[1] >= bottom + .28 * depth
            and .69 * depth <= box[2] - box[0] <= 1.47 * depth and .66 * depth <= box[3] - box[1] <= 1.49 * depth]
    return any(abs(a[0] - b[0]) <= .18 * depth and .86 * depth <= b[1] - a[1] <= 1.47 * depth
               for a in rail for b in rail)


def _underlined_surface(frame, proposal):
    left, top, right, bottom = proposal
    height, width = frame.shape[:2]
    span, depth = right - left, bottom - top
    start, end = max(1, top - round(.20 * depth)), min(height - 1, bottom + round(.20 * depth))
    a, b = left + round(.27 * span), right - round(.13 * span)
    band = frame[start:end, a:b:4].astype(np.int16)
    profile = np.median(band, axis=1)
    inner = np.median(profile[round(.36 * len(profile)):round(.62 * len(profile))], axis=0)
    matches = _bands(abs(profile - inner).max(axis=1) <= 4)
    flat = [(x + start, y + start) for x, y in matches if .65 * depth <= y - x <= 1.19 * depth
            and abs((x + y) / 2 + start - (top + bottom) / 2) <= .17 * depth]
    if len(flat) != 1:
        return None
    top_edge, interior_end = flat[0]
    thickness = interior_end - top_edge
    strip = frame[top_edge + round(.12 * thickness):interior_end - round(.17 * thickness),
                  max(0, left - round(.26 * thickness)):min(width, right + round(.26 * thickness))]
    ink = (abs(strip.astype(np.int16) - inner).max(axis=2) <= 5).mean(axis=0)
    offset = max(0, left - round(.26 * thickness))
    extents = [(a + offset, b + offset) for a, b in _bands(ink >= .63)
               if a + offset < (left + right) / 2 < b + offset]
    if len(extents) != 1:
        return None
    real_left, real_right = extents[0]
    if not (.90 * span <= real_right - real_left <= 1.06 * span
            and max(abs(real_left - left), abs(real_right - right)) <= .25 * thickness):
        return None
    pad = max(2, round(.035 * thickness))
    under = frame[interior_end - pad:interior_end + 2 * pad, real_left + pad:real_right - pad].astype(np.int16)
    contrast = abs(under - inner).max(axis=2)
    ridge = _bands((contrast >= 34).mean(axis=1) >= .92)
    if len(ridge) != 1 or not 1 <= ridge[0][1] - ridge[0][0] <= .082 * thickness:
        return None
    field_end = interior_end - pad + ridge[0][1]
    above = frame[top_edge - max(2, round(.12 * thickness)):top_edge - 2, a:b:4].astype(np.int16)
    below = frame[field_end + 2:field_end + max(4, round(.17 * thickness)), a:b:4].astype(np.int16)
    if not above.size or not below.size:
        return None
    paper = np.median(above, axis=(0, 1))
    if (not 9 <= abs(inner - paper).max() <= 97 or (abs(above - paper).max(axis=2) <= 7).mean() < .91
            or (abs(below - paper).max(axis=2) <= 8).mean() < .91):
        return None
    return (real_left, top_edge, real_right, field_end), inner


def _input_caption(frame, field, icon, paint, index, cache):
    left, top, right, bottom = field
    depth = bottom - top
    start = icon[2] + round(.13 * depth)
    end = right - round(.045 * depth)
    a, b = top + round(.21 * depth), bottom - round(.21 * depth)
    panel = frame[a:b, start:end].astype(np.int16)
    contrast = abs(panel - paint).max(axis=2)
    x, y, w, h = cv2.boundingRect(np.uint8(contrast >= 44))
    if (not .145 * depth <= h <= .44 * depth or not .65 * h <= w <= .47 * (right - left)
            or not .15 * depth <= start + x - icon[2] <= .67 * depth
            or abs(a + y + h / 2 - (top + bottom) / 2) > .095 * depth
            or abs(a + y + h / 2 - (icon[1] + icon[3]) / 2) > .16 * depth):
        return None
    pad = max(3, round(.24 * h))
    box = (max(start, start + x - pad), max(a, a + y - pad),
           min(end, start + x + w + pad), min(b, a + y + h + pad))
    read = read_caption(frame, box, index, cache=cache, choice=True)
    if read is None or len(read["text"]) < 2:
        return None
    return read["text"], (start + x + w // 2, a + y + h // 2)


def _available(field, regions):
    for control in regions.get("floating_controls", ()) or ():
        blocker = control.get("box") if isinstance(control, dict) else control
        if isinstance(blocker, (tuple, list)) and len(blocker) == 4:
            left, top, right, bottom = blocker
            if min(field[2], right) > max(field[0], left) and min(field[3], bottom) > max(field[1], top):
                return False
    return True


def landscape_search_field_actions(frame, regions, index):
    """Return one real search entry on a verified mobile main view."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or height < 223 or width < 1.45 * height:
        return []
    boxes = [tuple(row["box"]) for row in regions["candidates"] if row.get("visibility") == "page"
             and 0 <= row["box"][0] < row["box"][2] <= width and 0 <= row["box"][1] < row["box"][3] <= height]
    proposals = [box for box in boxes if .041 * height <= box[1] <= .248 * height and box[3] <= .361 * height
                 and .064 * height <= box[3] - box[1] <= .192 * height and box[2] - box[0] >= .63 * width]
    if not proposals or not _gesture_surface(frame):
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cache, actions = {}, []
    for proposal in proposals:
        result = _underlined_surface(frame, proposal)
        if not result:
            continue
        field, paint = result
        if not _available(field, regions) or not _body_families(boxes, field, height):
            continue
        left, top, right, bottom = field
        depth = bottom - top
        ink = 255 - gray if np.median(paint) >= 128 else gray
        lenses = []
        for x, y, r, b in boxes:
            if (not left - .035 * depth <= x <= left + .67 * depth or r > left + 1.12 * depth
                    or not top - .03 * depth <= y < b <= bottom + .05 * depth
                    or not .51 * depth <= b - y <= .96 * depth or not .69 <= (r - x) / (b - y) <= 1.48):
                continue
            mark = _magnifier(ink, (x, y, r, b))
            if mark and .15 * depth <= mark[0] - left <= .57 * depth:
                lenses.append(mark)
        if len(lenses) != 1:
            continue
        label = _input_caption(frame, field, lenses[0], paint, index, cache)
        if label:
            text, point = label
            actions.append({"role": "search", "function": "搜索", "preview_text": text,
                            "box": list(field), "click_point": list(point)})
    return actions if len(actions) == 1 else []
