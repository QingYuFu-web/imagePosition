"""Verify landscape gallery direction from separate header and footer controls."""

from itertools import combinations
from math import ceil, floor

import cv2
import numpy as np

from screen_candidate_glyphs import _overlap
from screen_landscape_edit_actions import _check_shape
from screen_landscape_header import _marks, _text_run


def _header(gray, width, height):
    size = max(5, round(.024 * height) | 1)
    chip = gray[:round(.102 * height)]
    contrast = cv2.morphologyEx(chip, cv2.MORPH_TOPHAT, np.ones((size, size), np.uint8))
    zone = (round(.013 * width), 0, round(.16 * width), len(chip))
    mark = _marks(contrast, zone, max(7, .022 * height), .077 * height)
    if mark is None:
        return None
    left, top, right, bottom = mark
    size = bottom - top
    if top > .045 * height or (left + right) / 2 > .135 * width:
        return None
    label = (right + round(.20 * size), max(0, top - round(.16 * size)),
             min(width, right + round(6.9 * size)), min(len(chip), bottom + round(.18 * size)))
    if not _text_run(contrast, label, size, (top + bottom) / 2, 2):
        return None
    return mark, (left, min(top, label[1]), label[2], label[3])


def _check_marks(paint, width, height, top):
    left, right = round(.72 * width), round(.97 * width)
    chip = paint[:, left:right]
    matches = []
    for threshold in (38, 51, 66, 82):
        _, labels, marks, _ = cv2.connectedComponentsWithStats(np.uint8(chip >= threshold))
        for number, (x, y, w, h, count) in enumerate(marks[1:], 1):
            if (.018 * height <= h <= .041 * height and 1.02 * h <= w <= 1.83 * h
                    and .23 * w * h <= count <= .62 * w * h
                    and .923 * height <= top + y + h / 2 <= .973 * height
                    and _check_shape(labels[y:y + h, x:x + w] == number)):
                matches.append((threshold, (left + x, top + y, left + x + w, top + y + h)))
    stable = [box for _, box in matches if len({level for level, peer in matches if _overlap(box, peer) >= .60}) >= 3]
    if stable and all(_overlap(box, stable[0]) >= .56 for box in stable):
        return stable[len(stable) // 2]
    return None


def _upper_rim(gray, cx, cy, width, height):
    left, right = cx - round(.055 * width), cx + round(.055 * width)
    top, bottom = cy - round(.054 * height), cy - round(.023 * height)
    tile = gray[top - 2:bottom + 2, left:right].astype(np.int16)
    ridge = tile[2:-2] - np.maximum(tile[:-4], tile[4:])
    rows = np.flatnonzero((ridge >= 10).mean(axis=1) >= .75)
    return (left, top + rows[0], right, top + rows[-1] + 1) if len(rows) else None


def _toolbar_peers(paint, cx, cy, width, height, top):
    left, right = round(.28 * width), cx - round(.13 * width)
    pad = round(.029 * height)
    band = paint[cy - top - pad:cy - top + pad + 1, left:right]
    _, _, marks, _ = cv2.connectedComponentsWithStats(np.uint8(band >= 51))
    peers = []
    for x, y, w, h, count in marks[1:]:
        row = cy - pad + y
        if (.028 * height <= h <= .058 * height and .73 * h <= w <= 1.38 * h
                and .27 * w * h <= count <= .79 * w * h and abs(row + h / 2 - cy) <= .012 * height
                and 0 < y < y + h < len(band)):
            peers.append((left + x, row, left + x + w, row + h))
    for group in combinations(sorted(peers), 3):
        first, second, last = [(left + right) / 2 for left, _, right, _ in group]
        one, two = second - first, last - second
        if (.070 * width <= min(one, two) <= max(one, two) <= .155 * width
                and abs(one - two) <= .026 * width and .17 * width <= cx - last <= .35 * width
                and max((top + bottom) / 2 for _, top, _, bottom in group)
                - min((top + bottom) / 2 for _, top, _, bottom in group) <= .010 * height):
            yield group


def gallery_back_box(frame, *, blocked=None):
    """Return the actual chevron when the navigation and editor controls agree."""
    source_h, source_w = frame.shape[:2]
    if source_h < 190 or not 1.70 * source_h <= source_w <= 2.52 * source_h:
        return None
    height, width = 360, round(source_w * 360 / source_h)
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA if source_h > height else cv2.INTER_CUBIC)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    top = round(.873 * height)
    size = max(7, round(.036 * height) | 1)
    outline = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    paint = cv2.morphologyEx(gray[top:], cv2.MORPH_TOPHAT, outline)
    check = _check_marks(paint, width, height, top)
    if check is None or (header := _header(gray, width, height)) is None:
        return None
    cx, cy = (check[0] + check[2]) // 2, (check[1] + check[3]) // 2
    rim = _upper_rim(gray, cx, cy, width, height)
    if rim is None:
        return None
    scale_x, scale_y = source_w / width, source_h / height

    def source_box(box):
        left, top, right, bottom = box
        return floor(left * scale_x), floor(top * scale_y), ceil(right * scale_x), ceil(bottom * scale_y)

    def visible(box):
        if blocked is None:
            return True
        left, top, right, bottom = source_box(box)
        return not blocked[top:bottom, left:right].any()

    arrow, title = header
    if not all(map(visible, (check, title, rim))) or not any(
            all(map(visible, group)) for group in _toolbar_peers(paint, cx, cy, width, height, top)):
        return None
    return source_box(arrow)


def gallery_orientation_cue(frame, *, blocked=None):
    """Accept the direction only when the navigation and editor controls agree."""
    return gallery_back_box(frame, blocked=blocked) is not None
