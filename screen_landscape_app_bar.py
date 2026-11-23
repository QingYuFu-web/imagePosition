"""Find a small Back action in a native landscape app bar."""

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_back_control import _title_run
from screen_candidate_glyphs import _overlap
from screen_component_captions import _bands


def _bar_surface(frame):
    height, width = frame.shape[:2]
    limit = round(.295 * height)
    strip = frame[:limit, round(.22 * width):round(.76 * width):4].astype(np.int16)
    paint = np.median(strip[:round(.063 * height)], axis=(0, 1))
    support = (abs(strip - paint).max(axis=2) <= 10).mean(axis=1)
    joined = _bands(support >= .90)
    if not joined or joined[0][0] > 1:
        return None
    depth = joined[0][1]
    if not .098 * height <= depth <= .255 * height:
        return None
    below = strip[depth + 2:depth + max(5, round(.105 * depth))]
    if not below.size or (abs(below - paint).max(axis=2) >= 15).mean() < .89:
        return None
    field = frame[round(.30 * depth):round(.52 * depth):2].astype(np.int16)
    present = (abs(field - paint).max(axis=2) <= 13).mean(axis=0) >= .61
    windows = [span for span in _bands(present) if span[0] < .33 * width and span[1] > .67 * width]
    if len(windows) != 1:
        return None
    first, last = windows[0]
    if first > .15 * width or last < .84 * width:
        return None
    gray = cv2.cvtColor(frame[:depth], cv2.COLOR_BGR2GRAY)
    tone = int(cv2.cvtColor(np.uint8([[paint]]), cv2.COLOR_BGR2GRAY)[0, 0])
    return gray, tone, first, last


def _clock_glyphs(tone_delta):
    height, width = tone_delta.shape
    if height < 8 or width < 8:
        return False
    for threshold in (49, 67):
        ink = np.uint8(tone_delta >= threshold)
        x, y, w, h = cv2.boundingRect(ink)
        if not .29 * height <= h <= .65 * height or not 1.58 * h <= w <= 4.35 * h:
            continue
        glyphs = ink[y:y + h, x:x + w]
        spans = _bands(glyphs.any(axis=0))
        if len(spans) not in (4, 5):
            continue
        colon_index = len(spans) - 3
        left, right = spans[colon_index]
        if right - left > .29 * h:
            continue
        _, _, parts, _ = cv2.connectedComponentsWithStats(glyphs[:, left:right], connectivity=8)
        if len(parts) != 3:
            continue
        first, second = sorted(parts[1:], key=lambda dot: dot[1])
        if (not .28 * h <= second[1] - first[1] <= .72 * h
                or not all(.063 * h <= dot[3] <= .29 * h and dot[4] >= 2 for dot in (first, second))):
            continue
        boxes = [cv2.boundingRect(glyphs[:, a:b]) for j, (a, b) in enumerate(spans) if j != colon_index]
        if all(.72 * h <= b[3] <= h and .83 * h <= b[1] + b[3] <= h and .12 * h <= b[2] <= .85 * h
               for b in boxes):
            return True
    return False


def _status_context(delta, left, right):
    depth = len(delta)
    bottom = round(.381 * depth)
    clock = delta[:bottom, left + round(.10 * depth):left + round(1.38 * depth)]
    if not _clock_glyphs(clock):
        return False
    edge = delta[:bottom, max(left, right - round(1.93 * depth)):right - round(.08 * depth)]
    _, _, shapes, _ = cv2.connectedComponentsWithStats(np.uint8(edge >= 60), connectivity=8)
    peers = [(x, y, w, h) for x, y, w, h, area in shapes[1:]
             if (.091 * depth <= h <= .234 * depth and .065 * depth <= w <= .28 * depth
                 and .23 * w * h <= area <= w * h and y > 0 and y + h < bottom)]
    return any(.08 * depth <= b[0] - a[0] <= .66 * depth
               and abs(a[1] + a[3] - b[1] - b[3]) <= .084 * depth
               for a in peers for b in peers)


def _arrow_marks(delta, left, right):
    depth = len(delta)
    x0, x1 = left + round(.075 * depth), min(right, left + round(1.22 * depth))
    y0 = round(.391 * depth)
    chip = delta[y0:round(.951 * depth), x0:x1]
    if min(chip.shape) < 12:
        return []
    steady = []
    for cut in (51, 74, 97):
        _, numbers, stats, _ = cv2.connectedComponentsWithStats(np.uint8(chip >= cut), connectivity=8)
        for i, (x, y, w, h, _) in enumerate(stats[1:], 1):
            if (not .118 * depth <= h <= .328 * depth or not .82 <= w / h <= 1.25
                    or not 0 < x < x + w < chip.shape[1] or not 0 < y < y + h < chip.shape[0]
                    or not .51 * depth <= y0 + y + h / 2 <= .80 * depth):
                continue
            if is_shaft_back_arrow(numbers[y:y + h, x:x + w] == i):
                steady.append((cut, (x0 + int(x), y0 + int(y), x0 + int(x + w), y0 + int(y + h))))
    return [box for cut, box in steady if any(
        peer_cut != cut and _overlap(box, peer) >= .79 for peer_cut, peer in steady)]


def _header_title(gray, tone, mark, right):
    _, top, end, bottom = mark
    size = bottom - top
    x0, x1 = end + round(.49 * size), min(right, end + round(11 * size))
    y0, y1 = max(0, top - round(.21 * size)), min(len(gray), bottom + round(.22 * size))
    crop = gray[y0:y1, x0:x1]
    run = _title_run(crop, tone, size, minimum=3, high=1.18)
    if run is None:
        return False
    a, b = run
    return .63 * size <= x0 + a - end <= 3.56 * size and 1.49 * size <= b - a <= 8.8 * size


def landscape_app_bar_actions(frame, regions):
    """Find an isolated shaft arrow with both status and title corroboration."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or width < 1.49 * height or height < 260:
        return []
    layer = _bar_surface(frame)
    if layer is None:
        return []
    gray, tone, left, right = layer
    if not .78 * width <= right - left <= width:
        return []
    markers = []
    residual = gray.astype(np.int16) - tone
    for sign in (1, -1):
        delta = sign * residual
        marks = _arrow_marks(delta, left, right)
        if not marks or not all(_overlap(marks[0], mark) >= .71 for mark in marks):
            continue
        mark = marks[len(marks) // 2]
        if _status_context(delta, left, right) and _header_title(gray, tone, mark, right):
            markers.append(mark)
    if len(markers) > 1:
        return []
    from screen_landscape_file_picker import file_picker_actions

    actions = file_picker_actions(frame, gray, tone, left, right)
    if markers:
        x0, y0, x1, y1 = markers[0]
        actions.insert(0, {"role": "back", "function": "返回", "preview_text": "返回", "box": list(markers[0]),
                           "click_point": [(x0 + x1) / 2, (y0 + y1) / 2]})
    return actions
