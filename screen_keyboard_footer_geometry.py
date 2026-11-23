"""Find a docked phone keyboard from its painted keys and footer boundary."""

from statistics import median

import cv2
import numpy as np


def _key_bands(keys, width):
    rows = []
    for top in sorted({box[1] for box in keys}):
        if any(abs(top - row[0]) <= .012 * width for row in rows):
            continue
        row = sorted(box for box in keys if abs(box[1] - top) <= .009 * width)
        if len(row) < 4:
            continue
        sizes = np.asarray([(box[2], box[3]) for box in row])
        key_width, key_height = np.median(sizes, axis=0)
        if np.any(abs(sizes - (key_width, key_height)) > .16 * np.array([key_width, key_height])):
            continue
        centers = np.asarray([x + w / 2 for x, _, w, _ in row])
        steps = np.diff(centers)
        step = median(steps)
        if max(abs(steps - step)) <= .10 * step and 1.034 * key_width <= step <= 1.38 * key_width:
            rows.append((median(box[1] for box in row), step, key_width, key_height, centers))
    return rows


def _row_triplets(rows):
    for first, second, third in zip(rows, rows[1:], rows[2:]):
        pitch, key_width, key_height = first[1:4]
        rise = second[0] - first[0]
        if (not 1.03 * key_height <= rise <= 1.48 * key_height
                or abs(third[0] - second[0] - rise) > .09 * rise):
            continue
        if all(abs(row[1] - pitch) <= .08 * pitch and abs(row[2] - key_width) <= .11 * key_width
               and abs(row[3] - key_height) <= .12 * key_height for row in (second, third)):
            yield first, second, third, rise


def _special_key(color, first, third, rise):
    y = round(third[0] + .35 * rise)
    end = round(third[0] + 1.55 * rise)
    x = round(first[4][-1])
    patch = color[y:end, max(0, x - 3):x + 4].astype(np.int16)
    if patch.size == 0:
        return False
    paint = np.median(patch, axis=(0, 1))
    return np.ptp(paint) >= 35 and (np.max(abs(patch - paint), axis=2) <= 18).mean() >= .78


def _upper_qwerty_row(rows, first, rise):
    top, pitch, width, height, centers = first
    aligned = [row[0] for row in rows if len(row[4]) == len(centers)
               and .64 * rise <= top - row[0] <= 1.16 * rise
               and abs(row[1] - pitch) <= .076 * pitch and abs(row[2] - width) <= .14 * width
               and .72 * height <= row[3] <= 1.16 * height
               and max(abs(row[4] - centers)) <= .12 * pitch]
    return aligned[0] if len(aligned) == 1 else top


def _keyboard_keys(boxes, color, t9):
    width = color.shape[1]
    span = (.145, .210) if t9 else (.056, .112)
    keys = [box for box in boxes if span[0] * width <= box[2] <= span[1] * width]
    rows = _key_bands(keys, width)
    for first, second, third, rise in _row_triplets(rows):
        counts = tuple(len(row[4]) for row in (first, second, third))
        if counts != ((5, 5, 4) if t9 else (10, 9, 7)):
            continue
        pitch, key_width, key_height, centers = first[1:]
        if centers[0] > .122 * width or centers[-1] < .878 * width:
            continue
        if t9:
            if (max(abs(second[4] - centers)) > .12 * pitch
                    or max(abs(third[4] - centers[:-1])) > .12 * pitch
                    or not _special_key(color, first, third, rise)):
                continue
        elif (max(abs(second[4] - (centers[:-1] + pitch / 2))) > .17 * pitch
              or max(abs(third[4] - second[4][1:-1])) > .17 * pitch):
            continue
        for x, y, w, h in boxes:
            if (abs(y - third[0] - rise) <= .18 * rise and .80 * key_height <= h <= 1.18 * key_height
                    and 1.83 * key_width <= w <= .61 * width and .27 * width <= x + w / 2 <= .69 * width):
                return first[0] if t9 else _upper_qwerty_row(rows, first, rise)
    return None


def _footer_edge(gray, first_key, t9):
    height, width = gray.shape
    left, right = round(.11 * width), round(.89 * width)
    section = gray[:, left:right]
    profile = np.median(section, axis=1)
    pad = max(3, round(.017 * width))
    start = max(2 * pad, round(first_key - (.31 if t9 else .22) * width))
    stop = round(first_key - (.13 if t9 else .070) * width)
    jumps = np.flatnonzero(abs(profile[2:] - profile[:-2]) >= 7) + 1
    matches = []
    for y in jumps:
        if y < start or y > stop or y + 2 * pad >= height:
            continue
        before = np.median(section[y - pad:y], axis=0)
        after = np.median(section[y:y + pad], axis=0)
        settled = np.median(section[y + pad:y + 2 * pad], axis=0)
        delta = after - before
        direction = np.sign(np.median(delta))
        if ((direction * delta >= 10).mean() < .77
                or (direction * (settled - before) >= 10).mean() < .77):
            continue
        matches.append((np.median(abs(delta)), int(y)))
    return max(matches, default=(0, None), key=lambda candidate: candidate[0])[1]


def _color_footer_edge(color, first_key, t9):
    edges = sorted(edge for plane in cv2.split(color)
                   if (edge := _footer_edge(plane, first_key, t9)) is not None)
    tolerance = max(2, round(.007 * color.shape[1]))
    return next((a for a, b in zip(edges, edges[1:]) if b - a <= tolerance), None)


def _key_surfaces(gray, start, level):
    width = gray.shape[1]
    _, _, components, _ = cv2.connectedComponentsWithStats(np.uint8(gray[start:] >= level))
    return [(int(x), int(y) + start, int(w), int(h)) for x, y, w, h, area in components[1:]
            if .053 * width <= w <= .61 * width and .065 * width <= h <= .15 * width and area >= .72 * w * h]


def _dark_canvas_paint(color):
    height, width = color.shape[:2]
    span = max(1, round(.005 * width))
    sides = color[round(.66 * height):round(.985 * height):3]
    sides = (sides[:, :span], sides[:, -span:])
    tones = [np.median(side, axis=(0, 1)) for side in sides]
    paint = np.mean(tones, axis=0)
    if (paint.max() <= 117 and np.ptp(paint) <= 27 and max(abs(tones[0] - tones[1])) <= 8
            and all((abs(side.astype(np.int16) - paint).max(axis=2) <= 9).mean() >= .86 for side in sides)):
        return paint
    return None


def _dark_canvas_edge(color, first_key, paint):
    width = color.shape[1]
    pad = max(4, round(.019 * width))
    start = max(0, round(first_key - .30 * width))
    stop = max(start + 1, round(first_key - .055 * width))
    match = abs(color[start:stop + pad, ::3].astype(np.int16) - paint).max(axis=2) <= 10
    sections = [lane.mean(axis=1) for lane in np.array_split(match, 5, axis=1)]
    floor = np.min(sections, axis=0)
    coverage = match.mean(axis=1)
    for y in range(pad, stop - start):
        if (floor[y] >= .65 and coverage[y - pad:y].mean() <= .49
                and coverage[y:y + pad].mean() >= .86
                and (floor[y:y + pad] >= .65).sum() >= .78 * pad):
            return start + y
    return None


def _dark_keyboard_top(color, gray, start):
    paint = _dark_canvas_paint(color)
    if paint is None:
        return None
    canvas = int(cv2.cvtColor(np.uint8(paint.reshape(1, 1, 3)), cv2.COLOR_BGR2GRAY)[0, 0])
    for delta in (9, 18, 27):
        boxes = _key_surfaces(gray, start, canvas + delta)
        first = _keyboard_keys(boxes, color, False)
        if first is not None and (top := _dark_canvas_edge(color, first, paint)) is not None:
            return top
    return None


def keyboard_footer_bounds(frame):
    """Return the docked keyboard and bottom inset in original xyxy coordinates."""
    height, width = frame.shape[:2]
    if width < 160 or not 1.55 * width <= height <= 2.85 * width:
        return None
    down = min(1, 480 / width)
    work = cv2.resize(frame[:, :, :3], (round(width * down), round(height * down)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    work_height, work_width = gray.shape
    start = round(.50 * work_height)
    for level in (237, 227, 246, 215, 252):
        boxes = _key_surfaces(gray, start, level)
        for t9 in (False, True):
            first = _keyboard_keys(boxes, work, t9)
            if first is None:
                continue
            top = _footer_edge(gray, first, t9)
            if top is None:
                # A tinted keyboard edge can be stronger in color channels than in luma.
                top = _color_footer_edge(work, first, t9)
            if top is not None:
                return (0, round(top * height / work_height), width, height)
    if (top := _dark_keyboard_top(work, gray, start)) is not None:
        return (0, round(top * height / work_height), width, height)
    return None
