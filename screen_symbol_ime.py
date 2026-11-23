"""Recognize a borderless dark symbol keyboard with a bent Return key."""

import cv2
import numpy as np

from screen_caption_transcript import _bands
from screen_keyboard_footer_geometry import _dark_canvas_paint
from screen_landscape_search_field import _available
from screen_stroke_ime import _bent_arrow


def _space_key(frame, canvas):
    height, width = frame.shape[:2]
    x0, x1 = round(.20 * width), round(.81 * width)
    y0 = height - round(.27 * width)
    band = frame[y0:, x0:x1]
    blue, green, red = (cv2.subtract(plane, round(paint))
                        for plane, paint in zip(cv2.split(band), canvas, strict=True))
    low = cv2.min(cv2.min(blue, green), red)
    high = cv2.max(cv2.max(blue, green), red)
    ink = np.uint8((low >= 10) & (cv2.subtract(high, low) <= 23))
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keys = []
    for x, y, w, h, area in stats[1:]:
        l, t, r, b = map(int, (x0 + x, y0 + y, x0 + x + w, y0 + y + h))
        if (.255 * width <= w <= .510 * width and .044 * width <= h <= .108 * width
                and .41 * width <= x0 + x + w / 2 <= .651 * width
                and .013 * width <= height - b <= .170 * width and area >= .85 * w * h):
            keys.append((l, t, r, b))
    return keys[0] if len(keys) == 1 else None


def _surface_start(sample, start, stop, paint, pad, tolerance):
    start = max(pad, start)
    color = sample[start - pad:stop + pad]
    close = abs(color.astype(np.int16) - paint).max(axis=2) <= tolerance
    support = close.mean(axis=1)
    floor = np.min([part.mean(axis=1) for part in np.array_split(close, 5, axis=1)], axis=0)
    matches = np.flatnonzero((support[pad:-pad] >= .89) & (support[pad - 1:-pad - 1] <= .37)) + pad
    for y in matches:
        if (support[y - pad:y].mean() <= .34 and support[y:y + pad].mean() >= .87
                and np.min(floor[y:y + pad]) >= .74):
            return start - pad + int(y)
    return None


def _toolbar_edge(frame, canvas, space):
    height, width = frame.shape[:2]
    step = max(2, round(.012 * width))
    sample = frame[:, ::step]
    pad = max(3, round(.010 * width))
    body = _surface_start(sample, space[1] - round(.720 * width), space[1] - round(.349 * width), canvas, pad, 6)
    if body is None:
        return None
    zone = sample[body - round(.080 * width):body - round(.022 * width)]
    paint = np.median(zone, axis=(0, 1))
    separation = np.max(abs(paint - canvas))
    if not 7 <= separation <= 71 or np.mean(paint) > 113:
        return None
    start = max(round(.39 * height), body - round(.167 * width))
    distance = np.max(abs(np.median(sample[start - pad:start], axis=(0, 1)) - paint))
    if distance < 5:
        return None
    jitter = max(2.0, min(7., .45 * distance))
    toolbar = _surface_start(sample, start, body - round(.059 * width), paint, pad, jitter)
    if toolbar is None:
        return None
    strip = frame[toolbar:body, ::max(1, round(.006 * width))]
    brightness = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    hits = (brightness >= np.median(brightness) + 38).mean(axis=0) >= .045
    panels = [run.mean() for run in np.array_split(hits, 6)]
    return (toolbar, body) if min(panels) >= .10 and .16 <= hits.mean() <= .66 else None


def _glyph_positions(ink, reach):
    support = np.uint8(ink.any(axis=0)[None, :])
    coupled = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((1, reach), np.uint8))[0]
    return np.array([(left + right) / 2 for left, right in _bands(coupled)])


def _symbol_rows(frame, canvas, body, space):
    width = frame.shape[1]
    stop = space[1] - round(.030 * width)
    gray = cv2.cvtColor(frame[body:stop], cv2.COLOR_BGR2GRAY)
    base = int(cv2.cvtColor(np.uint8(canvas.reshape(1, 1, 3)), cv2.COLOR_BGR2GRAY)[0, 0])
    bright = gray >= base + 44
    cover = bright.mean(axis=1)
    ink_rows = np.uint8(((cover >= .020) & (cover <= .43))[:, None])
    kernel = np.ones((max(3, round(.022 * width)), 1), np.uint8)
    joined = cv2.morphologyEx(ink_rows, cv2.MORPH_CLOSE, kernel)[:, 0]
    bands = [(a, b) for a, b in _bands(joined) if .026 * width <= b - a <= .087 * width]
    if len(bands) != 3:
        return False
    first, second, third = [(a + b) / 2 for a, b in bands]
    pitch = second - first
    if (not .043 * width <= first <= .132 * width or not .112 * width <= pitch <= .203 * width
            or abs(third - second - pitch) > .11 * pitch or not .076 * width <= space[1] - body - third <= .215 * width):
        return False
    reach = max(3, round(.018 * width))
    heads = [_glyph_positions(bright[max(0, a - round(.011 * width)):min(len(bright), b + round(.011 * width))], reach)
             for a, b in bands]
    if tuple(map(len, heads)) != (10, 10, 9):
        return False
    first, second, third = heads
    steps = np.diff(first)
    median = np.median(steps)
    return (max(abs(steps - median)) <= .12 * median and max(abs(first - second)) <= .135 * median
            and max(abs(third[:-1] - (first[:8] + median / 2))) <= .18 * median
            and .78 * width <= first[-1] - first[0] <= .948 * width
            and 1.02 * median <= third[-1] - third[-2] <= 1.47 * median)


def _outlined_return(mask):
    height, width = mask.shape
    if not .20 <= mask.mean() <= .46 or not mask.any(axis=0).all():
        return False
    columns = mask.sum(axis=0)
    centers = (np.arange(height) @ mask) / columns
    middle = slice(round(.45 * width), round(.82 * width))
    axis = np.median(centers[middle])
    if (not .32 * height <= axis <= .67 * height or np.ptp(centers[middle]) >= .116 * height
            or not .07 * height <= np.median(columns[middle]) <= .29 * height):
        return False
    top, joint, bottom = round(axis - .18 * height), round(axis), round(axis + .19 * height)
    tip, head = mask[:, :round(.125 * width)], mask[:, round(.20 * width):round(.38 * width)]
    stem = mask[:, round(.91 * width):]
    return (tip[:top].mean() <= .12 and tip[bottom:].mean() <= .12
            and head[:top].mean() >= .10 and head[bottom:].mean() >= .10
            and stem[:joint].mean() >= .45 and stem[bottom:].mean() <= .085)


def _check_bend(frame, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    xpad, ypad = round(.16 * width), round(.16 * height)
    chip = frame[top + ypad:bottom - ypad, left + xpad:right - xpad]
    gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY).astype(np.int16)
    paint = np.median(chip[::3, ::3], axis=(0, 1))
    if np.ptp(paint) < 41 or (abs(chip - paint).max(axis=2) <= 18).mean() < .77:
        return False
    delta = gray - np.median(gray)
    lower, upper = np.percentile(delta, (2, 98))
    contrast = max(-lower, upper)
    if contrast < 59:
        return False
    signed = delta if upper >= -lower else -delta
    boxes = []
    for level in (.43, .60, .76):
        ink = np.uint8(signed >= level * contrast)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
        if count < 2:
            continue
        largest = int(np.argmax(stats[1:, 4])) + 1
        x, y, w, h, area = stats[largest]
        if (not .30 * width <= w <= .657 * width or not .22 * height <= h <= .51 * height
                or not 1.16 <= w / h <= 2.61 or area < .90 * cv2.countNonZero(ink)
                or abs(x + xpad + w / 2 - width / 2) > .125 * width
                or abs(y + ypad + h / 2 - height / 2) > .125 * height):
            continue
        glyph = labels[y:y + h, x:x + w] == largest
        if _bent_arrow(glyph) or _outlined_return(glyph):
            boxes.append((x, y, x + w, y + h))
    return len(boxes) >= 2 and np.ptp(boxes, axis=0).max() <= max(2, .048 * height)


def _return_key(frame, canvas, space):
    height, width = frame.shape[:2]
    x0, y0 = round(.793 * width), space[1] - round(.075 * width)
    crop = frame[y0:min(height, space[3] + round(.040 * width)), x0:]
    ink = np.uint8((np.ptp(crop, axis=2) >= 40) & (abs(crop - canvas).max(axis=2) >= 48))
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    keys = []
    for x, y, w, h, area in stats[1:]:
        left, top, right, bottom = map(int, (x0 + x, y0 + y, x0 + x + w, y0 + y + h))
        if (.071 * width <= w <= .155 * width and .075 * width <= h <= .162 * width
                and .79 <= w / h <= 1.30 and right >= .944 * width and .60 <= area / (w * h) <= .94
                and abs(top + bottom - space[1] - space[3]) <= .13 * width):
            key = (left, top, right, bottom)
            if _check_bend(frame, key):
                keys.append(key)
    return keys[0] if len(keys) == 1 else None


def symbol_ime_layout(frame):
    """Measure the docked symbol keyboard and its separately verified Return key."""
    height, width = frame.shape[:2]
    if width < 200 or not 1.52 * width <= height <= 2.83 * width:
        return None
    canvas = _dark_canvas_paint(frame)
    if canvas is None or (space := _space_key(frame, canvas)) is None:
        return None
    edges = _toolbar_edge(frame, canvas, space)
    if edges is None or not _symbol_rows(frame, canvas, edges[1], space):
        return None
    return {"keyboard": (0, edges[0], width, height), "return_key": _return_key(frame, canvas, space)}


def symbol_ime_actions(frame, regions, *, layout=None):
    """Return the bent-arrow action only when the key is visible and active."""
    if regions.get("layer") != "page":
        return []
    evidence = layout if layout is not None else symbol_ime_layout(frame)
    if evidence is None or (key := evidence["return_key"]) is None or not _available(key, regions):
        return []
    left, top, right, bottom = key
    return [{"role": "keyboard_enter", "function": "键盘回车", "box": list(key),
             "click_point": [(left + right) // 2, (top + bottom) // 2], "preview_text": "键盘回车"}]
