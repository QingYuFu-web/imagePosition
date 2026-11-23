"""Locate a ruled Chinese stroke keyboard and verify its bent Return glyph."""

import cv2
import numpy as np

from screen_caption_transcript import _bands
from screen_landscape_search_field import _available


def _row_rails(frame):
    height, width = frame.shape[:2]
    start = round(.40 * height)
    inset = round(.025 * width)
    step = max(2, round(.020 * width))
    sample = cv2.cvtColor(frame[start:, inset:width - inset:step], cv2.COLOR_BGR2GRAY).astype(np.int16)
    pad = max(3, round(.008 * width))
    middle = sample[pad:-pad]
    before, after = middle - sample[:-2 * pad], middle - sample[2 * pad:]
    stable = ((before >= 19) & (after >= 19)) | ((before <= -19) & (after <= -19))
    return [start + pad + (a + b - 1) // 2 for a, b in _bands(stable.mean(axis=1) >= .76)
            if b - a <= .022 * width]


def _rail_columns(frame, top, bottom, paint, *, through_right=True):
    width = frame.shape[1]
    depth = bottom - top
    start, stop = top + round(.20 * depth), bottom - round(.20 * depth)
    strip = frame[start:stop:max(1, depth // 18)].astype(np.int16)
    support = (abs(strip - paint).max(axis=2) <= 24).mean(axis=0)
    bands = _bands(support >= .77)
    if not bands or bands[0][0] > .005 * width or (through_right and bands[-1][1] < .995 * width):
        return None
    return [(a + b) / (2 * width) for a, b in bands[1:] if b < .995 * width and b - a <= .023 * width]


def _even_columns(observed, count):
    return observed is not None and len(observed) == count - 1 and all(
        abs(x - i / count) <= .013 for i, x in enumerate(observed, 1))


def _grid(frame, rails):
    height, width = frame.shape[:2]
    top, first, second, third, bottom = rails
    rows = np.diff(rails) / width
    if (not .11 <= min(rows) <= max(rows) <= .213 or abs(rows[1] - rows[2]) > .025
            or not .023 * width <= height - bottom <= .24 * width):
        return None
    paint = np.median(frame[np.asarray(rails), ::max(2, round(.025 * width))], axis=(0, 1))
    digits = _rail_columns(frame, top, first, paint)
    upper = _rail_columns(frame, first, second, paint)
    lower = _rail_columns(frame, second, third, paint)
    # Compression can mix the edge rail with the rightmost colored action key.
    bottom_rules = _rail_columns(frame, third, bottom, paint, through_right=False)
    if not (_even_columns(digits, 10) and _even_columns(upper, 5) and _even_columns(lower, 5)
            and bottom_rules is not None and len(bottom_rules) == 4):
        return None
    lengths = np.diff([0., *bottom_rules, 1.])
    if (abs(bottom_rules[-1] - lower[-1]) > .014 or not .14 <= lengths[-1] <= .237
            or not .27 <= lengths[2] <= .49 or max(lengths[0], lengths[1], lengths[3]) >= .26):
        return None
    return round(bottom_rules[-1] * width), (upper, lower)


def _stroke_type(mask):
    height, width = mask.shape
    area = mask.mean()
    if width >= 4.7 * height and area >= .69:
        return "horizontal"
    if height >= 4.7 * width and area >= .69:
        return "vertical"
    if min(height, width) < 6 or not .10 <= area <= .48 or not mask.any(axis=0).all():
        return None
    line = (np.arange(height) @ mask) / (height * mask.sum(axis=0))
    end = max(2, round(.18 * width))
    motion = np.median(line[-end:]) - np.median(line[:end])
    if .58 <= width / height <= 1.65 and abs(motion) >= .47:
        expected = np.linspace(line[0], line[-1], width)
        if np.quantile(abs(line - expected), .88) <= .25:
            return "rising" if motion < 0 else "falling"
    if 1.58 <= width / height <= 4.45:
        body = mask[:, :round(.68 * width)].nonzero()[0]
        turn = mask[:, round(.77 * width):].nonzero()[0]
        if body.size and turn.size and np.quantile(body, .96) <= .39 * height and np.quantile(turn, .88) >= .68 * height:
            return "hook"
    return None


def _printed_stroke(chip):
    height, width = chip.shape[:2]
    blue, green, red = cv2.split(chip)
    delta = cv2.subtract(cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red))
    if delta.max() < 45:
        return None
    paint = np.median(chip[::3, ::3], axis=(0, 1))
    contrast = abs(chip.astype(np.int16) - paint).max(axis=2)
    ink = np.uint8((delta >= 45) & (contrast >= 59))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    if count < 2:
        return None
    largest = int(np.argmax(stats[1:, 4])) + 1
    x, y, w, h, area = stats[largest]
    if (area < max(8, .83 * cv2.countNonZero(ink)) or not .030 * width <= w <= .77 * width
            or not .030 * height <= h <= .79 * height or not 0 < x < x + w < width
            or not 0 < y < y + h < height):
        return None
    glyph = labels[y:y + h, x:x + w] == largest
    kind = _stroke_type(glyph)
    return (kind, np.median(chip[y:y + h, x:x + w][glyph], axis=0)) if kind is not None else None


def _stroke_palette(frame, rows, columns):
    width = frame.shape[1]
    strokes = {}
    for (top, bottom), grid in zip(zip(rows, rows[1:]), columns, strict=True):
        edges = [round(x * width) for x in (0., *grid, 1.)]
        for start, stop in zip(edges, edges[1:]):
            xpad, ypad = round(.19 * (stop - start)), round(.17 * (bottom - top))
            proof = _printed_stroke(frame[top + ypad:bottom - ypad, start + xpad:stop - xpad])
            if proof is None:
                continue
            kind, paint = proof
            if kind in strokes:
                return False
            strokes[kind] = paint
    return len(strokes) == 5 and np.ptp(list(strokes.values()), axis=0).max() <= 39


def _suggestion_edge(frame, rail):
    width = frame.shape[1]
    left, right = round(.27 * width), round(.96 * width)
    start, stop = rail - round(.177 * width), rail - round(.044 * width)
    pad = max(3, round(.007 * width))
    strip = cv2.cvtColor(frame[start - pad:stop + 2 * pad, left:right:3], cv2.COLOR_BGR2GRAY).astype(np.int16)
    profile = np.median(strip, axis=1)
    strongest = None
    for offset in np.flatnonzero(abs(profile[2:] - profile[:-2]) >= 20) + 1:
        if not pad <= offset <= stop - start + pad:
            continue
        before = np.median(strip[offset - pad:offset], axis=0)
        after = np.median(strip[offset:offset + pad], axis=0)
        delta = after - before
        direction = np.sign(np.median(delta))
        if (direction * delta >= 35).mean() < .81 or np.median(abs(delta)) < 49:
            continue
        weight = abs(profile[offset + 1] - profile[offset - 1])
        if strongest is None or weight > strongest[0]:
            strongest = (weight, start - pad + int(offset))
    if strongest is None:
        return None
    top = strongest[1]
    painted = frame[top + pad:rail - pad, ::max(1, round(.003 * width))]
    ink = np.ptp(painted, axis=2) >= 41
    visible = (ink.mean(axis=0) >= .075).mean()
    return top if .08 <= visible <= .69 else None


def _bent_arrow(mask):
    height, width = mask.shape
    if not .24 <= mask.mean() <= .57:
        return False
    tip = mask[:, round(.026 * width):round(.15 * width)]
    head = mask[:, round(.27 * width):round(.43 * width)]
    shaft = mask[:, round(.49 * width):round(.72 * width)]
    stem = mask[:, round(.81 * width):round(.962 * width)]
    if min(tip.shape[1], head.shape[1], shaft.shape[1], stem.shape[1]) == 0:
        return False
    return (.037 <= tip.mean() <= .24 and head.mean() >= tip.mean() + .19
            and .10 <= shaft.mean() <= .34 and stem.mean() >= .59
            and shaft[:round(.47 * height)].mean() <= .055
            and shaft[round(.56 * height):round(.84 * height)].mean() >= .49
            and stem[:round(.41 * height)].mean() >= .62
            and head[:round(.31 * height)].mean() <= .08)


def _active_return(frame, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    xpad, ypad = round(.15 * width), round(.15 * height)
    chip = frame[top + ypad:bottom - ypad, left + xpad:right - xpad]
    paint = np.median(chip[::3, ::3], axis=(0, 1))
    if np.ptp(paint) < 47 or (abs(chip.astype(np.int16) - paint).max(axis=2) <= 17).mean() < .74:
        return False
    gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY).astype(np.int16)
    delta = gray - np.median(gray)
    lo, hi = np.percentile(delta, (2, 98))
    peak = max(hi, -lo)
    if peak < 57:
        return False
    signed = delta if hi >= -lo else -delta
    marks = []
    for factor in (.43, .62, .78):
        ink = np.uint8(signed >= factor * peak)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
        if count < 2:
            continue
        largest = int(np.argmax(stats[1:, 4])) + 1
        x, y, w, h, area = stats[largest]
        if (.23 * width <= w <= .61 * width and .23 * height <= h <= .56 * height and 1.07 <= w / h <= 2.55
                and area >= .90 * cv2.countNonZero(ink) and abs(x + xpad + w / 2 - width / 2) <= .12 * width
                and abs(y + ypad + h / 2 - height / 2) <= .13 * height
                and _bent_arrow(labels[y:y + h, x:x + w] == largest)):
            marks.append((x, y, x + w, y + h))
    return len(marks) >= 2 and np.ptp(marks, axis=0).max() <= max(2, .041 * height)


def stroke_ime_layout(frame):
    """Measure the full docked keyboard surface and optional proven Return key."""
    height, width = frame.shape[:2]
    if width < 200 or not 1.52 * width <= height <= 2.78 * width:
        return None
    rails = _row_rails(frame)
    for group in zip(rails, rails[1:], rails[2:], rails[3:], rails[4:]):
        matrix = _grid(frame, group)
        if matrix is None:
            continue
        left, columns = matrix
        top, first, second, third, bottom = group
        keyboard_top = _suggestion_edge(frame, top)
        if keyboard_top is None or not _stroke_palette(frame, (first, second, third), columns):
            continue
        key = (left, third, width, bottom)
        return {"keyboard": (0, keyboard_top, width, height),
                "return_key": key if _active_return(frame, key) else None}
    return None


def stroke_ime_actions(frame, regions, *, layout=None):
    """Emit a native Return action when the key lies in the visible grid."""
    if regions.get("layer") != "page":
        return []
    result = layout if layout is not None else stroke_ime_layout(frame)
    if result is None or (key := result["return_key"]) is None or not _available(key, regions):
        return []
    left, top, right, bottom = key
    return [{"role": "keyboard_enter", "function": "键盘回车", "box": list(key),
             "click_point": [(left + right) // 2, (top + bottom) // 2], "preview_text": "键盘回车"}]
