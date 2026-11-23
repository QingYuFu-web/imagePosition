"""Recognize a checkmark on the action key of a docked phone keyboard."""

import cv2
import numpy as np

from screen_keyboard_footer_geometry import keyboard_footer_bounds
from screen_landscape_search_field import _available


def _check_stroke(mask):
    height, width = mask.shape
    ink_in_column = mask.sum(axis=0)
    if not ink_in_column.all() or not .13 <= mask.mean() <= .37:
        return False
    positions = np.arange(width)
    line = (np.arange(height) @ mask) / (height * ink_in_column)
    elbow = int(line.argmax())
    edge = max(1, round(.08 * width))
    left, joint, right = np.median(line[:edge]), line[elbow], np.median(line[-edge:])
    if not (.21 * width <= elbow <= .47 * width and .38 <= left <= .72
            and .73 <= joint <= .97 and 0 <= right <= .23):
        return False
    expected = np.interp(positions, (0, elbow, width - 1), (left, joint, right))
    return np.quantile(abs(line - expected), .90) <= .10


def _stable_mark(gray, minimum_contrast):
    height, width = gray.shape
    margin_x, margin_y = round(.13 * width), round(.14 * height)
    inner = gray[margin_y:height - margin_y, margin_x:width - margin_x].astype(np.int16)
    paint = np.median(inner)
    if (abs(inner - paint) <= 11).mean() < .79:
        return False
    delta = inner - paint
    low, high = np.percentile(delta, (2, 98))
    signed = -delta if -low > high else delta
    peak = max(-low, high)
    if peak < minimum_contrast:
        return False
    verified = []
    for level in (.41, .59, .76):
        ink = np.uint8(signed >= level * peak)
        ink_area = cv2.countNonZero(ink)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
        for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if (not .23 * height <= h <= .55 * height or not .21 * width <= w <= .49 * width
                    or not 1.02 <= w / h <= 1.82 or area < .89 * ink_area
                    or abs(x + margin_x + w / 2 - width / 2) > .12 * width
                    or abs(y + margin_y + h / 2 - height / 2) > .13 * height):
                continue
            if _check_stroke(labels[y:y + h, x:x + w] == index):
                verified.append((x, y, x + w, y + h))
    return len(verified) >= 2 and np.ptp(verified, axis=0).max() <= max(2, .045 * height)


def _caps_with_check(gray, mask, x0, y0, width, height, minimum_contrast):
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask), connectivity=8)
    keys = []
    for x, y, w, h, area in stats[1:]:
        left, top, right, bottom = map(int, (x0 + x, y0 + y, x0 + x + w, y0 + y + h))
        if (not .079 * width <= h <= .144 * width or not .097 * width <= w <= .181 * width
                or not 1.04 * h <= w <= 1.88 * h or not .67 * w * h <= area <= .981 * w * h
                or not .805 * width <= left <= .900 * width or not .957 * width <= right < width
                or not .010 * width <= height - bottom <= .264 * width):
            continue
        if _stable_mark(gray[y:y + h, x:x + w], minimum_contrast):
            keys.append((left, top, right, bottom))
    return keys


def _right_caps(frame):
    height, width = frame.shape[:2]
    x0 = round(.76 * width)
    y0 = max(round(.63 * height), height - round(.51 * width))
    crop = frame[y0:height - round(.007 * width), x0:]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blue, green, red = cv2.split(crop)
    chroma = cv2.subtract(cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red))
    if (chroma[::4, ::4] >= 64).mean() >= .055:
        keys = _caps_with_check(gray, chroma >= 64, x0, y0, width, height, minimum_contrast=44)
        if keys:
            return keys
    # Neutral action keys carry the same high-contrast icon as their row peers.
    for level in (245, 250):
        if (gray[::4, ::4] >= level).mean() >= .09:
            keys = _caps_with_check(gray, gray >= level, x0, y0, width, height, minimum_contrast=85)
            if keys:
                return keys
    return []


def _adjacent_spacebar(frame, key):
    left, top, _, bottom = key
    height, width = frame.shape[:2]
    depth = bottom - top
    x0, x1 = round(.15 * width), left - round(.005 * width)
    y0, y1 = max(0, top - round(.28 * depth)), min(height, bottom + round(.30 * depth))
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    edges = frame[top:bottom, -max(1, round(.004 * width)):]
    canvas = round(np.median(cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)))
    levels = (canvas + 15, canvas + 23, canvas + 30) if canvas < 130 else (237, 246, 252, 227)
    for level in levels:
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(gray >= level))
        for x, y, w, h, area in stats[1:]:
            if (.205 * width <= w <= .60 * width and .73 * depth <= h <= 1.30 * depth
                    and .80 * w * h <= area and 0 < x < x + w < gray.shape[1]
                    and .28 * width <= x0 + x + w / 2 <= .71 * width
                    and abs(y0 + y + h / 2 - (top + bottom) / 2) <= .22 * depth):
                return True
    return False


def ime_completion_actions(frame, regions, *, get_keyboard=None):
    """Return a keycap click after its mark, spacebar and keyboard agree."""
    height, width = frame.shape[:2]
    if regions.get("layer") != "page" or width < 160 or not 1.55 * width <= height <= 2.85 * width:
        return []
    keys = _right_caps(frame)
    if len(keys) != 1 or not _available(keys[0], regions):
        return []
    key = keys[0]
    if not _adjacent_spacebar(frame, key):
        return []
    keyboard = get_keyboard() if get_keyboard is not None else keyboard_footer_bounds(frame)
    left, top, right, bottom = key
    if keyboard is None or not .29 * width <= top - keyboard[1] <= .80 * width:
        return []
    return [{"role": "keyboard_done", "function": "完成输入", "box": list(key),
             "click_point": [(left + right) // 2, (top + bottom) // 2], "preview_text": "完成输入"}]
