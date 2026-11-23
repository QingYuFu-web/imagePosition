"""Verify the back mark next to an independently read landscape settings title."""

import cv2
import numpy as np

from inventory_back_evidence import _chevrons, is_shaft_back_arrow


def _title_ink(tile):
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    low, middle, high = np.percentile(gray, (12, 50, 88))
    polarity = 1 if high - middle >= middle - low else -1
    peak = max(high - middle, middle - low)
    if peak < 54:
        return None
    signal = polarity * (gray.astype(np.int16) - middle)
    marks = tile[signal >= .69 * peak]
    paint = tile[abs(gray.astype(np.int16) - middle) <= 12]
    if len(paint) < 14 or len(marks) < 14:
        return None
    pigment = np.median(marks, axis=0) - np.median(paint, axis=0)
    return polarity, peak, pigment


def _bold_left_angle(mask):
    height, width = mask.shape
    count = mask.sum(axis=1)
    if not .58 <= width / height <= 1.09 or not .248 <= mask.mean() <= .49 or not count.all():
        return False
    line = (mask @ np.arange(width)) / (width * count)
    vertex = int(line.argmin())
    reach = max(1, round(.105 * height))
    first, last = np.median(line[:reach]), np.median(line[-reach:])
    elbow = line[vertex]
    if (not .37 * height <= vertex <= .61 * height or not .73 <= min(first, last) <= max(first, last) <= 1
            or not .07 <= elbow <= .31 or min(first, last) - elbow < .46):
        return False
    model = np.interp(np.arange(height), (0, vertex, height - 1), (first, elbow, last))
    return np.quantile(abs(line - model), .91) <= .136 and np.max(abs(np.diff(line))) <= .22


def _left_triangle(mask):
    height, width = mask.shape
    if not .65 <= width / height <= 1.25 or not .42 <= mask.mean() <= .71:
        return False
    profile = mask.mean(axis=0)
    edge = max(1, round(.16 * width))
    return (np.median(profile[:edge]) <= .31 and np.median(profile[-edge:]) >= .73
            and np.mean(np.diff(profile) >= -.13) >= .83)


def _same_pigment(tile, mask, paint, title):
    key = np.median(tile[mask], axis=0) - paint
    size = np.linalg.norm(key)
    authority = np.linalg.norm(title)
    return (.49 * authority <= size <= 1.6 * authority
            and float(key @ title) >= .88 * size * authority)


def _measure(tile, mid, title_height, polarity, peak, title_pigment):
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    middle = float(np.median(gray))
    paper = abs(gray.astype(np.int16) - middle) <= 15
    if paper.mean() < .66:
        return []
    paint = np.median(tile[paper], axis=0)
    signal = polarity * (gray.astype(np.int16) - middle)
    votes = []
    for ratio in (.36, .47, .55, .64):
        _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(signal >= max(33, ratio * peak)))
        angles, tails = [], []
        for i, (x, y, w, h, _) in enumerate(stats[1:], 1):
            if (not 0 < x < x + w < gray.shape[1] or not 0 < y < y + h < gray.shape[0]
                    or abs(y + h / 2 - mid) > .19 * title_height or min(w, h) < 5):
                continue
            glyph = ids[y:y + h, x:x + w] == i
            box = tuple(map(int, (x, y, x + w, y + h)))
            if .50 * title_height <= h <= 1.25 * title_height:
                if (is_shaft_back_arrow(glyph) or _bold_left_angle(glyph)
                        or any(_chevrons(glyph, h / 2, h, h))):
                    if _same_pigment(tile[y:y + h, x:x + w], glyph, paint, title_pigment):
                        angles.append(box)
            elif .21 * title_height <= h < .50 * title_height and _left_triangle(glyph):
                if _same_pigment(tile[y:y + h, x:x + w], glyph, paint, title_pigment):
                    tails.append(box)
        if len(angles) != 1:
            continue
        a, b, r, t = angles[0]
        span = r - a
        adjacent = [tail for tail in tails if a + .56 * span <= tail[0] <= r + .09 * span
                    and .06 * span <= tail[2] - r <= .39 * span
                    and abs((tail[1] + tail[3] - b - t) / 2) <= .09 * title_height]
        votes.append((angles[0], adjacent[0] if len(adjacent) == 1 else None))
    return votes


def game_settings_back(frame, settings_label_box, navigation_width):
    """Return the actual left arrow in a verified settings header if one is visible."""
    height, width = frame.shape[:2]
    left, top, right, bottom = map(int, settings_label_box)
    title_height = bottom - top
    if (width < 1.42 * height or not .078 * width <= navigation_width <= .36 * width
            or not 0 <= top < bottom <= .16 * height or not .023 * height <= title_height <= .102 * height
            or not .24 * navigation_width <= left < right <= navigation_width):
        return None
    title = _title_ink(frame[top:bottom, left:right])
    if title is None:
        return None
    x0, x1 = max(0, left - round(2.35 * title_height)), left - round(.13 * title_height)
    midpoint = (top + bottom) / 2
    y0, y1 = max(0, round(midpoint - .91 * title_height)), min(height, round(midpoint + .91 * title_height))
    if min(x1 - x0, y1 - y0) < .69 * title_height:
        return None
    measured = _measure(frame[y0:y1, x0:x1], midpoint - y0, title_height, *title)
    if len(measured) < 2:
        return None
    boxes = [box for box, _ in measured]
    if np.ptp(boxes, axis=0).max() > max(2, .08 * title_height):
        return None
    a, b, r, t = boxes[len(boxes) // 2]
    tails = [tail for _, tail in measured if tail is not None]
    if len(tails) >= 2 and np.ptp(tails, axis=0).max() <= max(2, .08 * title_height):
        r = max(r, tails[len(tails) // 2][2])
    mark = (x0 + a, y0 + b, x0 + r, y0 + t)
    gap = left - mark[2]
    if not .15 * title_height <= gap <= .81 * title_height:
        return None
    return {"role": "back", "function": "返回", "preview_text": "返回", "box": list(mark),
            "click_point": [(mark[0] + mark[2]) // 2, (mark[1] + mark[3]) // 2]}
