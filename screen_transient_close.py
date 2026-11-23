"""Find a diagonal dismiss mark on its own tip or notification banner."""

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_dialog_close import _cross_footprint


def _mark(frame, bounds, scale):
    left, top, right, bottom = bounds
    tile = frame[top:bottom, left:right]
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).astype(np.int16)
    paper = int(np.median(gray))
    if (abs(gray - paper) <= 12).mean() < .82:
        return None
    height, width = gray.shape
    matches = []
    for sign in (1, -1):
        seen = set()
        votes = []
        for level in (41, 63, 83, 96, 112):
            ink = np.uint8(sign * (gray - paper) >= level)
            _, labels, components, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
            large = [(i, x, y, w, h, count) for i, (x, y, w, h, count) in enumerate(components[1:], 1)
                     if count >= .0025 * width * height]
            if len(large) != 1:
                continue
            i, x, y, w, h, size = large[0]
            if not (.021 * scale <= min(w, h) <= max(w, h) <= .061 * scale
                    and .77 <= w / h <= 1.30 and .21 * width <= w <= .78 * width
                    and .21 * height <= h <= .78 * height and .14 * w * h <= size <= .55 * w * h
                    and abs(x + w / 2 - width / 2) <= .14 * width
                    and abs(y + h / 2 - height / 2) <= .14 * height
                    and 0 < x < x + w < width and 0 < y < y + h < height):
                continue
            mark = labels[y:y + h, x:x + w] == i
            key = (mark.shape, mark.tobytes())
            if key not in seen and _cross_footprint(mark):
                seen.add(key)
                votes.append((left + x, top + y, left + x + w, top + y + h))
        if len(votes) >= 2 and np.ptp(np.asarray(votes), axis=0).max() <= max(2, .005 * scale):
            matches.append((votes[len(votes) // 2], sign))
    if len(matches) != 1:
        return None
    return (*matches[0], np.median(tile[::2, ::2], axis=(0, 1)).astype(np.int16))


def _trimmed_span(mask, origin, xy):
    x, y = xy
    height, width = mask.shape
    # Printed text can cut across the paint, so bridge only short runs within
    # the surface while keeping the taller outer margins distinct.
    area = mask[:, round(.10 * width):round(.92 * width)]
    active = np.uint8(area.mean(axis=1) >= .61)
    bridge = max(3, (round(.059 * width) | 1))
    solid = cv2.morphologyEx(active[:, None], cv2.MORPH_CLOSE, np.ones((bridge, 1), np.uint8)).ravel()
    spans = [(a, b) for a, b in _bands(solid) if a <= y - origin < b]
    if len(spans) != 1:
        return None
    a, b = spans[0]
    h = b - a
    if not (.088 * width <= h <= .265 * width and 3 < a < b < height - 3):
        return None
    outer = (area[max(0, a - round(.018 * width)):a - 2], area[b + 2:b + round(.018 * width)])
    if any(not view.size or view.mean() > .32 for view in outer):
        return None
    votes = mask[a + round(.19 * h):b - round(.19 * h)].mean(axis=0)
    points = np.flatnonzero(votes >= .55)
    if len(points) < .70 * width:
        return None
    first, last = int(points[0]), int(points[-1]) + 1
    if not (first <= x < last and .82 * width <= last - first <= width
            and max(first, width - last) <= .124 * width and abs(first + last - width) <= .063 * width):
        return None
    if not .21 * h <= last - x <= .81 * h or not .22 * h <= y - (a + origin) <= .74 * h:
        return None
    return first, a + origin, last, b + origin


def _surface(frame, center, paint, tolerance):
    height, width = frame.shape[:2]
    x, y = center
    reach = round(.33 * width)
    start, stop = max(0, y - reach), min(height, y + reach)
    minimum = tuple(np.maximum(paint - tolerance, 0).astype(float))
    maximum = tuple(np.minimum(paint + tolerance, 255).astype(float))
    mask = cv2.inRange(frame[start:stop], minimum, maximum) != 0
    return _trimmed_span(mask, start, center), mask, start


def _text_witness(gray, bounds, cross, sign):
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    x0 = left + round(.032 * width)
    x1 = min(cross[0] - round(.016 * width), right - round(.10 * width))
    y0, y1 = top + round(.05 * height), bottom - round(.07 * height)
    text = gray[y0:y1, x0:x1].astype(np.int16)
    paper = np.median(text)
    ink = sign * (text - paper) >= 43
    lines = []
    for a, b in _bands(ink.mean(axis=1) >= .028):
        if not .115 * height <= b - a <= .58 * height:
            continue
        x, _, w, _ = cv2.boundingRect(np.uint8(ink[a:b]))
        if not .025 * width <= w or x > .044 * width:
            continue
        _, _, parts, _ = cv2.connectedComponentsWithStats(np.uint8(ink[a:b]), connectivity=8)
        chars = sum(.24 * (b - a) <= h <= b - a and .11 * (b - a) <= w <= 2.4 * (b - a)
                    and size >= 4 for _, _, w, h, size in parts[1:])
        if chars >= 3:
            lines.append((y0 + a, y0 + b, w))
    return lines


def _pointer(mask, start, bounds):
    left, _, right, bottom = bounds
    frame_width = mask.shape[1]
    first = bottom - start
    last = min(len(mask), first + round(.076 * frame_width))
    middle = mask[first:last, left + round(.09 * frame_width):right - round(.09 * frame_width)]
    if not middle.size:
        return False
    widths = middle.sum(axis=1)
    support = np.flatnonzero(widths >= .013 * frame_width)
    if not (.016 * frame_width <= len(support) <= .066 * frame_width and support[0] <= 2
            and .031 * frame_width <= widths[:3].max() <= .14 * frame_width):
        return False
    points = [np.flatnonzero(middle[y]) for y in support]
    return (max(np.ptp(row) + 1 - len(row) for row in points) <= .006 * frame_width
            and max(row.mean() for row in points) - min(row.mean() for row in points) <= .012 * frame_width
            and widths[support[-1]] < .45 * widths[support[0]]
            and (np.diff(widths[support]) <= max(2, .006 * frame_width)).mean() >= .95)


def _toolbar_above(frame, regions, top, width):
    leaders, tools = [], []
    for row in regions["candidates"]:
        x, y = row["center"]
        a, t, b, d = row["box"]
        if (row["visibility"] != "page" or not .034 * width <= d - t <= .15 * width
                or not .041 * width <= top - y <= .128 * width or d > top):
            continue
        leading = x <= .27 * width and a < .11 * width
        if not leading and x < .81 * width:
            continue
        stop = min(b, a + round(1.14 * (d - t))) if leading else b
        tile = cv2.cvtColor(frame[t:d, a:stop], cv2.COLOR_BGR2GRAY).astype(np.int16)
        ink = abs(tile - np.median(tile)) >= 48
        if not .027 <= ink.mean() <= .42:
            continue
        if leading:
            leaders.append(y)
        else:
            tools.append(y)
    return any(abs(a - b) <= .033 * width for a in leaders for b in tools)


def transient_close_action(frame, regions):
    """Return one proven same-surface dismiss point below or above the page."""
    height, width = frame.shape[:2]
    if (regions.get("layer") != "page" or regions.get("regions") or regions.get("detected_close")
            or width < 240 or not 1.52 * width <= height <= 2.77 * width):
        return None
    owners = [row["box"] for row in regions["candidates"] if row.get("visibility") == "page"
              and .835 * width <= row["center"][0] <= .978 * width
              and (.076 * height <= row["center"][1] <= .255 * height or .66 * height <= row["center"][1] <= .92 * height)
              and .042 * width <= min(row["box"][2] - row["box"][0], row["box"][3] - row["box"][1])
              <= max(row["box"][2] - row["box"][0], row["box"][3] - row["box"][1]) <= .138 * width]
    actions = []
    gray = None
    for candidate in owners:
        proof = _mark(frame, candidate, width)
        if proof is None:
            continue
        cross, sign, paint = proof
        x, y = ((cross[0] + cross[2]) // 2, (cross[1] + cross[3]) // 2)
        if any(a <= x < b and t <= y < d for a, t, b, d in regions.get("floating_controls", ())):
            continue
        if any(abs(x - old_x) <= 2 and abs(y - old_y) <= 2 for _, (old_x, old_y) in actions):
            continue
        banner = np.ptp(paint) >= 52 and max(paint) >= 168 and y < .255 * height
        tip = np.ptp(paint) <= 18 and min(paint) >= 236 and y > .66 * height
        if not (banner or tip):
            continue
        bounds, mask, start = _surface(frame, (x, y), paint, 21 if banner else 8)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        if banner:
            if max(left, width - right) > .028 * width or not _toolbar_above(frame, regions, top, width):
                continue
        elif not (.025 * width <= min(left, width - right) <= .095 * width and _pointer(mask, start, bounds)):
            continue
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines = _text_witness(gray, bounds, cross, sign)
        if banner:
            if len(lines) != 1 or lines[0][2] < .19 * width or abs((lines[0][0] + lines[0][1]) / 2 - y) > .22 * (bottom - top):
                continue
        elif len(lines) < 2 or max(w for _, _, w in lines) < .29 * width:
            continue
        actions.append((cross, (x, y)))
    if not actions or np.ptp(np.array([point for _, point in actions]), axis=0).max() > .008 * width:
        return None
    box, point = actions[0]
    return {"role": "close", "function": "关闭", "box": list(map(int, box)), "click_point": list(map(int, point))}
