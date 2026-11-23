"""Verify a local floating video surface and its close and fullscreen controls."""

import cv2
import numpy as np

from screen_landscape_choice_dialog import _corner
from screen_landscape_media_bar import _triangle
from screen_wide_panel import _diagonal_cross


def _edge(gray, axis, anchor, first, last, reach, step, entering):
    field = np.moveaxis(gray, axis, 0)
    possible = []
    for pos in range(max(step, anchor - reach), min(len(field) - step, anchor + reach + 1)):
        before = field[pos - step:pos, first:last].mean(axis=0)
        after = field[pos:pos + step, first:last].mean(axis=0)
        outside, inside = (before, after) if entering else (after, before)
        diff = outside - inside
        support = float((diff >= 27).mean())
        if support >= .62 and np.median(diff) >= 44 and np.median(outside) >= 155 and outside.std() <= 38:
            possible.append((float(np.median(diff)) * support, pos))
    return max(possible)[1] if possible else None


def _shadow(gray, axis, edge, first, last, direction, height):
    field = np.moveaxis(gray, axis, 0)
    shifts = [max(2, round(factor * height)) for factor in (.007, .013, .022, .037, .056)]
    positions = [edge + direction * n for n in shifts]
    if min(positions) < 0 or max(positions) >= len(field):
        return False
    tones = [float(np.median(field[pos, first:last])) for pos in positions]
    return (tones[-1] - tones[0] >= 9 and np.median(np.diff(tones)) >= 2.5 and
            min(np.diff(tones)) >= -2 and tones[-1] >= 170)


def _pane(frame, gray, proposal):
    left, top, right, bottom = map(int, proposal)
    h, w = gray.shape
    width, height = right - left, bottom - top
    if (not 0 <= left < right <= w or not 0 <= top < bottom <= h or not .31 * w <= width <= .67 * w
            or not .24 * h <= height <= .69 * h or not 1.54 <= width / height <= 2.13):
        return None
    reach, step = round(.092 * height), max(2, round(.0075 * height))
    x0, x1 = left + round(.112 * width), right - round(.112 * width)
    y0, y1 = top + round(.125 * height), bottom - round(.125 * height)
    a = _edge(gray, 1, left, y0, y1, reach, step, True)
    r = _edge(gray, 1, right, y0, y1, reach, step, False)
    t = _edge(gray, 0, top, x0, x1, reach, step, True)
    b = _edge(gray, 0, bottom, x0, x1, reach, step, False)
    if any(edge is None for edge in (a, t, r, b)):
        return None
    ph, pw = b - t, r - a
    if (not .87 * height <= ph <= 1.10 * height or not .87 * width <= pw <= 1.10 * width
            or not 1.56 <= pw / ph <= 2.03 or min(a, w - r, t, h - b) < .027 * ph):
        return None
    checks = [(a, t, 1, 1), (r - 1, t, -1, 1), (a, b - 1, 1, -1), (r - 1, b - 1, -1, -1)]
    if sum(_corner(gray, x, y, dx, dy, ph) for x, y, dx, dy in checks) < 3:
        return None
    fades = [_shadow(gray, 0, pos, x0, x1, side, ph) for pos, side in ((t, -1), (b, 1))]
    fades += [_shadow(gray, 1, pos, y0, y1, side, ph) for pos, side in ((a, -1), (r, 1))]
    if sum(fades) < 2:
        return None
    inner = frame[t + round(.16 * ph):b - round(.14 * ph):4, a + round(.12 * pw):r - round(.12 * pw):4]
    if np.std(inner.astype(np.float32), axis=(0, 1)).max() < 25:
        return None
    return a, t, r, b


def _cog(contour, children, tree, bounds):
    x, y, width, height = bounds
    area = cv2.contourArea(contour)
    if not .53 <= area / (width * height) <= .83 or len(children) != 1:
        return False
    inner_index, inner = children[0]
    hole_area = cv2.contourArea(inner)
    descendants = [child for i, child in enumerate(tree[0]) if int(child[3]) == inner_index]
    if not .31 * area <= hole_area <= .88 * area or len(descendants) != 1:
        return False
    center = (x + (width - 1) / 2, y + (height - 1) / 2)
    mask = cv2.drawContours(np.zeros((height, width), np.uint8), [contour], 0, 1, cv2.FILLED, offset=(-x, -y))
    polar = cv2.warpPolar(mask, (48, 128), (center[0] - x, center[1] - y), .76 * max(width, height),
                          cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS)
    radius = (polar * np.arange(polar.shape[1])).max(axis=1)
    spectrum = np.square(np.abs(np.fft.rfft(radius - radius.mean())))
    teeth = int(spectrum.argmax())
    return 5 <= teeth <= 10 and spectrum[teeth] >= .48 * spectrum.sum()


def _shape(ink, contour, index, contours, tree, bounds, kind):
    x, y, width, height = bounds
    if kind == "close":
        return _diagonal_cross(ink[y:y + height, x:x + width] > 0)
    if kind == "play":
        polygon = cv2.approxPolyDP(contour, .059 * cv2.arcLength(contour, True), True).reshape(-1, 2)
        return (len(polygon) == 3 and .39 <= cv2.contourArea(contour) / (width * height) <= .63
                and _triangle((polygon - (x, y)) / (width - 1, height - 1)))
    children = [(off, child) for off, child in enumerate(contours) if tree[0, off, 3] == index]
    return _cog(contour, children, tree, bounds)


def _one_mark(bright, box, height, kind):
    l, t, r, b = box
    field = bright[t:b, l:r]
    found = []
    for level in (175, 195, 220, 239):
        ink = np.uint8(field >= level)
        contours, tree = cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if tree is None:
            continue
        for index, contour in enumerate(contours):
            if tree[0, index, 3] >= 0:
                continue
            x, y, w, h = bounds = cv2.boundingRect(contour)
            if (not .043 * height <= min(w, h) <= max(w, h) <= .100 * height or not .75 <= w / h <= 1.27
                    or not 0 < x < x + w < r - l or not 0 < y < y + h < b - t):
                continue
            if _shape(ink, contour, index, contours, tree, bounds, kind):
                found.append((level, (x + l, y + t, x + l + w, y + t + h)))
    return _stable(found)


def _stable(found):
    validated = [box for level, box in found if any(
        other != level and max(abs(a - b) for a, b in zip(box, near, strict=True)) <= 3
        for other, near in found)]
    if validated and all(max(abs(a - b) for a, b in zip(validated[0], near, strict=True)) <= 3 for near in validated):
        return validated[len(validated) // 2]
    return None


def _expand(bright, box, height):
    a, t, b, z = box
    found = []
    for level in (175, 195, 220, 239):
        blob = np.uint8(bright[t:z, a:b] >= level)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(blob)
        parts = [(i, int(x), int(y), int(w), int(h)) for i, (x, y, w, h, area) in enumerate(stats[1:], 1)
                 if .017 * height <= min(w, h) <= max(w, h) <= .048 * height and .79 <= w / h <= 1.22
                 and .37 <= area / (w * h) <= .79]
        if len(parts) != 4:
            continue
        by_y = sorted(parts, key=lambda mark: mark[2])
        parts = sorted(by_y[:2], key=lambda mark: mark[1]) + sorted(by_y[2:], key=lambda mark: mark[1])
        (_, x1, y1, w1, h1), (_, x2, y2, _, _), (_, x3, y3, _, _), (_, x4, y4, w4, h4) = parts
        size = (w1 + h1 + w4 + h4) / 4
        if (max(abs(x1 - x3), abs(x2 - x4), abs(y1 - y2), abs(y3 - y4)) > .24 * size or
                not 1.51 * size <= x2 - x1 <= 2.38 * size or not 1.51 * size <= y3 - y1 <= 2.38 * size):
            continue
        proof = []
        for j, (num, x, y, w, h) in enumerate(parts):
            region = labels[y:y + h, x:x + w] == num
            shape = region[::(-1 if j >= 2 else 1), ::(-1 if j % 2 else 1)]
            patch = max(2, round(.36 * min(w, h)))
            proof.append(min(shape[:patch].mean(), shape[:, :patch].mean()) >= .62 and
                         shape[round(.46 * h):, round(.46 * w):].mean() <= .15)
        if all(proof):
            found.append((level, (a + x1, t + min(y1, y2), a + x4 + w4, t + y4 + h4)))
    return _stable(found)


def _center(box):
    l, t, r, b = box
    return (l + r) / 2, (t + b) / 2


def _actions(frame, pane):
    left, top, right, bottom = pane
    height = bottom - top
    center = (left + right) / 2
    margin = round(.04 * height)
    reach = round(.213 * height)
    bright = frame.min(axis=2)
    gear = _one_mark(bright, (left + margin, top + margin, left + reach, top + reach), height, "gear")
    play = _one_mark(bright, (round(center - .100 * height), bottom - reach,
                              round(center + .100 * height), bottom - margin), height, "play")
    if gear is None or play is None or abs(_center(play)[0] - center) > .026 * height:
        return None
    close = _one_mark(bright, (right - reach, top + margin, right - margin, top + reach), height, "close")
    if close is not None:
        gx, gy = _center(gear)
        cx, cy = _center(close)
        if abs(gy - cy) > .029 * height or abs(left + right - gx - cx) > .041 * height:
            close = None
    expand = _expand(bright, (round(center - .118 * height), top + round(.345 * height),
                              round(center + .118 * height), top + round(.592 * height)), height)
    if expand is not None and abs(_center(expand)[0] - _center(play)[0]) > .027 * height:
        expand = None
    result = []
    for label, role, glyph in (("关闭画中画", "close", close), ("返回全屏", "fullscreen", expand)):
        if glyph is None:
            continue
        l, t, r, b = glyph
        pad = round(.037 * height)
        result.append({"role": role, "function": label, "preview_text": label,
                       "box": [l - pad, t - pad, r + pad, b + pad],
                       "click_point": [round((l + r) / 2), round((t + b) / 2)]})
    return result


def landscape_pip_actions(frame, regions):
    """Return the local foreground bounds and visible PiP actions."""
    h, w = frame.shape[:2]
    if regions.get("layer") != "page" or h < 390 or not 1.45 * h <= w <= 2.9 * h:
        return None
    candidates = [row["box"] for row in regions.get("candidates", ())
                  if .31 * w <= row["box"][2] - row["box"][0] <= .67 * w
                  and .24 * h <= row["box"][3] - row["box"][1] <= .69 * h]
    if not candidates:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    matches = {}
    for box in candidates[:12]:
        pane = _pane(frame, gray, box)
        if pane is not None and pane not in matches and (actions := _actions(frame, pane)) is not None:
            matches[pane] = actions
    return next(iter(matches.items())) if len(matches) == 1 else None
