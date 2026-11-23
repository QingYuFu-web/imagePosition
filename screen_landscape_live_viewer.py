"""Locate viewer chrome on a landscape stream without reading embedded video."""

from itertools import combinations

import cv2
import numpy as np

from screen_foreground import _centered_cross


def _size(box):
    x, y, r, b = box
    return (r - x + b - y) / 2


def _center(box):
    x, y, r, b = box
    return (x + r) / 2, (y + b) / 2


def _disc(shape, x, y, w, h):
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [shape], 0, 1, cv2.FILLED, offset=(-x, -y))
    yy, xx = np.ogrid[-1:1:complex(0, h), -1:1:complex(0, w)]
    disk = xx * xx + yy * yy <= 1
    return (mask & disk).sum() >= .89 * (mask | disk).sum()


def _pale(frame):
    blue, green, red = cv2.split(frame)
    low = cv2.min(cv2.min(blue, green), red)
    high = cv2.max(cv2.max(blue, green), red)
    return low, cv2.subtract(high, low)


def _footer(frame):
    height, width = frame.shape[:2]
    start = round(.82 * height)
    low, spread = _pale(frame[start:])
    faces = {}
    for level in (146, 170):
        pale = np.uint8((low >= level) & (spread <= 52))
        shapes, _ = cv2.findContours(pale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        row = []
        for shape in shapes:
            x, y, w, h = cv2.boundingRect(shape)
            if (.059 * height <= min(w, h) <= max(w, h) <= .128 * height and .82 <= w / h <= 1.20
                    and .855 * height <= start + y + h / 2 <= .975 * height
                    and 0 < x < x + w < width and start + y + h < height and _disc(shape, x, y, w, h)):
                row.append((x, start + y, x + w, start + y + h))
        faces[level] = row
    stable = sorted(box for box in faces[170] if any(
        max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= .081 * _size(box) for peer in faces[146]))
    groups = set()
    for box in stable:
        _, cy = _center(box)
        size = _size(box)
        row = [peer for peer in stable if abs(_center(peer)[1] - cy) <= .14 * size
               and abs(_size(peer) - size) <= .145 * size]
        if not 4 <= len(row) <= 7:
            continue
        xx = [(_center(peer)[0]) for peer in row]
        gaps = np.diff(xx)
        pitch = float(np.min(gaps))
        if (xx[-1] - xx[0] >= .60 * width and xx[-1] >= .88 * width and xx[0] <= .32 * width
                and 2.65 * size <= pitch <= 5.34 * size
                and np.max(np.abs(gaps - pitch * np.rint(gaps / pitch))) <= .18 * size):
            groups.add(tuple(row))
    return next(iter(groups)) if len(groups) == 1 else None


def _heart(frame, panel, cy, size):
    x, _, r, b = panel
    left, right = x + round(.38 * size), r - round(.42 * size)
    top = round(cy + .045 * size)
    if right <= left:
        return False
    field = frame[top:b, left:right]
    blue, green, red = np.int16(field).transpose(2, 0, 1)
    warm = (red >= 174) & (red - green >= 43) & (red - blue >= 39)
    light = cv2.cvtColor(field, cv2.COLOR_BGR2GRAY)
    fill = float(np.percentile(light, 75))
    # Luminance preserves the notch when JPEG subsamples the red edge.
    for drop in (52, 36, 19):
        ink = np.uint8(warm & (light <= fill - drop))
        _, components, records, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        for i, (u, v, w, h, _) in enumerate(records[1:], 1):
            if not .14 * size <= min(w, h) <= max(w, h) <= .36 * size or not .84 <= w / h <= 1.36:
                continue
            tile = components[v:v + h, u:u + w] == i
            top_row = tile[0]
            if (top_row[:w // 2].sum() >= .12 * w and top_row[w // 2:].sum() >= .12 * w
                    and not top_row[w // 2] and tile[round(.32 * h)].sum() >= .78 * w
                    and tile[-max(2, round(.12 * h)):].sum(axis=1).max() <= .48 * w):
                return True
    return False


def _head_outline(gray, face):
    x, y, r, b = face
    size = _size(face)
    cx, cy = _center(face)
    inset = round(.115 * size)
    x0, y0 = x + inset, y + inset
    chip = gray[y0:b - inset, x0:r - inset]
    ink = np.uint8(chip < float(np.median(chip)) - 76)
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    for i, hole in enumerate(contours):
        if hierarchy[0, i, 3] < 0:
            continue
        tx, ty, w, h = cv2.boundingRect(hole)
        if (.15 * size <= min(w, h) <= max(w, h) <= .33 * size and .79 <= w / h <= 1.27
                and abs(x0 + tx + w / 2 - cx) <= .14 * size and cy - .24 * size <= y0 + ty + h / 2 <= cy
                and chip[ty + h:].size and ink[ty + h:].sum() >= .014 * size * size):
            return True
    return False


def _host(frame, gray, diameter):
    height, width = frame.shape[:2]
    y0, y1 = round(.033 * height), round(.181 * height)
    low, spread = _pale(frame[y0:y1, :round(.226 * width)])
    pale = np.uint8((low >= 155) & (spread <= 49))
    shapes, _ = cv2.findContours(pale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels, rounds = [], []
    for shape in shapes:
        x, y, w, h = cv2.boundingRect(shape)
        if not .81 * diameter <= h <= 1.20 * diameter or not 0 < y < y + h < pale.shape[0]:
            continue
        box = (x, y0 + y, x + w, y0 + y + h)
        if .85 <= w / h <= 1.18 and _disc(shape, x, y, w, h):
            rounds.append(box)
        if 1.46 <= w / h <= 3.13 and cv2.contourArea(shape) >= .70 * w * h:
            panels.append(box)
    for panel in panels:
        _, cy = _center(panel)
        size = panel[3] - panel[1]
        for badge in rounds:
            if (not .08 * size <= badge[0] - panel[2] <= .37 * size or abs(_center(badge)[1] - cy) > .13 * size
                    or not _heart(frame, panel, cy, size) or not _head_outline(gray, badge)):
                continue
            left, right = max(0, panel[0] - round(.84 * size)), panel[0] + round(.73 * size)
            top, bottom = max(0, round(cy - .70 * size)), round(cy + .70 * size)
            chip = cv2.GaussianBlur(gray[top:bottom, left:right], (5, 5), 1.10)
            circle = cv2.HoughCircles(chip, cv2.HOUGH_GRADIENT, 1.2, .49 * size,
                                      param1=73, param2=19, minRadius=round(.36 * size),
                                      maxRadius=round(.58 * size))
            if circle is not None and any(abs(left + x - panel[0]) <= .28 * size
                                            and abs(top + y - cy) <= .17 * size and x - r >= 0
                                            for x, y, r in circle[0]):
                return cy, size
    return None


def _bubble_outline(ink):
    h, w = ink.shape
    cap = max(2, round(.122 * h))
    edge = max(2, round(.14 * w))
    body = ink[round(.17 * h):round(.73 * h)]
    tail = ink[round(.85 * h):]
    ys, xs = np.nonzero(tail)
    if (ink[:cap].any(axis=0).mean() < .73 or min(body[:, :edge].any(axis=1).mean(),
                                                body[:, -edge:].any(axis=1).mean()) < .78
            or not xs.size or xs.min() < .15 * w or xs.max() > .60 * w
            or not .12 * w <= np.ptp(xs) + 1 <= .39 * w):
        return False
    by = ys == ys.max()
    ty = ys <= ys.min() + 1
    return float(xs[ty].mean() - xs[by].mean()) >= max(1., .052 * w)


def _speech(gray, box):
    l, t, r, b = box
    size = _size(box)
    inset = max(2, round(.092 * size))
    x0, y0 = l + inset, t + inset
    chip = gray[y0:b - inset, x0:r - inset]
    floor = float(np.percentile(chip, 62))
    matches = []
    for level in (50, 60, 71, 92):
        _, ids, stats, points = cv2.connectedComponentsWithStats(np.uint8(chip <= floor - level), connectivity=8)
        for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if (not .39 * size <= min(w, h) <= max(w, h) <= .66 * size or not .85 <= w / h <= 1.25
                    or not .09 <= area / (w * h) <= .30 or not 0 < x < x + w < chip.shape[1]
                    or not 0 < y < y + h < chip.shape[0]):
                continue
            if not _bubble_outline(ids[y:y + h, x:x + w] == i):
                continue
            dots = sorted(((u - x) / w, (v - y) / h) for (_, _, a, d, count), (u, v) in
                          zip(stats[1:], points[1:], strict=True)
                          if .0027 * w * h <= count <= .023 * w * h and .63 <= a / d <= 1.58
                          and .15 * w <= u - x <= .85 * w and .29 * h <= v - y <= .61 * h)
            if (len(dots) == 3 and np.ptp([v for _, v in dots]) <= .09
                    and all(.15 <= q[0] - p[0] <= .34 for p, q in zip(dots[:-1], dots[1:], strict=True))):
                matches.append((level, (x0 + x, y0 + y, x0 + x + w, y0 + y + h)))
    return len({level for level, _ in matches}) >= 2 and np.ptp([b for _, b in matches], axis=0).max() <= .08 * size


def _exit(gray, frame, cy, size):
    _, width = gray.shape
    left, right = width - round(1.46 * size), width - max(1, round(.045 * size))
    top, bottom = round(cy - .63 * size), round(cy + .63 * size)
    pale, spread = _pale(frame[top:bottom, left:right])
    matches = []
    for level in (155, 170, 186, 204):
        _, marks, records, _ = cv2.connectedComponentsWithStats(np.uint8((pale >= level) & (spread <= 55)))
        for i, (x, y, w, h, area) in enumerate(records[1:], 1):
            if (not .29 * size <= min(w, h) <= max(w, h) <= .64 * size or not .79 <= w / h <= 1.27
                    or not .080 <= area / (w * h) <= .37 or not 0 < x < x + w < pale.shape[1] - 1
                    or abs(top + y + h / 2 - cy) > .20 * size):
                continue
            main = marks[y:y + h, x:x + w] == i
            yy, xx = np.ogrid[-1:1:complex(0, h), -1:1:complex(0, w)]
            near_diagonal = abs(abs(xx) - abs(yy)) <= .20
            if (main & near_diagonal).sum() < .89 * area:
                continue
            # Keep a separately drawn tip on the same diagonal if strokes meet through a shadow.
            ink = (marks[y:y + h, x:x + w] > 0) & near_diagonal
            arms = [np.count_nonzero(ink & (xx * a >= .43) & (yy * b >= .43))
                    for a in (-1, 1) for b in (-1, 1)]
            if min(arms) < .084 * area or min(arms) < .43 * max(arms):
                continue
            matches.append((level, (left + x, top + y, left + x + w, top + y + h)))
    if len({level for level, _ in matches}) < 2 or np.max(np.ptp([box for _, box in matches], axis=0)) > .083 * size:
        return None
    box = matches[len(matches) // 2][1]
    x, y, r, b = box
    pads = [max(2, round(fraction * size)) for fraction in (.11, .15, .20)]
    if any(_centered_cross(gray[y - pad:b + pad, x - pad:r + pad]) for pad in pads):
        return box
    return None


def _action(role, function, box):
    return {"role": role, "function": function, "preview_text": function, "box": [int(value) for value in box],
            "click_point": [round(v) for v in _center(box)]}


def landscape_live_viewer_actions(frame, overlay, index=None):
    """Read separately visible header and footer controls from a verified viewer."""
    height, width = frame.shape[:2]
    if overlay is not None or height < 250 or not 1.48 * height <= width <= 2.92 * height:
        return []
    row = _footer(frame)
    if row is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    host = _host(frame, gray, float(np.median([_size(box) for box in row])))
    if host is None:
        return []
    first = row[0]
    result = []
    if (close := _exit(gray, frame, *host)) is not None:
        result.append(_action("close", "退出直播间", close))
    if _center(first)[0] <= .122 * width and _speech(gray, first):
        result.append(_action("comments", "打开评论输入框", first))
    return result
