"""Ground an outlined comment bubble in a measured, aligned post action strip."""

from itertools import combinations

import cv2
import numpy as np

from screen_horizontal_post_actions import _center, _close, _size


def _cuts(gray, box):
    left, top, right, bottom = box
    tile = gray[top:bottom, left:right]
    if not tile.size:
        return
    paper = np.median(tile)
    if paper < 207 or (tile >= paper - 16).mean() < .81:
        return
    seen = set()
    for value in (37, 56, 78, 100):
        ink = np.uint8(tile <= paper - value)
        fingerprint = ink.tobytes()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        contours, links = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if links is not None:
            yield value, contours, links[0]


def _outline(contour, bounds):
    x, y, w, h = bounds
    return cv2.drawContours(np.zeros((h, w), np.uint8), [contour], 0, 1, cv2.FILLED, offset=(-x, -y))


def _speech_bubble(contour, holes, bounds):
    x, y, w, h = bounds
    if not .87 <= w / h <= 1.34 or not .60 <= cv2.contourArea(contour) / (w * h) <= .86:
        return False
    prominent = [hole for hole in holes if cv2.contourArea(hole) >= .025 * w * h]
    if len(prominent) != 1:
        return False
    hole = prominent[0]
    hx, hy, hw, hh = cv2.boundingRect(hole)
    headroom, footroom = hy - y, y + h - hy - hh
    if not (.65 * w <= hw <= .91 * w and .59 * h <= hh <= .87 * h
            and .30 <= cv2.contourArea(hole) / (w * h) <= .67
            and abs(hx + hw / 2 - x - w / 2) <= .12 * w and .07 * h <= footroom <= .31 * h):
        return False
    body = _outline(contour, bounds)
    toe = np.flatnonzero(body[-1])
    if not 0 < len(toe) <= .43 * w:
        return False
    # Rounded outlines also have cavities. Require a separated lower tip or
    # a visibly longer, narrower extension below the cavity.
    forks = np.flatnonzero(np.diff(toe) > max(2, .13 * w))
    if len(forks) == 1 and toe[-1] - toe[0] >= .38 * w and abs(toe.mean() - .5 * w) >= .12 * w:
        return True
    return (not len(forks) and len(toe) <= .29 * w and footroom - headroom >= .10 * h
            and (body[-max(2, round(.16 * h)):].sum(axis=1) <= .47 * w).all())


def _marks(gray, box):
    left, top, right, bottom = box
    w, h = _size(box)
    found = []
    for level, contours, links in _cuts(gray, box):
        for i, (contour, (_, _, child, parent)) in enumerate(zip(contours, links, strict=True)):
            if child < 0 or parent >= 0:
                continue
            x, y, width, height = bounds = cv2.boundingRect(contour)
            if (not .26 * h <= height <= .51 * h or abs(x + width / 2 - w / 2) > .11 * w
                    or not .30 * h <= y + height / 2 <= .57 * h):
                continue
            holes = [hole for hole, link in zip(contours, links, strict=True) if link[3] == i]
            if _speech_bubble(contour, holes, bounds):
                found.append((level, (left + x, top + y, left + x + width, top + y + height)))
    stable = [mark for level, mark in found if any(
        other != level and _close(mark, peer, max(2, .09 * h)) for other, peer in found)]
    if stable and all(_close(stable[0], mark, max(2, .09 * h)) for mark in stable):
        return stable[len(stable) // 2]
    return None


def _leading_icon(gray, box, mark):
    _, top, _, _ = box
    _, cy = _center(mark)
    size = _size(mark)[1]
    levels = 0
    for _, contours, links in _cuts(gray, box):
        for contour, (_, _, _, parent) in zip(contours, links, strict=True):
            if parent >= 0:
                continue
            _, y, w, h = cv2.boundingRect(contour)
            if (.65 * size <= h <= 1.31 * size and .62 <= w / h <= 1.43
                    and abs(top + y + h / 2 - cy) <= .22 * size
                    and .33 <= cv2.contourArea(contour) / (w * h) <= .85):
                levels += 1
                break
    return levels >= 2


def _opposed_arrows(gray, box, mark):
    _, top, _, _ = box
    w_box, _ = _size(box)
    _, cy = _center(mark)
    size = _size(mark)[1]
    levels = 0
    for _, contours, links in _cuts(gray, box):
        shapes = []
        for contour, (_, _, child, parent) in zip(contours, links, strict=True):
            if child >= 0 or parent >= 0:
                continue
            x, y, w, h = bounds = cv2.boundingRect(contour)
            if (.40 * size <= h <= .77 * size and .62 <= w / h <= 2.38
                    and .20 <= cv2.contourArea(contour) / (w * h) <= .48
                    and abs(x + w / 2 - w_box / 2) <= .14 * w_box):
                shapes.append((bounds, contour))
        for ((ax, ay, aw, ah), a), ((bx, by, bw, bh), b) in combinations(shapes, 2):
            first_center = ax + aw / 2, ay + ah / 2
            second_center = bx + bw / 2, by + bh / 2
            if (abs(first_center[0] - second_center[0]) > .25 * max(aw, bw)
                    or not .51 * max(ah, bh) <= abs(first_center[1] - second_center[1]) <= 1.16 * max(ah, bh)
                    or abs(top + (first_center[1] + second_center[1]) / 2 - cy) > .21 * size
                    or not .79 <= aw / bw <= 1.27 or not .80 <= ah / bh <= 1.23):
                continue
            first = _outline(a, (ax, ay, aw, ah))
            second = _outline(b, (bx, by, bw, bh))
            reverse = cv2.resize(second[::-1, ::-1], (aw, ah), interpolation=cv2.INTER_NEAREST)
            if (first & reverse).sum() >= .77 * (first | reverse).sum():
                levels += 1
                break
    return levels >= 2


def _separator(gray, first, last, size):
    width = gray.shape[1]
    for edge in (first, last):
        reach = max(3, round(.29 * size))
        strip = gray[max(0, edge - reach):min(len(gray), edge + reach + 1), round(.08 * width):round(.92 * width):3]
        if not strip.size:
            continue
        line = ((strip >= 164) & (strip <= 245)).mean(axis=1) >= .84
        quiet = (strip >= 247).mean(axis=1) >= .81
        if line.any() and quiet.any() and line.sum() <= .17 * size:
            return True
    return False


def inline_post_comments(frame, regions):
    """Expose only the confirmed comment action, independent of app names."""
    height, width = frame.shape[:2]
    if (regions["layer"] != "page" or regions.get("regions") or min(height, width) < 240
            or not .57 * width <= height <= 2.85 * width):
        return []
    boxes = [tuple(entry["box"]) for entry in regions["candidates"] if entry["visibility"] == "page"
             and .044 * width <= _size(entry["box"])[1] <= .128 * width
             and .052 * width <= _size(entry["box"])[0] <= .285 * width
             and .12 * height <= entry["center"][1] <= .94 * height]
    rows = []
    for center in boxes:
        x, y = _center(center)
        h = _size(center)[1]
        if not .23 * width <= x <= .67 * width:
            continue
        peers = [box for box in boxes if abs(_center(box)[1] - y) <= .15 * h and .71 <= _size(box)[1] / h <= 1.25]
        left = [box for box in peers if .075 * width <= x - _center(box)[0] <= .25 * width]
        right = [box for box in peers if .075 * width <= _center(box)[0] - x <= .25 * width]
        for a in left:
            for b in right:
                if (abs(_center(a)[0] + _center(b)[0] - 2 * x) <= .21 * (x - _center(a)[0])
                        and a[2] <= center[0] + .19 * h and b[0] >= center[2] - .19 * h):
                    rows.append((a, center, b))
    if not rows:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    accepted = []
    for left, center, right in rows:
        if not _separator(gray, center[1], center[3], _size(center)[1]):
            continue
        mark = _marks(gray, center)
        if mark is None or not _leading_icon(gray, left, mark) or not _opposed_arrows(gray, right, mark):
            continue
        if any(_close(mark, prior, max(3, .15 * _size(mark)[1])) for prior in accepted):
            continue
        accepted.append(mark)
    actions = []
    for mark in accepted:
        x, y = map(round, _center(mark))
        if any(a <= x < b and t <= y < d for a, t, b, d in regions.get("floating_controls", ())):
            continue
        actions.append({"role": "comments", "function": "打开评论", "box": list(mark),
                        "click_point": [x, y], "preview_text": "Comments"})
    return actions
