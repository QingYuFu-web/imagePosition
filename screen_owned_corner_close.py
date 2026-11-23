"""Recognize a page dismissal or a painted dismiss button on a card rim."""

import cv2
import numpy as np

from screen_back_control import _title_run
from screen_dialog_close import _cross_footprint
from screen_fullscreen_close import _cross


_ANGLES = np.arange(40) * (np.pi / 20)
_UNIT = np.stack((np.cos(_ANGLES), np.sin(_ANGLES)))


def _center(box):
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def _quiet(tile, paper, tolerance):
    if not tile.size:
        return False
    return (np.max(np.abs(tile.astype(np.int16) - paper), axis=2) <= tolerance).mean() >= .89


def _vertical_more(tile, size):
    paper = float(np.median(tile))
    for sign in (1, -1):
        votes = 0
        for cutoff in (49, 73, 96):
            _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(sign * (tile - paper) >= cutoff))
            dots = sorted((tuple(row) for row in stats[1:] if row[4] >= max(3, .001 * size * size)), key=lambda row: row[1])
            if len(dots) != 3 or not all(.047 * size <= min(w, h) <= max(w, h) <= .135 * size
                                           and .76 <= w / h <= 1.30 and area >= .53 * w * h
                                           for _, _, w, h, area in dots):
                continue
            x = [a + w / 2 for a, _, w, _, _ in dots]
            y = [b + h / 2 for _, b, _, h, _ in dots]
            width = np.median([w for _, _, w, _, _ in dots])
            if (np.ptp(x) <= .32 * width and np.ptp(np.diff(y)) <= .24 * np.mean(np.diff(y))
                    and .18 * size <= np.ptp(y) <= .47 * size
                    and abs(np.mean(x) - tile.shape[1] / 2) <= .13 * size
                    and abs(np.mean(y) - tile.shape[0] / 2) <= .13 * size):
                votes += 1
        if votes >= 2:
            return sign
    return None


def _toolbar_close(frame, gray, box, boxes):
    left, top, right, bottom = box
    height, width = gray.shape
    x, y = _center(box)
    size = min(right - left, bottom - top)
    leading = x < width / 2
    if not (.04 * width <= min(x, width - x) <= .146 * width and .045 * height <= y <= .104 * height
            and .054 * width <= size <= .14 * width):
        return None
    owner = frame[top:bottom, left:right]
    point = _cross(owner)
    if point is None:
        return None
    paint = np.median(owner[::2, ::2], axis=(0, 1))
    paper = float(np.median(gray[top:bottom, left:right]))
    sign = 1 if paper <= 76 else -1
    titles, tools = [], []
    for a, t, b, d in boxes:
        cx, cy = (a + b) / 2, (t + d) / 2
        opposite = cx if leading else width - cx
        if abs(cy - (top + point[1])) > .24 * size:
            continue
        tile = gray[t:d, a:b]
        if abs(float(np.median(tile)) - paper) > 21:
            continue
        if (.46 * width <= cx <= .54 * width and .20 * width <= b - a <= .60 * width
                and .37 * size <= d - t <= 1.10 * size):
            if (title := _title_run(tile, paper, .43 * size, minimum=4, high=1.34)) is not None:
                if title[1] - title[0] >= .66 * (b - a):
                    titles.append((a, b))
        elif (.84 * width <= opposite <= .965 * width and .60 * size <= min(b - a, d - t)
              <= max(b - a, d - t) <= 1.33 * size and _vertical_more(tile, size) == sign):
            tools.append((a, b))
    if len(titles) != 1 or len(tools) != 1:
        return None
    title_left, title_right = titles[0]
    tool_left, tool_right = tools[0]
    gaps = ((right, title_left), (title_right, tool_left)) if leading else ((tool_right, title_left), (title_right, left))
    if any(b - a < .055 * width for a, b in gaps):
        return None
    first = max(0, round(top + point[1] - .32 * size))
    stop = min(height, round(top + point[1] + .32 * size))
    return (left + point[0], top + point[1]) if all(_quiet(frame[first:stop:2, a:b:3], paint, 25) for a, b in gaps) else None


def _card_boundaries(frame, box):
    left, top, right, bottom = box
    height, width = frame.shape[:2]
    reach, offset = round(.017 * width), max(2, round(.008 * width))
    if min(left, width - right, top, height - bottom) < reach + offset:
        return False
    span = right - left
    xs = np.arange(left + round(.14 * span), right - round(.14 * span), max(2, offset))
    ys = np.arange(top + round(.14 * (bottom - top)), bottom - round(.14 * (bottom - top)), max(2, offset))
    paper = []
    for edge, direction, horizontal in ((left, 1, False), (top, 1, True), (right, -1, False), (bottom, -1, True)):
        probes = np.arange(edge - reach, edge + reach + 1, max(1, offset // 3))[:, None]
        inner, outer = probes + direction * offset, probes - direction * offset
        a, b = (frame[inner, xs], frame[outer, xs]) if horizontal else (frame[ys, inner], frame[ys, outer])
        baseline = np.median(b.astype(np.int16), axis=1)[:, None, :]
        quiet = (np.max(np.abs(b - baseline), axis=2) <= 13).mean(axis=1)
        support = (np.max(np.abs(a.astype(np.int16) - b.astype(np.int16)), axis=2) >= 46).mean(axis=1)
        supported = np.flatnonzero((quiet >= .89) & (support >= .81))
        if not supported.size:
            return False
        paper.append(baseline[supported[np.argmax(support[supported])], 0])
    return np.ptp(paper, axis=0).max() <= 28


def _circular_cross(frame, gray, box):
    left, top, right, bottom = box
    height, width = gray.shape
    x, y = _center(box)
    size = min(right - left, bottom - top)
    if min(x, width - 1 - x, y, height - 1 - y) <= .62 * size:
        return None
    rings = []
    for fraction in (.36, .58):
        xx, yy = np.rint(np.array([[x], [y]]) + fraction * size * _UNIT).astype(int)
        rings.append(frame[yy, xx].astype(np.int16))
    inside, outside = rings
    paint = np.median(inside, axis=0)
    if ((np.max(abs(inside - paint), axis=1) <= 17).mean() < .75
            or (np.max(abs(outside - paint), axis=1) >= 32).mean() < .73):
        return None
    tile = gray[top:bottom, left:right]
    paper = int(cv2.cvtColor(np.uint8(paint)[None, None], cv2.COLOR_BGR2GRAY)[0, 0])
    xx, yy = np.meshgrid(np.arange(right - left), np.arange(bottom - top))
    radial = np.hypot(xx - (right - left) / 2, yy - (bottom - top) / 2)
    core = radial <= .365 * size
    verified, silhouettes = [], set()
    for sign in (1, -1):
        ink = sign * (tile - paper)
        for cutoff in (66, 94, 126):
            mask = np.uint8(core & (ink >= cutoff))
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            parts = [(i, a, b, w, h, area) for i, (a, b, w, h, area) in enumerate(stats[1:], 1)
                     if area >= .011 * size * size]
            if len(parts) != 1:
                continue
            identity, a, b, w, h, area = parts[0]
            glyph = labels[b:b + h, a:a + w] == identity
            if (.29 * size <= min(w, h) <= max(w, h) <= .59 * size and .87 <= w / h <= 1.15
                    and .18 <= area / (w * h) <= .53 and abs(a + w / 2 - (right - left) / 2) <= .10 * size
                    and abs(b + h / 2 - (bottom - top) / 2) <= .10 * size
                    and _cross_footprint(glyph)):
                silhouettes.add((glyph.shape, glyph.tobytes()))
                verified.append((left + a + w / 2, top + b + h / 2))
    return verified[0] if len(silhouettes) >= 2 and np.ptp(verified, axis=0).max() <= .07 * size else None


def owned_corner_close_actions(frame, regions):
    """Return close controls with independent toolbar or artwork-card ownership."""
    height, width = frame.shape[:2]
    if (regions.get("layer") != "page" or regions.get("regions") or regions.get("detected_close")
            or width < 300 or not 1.46 * width <= height <= 2.90 * width):
        return []
    boxes = [tuple(row["box"]) for row in regions["candidates"] if row.get("visibility") == "page"]
    cards = [box for box in boxes if .69 * width <= box[2] - box[0] <= .985 * width
             and .29 * width <= box[3] - box[1] <= .84 * width and .12 * height <= box[1] <= .61 * height]
    small = [box for box in boxes if .041 * width <= min(box[2] - box[0], box[3] - box[1])
             <= max(box[2] - box[0], box[3] - box[1]) <= .145 * width
             and .74 <= (box[2] - box[0]) / (box[3] - box[1]) <= 1.34]
    gray, accepted = None, []
    for box in small:
        x, y = _center(box)
        parents = [card for card in cards if .022 * width <= min(x - card[0], card[2] - x) <= .115 * width
                   and .024 * width <= y - card[1] <= .123 * width]
        if not parents and not (min(x, width - x) <= .15 * width and y <= .12 * height):
            continue
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        point = _toolbar_close(frame, gray, box, boxes)
        if point is None and parents:
            point = _circular_cross(frame, gray, box)
            if point is not None and not any(_card_boundaries(frame, card) for card in parents):
                point = None
        if point is None or any(a <= point[0] < b and t <= point[1] < d for a, t, b, d in regions.get("floating_controls", ())):
            continue
        accepted.append({"role": "close", "function": "关闭", "box": list(box), "click_point": list(point)})
    if not accepted:
        from screen_inline_card_close import inline_card_close_actions

        return inline_card_close_actions(frame, regions)
    return accepted if len(accepted) <= 1 else []
