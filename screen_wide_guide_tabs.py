"""Measure a wide guide overlay with a marked navigation rail."""

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts, read_caption
from screen_component_captions import _bands
from screen_dialog_close import _cross_footprint
from screen_foreground import BlockingOverlay
from screen_landscape_search_field import _available


_HEADINGS = {"攻略", "英雄攻略", "角色攻略", "指南", "游戏指南", "玩法说明", "帮助", "图鉴", "英雄图鉴", "角色图鉴"}


def _rim(gray, window, sign):
    height, width = gray.shape
    gap = max(2, round(.005 * height))
    first, last = (round(value * height) for value in window)
    points = gray[:, ::max(2, round(.017 * width))].astype(np.int16)
    delta = sign * (points[first + gap:last + gap] - points[first - gap:last - gap])
    support = (delta >= 16).mean(axis=1)
    strength = np.median(delta, axis=1)
    best = int(np.argmax(support * np.maximum(strength, 0)))
    row = first + best
    if support[best] < .62 or strength[best] < 19 or min(
            (side >= 16).mean() for side in np.array_split(delta[best], 2)) < (.50 if sign > 0 else .39):
        return None
    # Recenter the wide rising edge within the sampling window.
    segment = points[row - gap:row + gap + 1]
    steps = np.median(sign * np.diff(segment, axis=0), axis=1)
    return row - gap + int(np.argmax(steps)) + 1


def _body(gray):
    height, width = gray.shape
    top = _rim(gray, (.065, .245), 1)
    if top is None:
        return None
    bottom = _rim(gray, (.715, .955), -1)
    if bottom is None or not .60 * height <= bottom - top <= .83 * height:
        return None
    floor = gray[bottom + round(.055 * height):, round(.04 * width):round(.96 * width):5]
    return (top, bottom) if floor.size and np.percentile(floor, 86) <= 72 else None


def _cross(gray, header):
    height, width = gray.shape
    start = round(.83 * width)
    tile = gray[round(.012 * height):header, start:round(.979 * width)]
    offset = round(.012 * height)
    matches = []
    for threshold in (180, 201, 223):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(tile >= threshold))
        for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if (.030 * height <= min(w, h) <= max(w, h) <= .076 * height and .81 <= w / h <= 1.22
                    and .15 <= area / (w * h) <= .43 and offset + y + .5 * h >= .43 * header
                    and _cross_footprint(labels[y:y + h, x:x + w] == i)):
                matches.append((threshold, (start + x, offset + y, start + x + w, offset + y + h)))
    stable = [box for level, box in matches if any(other != level and max(abs(a - b) for a, b in
                        zip(box, peer, strict=True)) <= 2 for other, peer in matches)]
    if not stable or np.ptp(stable, axis=0).max() > 3:
        return None
    return tuple(np.median(stable, axis=0).astype(int))


def _selected(gray, body):
    height, width = gray.shape
    top, bottom = body
    y = top + round(.055 * height)
    left, right = round(.024 * width), round(.065 * width)
    passage = gray[y:min(bottom, top + round(.61 * height)), left:right]
    bright = (passage >= 160).mean(axis=1) >= .84
    choices = [(y + a, y + b) for a, b in _bands(bright) if .056 * height <= b - a <= .144 * height]
    if len(choices) != 1:
        return None
    first, last = choices[0]
    depth = last - first
    tile = gray[first + round(.060 * depth):first + round(.24 * depth), :round(.32 * width)]
    across = (tile >= 157).mean(axis=0) >= .86
    strip = [(a, b) for a, b in _bands(across) if a <= .028 * width and .109 * width <= b <= .29 * width]
    if len(strip) != 1:
        return None
    rail = strip[0][1] - 1
    return first, last, rail


def _rail(gray, body, selected):
    height, width = gray.shape
    t, b = body
    first, last, nominal = selected
    start, stop = t + round(.034 * height), b - round(.077 * height)
    reach = max(4, round(.013 * height))
    patch = gray[start:stop, nominal - reach:nominal + reach + 1].astype(np.int16)
    sample = max(1, round(.013 * patch.shape[1]))
    flanks = np.concatenate((patch[:, :sample], patch[:, -sample:]), axis=1)
    distance = patch - np.median(flanks, axis=1)[:, None]
    size = max(2, round(.009 * height))
    cut = 34 if height >= 470 else 28
    levels = np.uint8(distance >= cut).sum(axis=1)
    marker_rows = []
    for a, z in _bands(levels >= max(2, round(.67 * size))):
        if a + start <= last + round(.023 * height) and z + start >= first - round(.023 * height):
            continue
        if z - a > max(4, .018 * height):
            continue
        body_ink = distance[a:z] >= cut
        _, _, w, h = cv2.boundingRect(np.uint8(body_ink))
        if (not .56 * h <= w <= max(2.2 * h, 4 if height < 430 else 0) or np.max(distance[a:z]) < 55
                or abs(np.argmax(distance[a:z].max(axis=0)) - reach) > max(2, .007 * height)):
            continue
        marker_rows.append(start + (a + z - 1) / 2)
    if not 2 <= len(marker_rows) <= 5:
        return None
    center = (first + last - 1) / 2
    lattice = sorted([center, *marker_rows])
    gaps = np.diff(lattice)
    pitch = float(np.median(gaps))
    if not .079 * height <= pitch <= .177 * height or np.max(np.abs(gaps - pitch)) > .14 * pitch:
        return None
    upper = max(0, round(lattice[0] - .38 * pitch) - start)
    lower = min(len(distance), round(lattice[-1] + .38 * pitch) - start)
    spine = distance[upper:lower, reach - 1:reach + 2].max(axis=1)
    if not len(spine) or (spine >= 8).mean() < .72:
        return None
    return marker_rows, center, pitch, nominal


def _caption(frame, box, index, cache):
    left, top, right, bottom = box
    title = read_caption(frame, box, index, cache=cache, choice=True, join_radicals=True)
    if title is None and bottom - top < 36:
        tile = frame[top:bottom, left:right]
        enlarged = cv2.resize(tile, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
        # The reader's glyph height budget is expressed in terms of full-screen width.
        canvas = np.zeros((len(enlarged), frame.shape[1], 3), np.uint8)
        canvas[:, :enlarged.shape[1]] = enlarged
        title = read_caption(canvas, (0, 0, enlarged.shape[1], len(enlarged)), index,
                            cache=cache, choice=True, join_radicals=True)
    if title and 2 <= len(title["text"]) <= 7 and all("\u3400" <= glyph <= "\u9fff" for glyph in title["text"]):
        return title["text"]
    return None


def _selected_caption(frame, box):
    left, top, right, bottom = box
    gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    paper = float(np.median(gray))
    seen = set()
    for threshold in (51, 73, 97, 109, 127):
        ink = np.uint8(paper - gray >= threshold)
        x, y, w, h = cv2.boundingRect(ink)
        if (7 <= h <= .87 * len(gray) and 1.6 * h <= w <= 7.3 * h
                and min(x, y, gray.shape[1] - x - w, len(gray) - y - h) >= 1):
            run = ink[y:y + h, x:x + w]
            if 2 <= len(_cjk_parts(run, h, join_radicals=True)) <= 7:
                seen.add(run.tobytes())
    return len(seen) >= 2


def wide_guide_tabs(frame, index, *, regions=None):
    """Return the confirmed overlay and its literal inactive tabs."""
    height, width = frame.shape[:2]
    if index is None or height < 254 or not 1.56 * height <= width <= 2.65 * height:
        return None
    overview = frame if height <= 560 else cv2.resize(frame, (round(560 * width / height), 560),
                                                        interpolation=cv2.INTER_AREA)
    small_h, small_w = overview.shape[:2]
    gray = cv2.cvtColor(overview, cv2.COLOR_BGR2GRAY)
    if (body := _body(gray)) is None or (cross := _cross(gray, body[0])) is None:
        return None
    if (selected := _selected(gray, body)) is None or (rail := _rail(gray, body, selected)) is None:
        return None
    rows, current, pitch, edge = rail
    sx, sy = width / small_w, height / small_h

    def project(box):
        return tuple(round(value * scale) for value, scale in zip(box, (sx, sy, sx, sy), strict=True))

    heading_tile = gray[round(.30 * body[0]):body[0], round(.091 * small_w):round(.28 * small_w)]
    x, y, w, h = cv2.boundingRect(np.uint8(heading_tile >= 176))
    if not .029 * small_h <= h <= .087 * small_h or not 1.4 * h <= w <= 6.8 * h:
        return None
    cache = {}
    a = round(.091 * small_w) + x
    t = round(.30 * body[0]) + y
    pad = max(2, round(.13 * h))
    title = project((a - pad, t - pad, a + w + pad, t + h + pad))
    close = project(cross)
    if regions is not None and not all(_available(box, regions) for box in (title, close)):
        return None
    heading = _caption(frame, title, index, cache)
    if heading not in _HEADINGS:
        return None

    def row_box(y):
        return project((round(.13 * edge), round(y - .35 * pitch), edge - round(.06 * edge), round(y + .35 * pitch)))

    def label_box(y):
        return project((round(.29 * edge), round(y - .245 * pitch), edge - round(.073 * edge), round(y + .245 * pitch)))

    if regions is not None and not _available(label_box(current), regions):
        return None
    if not _selected_caption(frame, label_box(current)):
        return None
    entries = []
    for y in rows:
        label_area = label_box(y)
        if regions is not None and not _available(label_area, regions):
            continue
        label = _caption(frame, label_area, index, cache)
        if label:
            l, t, r, b = row_box(y)
            left, _, right, _ = label_area
            entries.append({"role": "switch_tab", "function": f"切换到{label}", "preview_text": label,
                            "box": [l, t, r, b], "click_point": [round((left + right) / 2), round((t + b) / 2)]})
    l, t, r, b = close
    close_action = {"role": "close", "function": f"关闭{heading}", "box": list(close),
                    "click_point": [round((l + r) / 2), round((t + b) / 2)]}
    overlay = BlockingOverlay("dialog", (0, 0, width, round(body[1] * sy)), close, round(body[0] * sy), "attached")
    return overlay, [close_action, *entries]
