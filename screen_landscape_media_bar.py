"""Identify miniature media toolbar controls from mutually supporting glyphs."""

from itertools import combinations

import cv2
import numpy as np

from screen_back_control import _title_run
from screen_candidate_glyphs import _overlap
from screen_landscape_edit_actions import _check_shape


def _action(role, function, box):
    left, top, right, bottom = box
    return {"role": role, "function": function, "preview_text": function, "box": list(box),
            "click_point": [(left + right) / 2, (top + bottom) / 2]}


def _slot_groups(boxes, width, height):
    choices = sorted(box for box in boxes if box[1] <= .052 * height and box[3] <= .143 * height
                     and .058 * height <= box[3] - box[1] <= .135 * height
                     and .76 <= (box[2] - box[0]) / (box[3] - box[1]) <= 1.57)
    for row in combinations(choices, 3):
        first, middle, last = row
        depth = float(np.median([bottom - top for _, top, _, bottom in row]))
        span = last[2] - first[0]
        if not (.108 * width <= span <= .245 * width and .32 * width <= (first[0] + last[2]) / 2 <= .68 * width):
            continue
        if (np.ptp([box[1] for box in row]) <= .18 * depth and np.ptp([box[3] for box in row]) <= .18 * depth
                and -.095 * depth <= middle[0] - first[2] <= .14 * depth
                and -.095 * depth <= last[0] - middle[2] <= .14 * depth):
            yield row


def _solid_group(gray, boxes):
    values = []
    for left, top, right, bottom in boxes:
        h = bottom - top
        sample = gray[top:bottom, left:right].astype(np.int16)
        below = gray[bottom + 1:bottom + max(3, round(.29 * h)), left:right]
        tone = float(np.median(sample))
        if (abs(sample - tone) <= 9).mean() < .65 or np.median(below) - tone < 7:
            return False
        values.append(tone)
    return np.ptp(values) <= 10


def _triangle(polygon):
    a, b, tip = polygon[np.argsort(polygon[:, 0])]
    return (max(a[0], b[0]) <= .23 and tip[0] >= .79 and .29 <= tip[1] <= .70
            and min(a[1], b[1]) <= .17 and max(a[1], b[1]) >= .83)


def _shape_mark(frame, gray, box, icon):
    left, top, right, bottom = box
    tile = gray[top:bottom, left:right].astype(np.int16)
    delta = tile - np.median(tile)
    if icon == "record":
        blue, green, red = frame[top:bottom, left:right].astype(np.int16).transpose(2, 0, 1)
        delta = red - np.maximum(green, blue)
    marks = []
    for cut in ((88, 113, 133) if icon == "record" else (42, 59, 75)):
        ink = np.uint8(delta >= cut)
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            continue
        contour = contours[0]
        x, y, width, height = cv2.boundingRect(contour)
        if not (.32 * tile.shape[0] <= height <= .72 * tile.shape[0] and .71 <= width / height <= 1.23
                and abs(x + width / 2 - tile.shape[1] / 2) <= .13 * tile.shape[1]
                and abs(y + height / 2 - tile.shape[0] / 2) <= .13 * tile.shape[0]):
            continue
        fill = cv2.contourArea(contour) / (width * height)
        poly = cv2.approxPolyDP(contour, .052 * cv2.arcLength(contour, True), True).reshape(-1, 2)
        if icon == "stop":
            if not (.89 <= width / height <= 1.12 and .80 <= fill <= .99 and len(poly) == 4):
                continue
            aligned = abs(np.roll(poly, 1, axis=0) - poly)
            if aligned.min(axis=1).max() > .17 * width:
                continue
        elif icon == "play":
            if not (.40 <= fill <= .62 and len(poly) == 3 and _triangle((poly - (x, y)) / (width - 1, height - 1))):
                continue
        elif not (.60 <= fill <= .83 and .89 <= width / height <= 1.13 and len(poly) >= 5):
            continue
        marks.append((left + x, top + y, left + x + width, top + y + height))
    if len(marks) >= 2 and all(_overlap(marks[0], mark) >= .76 for mark in marks):
        return marks[len(marks) // 2]
    return None


def _filled_check(frame, box):
    left, top, right, bottom = box
    patch = frame[top:bottom, left:right].astype(np.int16)
    low, high = patch.min(axis=2), patch.max(axis=2)
    fill = np.uint8((high >= 140) & (high - low >= 71))
    bodies, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rims = []
    for body in bodies:
        x, y, width, height = cv2.boundingRect(body)
        density = cv2.contourArea(body) / (width * height)
        if (.81 <= width / height <= 1.22 and .64 <= density <= .85 and .66 * len(fill) <= height <= .98 * len(fill)
                and abs(x + width / 2 - fill.shape[1] / 2) <= .12 * len(fill)
                and abs(y + height / 2 - fill.shape[0] / 2) <= .12 * len(fill)):
            rims.append((x, y, width, height))
    if len(rims) != 1:
        return None
    x, y, width, height = rims[0]
    inner = low[y:y + height, x:x + width]
    cuts = []
    for level in (181, 201, 221):
        ink = np.uint8(inner >= level)
        a, b, w, h = cv2.boundingRect(ink)
        if not (.29 * height <= h <= .57 * height and .72 <= w / max(h, 1) <= 1.47
                and abs(a + w / 2 - width / 2) <= .13 * height and abs(b + h / 2 - height / 2) <= .13 * height):
            continue
        mark = ink[b:b + h, a:a + w]
        if mark.mean() < .22:
            mark = cv2.dilate(np.pad(mark, 1), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        if _check_shape(mark):
            cuts.append((a, b, w, h))
    return (left + x, top + y, left + x + width, top + y + height) if len(cuts) >= 2 else None


def _dismiss_button(frame, gray, boxes):
    height, width = frame.shape[:2]
    matches = []
    for box in boxes:
        left, top, right, bottom = box
        size = bottom - top
        if (left < .82 * width or bottom > .207 * height or not .071 * height <= size <= .155 * height
                or not .79 <= (right - left) / size <= 1.26):
            continue
        opposite = [peer for peer in boxes if peer[2] <= .26 * width and 2.21 * size <= peer[2] - peer[0] <= 4.65 * size
                    and max(abs(peer[1] - top), abs(peer[3] - bottom)) <= .24 * size]
        if len(opposite) != 1:
            continue
        start, end = round(.31 * width), round(.69 * width)
        title = gray[top:bottom, start:end]
        paper = np.median(title)
        if (abs(title.astype(np.int16) - paper) <= 13).mean() < .80 or (glyph := _filled_check(frame, box)) is None:
            continue
        run = _title_run(title, paper, size * .45, minimum=4, high=1.18)
        if run is None:
            continue
        a, b = run
        if abs(start + (a + b) / 2 - .5 * width) > .042 * width or b - a < 1.33 * size:
            continue
        matches.append(glyph)
    return matches[0] if len(matches) == 1 else None


def landscape_media_actions(frame, regions):
    """Map standard transport shapes and a checked header button in their owning toolbar."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or width < 1.52 * height or height < 220:
        return []
    boxes = [tuple(row["box"]) for row in regions["candidates"] if row.get("visibility") == "page"
             and 0 <= row["box"][0] < row["box"][2] <= width and 0 <= row["box"][1] < row["box"][3] <= .215 * height]
    if len(boxes) < 2:
        return []
    gray = cv2.cvtColor(frame[:round(.225 * height)], cv2.COLOR_BGR2GRAY)
    transport = []
    for row in _slot_groups(boxes, width, height):
        if not _solid_group(gray, row):
            continue
        marks = [_shape_mark(frame, gray, box, icon) for icon, box in zip(("stop", "play", "record"), row, strict=True)]
        if all(marks):
            transport.append(marks)
    done = _dismiss_button(frame, gray, boxes)
    if len(transport) == 1 and done is None:
        return [_action(role, text, box) for (role, text), box in zip((
            ("stop_playback", "停止播放"), ("toggle_playback", "切换播放"), ("record", "录制")), transport[0], strict=True)]
    return [_action("close", "完成", done)] if not transport and done is not None else []
