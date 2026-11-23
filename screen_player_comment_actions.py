"""Recover an audio or video player's toolbar comment action from pixel evidence."""

import cv2
import numpy as np

from inventory_miniplayer import _right_triangle
from screen_video_controls import _speech


def _translate(box, left, top):
    x, y, width, height = box
    return left + x, top + y, left + x + width, top + y + height


def _agreed(matches, tolerance):
    if len(matches) < 2:
        return None
    points = np.asarray(matches)
    if np.ptp(points, axis=0).max() > tolerance:
        return None
    return tuple(map(int, np.rint(np.median(points, axis=0))))


def _player_ring(gray, box, width):
    left, top, right, bottom = box
    margin = max(2, round(.011 * width))
    left, top = max(0, left - margin), max(0, top - margin)
    tile = gray[top:bottom + margin, left:right + margin]
    paper = np.median(tile)
    matches = []
    for lift in (32, 56, 81):
        outlines, tree = cv2.findContours(np.uint8(tile >= paper + lift), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if tree is None:
            continue
        for ring, relation in zip(outlines, tree[0]):
            child, parent = relation[2:]
            if parent >= 0 or child < 0:
                continue
            x, y, w, h = bounds = cv2.boundingRect(ring)
            if not (.10 * width <= min(w, h) <= max(w, h) <= .21 * width and .92 <= w / h <= 1.08
                    and x > 0 and y > 0 and x + w < tile.shape[1] and y + h < tile.shape[0]):
                continue
            area = cv2.contourArea(ring)
            if not (.69 <= area / (w * h) <= .83 and .72 <= cv2.contourArea(outlines[child]) / area <= .96):
                continue
            inside = tree[0, child, 2]
            if inside < 0:
                continue
            parts = []
            while inside >= 0:
                glyph = outlines[inside]
                a, b, c, d = cv2.boundingRect(glyph)
                if .23 * h <= d <= .46 * h and abs(b + d / 2 - y - h / 2) <= .09 * h:
                    parts.append((glyph, a, c, d))
                inside = tree[0, inside, 0]
            if len(parts) == 1:
                glyph, a, c, d = parts[0]
                if abs(a + c / 2 - x - w / 2) <= .10 * w and _right_triangle(glyph, c, d):
                    matches.append(_translate(bounds, left, top))
            elif len(parts) == 2:
                parts.sort(key=lambda part: part[1])
                (_, a, c, d), (_, b, e, f) = parts
                if (.11 <= c / d <= .39 and .11 <= e / f <= .39 and .87 <= d / f <= 1.15
                        and .075 * w <= b - a - c <= .23 * w and abs((a + b + e) / 2 - x - w / 2) <= .09 * w):
                    matches.append(_translate(bounds, left, top))
    return _agreed(matches, max(2, .012 * width))


def _seekbar(gray, playback, width):
    top = playback[1]
    radius = max(4, round(.010 * width))
    first, end = max(radius, top - round(.17 * width)), top - round(.034 * width)
    if first >= end:
        return None
    x0, x1 = round(.08 * width), round(.92 * width)
    center = gray[first:end, x0:x1:3].astype(np.int16)
    before = gray[first - radius:end - radius, x0:x1:3]
    after = gray[first + radius:end + radius, x0:x1:3]
    strength = (center - np.maximum(before, after) >= 16).mean(axis=1)
    rows = np.flatnonzero(strength >= .76)
    if not len(rows) or rows[-1] - rows[0] > .015 * width:
        return None
    y = first + int(rows[len(rows) // 2])
    extent = max(5, round(.021 * width))
    band = gray[y - extent:y + extent + 1, x0:x1]
    floor = np.percentile(gray[y, x0:x1], 65)
    thumb = np.uint8(band >= min(242, floor + 30))
    diameter = max(3, round(.009 * width) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    thumb = cv2.morphologyEx(thumb, cv2.MORPH_OPEN, kernel)
    marks, _ = cv2.findContours(thumb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for mark in marks:
        _, a, w, h = cv2.boundingRect(mark)
        perimeter = cv2.arcLength(mark, True)
        if (.011 * width <= min(w, h) <= max(w, h) <= .042 * width
                and .72 <= w / h <= 1.35 and abs(a + h / 2 - extent) <= .22 * h
                and 4 * np.pi * cv2.contourArea(mark) >= .73 * perimeter ** 2):
            points.append(mark)
    return y if len(points) == 1 else None


def _comment_bubble(gray, box, width):
    left, top, right, bottom = box
    tile = gray[top:bottom, left:right]
    paper = np.median(tile)
    matches = []
    for lift in (34, 46, 58, 70, 82):
        outlines, tree = cv2.findContours(np.uint8(tile >= paper + lift), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if tree is None:
            continue
        parents = tree[0, :, 3]
        for index, outline in enumerate(outlines):
            x, y, w, h = bounds = cv2.boundingRect(outline)
            if (parents[index] >= 0 or not .030 * width <= min(w, h) <= max(w, h) <= .097 * width
                    or not 0 < x < x + w < tile.shape[1] or not 0 < y < y + h < tile.shape[0]):
                continue
            holes = [contour for contour, parent in zip(outlines, parents, strict=True) if parent == index]
            if len(holes) < 3:
                continue
            shape = cv2.drawContours(np.zeros((h, w), np.uint8), [outline], 0, 1, cv2.FILLED, offset=(-x, -y))
            if _speech(shape, bounds, cv2.contourArea(outline), holes) != 3:
                continue
            toe = np.nonzero(shape[-max(1, round(.13 * h)):])[1]
            if toe.size and abs(toe.mean() - w / 2) >= .20 * w:
                matches.append(_translate(bounds, left, top))
    return _agreed(matches, max(3, .008 * width))


def player_comment_actions(frame, regions):
    """Emit one comment point above verified playback controls and seek thumb."""
    height, width = frame.shape[:2]
    if (regions["layer"] != "page" or regions.get("regions") or width < 240
            or not 1.50 * width <= height <= 2.82 * width):
        return []
    candidates = [row for row in regions["candidates"] if row.get("visibility") == "page"]
    owners = []
    for row in candidates:
        x, y = row["center"]
        a, t, b, d = row["box"]
        if (.36 * width <= x <= .64 * width and .69 * height <= y <= .95 * height
                and .085 * width <= min(b - a, d - t) <= max(b - a, d - t) <= .26 * width):
            owners.append(row["box"])
    if not owners:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    negative = None
    tracks = []
    for box in owners:
        a, t, b, d = box
        inverted = np.median(gray[t:d:3, a:b:3]) >= 150
        if inverted and negative is None:
            negative = cv2.bitwise_not(gray)
        light = negative if inverted else gray
        if (ring := _player_ring(light, box, width)) and (track := _seekbar(light, ring, width)):
            tracks.append((track, inverted))
    if not tracks or len({inverted for _, inverted in tracks}) != 1:
        return []
    positions = [track for track, _ in tracks]
    if max(positions) - min(positions) > .012 * width:
        return []
    track_y = round(float(np.median(positions)))
    light = negative if tracks[0][1] else gray
    matches = []
    for row in candidates:
        x, y = row["center"]
        a, t, b, d = row["box"]
        if (not .10 * width <= x <= .92 * width or not .042 * width <= track_y - y <= .18 * width
                or not .047 * width <= min(b - a, d - t) <= max(b - a, d - t) <= .21 * width):
            continue
        box = _comment_bubble(light, (a, t, b, min(d, track_y - round(.014 * width))), width)
        if box and .040 * width <= track_y - (box[1] + box[3]) / 2 <= .16 * width:
            matches.append(box)
    if not matches or np.ptp(np.asarray(matches), axis=0).max() > .011 * width:
        return []
    box = matches[0]
    x, y = round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)
    if any(a <= x < b and t <= y < d for a, t, b, d in regions.get("floating_controls", ())):
        return []
    return [{"role": "comments", "function": "打开评论", "box": list(box), "click_point": [x, y], "preview_text": "评论"}]
