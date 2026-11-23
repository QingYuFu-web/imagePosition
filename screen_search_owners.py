"""Confirm a search point using a measured field or a first-row tool group."""

from itertools import combinations
import re

import cv2
import numpy as np

from screen_back_control import _center, _status_band
from screen_candidate_glyphs import _overlap, _search
from screen_caption_transcript import read_caption
from screen_component_captions import _contrast


_SEARCH = re.compile(r"(?:search(?:\s|$)|搜索|搜尋|查找|検索|rechercher(?:\s|$)|suchen(?:\s|$)|buscar(?:\s|$))")


def _bounds(frame, box, width):
    left, top, right, bottom = box
    crop = frame[top:bottom, left:right]
    delta, paper = _contrast(crop)
    if (delta <= 12).mean() < .66:
        return None, paper
    seen, stable = set(), []
    span, depth = right - left, bottom - top
    for level in (37, 49, 65, 78, 105, 148):
        ink = np.uint8(delta >= level)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        for i, stats_row in enumerate(stats[1:], 1):
            x, y, w, h = map(int, stats_row[:4])
            if (not .025 * width <= min(w, h) <= max(w, h) <= .087 * width or not .79 <= w / h <= 1.26
                    or not 0 < x < x + w < span or not 0 < y < y + h < depth
                    or abs(x + w / 2 - span / 2) > .22 * span or abs(y + h / 2 - depth / 2) > .22 * depth):
                continue
            part = np.uint8(labels[y:y + h, x:x + w] == i)
            identity = (x, y, part.shape, part.tobytes())
            if identity not in seen and _search(part):
                seen.add(identity)
                stable.append((left + x, top + y, left + x + w, top + y + h))
    if len(stable) >= 2 and all(_overlap(stable[0], other) >= .79 for other in stable[1:]):
        return stable[len(stable) // 2], paper
    return None, paper


def _surface(frame, glyph, paper):
    height, width = frame.shape[:2]
    gx, gy = _center(glyph)
    size = glyph[3] - glyph[1]
    top, bottom = max(0, round(gy - 1.72 * size)), min(height, round(gy + 1.72 * size))
    patch = frame[top:bottom]
    reads = []
    for tolerance in (3, 7):
        ink = cv2.inRange(patch, np.maximum(0, paper - tolerance).astype(np.uint8),
                          np.minimum(255, paper + tolerance).astype(np.uint8))
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (not .47 * width <= w <= .985 * width or not 1.64 * size <= h <= 3.07 * size
                    or not 0 < y < y + h < len(patch) or not 0 < x < x + w < width
                    or not .022 * height <= top + y <= .145 * height or top + y + h > .183 * height
                    or not .26 * h <= gx - x <= .98 * h or abs(gy - (top + y + h / 2)) > .13 * h
                    or cv2.contourArea(contour) < .94 * w * h):
                continue
            side = max(3, round(.10 * h))
            windows = (ink[y:y + side, x:x + side], ink[y:y + side, x + w - side:x + w],
                       ink[y + h - side:y + h, x:x + side], ink[y + h - side:y + h, x + w - side:x + w])
            if min(part.mean() for part in windows) < 251:
                reads.append((tolerance, (x, top + y, x + w, top + y + h)))
    validated = [box for level, box in reads if any(k != level and _overlap(box, other) >= .925 for k, other in reads)]
    if not validated or any(_overlap(validated[0], other) < .925 for other in validated[1:]):
        return None
    return validated[-1]


def _peer_mark(frame, box, paper):
    x, y, r, b = box
    tile = frame[y:b, x:r]
    off, surface = _contrast(tile)
    if np.max(abs(surface - paper)) > 13 or (off <= 14).mean() < .77:
        return None
    ink = np.uint8(off >= 46)
    a, t, w, h = cv2.boundingRect(ink)
    width, height = r - x, b - y
    if (not .22 * height <= h <= .69 * height or not .05 * width <= w <= .70 * width or not 0 < a < a+w < width
            or not 0 < t < t+h < height or abs(a + w / 2 - width / 2) > .13 * width
            or abs(t + h / 2 - height / 2) > .15 * height):
        return None
    return (x + a, y + t, x + a + w, y + t + h), ink[t:t + h, a:a + w]


def _field_text(frame, field, glyph, paper, boxes, index, cache):
    x, y, r, b = field
    depth, span = b - y, r - x
    _, gy = _center(glyph)
    trailing = []
    for left, top, right, bottom in boxes:
        w, h = right - left, bottom - top
        if (.62 * span <= left - x <= .96 * span and right < r + .08 * depth
                and .42 * depth <= min(w, h) <= max(w, h) <= 1.27 * depth
                and abs((top + bottom) / 2 - gy) < .18 * depth):
            if _peer_mark(frame, (left, top, right, bottom), paper):
                trailing.append(left)
    text_start = glyph[2] + max(2, round(.035 * depth))
    text_limit = min(trailing, default=r) - max(3, round(.12 * depth))
    if text_limit - text_start < 1.75 * depth:
        return None
    text_y = round(gy - .30 * depth)
    text_b = round(gy + .32 * depth)
    reading = read_caption(frame, (text_start, text_y, text_limit, text_b), index,
                           cache=cache, choice=True, sentence_case=True, join_letters=True, join_radicals=True)
    if not reading or not _SEARCH.match(reading["text"].casefold().strip()):
        return None
    a, t, end, last = reading["line_box"]
    if not .24 * depth <= last - t <= .62 * depth or abs((t + last) / 2 - gy) > .13 * depth:
        return None
    if a <= text_start + .56 * depth:
        return (x, y, min(trailing, default=r), b), reading["text"]
    parsed = reading["text"].casefold().strip()
    if (parsed not in {"search", "搜索", "搜尋", "查找", "検索", "rechercher", "suchen", "buscar"}
            or a - x < .66 * span or r - end < .20 * depth):
        return None
    # Accept a separate submit word only if separated from a truly empty editable field.
    field_fill = frame[text_y:text_b, text_start:a - round(.24 * depth):3]
    if (np.max(abs(field_fill.astype(np.int16) - paper), axis=2) <= 12).mean() < .989:
        return None
    top, bottom = y + round(.30 * depth), b - round(.31 * depth)
    columns = np.max(abs(frame[top:bottom, a - round(.44 * depth):a].astype(np.int16) - paper), axis=2)
    ridges = np.flatnonzero((columns >= 10).mean(axis=0) >= .72)
    return ((x, y, a - round(.44 * depth) + int(ridges[0]), b), reading["text"]) if 1 <= len(ridges) <= .07 * depth else None


def _tool_row(frame, glyph, owner, paper, boxes):
    height, width = frame.shape[:2]
    x, y = _center(glyph)
    size = glyph[3] - glyph[1]
    if not (.51 * width <= x <= .76 * width and .050 * height <= y <= .113 * height):
        return False
    a, t, r, b = owner
    depth, span = b - t, r - a
    peers = []
    for box in boxes:
        cx, cy = _center(box)
        w, h = box[2] - box[0], box[3] - box[1]
        if (not .78 * span <= w <= 1.26 * span or not .78 * depth <= h <= 1.26 * depth
                or not .72 * span <= cx - x <= 2.68 * span or abs(cy - y) > .16 * depth):
            continue
        if (mark := _peer_mark(frame, box, paper)) is not None:
            peers.append((box, mark))
    groups = []
    for (first, m1), (second, m2) in combinations(sorted(peers, key=lambda p: p[0][0]), 2):
        one, two = (_center(box)[0] - center for box, center in ((first, x), (second, _center(first)[0])))
        if not .76 * span <= min(one, two) <= max(one, two) <= 1.28 * span or abs(one - two) > .16 * span:
            continue
        if second[2] < .916 * width or np.array_equal(m1[1], m2[1]):
            continue
        groups.append((first, second))
    if not groups:
        return False
    if not _status_band(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), glyph):
        return False
    # A separated body action can retain the same size and peers. Inspect
    # every row between the system status band and the first tool glyph.
    start, stop = round(.040 * height), min(glyph[1], *(mark[0][1] for _, mark in peers)) - 1
    if start >= stop:
        return False
    strip = frame[start:stop, round(.055 * width):round(.945 * width):3]
    distance = np.max(abs(strip.astype(np.int16) - paper), axis=2)
    if (distance <= 17).mean() < .85 or (distance >= 22).mean(axis=1).max() > .58:
        return False
    quiet = frame[round(y - .55 * size):round(y + .55 * size), round(.04 * width):a - round(.16 * depth):3]
    if (np.max(abs(quiet.astype(np.int16) - paper), axis=2) <= 14).mean() < .96:
        return False
    return True


def search_owner_actions(frame, regions, index, *, diagnostic=False):
    height, width = frame.shape[:2]
    if (index is None or regions.get("layer") != "page" or regions.get("regions") or width < 240
            or not 1.45 * width <= height <= 2.83 * width):
        return []
    boxes = [tuple(row["box"]) for row in regions["candidates"] if row.get("visibility") == "page"]
    if not boxes:
        return []
    result, used, cache = [], [], {}
    for box in boxes:
        a, t, r, b = box
        x, y = _center(box)
        if (not 0 <= a < r <= width or not 0 <= t < b <= height or not .023 * height <= y <= .171 * height
                or not .014 * width <= x <= .76 * width or not .032 * width <= min(r-a, b-t) <= max(r-a, b-t) <= .183 * width):
            continue
        glyph, paper = _bounds(frame, box, width)
        if not glyph or any(_overlap(glyph, other) >= .78 for other in used):
            continue
        gx, gy = _center(glyph)
        if any(l <= gx < r and t <= gy < b for l, t, r, b in regions.get("floating_controls", ())):
            continue
        if x <= .36 * width and (field := _surface(frame, glyph, paper)):
            verified = _field_text(frame, field, glyph, paper, boxes, index, cache)
            if verified:
                bounds, caption = verified
                point = _center(bounds)
                if any(l <= point[0] < r and t <= point[1] < b for l, t, r, b in regions.get("floating_controls", ())):
                    continue
                used.append(glyph)
                result.append({"role": "search", "function": "搜索", "box": list(bounds), "click_point": list(point),
                               **({"witness": "field", "caption": caption, "glyph": list(glyph)} if diagnostic else {})})
        elif _tool_row(frame, glyph, box, paper, boxes):
            used.append(glyph)
            result.append({"role": "search", "function": "搜索", "box": list(box), "click_point": [gx, gy],
                           **({"witness": "tools", "glyph": list(glyph)} if diagnostic else {})})
    return result if len(result) == 1 else []
