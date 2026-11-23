"""Recover leading navigation controls with visual evidence of toolbar ownership."""

from itertools import combinations

import cv2
import numpy as np

from inventory_back_evidence import _chevrons, is_shaft_back_arrow
from screen_back_control import _center, _status_band, _title_run, _unique
from screen_candidate_glyphs import _overlap


_ANGLES = np.arange(24) * (np.pi / 12)
_RING = np.vstack((np.cos(_ANGLES), np.sin(_ANGLES)))


def _circle(gray, box):
    left, top, right, bottom = box
    x, y = _center(box)
    diameter = min(right - left, bottom - top)
    chip = gray[top:bottom, left:right]
    def samples(radius):
        xx, yy = np.rint(np.array([[x], [y]]) + radius * diameter * _RING).astype(int)
        return gray[yy, xx].astype(np.int16)

    height, width = gray.shape
    if min(x, width - 1 - x, y, height - 1 - y) <= .54 * diameter:
        return None
    inner, outer = samples(.32), samples(.54)
    dark = float(np.median(inner))
    gap = float(np.median(outer - inner))
    if np.ptp(inner) > 21 or gap < 25 or (outer - inner >= 18).mean() < .79:
        return None
    stable = []
    for cut in (.32, .60):
        mask = np.uint8(chip <= dark + cut * gap)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for outline in contours:
            a, t, w, h = cv2.boundingRect(outline)
            if (not .71 * diameter <= min(w, h) <= max(w, h) <= 1.03 * diameter
                    or not .91 <= w / h <= 1.10 or abs(a + w / 2 + left - x) > .11 * diameter
                    or abs(t + h / 2 + top - y) > .11 * diameter):
                continue
            area, perimeter = cv2.contourArea(outline), cv2.arcLength(outline, True)
            if .70 <= area / (w * h) <= .85 and 4 * np.pi * area >= .84 * perimeter * perimeter:
                stable.append((left + a, top + t, left + a + w, top + t + h))
    return (stable[0], dark) if len(stable) == 2 and _overlap(*stable) >= .85 else None


def _left_mark(mask):
    x, y, w, h = cv2.boundingRect(mask)
    size = mask.shape[0]
    if not .35 * size <= h <= .65 * size or not .39 <= w / h <= .73:
        return False
    body = mask[y:y + h, x:x + w]
    if not .10 <= body.mean() <= .39:
        return False
    # Thin the stroke by one edge pixel when a filled toolbar has heavy ink.
    thin = cv2.erode(body, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    for candidate in (body, thin):
        _, _, _, height = cv2.boundingRect(candidate)
        if .79 * h <= height <= h and list(_chevrons(candidate, h / 2, height, height)):
            return True
    return False


def _plus_mark(mask):
    x, y, w, h = cv2.boundingRect(mask)
    if not .28 * mask.shape[0] <= max(w, h) <= .63 * mask.shape[0] or not .79 <= w / max(1, h) <= 1.27:
        return False
    body = mask[y:y + h, x:x + w]
    hx, hy = max(2, round(.27 * w)), max(2, round(.27 * h))
    return (.13 <= body.mean() <= .48 and body[h // 2].mean() >= .80 and body[:, w // 2].mean() >= .80
            and not any(chip.any() for chip in (body[:hy, :hx], body[:hy, -hx:], body[-hy:, :hx], body[-hy:, -hx:])))


def _dots_mark(mask):
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dots = sorted(tuple(row) for row in stats[1:] if row[4] >= 3)
    if len(dots) != 3:
        return False
    columns = np.array([x + w / 2 for x, _, w, _, _ in dots])
    rows = [y + h / 2 for _, y, _, h, _ in dots]
    spans = [w for _, _, w, _, _ in dots]
    return (all(.055 * mask.shape[0] <= min(w, h) <= max(w, h) <= .17 * mask.shape[0] and area >= .45 * w * h
                for _, _, w, h, area in dots)
            and np.ptp(rows) <= .21 * np.median(spans) and np.ptp(np.diff(columns)) <= .22 * np.mean(np.diff(columns))
            and .31 * mask.shape[1] <= columns[-1] - columns[0] <= .60 * mask.shape[1])


def _circle_kind(gray, box, fill):
    l, t, r, b = box
    chip = gray[t:b, l:r].astype(np.int16)
    yy, xx = np.ogrid[:b - t, :r - l]
    core = np.hypot(yy - (b - t - 1) / 2, xx - (r - l - 1) / 2) <= .40 * min(b - t, r - l)
    peak = float(chip[core].max())
    if peak - fill < 52:
        return None
    kinds = [kind for kind, proof in (("back", _left_mark), ("add", _plus_mark), ("more", _dots_mark))
             if sum(proof(np.uint8(core & (chip >= fill + fraction * (peak - fill))))
                    for fraction in (.55, .70, .83)) >= 2]
    return kinds[0] if len(kinds) == 1 else None


def _floating_toolbar_back(gray, boxes):
    height, width = gray.shape
    buttons = []
    for box in boxes:
        l, t, r, b = box
        x, y = _center(box)
        if (not (.018 * width <= x <= .142 * width or .66 * width <= x <= .985 * width)
                or not .034 * height <= y <= .12 * height or not .049 * width <= b - t <= .135 * width
                or not .80 <= (r - l) / (b - t) <= 1.25):
            continue
        if (witness := _circle(gray, box)) is not None:
            owner, fill = witness
            if kind := _circle_kind(gray, owner, fill):
                buttons.append((kind, owner))
    found = []
    for kind, target in buttons:
        x, y = _center(target)
        size = target[3] - target[1]
        if kind != "back" or x > .142 * width or not _status_band(gray, target):
            continue
        peers = [(role, other) for role, other in buttons if _center(other)[0] >= .66 * width
                 and abs(_center(other)[1] - y) <= .16 * size and .83 <= (other[3] - other[1]) / size <= 1.18]
        if any({one[0], two[0]} == {"add", "more"} and .99 * size <= abs(_center(one[1])[0] - _center(two[1])[0]) <= 2.36 * size
               for one, two in combinations(peers, 2)):
            found.append(target)
    return found


def _long_shaft(body):
    h, w = body.shape
    if not 1.24 <= w / h <= 1.73:
        return False
    core = round(1.18 * h)
    tail = body[:, core:]
    a, b = round(.36 * h), round(.64 * h)
    return (tail[a:b].any() and not (tail[:a].any() or tail[b:].any())
            and is_shaft_back_arrow(body[:, :core]))


def _long_shaft_marks(gray, box, width):
    l, t, r, b = box
    chip = gray[t:b, l:r].astype(np.int16)
    paper = float(np.median(chip))
    if (abs(chip - paper) <= 12).mean() < .86:
        return []
    found = []
    for cut in (55, 76, 98):
        mask = np.uint8(abs(chip - paper) >= cut)
        x, y, w, h = cv2.boundingRect(mask)
        if (not .024 * width <= h <= .069 * width or not 0 < x < x + w < r - l or not 0 < y < y + h < b - t
                or abs(x + w / 2 - (r - l) / 2) > .17 * (r - l)
                or abs(y + h / 2 - (b - t) / 2) > .17 * (b - t)):
            continue
        if _long_shaft(mask[y:y + h, x:x + w]):
            found.append((l + x, t + y, l + x + w, t + y + h))
    return found if len(found) >= 2 and all(_overlap(found[0], mark) >= .84 for mark in found[1:]) else []


def _compact_status(gray, target, paper):
    height, width = gray.shape
    _, y = _center(target)
    h = target[3] - target[1]
    end = round(min(.061 * height, y - 1.06 * h))
    if end <= h:
        return False
    ink = abs(gray[:end:2, ::2].astype(np.int16) - paper) >= 40
    lane = ink.shape[1]
    a = ink[:, round(.019 * lane):round(.28 * lane)].mean(axis=1)
    b = ink[:, round(.71 * lane):round(.982 * lane)].mean(axis=1)
    middle = ink[:, round(.35 * lane):round(.63 * lane)].mean(axis=1)
    rows = 2 * np.flatnonzero((a >= .071) & (b >= .14) & (middle <= .07))
    return (.25 * h <= len(rows) * 2 <= 1.24 * h and np.ptp(rows) <= 1.25 * h
            and 1.88 * h <= y - np.median(rows) <= 4.47 * h)


def _large_title_toolbar(gray, target, boxes):
    height, width = gray.shape
    _, y = _center(target)
    h = target[3] - target[1]
    if not any(l <= target[0] and r - l >= .85 * width and t < y < b and abs((t + b) / 2 - y) <= .5 * h
               for l, t, r, b in boxes):
        return False
    l, r = round(.17 * width), round(.96 * width)
    strip = gray[round(y - .74 * h):round(y + 2.20 * h):2, l:r:2].astype(np.int16)
    surface = float(np.median(strip))
    if (abs(strip - surface) <= 10).mean() < .985 or not (_status_band(gray, target) or _compact_status(gray, target, surface)):
        return False

    title_top, title_end = round(y + 1.81 * h), round(y + 7.15 * h)
    a, b = round(.030 * width), round(.80 * width)
    title = _title_run(gray[title_top:title_end, a:b], surface, 1.42 * h, minimum=3, high=2.19)
    if title is None or not (.042 * width <= a + title[0] <= .13 * width and 3.16 * h <= title[1] - title[0] <= .55 * width):
        return False
    rows = [(l, t, r, b) for l, t, r, b in boxes if l <= .074 * width and r >= .90 * width
            and .78 * width <= r - l and 2.3 * h <= b - t <= 6.8 * h and title_end <= t <= title_end + 12 * h]
    return any(abs(first[0] - second[0]) <= .041 * width and abs(first[2] - second[2]) <= .052 * width
               and .74 <= (first[3] - first[1]) / (second[3] - second[1]) <= 1.36
               and abs(second[1] - first[3]) <= .88 * h for first, second in combinations(sorted(rows, key=lambda b: b[1]), 2))


def _aligned_body_starts(boxes, width, height, mark):
    size = mark[3] - mark[1]
    peers = [box for box in boxes if .19 * width <= box[2] - box[0] <= .55 * width
             and box[3] - box[1] >= 1.65 * size and mark[3] + 1.2 * size <= box[1] <= .32 * height]
    for left, right in combinations(sorted(peers), 2):
        left_height, right_height = left[3] - left[1], right[3] - right[1]
        if (_center(left)[0] < .41 * width < .59 * width < _center(right)[0]
                and -.025 * width <= right[0] - left[2] <= .34 * width
                and abs(left[1] - right[1]) <= max(.6 * size, .09 * min(left_height, right_height))
                and .60 <= left_height / right_height <= 1.66):
            yield min(left[1], right[1])


def _spaced_letter_run(ink):
    """Recognize separated typographic cells, including multistroke CJK glyphs."""
    height = ink.shape[0]
    columns = ink.sum(axis=0) >= max(2, round(.055 * height))
    spans = np.flatnonzero(np.diff(np.r_[False, columns, False])).reshape(-1, 2)
    cells = [ink[:, left:right] for left, right in spans if .28 * height <= right - left <= 1.32 * height]
    return sum(.09 <= cell.mean() <= .68 and cell.any(axis=1).sum() >= .57 * height for cell in cells) >= 3


def _title_and_subline(gray, mark, body_top, paper):
    _, width = gray.shape
    _, center_y = _center(mark)
    size = mark[3] - mark[1]
    x0, x1 = round(.018 * width), round(.97 * width)
    y0 = round(center_y - 1.40 * size)
    y1 = body_top - round(.24 * size)
    if not 0 <= y0 < y1:
        return False
    tile = gray[y0:y1, x0:x1].copy()
    left, top, right, bottom = mark
    tile[max(0, top - y0 - 2):bottom - y0 + 2, max(0, left - x0 - 2):right - x0 + 2] = paper
    delta = abs(tile.astype(np.int16) - paper)
    if (delta <= 12).mean() < .84:
        return False
    ink = delta >= 42
    rows = ink.sum(axis=1) >= max(3, round(.12 * size))
    spans = np.flatnonzero(np.diff(np.r_[False, rows, False])).reshape(-1, 2)
    bands = []
    for top, bottom in spans:
        run_height = bottom - top
        if not .4 * size <= run_height <= 3.05 * size:
            continue
        offset, _, length, _ = cv2.boundingRect(np.uint8(ink[top:bottom]))
        if not .072 * width <= length <= .66 * width:
            continue
        pad = max(2, round(.12 * size))
        tile_top, tile_bottom = max(0, top - pad), min(tile.shape[0], bottom + pad)
        run_tile = tile[tile_top:tile_bottom, max(0, offset - pad):offset + length + pad]
        if (_title_run(run_tile, paper, max(.70 * size, .53 * run_height), minimum=3, high=2.6) is not None
                or _spaced_letter_run(ink[top:bottom, offset:offset + length])):
            bands.append((y0 + top, y0 + bottom, x0 + offset, x0 + offset + length / 2))
    if len(bands) != 2:
        return False
    title, subline = bands
    if not (.17 * size <= subline[0] - title[1] <= 3.0 * size
            and .27 * size <= body_top - subline[1] <= 3.5 * size):
        return False
    if title[0] <= center_y <= title[1]:
        return max(abs(title[3] - .5 * width), abs(subline[3] - .5 * width)) <= .067 * width
    return (1.28 * size <= title[0] - center_y <= 4.6 * size
            and max(title[2], subline[2]) <= .145 * width and abs(title[2] - subline[2]) <= .046 * width
            and title[1] - title[0] >= 1.22 * (subline[1] - subline[0]))


def _double_title_toolbar(gray, mark, boxes):
    height, width = gray.shape
    if not _status_band(gray, mark):
        return False
    left, top, right, bottom = mark
    paper = float(np.median(gray[max(0, 2 * top - bottom):bottom, left:right]))
    return (_compact_status(gray, mark, paper)
            and any(_title_and_subline(gray, mark, start, paper)
                    for start in _aligned_body_starts(boxes, width, height, mark)))


def context_back(frame, regions, *, gray=None):
    """Return one uniquely corroborated leading mark on a portrait page."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or regions.get("regions") or width < 240 or not 1.38 * width <= height <= 2.8 * width:
        return None
    boxes = [tuple(row["box"]) for row in regions["candidates"] if row["visibility"] == "page"]
    left = [box for box in boxes if .026 * width <= _center(box)[0] <= .136 * width
            and .036 * height <= _center(box)[1] <= .113 * height
            and .042 * width <= min(box[2] - box[0], box[3] - box[1]) <= max(box[2] - box[0], box[3] - box[1]) <= .156 * width]
    if not left:
        return None
    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = _floating_toolbar_back(gray, boxes)
    shafts = [mark for box in left for mark in _long_shaft_marks(gray, box, width)]
    found.extend(mark for mark in shafts if _large_title_toolbar(gray, mark, boxes))
    if not found:
        found = [mark for mark in _unique(shafts) if _double_title_toolbar(gray, mark, boxes)]
    unique = _unique(found)
    if len(unique) == 1:
        box = unique[0]
        x, y = _center(box)
        if not any(l <= x < r and t <= y < b for l, t, r, b in regions.get("floating_controls", ())):
            return {"role": "back", "function": "返回上一级", "box": list(box), "click_point": [x, y]}
    return None
