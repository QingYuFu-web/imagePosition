"""Read a mobile search field from its painted surface, lens, and placeholder."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_component_captions import _bands
from screen_icon_create import _cross
from screen_landscape_search_field import _available
from screen_native_tab_bar import _distance


def _close(first, second, tolerance):
    return max(abs(a - b) for a, b in zip(first, second, strict=True)) <= tolerance


def _plate(frame, proposal):
    height, width = frame.shape[:2]
    l, t, r, b = map(int, proposal)
    w, h = r - l, b - t
    sx, sy = round(.13 * w), round(.28 * h)
    paint = np.median(frame[t + sy:b - sy:3, l + sx:r - sx:4], axis=(0, 1))
    if paint.mean() < 202 or np.ptp(paint) > 23:
        return None
    pad = max(3, round(.147 * h))
    a, y = max(0, l - pad), max(0, t - pad)
    chip = frame[y:min(height, b + pad), a:min(width, r + pad)]
    distance = _distance(chip, paint)
    witnessed = []
    for threshold in (3, 6, 9):
        surface = np.uint8(distance <= threshold)
        contours, _ = cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, top, span, depth = cv2.boundingRect(contour)
            box = (a + x, y + top, a + x + span, y + top + depth)
            if (not .88 * w <= span <= 1.08 * w or not .69 * h <= depth <= 1.06 * h
                    or not _close(box, proposal, .29 * h) or not 1 <= x < x + span < chip.shape[1]
                    or not 1 <= top < top + depth < chip.shape[0]
                    or cv2.contourArea(contour) / (span * depth) < .90):
                continue
            corner = max(3, round(.20 * depth))
            tl = surface[top:top + corner, x:x + corner]
            tr = surface[top:top + corner, x + span - corner:x + span]
            if .48 <= min(tl.mean(), tr.mean()) <= max(tl.mean(), tr.mean()) <= .94:
                witnessed.append((threshold, box))
    stable = [box for level, box in witnessed if any(k != level and _close(box, other, .060 * h)
                                                     for k, other in witnessed)]
    if not stable or not all(_close(stable[0], box, .067 * h) for box in stable):
        return None
    left, top, right, bottom = stable[len(stable) // 2]
    flank = round(.095 * (right - left))
    interior = frame[top + 4:bottom - 4:3, left + flank:right - flank:4]
    if (_distance(interior, paint) <= 8).mean() < .82:
        return None
    margin_y = max(3, round(.065 * (bottom - top)))
    above = frame[top - margin_y:top, left + flank:right - flank]
    below = frame[bottom:bottom + margin_y, left + flank:right - flank]
    if min((_distance(above, paint) >= 9).mean(), (_distance(below, paint) >= 9).mean()) < .78:
        return None
    return (left, top, right, bottom), paint


def _lens(distance, size):
    proposals = []
    for level in (31, 41, 54, 65):
        ink = np.uint8(distance >= level)
        contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for contour, (_, _, child, parent) in zip(contours, hierarchy[0], strict=True):
            x, y, w, h = cv2.boundingRect(contour)
            if (parent != -1 or child < 0 or not .32 * size <= min(w, h) <= max(w, h) <= .58 * size
                    or not .82 <= w / h <= 1.22 or not 0 < y < y + h < ink.shape[0]):
                continue
            hole = contours[child]
            a, t, span, depth = cv2.boundingRect(hole)
            area = cv2.contourArea(hole)
            if not (.82 <= span / depth <= 1.19 and .397 * w * h <= area <= .71 * w * h
                    and .68 <= area / (span * depth) <= .91):
                continue
            cx, cy = a + (span - 1) / 2, t + (depth - 1) / 2
            q = .5 * (span + depth)
            x0, y0 = round(cx + .45 * q), round(cy + .45 * q)
            x1, y1 = round(cx + .85 * q) + 1, round(cy + .85 * q) + 1
            ys, xs = np.nonzero(ink[y0:y1, x0:x1])
            if len(xs) < .17 * q or np.count_nonzero(np.abs(x0 + xs - cx - (y0 + ys - cy)) <= .13 * q) < .14 * q:
                continue
            if abs(y + h / 2 - ink.shape[0] / 2) > .15 * size:
                continue
            proposals.append((level, (x, y, max(x + w, x1), y + h)))
    return [box for cut, box in proposals if any(k != cut and _close(box, other, .065 * size) for k, other in proposals)]


def _placeholder(frame, box, paint, index):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    inset = round(.15 * height)
    x, y = left + inset, top + inset
    distance = _distance(frame[y:bottom - inset, x:right - inset], paint)
    options = _lens(distance, height)
    if not options or not all(_close(options[0], lens, .07 * height) for lens in options):
        return None
    l, t, r, b = options[0]
    x0 = x + r
    if x0 > left + .60 * width or x0 + .76 * height > right - inset:
        return None
    clip = (x0, max(y, y + t - round(.08 * height)),
            right - inset, min(bottom, y + b + round(.12 * height)))
    reading = read_caption(frame, clip, index, cache={}, choice=True, join_radicals=True, sentence_case=True)
    if not reading:
        return None
    word = reading["text"].casefold().strip()
    a, first, end, last = reading["line_box"]
    if not word.startswith(("搜索", "搜尋", "查找", "search")) or len(word) > 22:
        return None
    if not .27 * height <= last - first <= .58 * height or not 0 <= a - x0 <= .45 * height:
        return None
    return word, ((l + x + end) // 2, (top + bottom) // 2)


def _circled_plus(frame, size, proposal):
    left, top, right, bottom = proposal
    field = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY).astype(np.int16)
    paint = np.median(field)
    alert = []
    for threshold in (55, 79, 104):
        mask = np.uint8(paint - field >= threshold)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        links = hierarchy[0]
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            child, parent = links[i, 2:]
            if (parent != -1 or child < 0 or not .41 * size <= min(w, h) <= max(w, h) <= .69 * size
                    or not .87 <= w / h <= 1.14 or not .69 <= cv2.contourArea(contour) / (w * h) <= .88):
                continue
            inner = links[child, 2]
            if inner < 0 or links[inner, 0] != -1 or not .69 <= cv2.contourArea(contours[child]) / cv2.contourArea(contour) <= .96:
                continue
            a, t, k, d = cv2.boundingRect(contours[inner])
            if (not .37 * w <= k <= .64 * w or not .36 * h <= d <= .66 * h
                    or abs(a + k / 2 - x - w / 2) > .11 * w or abs(t + d / 2 - y - h / 2) > .11 * h):
                continue
            if _cross(mask[t:t + d, a:a + k]):
                alert.append((threshold, (left + x, top + y, left + x + w, top + y + h)))
    return [box for cut, box in alert if any(q != cut and _close(box, other, .05 * size) for q, other in alert)]


def _menu(frame, regions, field, index):
    width = frame.shape[1]
    left, top, right, bottom = field
    if right - left < .86 * width:
        return None
    h = bottom - top
    owners = [row["box"] for row in regions["candidates"] if row["visibility"] == "page"
              and .84 * width <= row["center"][0] <= .97 * width
              and .50 * h <= top - row["center"][1] <= 1.23 * h
              and .55 * h <= row["box"][3] - row["box"][1] <= 1.48 * h]
    marks = [mark for crop in owners for mark in _circled_plus(frame, h, crop)]
    if not marks or not all(_close(marks[0], mark, .065 * h) for mark in marks):
        return None
    a, t, b, e = marks[0]
    cy = (t + e) / 2
    title = (round(.35 * width), round(cy - .39 * h), round(.65 * width), round(cy + .41 * h))
    reading = read_caption(frame, title, index, cache={}, choice=True, join_radicals=True)
    if reading and reading["text"].casefold() in {"微信", "wechat"} and _available(marks[0], regions):
        return {"role": "open_menu", "function": "打开添加菜单", "box": list(marks[0]),
                "click_point": [(a + b) // 2, (t + e) // 2], "preview_text": "添加菜单"}
    return None


def _unproposed_plate(frame):
    """Find a wide, bright header surface that a larger proposal may enclose."""
    height, width = frame.shape[:2]
    start, stop = round(.025 * height), round(.181 * height)
    tile = frame[start:stop]
    if tile.dtype == np.uint8 and tile.shape[2] == 3:
        bright = cv2.inRange(tile, (245, 245, 245), (255, 255, 255)) != 0
    else:
        bright = tile.min(axis=2) >= 245
    inset = round(.07 * width)
    rows = bright[:, inset:width - inset].mean(axis=1) >= .84
    seeds = []
    for top, bottom in _bands(rows):
        depth = bottom - top
        if not .052 * width <= depth <= .126 * width or start + top > .13 * height:
            continue
        middle = bright[top + depth // 4:bottom - depth // 4]
        runs = [(left, right) for left, right in _bands(middle.mean(axis=0) >= .72)
                if right - left >= .095 * width]
        if not runs:
            continue
        left, right = runs[0][0], runs[-1][1]
        if (.86 * width <= right - left <= .995 * width and min(left, width - right) >= .008 * width
                and bright[top:bottom, left:right].mean() >= .90):
            seeds.append((left, start + top, right, start + bottom))
    return seeds[0] if len(seeds) == 1 else None


def header_search_plate_actions(frame, regions, index, *, verify_menu=True):
    """Return a visually owned search field and any independently verified add menu."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or width < 240 or not 1.41 * width <= height <= 2.85 * width:
        return []
    found = []
    for row in regions["candidates"]:
        l, t, r, b = row["box"]
        if (row["visibility"] != "page" or not .022 * height <= t <= .13 * height or b > .181 * height
                or not .46 * width <= r - l <= .995 * width or not .052 * width <= b - t <= .126 * width):
            continue
        measured = _plate(frame, row["box"])
        if measured is None:
            continue
        box, paint = measured
        if (_available(box, regions) and not any(_close(box, old, .028 * width) for old, _ in found)
                and (witness := _placeholder(frame, box, paint, index))):
            found.append((box, witness))
    if not found and (seed := _unproposed_plate(frame)) and (measured := _plate(frame, seed)):
        box, paint = measured
        if _available(box, regions) and (witness := _placeholder(frame, box, paint, index)):
            found.append((box, witness))
    if len(found) != 1:
        return []
    box, (caption, point) = found[0]
    actions = [{"role": "search", "function": "搜索", "box": list(box), "click_point": list(point), "preview_text": caption}]
    if verify_menu and (menu := _menu(frame, regions, box, index)):
        actions.append(menu)
    return actions
