"""Locate miniature navigation glyphs owned by a landscape toolbar."""

import cv2
import numpy as np

from screen_candidate_glyphs import _overlap, _search


def _chevron(stroke):
    height, width = stroke.shape
    if not .37 <= width / height <= 1.12 or not .16 <= stroke.mean() <= .61:
        return False
    ys, xs = stroke.nonzero()
    means = np.bincount(ys, weights=xs, minlength=height) / np.maximum(1, np.bincount(ys, minlength=height))
    means /= max(1, width - 1)
    reach = max(1, round(.22 * height))
    center = means[round(.35 * height):round(.65 * height) + 1].min()
    first, last = means[:reach].mean(), means[-reach:].mean()
    if min(first, last) - center < .39 or abs(first - last) > .35:
        return False
    vertex = reach + int(np.argmin(means[reach:-reach]))
    start = np.linspace(means[0], means[vertex], vertex + 1)
    end = np.linspace(means[vertex], means[-1], height - vertex)
    shape = np.r_[start, end[1:]]
    return .30 * height <= vertex <= .66 * height and np.sqrt(np.mean(np.square(means - shape))) <= .148


def _marks(contrast, box, low, high):
    left, top, right, bottom = box
    paint = contrast[top:bottom, left:right]
    candidates = []
    for level in (51, 69, 84, 102, 119):
        _, groups, stats, _ = cv2.connectedComponentsWithStats(np.uint8(paint >= level))
        for i, (x, y, width, height, _) in enumerate(stats[1:], 1):
            if not (low <= height <= high and 0 < x < x + width < paint.shape[1]
                    and 0 < y < y + height < paint.shape[0]):
                continue
            if _chevron(groups[y:y + height, x:x + width] == i):
                candidates.append((level, (int(left + x), int(top + y), int(left + x + width), int(top + y + height))))
    stable = [box for level, box in candidates if any(
        peer_level != level and _overlap(box, peer) >= .61 for peer_level, peer in candidates)]
    if stable and all(_overlap(box, stable[0]) >= .56 for box in stable):
        return stable[len(stable) // 2]
    return None


def _text_run(contrast, box, size, centered_y, minimum):
    left, top, right, bottom = box
    patch = contrast[top:bottom, left:right]
    if min(patch.shape) <= 2:
        return False
    for threshold in (46, 53, 64):
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(patch >= threshold))
        letters = [(x, x + w) for x, y, w, h, area in stats[1:] if (
            .23 * size <= w <= 1.93 * size and .40 * size <= h <= 1.1 * size
            and .14 * w * h <= area <= .87 * w * h and abs(top + y + h / 2 - centered_y) <= .29 * size)]
        if not letters:
            continue
        runs = []
        for start, end in sorted(letters):
            if runs and start - runs[-1][-1][1] <= .61 * size:
                runs[-1].append((start, end))
            else:
                runs.append([(start, end)])
        if any(len(run) >= minimum and run[0][0] <= .96 * size
               and run[-1][1] - run[0][0] >= .57 * minimum * size for run in runs):
            return True
    return False


def _leading_arrow(contrast, frame_height, width):
    band = (round(.013 * width), 0, round(.184 * width), round(.102 * frame_height))
    mark = _marks(contrast, band, max(7, .022 * frame_height), .077 * frame_height)
    if mark is None:
        return None
    left, top, right, bottom = mark
    size = bottom - top
    if (left + right) / 2 > .149 * width or top > .043 * frame_height:
        return None
    label = (right + round(.20 * size), max(0, top - round(.16 * size)),
             min(width, right + round(6.9 * size)), min(len(contrast), bottom + round(.18 * size)))
    return mark if _text_run(contrast, label, size, (top + bottom) / 2, 3) else None


def _button_plane(gray, box, *, glyph_only):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    patch = gray[top:bottom, left:right].astype(np.int16)
    first, last = round(.30 * height), round(.76 * height)
    left_shoulder = patch[first:last, round(.11 * width):round(.23 * width)]
    right_shoulder = patch[first:last, round(.81 * width):round(.93 * width)]
    tone = np.median(np.concatenate((left_shoulder, right_shoulder), axis=1))
    shoulders = (abs(left_shoulder - tone) <= 9).mean(), (abs(right_shoulder - tone) <= 9).mean()
    if min(shoulders) < (.85 if glyph_only else .68):
        return None
    clean = patch[:, round(.12 * width):round(.88 * width)]
    ridges = (clean - tone >= 9).mean(axis=1)
    edge = max(2, round(.21 * height))
    if ridges[:edge + 2].max() < .49 or ridges[-edge - 2:].max() < .49:
        return None
    return tone


def _opposite_arrow(gray, contrast, candidates, frame_height, width):
    controls = []
    for row in candidates:
        left, top, right, bottom = row["box"]
        height = bottom - top
        if (.70 * width <= left < right <= width and 0 <= top < bottom <= .112 * frame_height
                and .037 * frame_height <= height <= .097 * frame_height and 1.85 <= (right - left) / height <= 3.65):
            controls.append((left, top, right, bottom))
    arrows = []
    for box in controls:
        left, top, right, bottom = box
        size = bottom - top
        if (left + right) / 2 < .865 * width or (tone := _button_plane(gray, box, glyph_only=True)) is None:
            continue
        mid = (right + left) / 2
        icon = (round(mid - .50 * size), top + round(.18 * size), round(mid + .50 * size), bottom - round(.10 * size))
        mark = _marks(contrast, icon, max(7, .31 * size), .75 * size)
        if mark is None:
            continue
        peers = []
        for peer in controls:
            x, y, r, b = peer
            if not (0 <= left - r <= .62 * size and abs((r - x) - (right - left)) <= .35 * size
                    and max(abs(y - top), abs(b - bottom)) <= .27 * size):
                continue
            if (paper := _button_plane(gray, peer, glyph_only=False)) is None or abs(paper - tone) > 11:
                continue
            center = (r + x) / 2
            pad = round(.86 * size)
            text = (round(center - pad), y + round(.17 * size), round(center + pad), b - round(.10 * size))
            if _text_run(contrast, text, .60 * size, (y + b) / 2, 2):
                peers.append(peer)
        if len(peers) == 1:
            arrows.append(mark)
    return arrows[0] if len(arrows) == 1 else None


def _outlined_field(gray, box):
    left, top, right, bottom = box
    pad = max(2, round(.10 * (bottom - top)))
    tile = gray[max(0, top - pad):bottom + pad, max(0, left - pad):right + pad].astype(np.int16)
    height, width = tile.shape
    middle_x = tile[:, round(.08 * width):round(.93 * width)]
    middle_y = tile[round(.18 * height):round(.83 * height)]

    def ridges(strip, leading, trailing):
        pad = max(2, round(.24 * height))
        edges = strip[2:-2] - np.maximum(strip[:-4], strip[4:])
        support = (edges >= 11).mean(axis=1)
        return support[:pad].max() >= leading and support[-pad:].max() >= trailing

    return ridges(middle_x, .76, .54) and ridges(middle_y.T, .68, .68)


def _magnifier(gray, box):
    left, top, right, bottom = box
    tile = gray[top:bottom, left:right]
    shadow, peak = np.percentile(tile, (50, 96))
    if peak - shadow < 64 or (abs(tile.astype(np.int16) - shadow) <= 16).mean() < .68:
        return None
    fine = cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    marks = []
    for relative in (.53, .64, .73, .82):
        ink = np.uint8(fine >= shadow + relative * (peak - shadow))
        x, y, width, height = cv2.boundingRect(ink)
        if (.79 <= width / max(1, height) <= 1.20 and .28 * fine.shape[0] <= height <= .74 * fine.shape[0]
                and min(x, y, fine.shape[1] - x - width, fine.shape[0] - y - height) >= 2
                and _search(ink[y:y + height, x:x + width])):
            marks.append((left + x // 2, top + y // 2, left + (x + width + 1) // 2, top + (y + height + 1) // 2))
    if len(marks) >= 2 and all(_overlap(marks[0], mark) >= .68 for mark in marks):
        return marks[len(marks) // 2]
    return None


def _header_search(gray, arrow, candidates, frame_height, width):
    arrow_x = (arrow[0] + arrow[2]) / 2
    fields = []
    for item in candidates:
        left, top, right, bottom = item["box"]
        w, h = right - left, bottom - top
        if not (.065 * width <= w <= .27 * width and .038 * frame_height <= h <= .12 * frame_height
                and 2.35 * h <= w <= 6.9 * h and .11 * frame_height <= (top + bottom) / 2 <= .34 * frame_height
                and abs(left - arrow_x) <= .66 * h and right <= .30 * width):
            continue
        if _outlined_field(gray, item["box"]):
            fields.append(item["box"])
    matches = []
    for left, top, right, bottom in fields:
        size = bottom - top
        for item in candidates:
            x, y, r, b = item["box"]
            if (-.12 * size <= x - right <= .47 * size and max(abs(y - top), abs(b - bottom)) <= .26 * size
                    and .58 * size <= r - x <= 1.37 * size and .68 * size <= b - y <= 1.33 * size):
                mark = _magnifier(gray, item["box"])
                if mark is not None and all(_overlap(mark, previous) < .68 for previous in matches):
                    matches.append(mark)
    return matches[0] if len(matches) == 1 else None


def landscape_header_actions(frame, regions):
    """Return a Back glyph and, if separately grounded, the header's small Search."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or height < 195 or width < 1.53 * height:
        return []
    gray = cv2.cvtColor(frame[:round(.36 * height)], cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, round(.024 * height) | 1),) * 2)
    contrast = cv2.subtract(gray, cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel))
    candidates = [row for row in regions["candidates"] if row.get("visibility") == "page"
                  and 0 <= row["box"][0] < row["box"][2] <= width and 0 <= row["box"][1] < row["box"][3] <= len(gray)]
    primary = _leading_arrow(contrast, height, width)
    opposite = _opposite_arrow(gray, contrast, candidates, height, width)
    if (primary is None) == (opposite is None):
        return []
    boxes = [("back", "返回", primary if primary is not None else opposite)]
    if primary is not None and (search := _header_search(gray, primary, candidates, height, width)) is not None:
        boxes.append(("search", "搜索", search))
    return [{"role": role, "function": function, "preview_text": function, "box": list(box),
             "click_point": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]} for role, function, box in boxes]
