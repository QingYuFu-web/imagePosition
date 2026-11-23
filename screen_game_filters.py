"""Read checked filter controls inside a visually bounded landscape overlay."""

from itertools import combinations

import cv2
import numpy as np

from inventory_choice_glyphs import _tick
from screen_caption_transcript import read_caption
from screen_game_primary import _button_box
from screen_landscape_segmented_choices import _centered_ink


def _diamonds(image, gray):
    blue, green, red = image.astype(np.int16).transpose(2, 0, 1)
    colored = np.uint8((red >= 110) & (red - blue >= 55) & (green - blue >= 30) & (red - green >= 24))
    outlines, _ = cv2.findContours(colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height = gray.shape[0]
    matches = []
    for outline in outlines:
        x, y, w, h = cv2.boundingRect(outline)
        if (not .021 * height <= h <= .047 * height or not .83 <= w / h <= 1.20
                or not .36 <= cv2.contourArea(outline) / (w * h) <= .59):
            continue
        polygon = cv2.approxPolyDP(outline, .052 * cv2.arcLength(outline, True), True).reshape(-1, 2)
        if len(polygon) != 4:
            continue
        nx, ny = (polygon - (x, y)).T / np.array([[w], [h]])
        if (not np.allclose(np.sort(nx), [0, .5, .5, 1], atol=.22)
                or not np.allclose(np.sort(ny), [0, .5, .5, 1], atol=.22)):
            continue
        sy, sx = np.ogrid[:h, :w]
        inside = abs((sx + .5) / w - .5) + abs((sy + .5) / h - .5) <= .40
        tile = gray[y:y + h, x:x + w]
        light = float(np.median(tile[inside]))
        strength = np.float32(light) - tile
        mark = np.uint8(inside & (strength >= max(25, .31 * light)))
        u, v, mw, mh = cv2.boundingRect(mark)
        if not (.24 * w <= mw <= .67 * w and .23 * h <= mh <= .67 * h):
            continue
        roi = np.s_[v:v + mh, u:u + mw]
        if _tick(mark[roi] != 0, strength[roi], minimum_fill=.19):
            matches.append((x + w / 2, y + h / 2, (w + h) / 2))
    return matches


def _checked_row(marks, width):
    if not 4 <= len(marks) <= 18:
        return None
    rows = []
    for row in combinations(sorted(marks), 4):
        centers, heights, sizes = np.array(row).T
        stride = np.diff(centers)
        unit = np.median(sizes)
        if (np.ptp(heights) <= .26 * unit and np.ptp(sizes) <= .20 * unit
                and .06 * width <= stride.min() <= stride.max() <= .19 * width
                and np.ptp(stride) <= .075 * stride.min()):
            rows.append(row)
    return rows[0] if len(rows) == 1 else None


def _edge_peak(edges, low, high, start, end):
    start, end = max(0, round(start)), min(edges.shape[0], round(end))
    low, high = max(0, round(low)), min(edges.shape[1], round(high))
    if start >= end or low >= high:
        return None
    support = (edges[start:end, low:high] != 0).mean(axis=0)
    strongest = int(np.argmax(support))
    return low + strongest if support[strongest] >= .82 else None


def _panel_box(gray, row):
    height, width = gray.shape
    first, y, _ = row[0]
    last = row[-1][0]
    pitch = (last - first) / (len(row) - 1)
    edges = cv2.Canny(gray, 18, 41)
    left = _edge_peak(edges, first - 1.44 * pitch, first - .69 * pitch, y - .18 * height, y + .24 * height)
    right = _edge_peak(edges, last + .65 * pitch, last + 1.44 * pitch, y - .18 * height, y + .24 * height)
    if left is None or right is None or not .38 * width <= right - left <= .72 * width:
        return None
    inset = round(.07 * (right - left))
    top = _edge_peak(edges.T, y - .43 * height, y - .20 * height, left + inset, right - inset)
    bottom = _edge_peak(edges.T, y + .22 * height, y + .46 * height, left + inset, right - inset)
    if top is None or bottom is None or not .53 * height <= bottom - top <= .84 * height:
        return None
    interior = gray[top + 5:bottom - 5, left + inset:right - inset]
    if np.percentile(interior, 76) > 43:
        return None
    return left, top, right, bottom


def _caption(frame, box, index, cache, margin=.017):
    reading = read_caption(frame, box, index, cache=cache)
    if reading and reading["min_margin"] >= margin and reading["max_distance"] <= .16:
        return reading["text"], reading["line_box"]
    return None


def _filter_entry(frame, panel, mark, index, cache, *, row_choice):
    x, y, size = mark
    left, top, right, bottom = panel
    label = _caption(frame, (round(x + .73 * size), max(top, round(y - .87 * size)),
                             min(right, round(x + 3.70 * size)), min(bottom, round(y + .87 * size))), index, cache)
    if label is None:
        return None
    name, (label_left, label_top, label_right, label_bottom) = label
    if not row_choice:
        if name not in {"开启", "启用", "打开"}:
            return None
        owner = _caption(frame, (max(left, round(x - 6.1 * size)), max(top, round(y - .87 * size)),
                                round(x - 1.02 * size), min(bottom, round(y + .87 * size))), index, cache, margin=.014)
        if owner is None:
            return None
        name, (label_left, heading_top, _, heading_bottom) = owner
        label_top = min(label_top, heading_top)
        label_bottom = max(label_bottom, heading_bottom)
    function = f"取消勾选{name}" if row_choice else f"关闭{name}筛选"
    chip = .77 * size
    return {"role": "checkbox_setting", "function": function, "preview_text": function,
            "box": [min(label_left, round(x - chip)), min(label_top, round(y - chip)), label_right,
                    max(label_bottom, round(y + chip))],
            "click_point": [round(x), round(y)]}


def _visual_filter_entry(frame, panel, mark):
    """Keep a visible checkbox and caption when the caption cannot be read."""
    x, y, size = mark
    left, top, right, bottom = panel
    a, t = round(x + .73 * size), max(top, round(y - .87 * size))
    b, z = min(right, round(x + 3.70 * size)), min(bottom, round(y + .87 * size))
    if a >= b or t >= z:
        return None
    gray = cv2.cvtColor(frame[t:z, a:b], cv2.COLOR_BGR2GRAY)
    contrast = gray.astype(np.int16) - np.median(gray)
    peak = float(np.percentile(contrast, 96))
    if peak < 45:
        return None
    ink = np.uint8(contrast >= max(26, .53 * peak))
    offset, start, span, height = cv2.boundingRect(ink)
    if (not 1.10 * size <= span <= 2.65 * size or not .62 * size <= height <= 1.35 * size
            or abs(t + start + height / 2 - y) > .26 * size or not .06 <= ink.mean() <= .27
            or not .80 * size <= a + offset - x <= 1.60 * size):
        return None
    pad = max(2, round(.19 * size))
    box = [round(x - .77 * size) - pad, round(min(y - .77 * size, t + start)) - pad,
           a + offset + span + pad, round(max(y + .77 * size, t + start + height)) + pad]
    if not (left <= box[0] < box[2] <= right and top <= box[1] < box[3] <= bottom):
        return None
    return {"box": box, "click_point": [round(x), round(y)], "fixed_center": True}


def _search(frame, panel, regions, index, cache, *, include_visual_candidates):
    left, top, right, bottom = panel
    width, height = right - left, bottom - top
    matches = []
    visual = []
    for candidate in regions["candidates"]:
        x0, y0, x1, y1 = candidate["box"]
        if (candidate["visibility"] != "page" or not (left + .49 * width <= x0 < x1 <= right
                                                       and top + .79 * height <= y0 < y1 <= bottom)
                or not (.09 * width <= x1 - x0 <= .24 * width and .05 * height <= y1 - y0 <= .15 * height)):
            continue
        if (box := _button_box(frame, candidate["box"])) is None:
            continue
        x0, y0, x1, y1 = box
        pad_x, pad_y = round(.27 * (x1 - x0)), round(.13 * (y1 - y0))
        label = _caption(frame, (x0 + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y), index, cache)
        if label is not None and label[0] == "搜索":
            matches.append({"role": "search", "function": "搜索", "preview_text": "搜索", "box": box,
                            "click_point": [(x0 + x1) // 2, (y0 + y1) // 2]})
        elif include_visual_candidates:
            chip = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            if _centered_ink(chip, (0, 0, x1 - x0, y1 - y0)) is not None:
                visual.append({"box": box, "click_point": [(x0 + x1) // 2, (y0 + y1) // 2], "fixed_center": True})
    return (matches, []) if len(matches) == 1 else ([], visual if not matches and len(visual) == 1 else [])


def _small_filter_layout(small):
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    marks = _diamonds(small, gray)
    if (row := _checked_row(marks, small.shape[1])) is None or (area := _panel_box(gray, row)) is None:
        return None
    return area, row, marks


def game_filter_layout(frame, *, camera=False):
    """Find the verified panel, its checked row and all tick marks in image coordinates."""
    height, width = frame.shape[:2]
    if width < 1.42 * height:
        return None
    scale = min(1., 1100 / width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    layout = None
    if camera:
        from screen_photo_geometry import prepare_camera_screen

        layout = _small_filter_layout(prepare_camera_screen(small))
    if layout is None and (layout := _small_filter_layout(small)) is None:
        return None
    area, row, marks = layout

    def native(mark):
        return tuple(value / scale for value in mark)

    return native(area), tuple(map(native, row)), tuple(map(native, marks))


def game_filter_actions(frame, regions, index, *, include_visual_candidates=False, camera=False):
    if regions["layer"] != "page" or (layout := game_filter_layout(frame, camera=camera)) is None:
        return None
    area, row, marks = layout
    left, top, right, bottom = area
    panel = [round(value) for value in area]
    cache = {}
    actions, visual_candidates = [], []
    for mark in row:
        if (action := _filter_entry(frame, panel, mark, index, cache, row_choice=True)) is not None:
            actions.append(action)
        elif include_visual_candidates and (candidate := _visual_filter_entry(frame, panel, mark)) is not None:
            visual_candidates.append(candidate)
    complete = include_visual_candidates and len(actions) + len(visual_candidates) == len(row)
    if (len(actions) < 2 and not complete) or len({action["function"] for action in actions}) != len(actions):
        return None
    x0, y0, size = row[0]
    same_column = [mark for mark in marks if left < mark[0] < right and top < mark[1] < bottom
                   and abs(mark[0] - x0) <= .24 * size and 4.8 * size <= mark[1] - y0 <= 10.3 * size]
    if len(same_column) == 1:
        if (entry := _filter_entry(frame, panel, same_column[0], index, cache, row_choice=False)) is not None:
            actions.append(entry)
    search, search_visual = _search(frame, panel, regions, index, cache, include_visual_candidates=include_visual_candidates)
    actions.extend(search)
    if include_visual_candidates:
        from screen_filter_dropdowns import filter_dropdown_candidates
        from screen_filter_pairs import filter_pair_candidates
        from screen_filter_radios import filter_radio_candidates

        visual_candidates.extend(search_visual)
        visual_candidates.extend(filter_radio_candidates(frame, panel, row))
        visual_candidates.extend(filter_pair_candidates(frame, panel, row))
        visual_candidates.extend(filter_dropdown_candidates(frame, panel, regions["candidates"]))
        return panel, actions, visual_candidates
    return panel, actions
