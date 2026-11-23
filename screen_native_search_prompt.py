"""Read an empty native search toolbar from a printed prompt and filter shelf."""

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_back_control import _status_band
from screen_caption_transcript import read_caption
from screen_landscape_search_field import _available
from screen_native_tab_bar import _distance


_SEARCH_PROMPTS = {
    "搜索", "搜尋", "查找", "查詢", "查询", "搜索文件", "搜尋檔案", "查找文件",
    "搜索联系人", "搜尋聯絡人", "搜索应用", "搜尋應用程式", "输入关键词", "輸入關鍵詞",
    "search", "searchfiles", "searchcontacts", "searchapps", "findfiles",
}


def _leading_arrow(frame, paint):
    height, width = frame.shape[:2]
    left, top = round(.016 * width), round(.041 * height)
    tile = _distance(frame[top:round(.145 * height), left:round(.19 * width)], paint)
    _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(tile >= 52), connectivity=8)
    hits = []
    for i, (x, y, w, h, _) in enumerate(stats[1:], 1):
        if (.026 * width <= h <= .072 * width and .78 <= w / h <= 1.32
                and 0 < x < x + w < tile.shape[1] and 0 < y < y + h < tile.shape[0]
                and is_shaft_back_arrow(ids[y:y + h, x:x + w] == i)):
            hits.append(tuple(map(int, (left + x, top + y, left + x + w, top + y + h))))
    return hits[0] if len(hits) == 1 else None


def _prompt_geometry(ink, arrow):
    _, top, end, bottom = arrow
    width = ink.shape[1]
    size = bottom - top
    baseline = (top + bottom) / 2
    x0 = end + round(.62 * size)
    y0, y1 = round(baseline - .82 * size), round(baseline + .82 * size)
    tile = ink[y0:y1, x0:round(.93 * width)]
    mask = np.uint8(tile >= 53)
    if mask.mean() > .175:
        return None
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pieces = sorted((int(x), int(y), int(x + w), int(y + h), int(n)) for x, y, w, h, n in stats[1:]
                    if n >= max(2, .0009 * size * size))
    if not 5 <= len(pieces) <= 29:
        return None
    letters, dots = pieces[:-3], pieces[-3:]
    l, t = min(mark[0] for mark in letters), min(mark[1] for mark in letters)
    r, b = max(mark[2] for mark in letters), max(mark[3] for mark in letters)
    if (not .64 * size <= b - t <= 1.44 * size or not .95 * size <= r - l <= 8.3 * size
            or l >= 4.3 * size or abs(y0 + (t + b) / 2 - baseline) > .23 * size):
        return None
    spans = [(a, c, k - a, d - c, n) for a, c, k, d, n in dots]
    if not all(.055 * size <= min(w, h) <= max(w, h) <= .205 * size and area >= .53 * w * h
               and t + .70 * (b - t) <= y + h / 2 <= b + .09 * size for _, y, w, h, area in spans):
        return None
    pitch = np.diff([x + w / 2 for x, _, w, _, _ in spans])
    if (not .105 * size <= min(pitch) <= max(pitch) <= .34 * size or max(pitch) > 1.31 * min(pitch)
            or not .015 * size <= dots[0][0] - r <= .37 * size
            or max(dot[1] for dot in dots) - min(dot[1] for dot in dots) > .07 * size):
        return None
    pad = max(2, round(.075 * size))
    reading = (x0 + max(0, l - pad), y0 + max(0, t - pad), x0 + min(r + pad, dots[0][0]),
               y0 + min(y1 - y0, b + pad))
    return reading, x0 + dots[-1][2]


def _filter_shelf(ink, arrow):
    _, top, _, bottom = arrow
    height, width = ink.shape
    size = bottom - top
    y0 = bottom + round(.72 * size)
    y1 = min(height, bottom + round(5.35 * size))
    mask = np.uint8(ink[y0:y1] >= 22)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    chips = []
    for contour, row in zip(contours, hierarchy[0], strict=True):
        hole = row[2]
        if row[3] != -1 or hole < 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if (not .10 * width <= w <= .37 * width or not 1.12 * size <= h <= 2.66 * size
                or not 1.6 <= w / h <= 4.8 or y < .18 * size or y + h >= len(mask)):
            continue
        outer, inner = cv2.contourArea(contour), cv2.contourArea(contours[hole])
        if .73 * w * h <= outer <= .966 * w * h and .72 * outer <= inner <= .976 * outer:
            chips.append((x, y, w, h))
    if len(chips) < 3:
        return None
    for _, y, _, h in chips:
        row = sorted(chip for chip in chips if abs(chip[1] - y) <= .14 * h and abs(chip[3] - h) <= .17 * h)
        if len(row) < 3 or row[-1][0] + row[-1][2] - row[0][0] < .70 * width:
            continue
        gaps = [b[0] - a[0] - a[2] for a, b in zip(row, row[1:])]
        if .065 * h <= min(gaps) <= max(gaps) <= .63 * h and max(gaps) - min(gaps) <= .23 * h:
            return y0 + min(chip[1] for chip in row)
    return None


def native_search_prompt_actions(frame, regions, font_index):
    """Return Back and an editable search target when independent toolbar cues agree."""
    height, width = frame.shape[:2]
    if (font_index is None or regions.get("layer") != "page" or width < 244
            or not 1.49 * width <= height <= 2.86 * width):
        return []
    limit = min(round(.32 * height), round(.55 * width))
    paint = np.median(frame[round(.067 * height):round(.121 * height):4,
                            round(.46 * width):round(.74 * width):6], axis=(0, 1))
    arrow = _leading_arrow(frame, paint)
    if arrow is None:
        return []
    ink = _distance(frame[:limit], paint)
    prompt = _prompt_geometry(ink, arrow)
    if prompt is None:
        return []
    left, top, right, bottom = arrow
    size = bottom - top
    shelf = _filter_shelf(ink, arrow)
    if shelf is None or not 1.23 * size <= shelf - bottom <= 3.3 * size:
        return []
    gap = ink[bottom + round(.22 * size):shelf - round(.21 * size)]
    if (gap <= 13).mean() < .966:
        return []
    if not _status_band(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), arrow):
        return []
    label_box, prompt_end = prompt
    reading = read_caption(frame, label_box, font_index, cache={}, choice=True)
    if reading is None or "".join(reading["text"].casefold().split()) not in _SEARCH_PROMPTS:
        return []
    pad = round(.32 * size)
    search = [label_box[0] - round(.82 * size), top - round(.58 * size),
              min(width - pad, prompt_end + round(6.10 * size)), bottom + round(.62 * size)]
    back = [left - pad, top - pad, right + pad, bottom + pad]
    actions = [
        {"role": "back", "function": "返回", "preview_text": "返回", "box": back,
         "click_point": [(left + right) // 2, (top + bottom) // 2]},
        {"role": "search", "function": "搜索", "preview_text": f"{reading['text']}…", "box": search,
         "click_point": [(search[0] + search[2]) // 2, (top + bottom) // 2]},
    ]
    return [action for action in actions if _available(action["box"], regions)]
