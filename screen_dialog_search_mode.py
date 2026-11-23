"""Recognize the inline match selector of a verified search dialog."""

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_landscape_search_field import _available


_MATCH_MODES = frozenset({
    "contains", "does not contain", "doesn't contain", "not contains",
    "starts with", "begins with", "ends with", "exact", "exact match",
    "equals", "matches", "pattern", "wildcard", "regex", "regular expression",
    "包含", "不包含", "開頭是", "开头是", "結尾是", "结尾是",
    "等于", "等於", "精确匹配", "精確匹配", "完全匹配", "正则", "正則", "正则表达式", "正則表達式",
})


def _down_triangle(mask):
    height, width = mask.shape
    counts = mask.sum(axis=1)
    if not .36 <= mask.mean() <= .65 or not counts.all():
        return False
    row_centers = (mask @ np.arange(width)) / counts
    tip = max(1, round(.19 * height))
    expected = np.linspace(.98, .052, height)
    return (np.median(counts[:tip]) >= .76 * width and np.median(counts[-tip:]) <= .27 * width
            and np.quantile(abs(counts / width - expected), .90) <= .142
            and np.quantile(abs(row_centers - (width - 1) / 2), .90) <= .093 * width)


def _inline_arrow(frame, panel, field, paper):
    left, _, right, _ = panel
    span = right - left
    start, top, _, bottom = field.input_box
    row_height = bottom - top
    mid = (top + bottom) / 2
    x0, x1 = max(left, start - round(.139 * span)), start - round(.008 * span)
    y0, y1 = max(panel[1], top - round(.09 * row_height)), min(panel[3], bottom + round(.09 * row_height))
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    consistent = []
    for cut in (20, 29, 38):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(gray <= paper - cut), connectivity=8)
        for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if (not .023 * span <= w <= .060 * span or not .011 * span <= h <= .034 * span
                    or not 1.57 <= w / h <= 2.45 or not .33 * w * h <= area <= .66 * w * h
                    or not 0 < x < x + w < gray.shape[1] or not 0 < y < y + h < gray.shape[0]
                    or abs(y0 + y + h / 2 - mid) > .38 * row_height):
                continue
            if _down_triangle(labels[y:y + h, x:x + w] == index):
                consistent.append((cut, tuple(map(int, (x0 + x, y0 + y, x0 + x + w, y0 + y + h)))))
    if len({cut for cut, _ in consistent}) < 2:
        return None
    bounds = [box for _, box in consistent]
    return bounds[len(bounds) // 2] if np.ptp(bounds, axis=0).max() <= max(2, .005 * span) else None


def _owner_text(frame, panel, field, arrow, paper, index):
    left, _, right, _ = panel
    span = right - left
    _, top, _, bottom = field.input_box
    gap = max(2, round(.005 * span))
    x0, x1 = left + round(.033 * span), arrow[0] - gap
    if x1 - x0 < .077 * span:
        return None
    gray = cv2.cvtColor(frame[top:bottom, x0:x1], cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(np.uint8(gray <= paper - 78))
    if (not .069 * span <= w <= .294 * span or not .024 * span <= h <= .064 * span
            or not .007 * span <= arrow[0] - x0 - x - w <= .070 * span
            or abs(top + y + h / 2 - (arrow[1] + arrow[3]) / 2) > .47 * h):
        return None
    box = (x0 + x, top + y, x0 + x + w, top + y + h)
    area = (max(x0, box[0] - gap), max(top, box[1] - gap), min(x1, box[2] + gap), min(bottom, box[3] + gap))
    words = read_caption(frame, area, index, cache={})
    if not words or words["text"].casefold().strip() not in _MATCH_MODES:
        return None
    return box, words["text"]


def _footer_control(frame, panel, paper):
    left, top, right, bottom = panel
    wide, tall = right - left, bottom - top
    y0 = top + round(.845 * tall)
    x0 = left + round(.53 * wide)
    gray = cv2.cvtColor(frame[y0:bottom - round(.015 * tall), x0:right - round(.014 * wide)], cv2.COLOR_BGR2GRAY)
    delta = paper - gray.astype(np.int16)
    neutral = np.uint8((delta >= 6) & (delta <= 38))
    _, _, stats, _ = cv2.connectedComponentsWithStats(neutral, connectivity=8)
    for x, y, w, h, area in stats[1:]:
        if (not .106 * wide <= w <= .337 * wide or not .045 * tall <= h <= .115 * tall
                or not .81 * w * h <= area <= w * h or w < 1.22 * h
                or not 0 < x < x + w < gray.shape[1] or not 0 < y < y + h < gray.shape[0]):
            continue
        tile = gray[y:y + h, x:x + w]
        words = np.uint8(tile <= paper - 75)
        l, t, sw, sh = cv2.boundingRect(words)
        if (.19 * h <= sh <= .51 * h and .16 * w <= sw <= .72 * w and len(_bands(words.any(axis=0))) >= 2
                and abs(l + sw / 2 - w / 2) <= .108 * w and abs(t + sh / 2 - h / 2) <= .14 * h):
            return True
    return False


def dialog_search_mode_actions(frame, regions, font_index, *, field):
    """Add one match mode beside a separately verified search field."""
    if regions.get("layer") != "dialog" or font_index is None or field is None:
        return []
    left, top, right, bottom = field.panel
    w, h = right - left, bottom - top
    start, row_top, _, _ = field.input_box
    heading_left, heading_top, _, heading_bottom = field.title_box
    if (not .155 * w <= start - left <= .50 * w or not .087 * h <= row_top - top <= .34 * h
            or not .024 * w <= heading_left - left <= .143 * w
            or not top < heading_top < heading_bottom < row_top - .021 * w):
        return []
    sample = frame[top + round(.31 * h):bottom - round(.078 * h):7, left + round(.1 * w):right - round(.1 * w):9]
    paint = np.median(sample, axis=(0, 1))
    paper = int(cv2.cvtColor(np.uint8(paint.reshape(1, 1, 3)), cv2.COLOR_BGR2GRAY)[0, 0])
    if paper < 198 or np.ptp(paint) > 24:
        return []
    arrow = _inline_arrow(frame, field.panel, field, paper)
    if arrow is None or not _footer_control(frame, field.panel, paper):
        return []
    owner = _owner_text(frame, field.panel, field, arrow, paper, font_index)
    if owner is None:
        return []
    word, name = owner
    pad_x, pad_y = round(.010 * w), round(.012 * w)
    box = [word[0] - pad_x, min(word[1], arrow[1]) - pad_y,
           min(start - 1, arrow[2] + pad_x), max(word[3], arrow[3]) + pad_y]
    if not _available(box, regions):
        return []
    return [{"role": "dropdown", "function": "选择匹配方式", "preview_text": name, "box": box,
             "click_point": [(word[0] + arrow[2]) // 2, (word[1] + word[3]) // 2]}]
