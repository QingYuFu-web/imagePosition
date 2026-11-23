"""Require content pixels inside a measured comment entry before exposing Send."""

import cv2
import numpy as np

from screen_comment_sheet_modal import _phrase
from screen_component_captions import _bands
from screen_colored_settings_panel import _printed


_PLACEHOLDERS = ("Add a comment", "Write a comment", "Type a comment", "Enter a comment", "Leave a comment",
                 "Your comment", "Comment as", "Reply to", "写评论", "输入评论", "发表评论", "添加评论", "写下评论",
                 "请输入评论", "请输入内容", "写下你的评论", "说点什么", "说说你的想法", "说说你的看法",
                 "留下你的评论", "新增留言", "寫評論", "發表評論", "添加評論", "輸入留言")
_RIVALS = ("Comments", "Replies", "Contents", "Community", "添加说明", "说说你的故事", "回复评论", "我的看法")
_PREFIXES = ("Comment as", "Reply to")


def _line_boxes(signal, threshold, span):
    ink = np.uint8(signal >= threshold)
    minimum = max(1, round(.011 * span))
    rows = ink.sum(axis=1) >= minimum
    bridge = max(1, round(.037 * span) | 1)
    rows = cv2.morphologyEx(np.uint8(rows[:, None]), cv2.MORPH_CLOSE, np.ones((bridge, 1), np.uint8))[:, 0]
    for start, stop in _bands(rows != 0):
        patch = ink[start:stop]
        x, y, w, h = cv2.boundingRect(patch)
        if not max(7, .16 * span) <= h <= .70 * span or not .30 * h <= w <= 26 * h:
            continue
        shaped = patch[y:y + h, x:x + w]
        glyphs = []
        for a, b in _bands(shaped.any(axis=0)):
            part = shaped[:, a:b]
            _, _, _, height = cv2.boundingRect(part)
            if (.17 * h <= b - a <= 1.67 * h and .44 * h <= height <= h
                    and .10 <= part.mean() <= .84):
                glyphs.append((a, b, height))
        if len(glyphs) >= 2 or (len(glyphs) == 1 and .36 * h <= glyphs[0][1] - glyphs[0][0]
                               and glyphs[0][2] >= .82 * h and .14 <= shaped.mean() <= .74):
            yield x, start + y, x + w, start + y + h


def _colored_token(frame, gray, span, paper):
    blue, green, red = cv2.split(frame)
    high, low = cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red)
    chroma = cv2.subtract(high, low)
    contrast = cv2.absdiff(gray, np.full_like(gray, paper))
    mask = cv2.bitwise_and(cv2.compare(chroma, 62, cv2.CMP_GE), cv2.compare(contrast, 49, cv2.CMP_GE))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for component, (x, y, w, h, count) in enumerate(stats[1:], 1):
        if (.13 * span <= min(w, h) <= max(w, h) <= .59 * span and .63 <= w / h <= 1.65
                and .39 * w * h <= count <= .92 * w * h):
            return True
        if (.20 * span <= w <= .44 * span and .080 * span <= h <= .24 * span and 1.65 <= w / h <= 3.1
                and .56 * w * h <= count <= .87 * w * h):
            cap = labels[y:y + h, x:x + w] == component
            first, last = np.flatnonzero(cap[0]), np.flatnonzero(cap[-1])
            if (.15 * w <= len(first) <= .49 * w and len(last) >= .70 * w
                    and abs(first.mean() - w / 2) <= .20 * w):
                return True
    return False


def _placeholder(gray, signal, bounds, level, index, light):
    if index is None:
        return False
    x, y, right, bottom = bounds
    height = bottom - y
    crop = signal[y:bottom, x:right] >= level
    columns = _bands(crop.any(axis=0))
    if len(columns) >= 2:
        l, r = columns[-1]
        tick = crop[:, l:r]
        if r - l <= .13 * height and tick.mean() >= .69 and l - columns[-2][1] >= max(1, .039 * height):
            right = x + columns[-2][1]
    aspect = (right - x) / height
    choices = tuple(label for label in _PLACEHOLDERS if label in _PREFIXES and aspect >= 2.7 or
                    (.28 * len(label) <= aspect <= .91 * len(label) if label.isascii()
                     else .67 * len(label) <= aspect <= 1.46 * len(label)))
    if not choices:
        return False
    pad = max(2, round(.23 * height))
    rim = min(gray.shape[1], right + pad)
    if right != bounds[2]:
        rim = min(rim, x + columns[-1][0])
    region = (max(0, x - pad), max(0, y - pad), rim, min(gray.shape[0], bottom + pad))
    source = gray if light else cv2.bitwise_not(gray)
    if _phrase(source, region, choices, index, piece="account") is not None:
        return True
    if aspect >= 3.2:
        gaps = [next_[0] - prev[1] for prev, next_ in zip(columns, columns[1:])]
        word_break = max(.17 * height, 1.65 * np.median(gaps)) if gaps else 0
        breaks = [a[1] for a, b in zip(columns, columns[1:]) if b[0] - a[1] >= word_break]
        if len(breaks) >= 2:
            prefix = (region[0], region[1], min(region[2], x + breaks[1] + max(1, round(.08 * height))), region[3])
            if _printed(gray, prefix, index, vocabulary=(*_PREFIXES, *_RIVALS), light=not light,
                        minimum_score=.82, minimum_margin=.027) in _PREFIXES:
                return True
    return _printed(gray, region, index, vocabulary=(*choices, *_RIVALS), light=not light,
                    minimum_score=.82, minimum_margin=.027) in choices


def has_visible_draft(frame, field, index, *, trailing_fields=1.20, line_span=None):
    """Read draft evidence in the writing area, excluding borders and other controls."""
    left, top, right, bottom = field
    span = min(bottom - top, line_span) if line_span is not None else bottom - top
    margin = round(.21 * span)
    body = frame[top + round(.12 * span):bottom - round(.10 * span),
                 left + margin:right - round(trailing_fields * span)]
    if min(body.shape[:2]) < 12:
        return False
    gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
    paper = int(np.median(gray))
    light = paper >= 128
    signal = cv2.subtract(paper, gray) if light else cv2.subtract(gray, paper)
    room = paper if light else 255 - paper
    high = max(85, round(.50 * room))
    candidates = [list(_line_boxes(signal, level, span)) for level in (max(55, round(.28 * room)), high)]
    for x, y, r, b in candidates[-1]:
        h = b - y
        if not any(max(abs(a - c) for a, c in zip((x, y, r, b), peer, strict=True)) <= max(2, .20 * h)
                   for peer in candidates[0]):
            continue
        if not _placeholder(gray, signal, (x, y, r, b), high, index, light):
            return True
    return _colored_token(body, gray, span, paper)
