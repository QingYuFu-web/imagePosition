"""Locate visible close and start controls on a portrait live setup page."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_component_captions import _bands
from screen_dialog_close import _cross_footprint
from screen_landscape_header import _chevron


_START = "startlive"
_SETUP = {"basic information", "basic info", "stream information", "直播信息", "基本信息", "基本資訊"}


def _action(box, function, role):
    left, top, right, bottom = map(int, box)
    return {"role": role, "function": function, "box": [left, top, right, bottom],
            "click_point": [(left + right) // 2, (top + bottom) // 2]}


def _color_extremes(crop):
    blue, green, red = (crop[..., i] for i in range(3))
    return np.maximum(np.maximum(blue, green), red), np.minimum(np.minimum(blue, green), red)


def _start_carrier(frame):
    height, width = frame.shape[:2]
    top = height - round(.235 * width)
    stride = max(1, width // 540)
    view = frame[top::stride, ::stride]
    span = view.shape[1]
    light, low = _color_extremes(view)
    ink = np.uint8((light >= 125) & (light - low >= 52))
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (.53 * span <= w <= .89 * span and .078 * span <= h <= .168 * span
                and abs(x + w / 2 - span / 2) <= .036 * span and .016 * span <= len(view) - y - h <= .095 * span
                and cv2.contourArea(contour) >= .93 * w * h):
            offset = max(2, round(.028 * h))
            if (y >= 2 * offset + 2 and ink[y + h // 6:y + 5 * h // 6, x + h // 8:x + w - h // 8].mean() >= .90
                    and (light[y - 2 * offset - 2:y - offset, x:x + w:4] <= 78).mean() >= .80):
                marks.append((x * stride, top + y * stride, (x + w) * stride, top + (y + h) * stride))
    return marks[0] if len(marks) == 1 else None


def _letter(mask, index, expected):
    hits = index.rank(mask, alphabet="latin", limit=4)
    if not hits:
        return False
    top, distance = hits[0]["text"], hits[0]["distance"]
    rows = _bands(mask.any(axis=1))
    _, _, w, h = cv2.boundingRect(mask)
    if expected == "l":
        # A plain sans-serif lowercase l has the same stem as uppercase I.
        return (any(hit["text"] == "l" and hit["distance"] <= .15 and hit["distance"] - distance <= .032
                    for hit in hits) and len(rows) == 1 and .09 <= w / h <= .29)
    if expected == "i":
        if len(rows) != 2 or not .055 * h <= rows[1][0] - rows[0][1] <= .28 * h:
            return False
    alternative = next((hit["distance"] for hit in hits if hit["text"].casefold() != expected), 1.)
    return top.casefold() == expected and distance <= .19 and alternative - distance >= .011


def _start_label(frame, carrier, index):
    left, top, right, bottom = carrier
    depth = bottom - top
    cx, cy = (left + right) // 2, (top + bottom) // 2
    dx, dy = round(1.24 * depth), round(.37 * depth)
    _, crop = _color_extremes(frame[cy - dy:cy + dy, cx - dx:cx + dx])
    votes = 0
    for threshold in (145, 175, 195, 115):
        ink = np.uint8(crop >= threshold)
        x, y, w, h = cv2.boundingRect(ink)
        if (not .18 * depth <= h <= .37 * depth or not 3.6 * h <= w <= 6.5 * h
                or abs(x + w / 2 - dx) > .13 * depth or abs(y + h / 2 - dy) > .10 * depth):
            continue
        line = ink[y:y + h, x:x + w]
        spans = _bands(line.any(axis=0))
        if len(spans) != len(_START):
            continue
        gaps = [b[0] - a[1] for a, b in zip(spans, spans[1:])]
        if (not .23 * h <= gaps[4] <= .63 * h or max(gaps[:4] + gaps[5:]) > .24 * h):
            continue
        if all(_letter(line[:, a:b], index, expected) for (a, b), expected in zip(spans, _START, strict=True)):
            votes += 1
            if votes >= 2:
                return True
    return False


def _setup_option(frame, index):
    height, width = frame.shape[:2]
    left, top = round(.21 * width), round(.409 * width)
    crop = cv2.cvtColor(frame[top:round(.493 * width), left:round(.588 * width)], cv2.COLOR_BGR2GRAY)
    ink = np.uint8(crop >= 98)
    _, y, _, h = cv2.boundingRect(ink)
    if not .017 * width <= h <= .044 * width or np.median(crop) > 48:
        return False
    words = _bands(ink.any(axis=0))
    if len(words) < 5:
        return False
    a, b = words[-1]
    if not .23 * h <= b - a <= .63 * h or not .25 * h <= a - words[-2][1] <= .99 * h:
        return False
    _, t, _, mark_height = cv2.boundingRect(ink[:, a:b])
    if not .46 * h <= mark_height <= .91 * h or not _chevron(ink[t:t + mark_height, a:b][:, ::-1]):
        return False
    pad = max(3, round(.16 * h))
    area = (max(0, left + words[0][0] - pad), max(0, top + y - pad),
            min(width, left + words[-2][1] + pad), min(height, top + y + h + pad))
    reading = read_caption(frame, area, index, cache={}, choice=True, join_radicals=True, sentence_case=True,
                           join_letters=True)
    return reading is not None and reading["text"].casefold() in _SETUP


def _close(frame):
    width = frame.shape[1]
    top = round(.10 * width)
    crop = cv2.cvtColor(frame[top:round(.285 * width), :round(.157 * width)], cv2.COLOR_BGR2GRAY)
    if np.median(crop) > 48:
        return None
    marks = []
    for threshold in (108, 153):
        ink = np.uint8(crop >= threshold)
        x, y, w, h = cv2.boundingRect(ink)
        if (not .029 * width <= min(w, h) <= max(w, h) <= .057 * width or not .84 <= w / h <= 1.19
                or not .046 * width <= x + w / 2 <= .112 * width or .23 > ink[y:y + h, x:x + w].mean()
                or not _cross_footprint(ink[y:y + h, x:x + w].astype(bool))):
            return None
        marks.append((x, y, w, h))
    if np.max(np.abs(np.subtract(*marks))) > .10 * marks[0][2]:
        return None
    x, y, w, h = marks[0]
    pad = round(.18 * w)
    return (x - pad, top + y - pad, x + w + pad, top + y + h + pad)


def live_producer_preview_actions(frame, regions, font_index):
    """Require the literal start CTA and an independent setup option on a page."""
    height, width = frame.shape[:2]
    if (font_index is None or regions.get("layer") != "page" or regions.get("regions")
            or not 1.74 * width <= height <= 2.96 * width or width < 380):
        return []
    carrier = _start_carrier(frame)
    if carrier is None or not _start_label(frame, carrier, font_index) or not _setup_option(frame, font_index):
        return []
    result = [_action(carrier, "开始直播", "start_live")]
    close = _close(frame)
    if close is not None:
        result.append(_action(close, "退出直播预览", "close"))
    return result
