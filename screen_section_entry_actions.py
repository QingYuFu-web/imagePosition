"""Locate add links attached to repeated bright section cards."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_component_captions import _bands


def _accent_labels(frame, gray):
    height, width = gray.shape
    step = max(2, round(width / 590))
    left, right = round(.66 * width), round(.975 * width)
    top, bottom = round(.056 * height), round(.933 * height)
    view = frame[top:bottom:step, left:right:step]
    blue, green, red = cv2.split(view)
    high = cv2.max(cv2.max(blue, green), red)
    low = cv2.min(cv2.min(blue, green), red)
    ink = ((cv2.subtract(high, low) >= 25) & (high <= 192)
           & (gray[top:bottom:step, left:right:step] <= 174))
    rows = _bands(ink.sum(axis=1) >= max(3, round(.009 * width / step)))
    lines = []
    for start, stop in rows:
        if lines and start - lines[-1][1] <= max(1, round(.0045 * width / step)):
            lines[-1] = lines[-1][0], stop
        else:
            lines.append((start, stop))
    for start, stop in lines:
        field = ink[start:stop]
        x, y, w, h = cv2.boundingRect(np.uint8(field))
        if not (.021 * width <= h * step <= .059 * width and 1.65 * h <= w <= 8.4 * h
                and x > 1 and .82 * width <= left + (x + w) * step <= .968 * width):
            continue
        color = np.median(view[start:stop][field], axis=0)
        yield (left + x * step, top + (start + y) * step,
               left + (x + w) * step, top + (start + y + h) * step), color


def _heading(gray, label):
    height, width = gray.shape
    _, top, _, bottom = label
    size = bottom - top
    y0, y1 = max(0, top - round(.75 * size)), min(height, bottom + round(.75 * size))
    patch = gray[y0:y1, :round(.51 * width)]
    tone = float(np.median(patch))
    if tone < 190:
        return None
    x, y, w, h = cv2.boundingRect(np.uint8(patch <= min(117, tone - 85)))
    if (not .022 * width <= x <= .14 * width or not .10 * width <= w <= .465 * width
            or not 1.18 * size <= h <= 2.24 * size
            or abs(y0 + y + h / 2 - (top + bottom) / 2) > .32 * size):
        return None
    return x, y0 + y, x + w, y0 + y + h


def _wings(gray, top, width, start, stop):
    profile = np.median(gray[top + round(start * width):top + round(stop * width)], axis=0)
    gap = max(2, round(.0034 * width))
    step = np.pad(profile[2 * gap:] - profile[:-2 * gap], (gap, gap))
    a, b, c, d = (round(factor * width) for factor in (.035, .21, .79, .965))
    left = a + int(np.argmax(step[a:b]))
    right = c + int(np.argmin(step[c:d]))
    if min(step[left], -step[right]) < 4:
        return None
    return left, right


def _section_panel(gray, label, heading):
    height, width = gray.shape
    _, _, _, bottom = label
    start, stop = bottom + round(.035 * width), bottom + round(.136 * width)
    gap = max(2, round(.0045 * width))
    reach = round(.067 * width)
    if stop + reach >= height:
        return None
    strip = gray[start - gap:stop + gap, round(.21 * width):round(.79 * width):3].astype(np.int16)
    delta = strip[2 * gap:] - strip[:-2 * gap]
    rows = np.flatnonzero((delta >= 5).mean(axis=1) >= .78)
    if rows.size == 0 or np.ptp(rows) > .042 * width:
        return None
    transition = start + int(rows[np.argmax(np.median(delta[rows], axis=1))])
    shallow = _wings(gray, transition, width, .007, .014)
    deep = _wings(gray, transition, width, .046, .061)
    if shallow is None or deep is None:
        return None
    left, right = deep
    turns = (shallow[0] - left, right - shallow[1])
    if (abs(left - heading[0]) <= .041 * width and right - left >= .77 * width
            and abs(left + right - width) <= .057 * width
            and min(turns) >= .020 * width and max(turns) <= .104 * width
            and abs(turns[0] - turns[1]) <= .041 * width):
        return left, transition, right
    return None


def _similar(first, second, width):
    box, paint, panel = first
    peer, color, other = second
    heights = box[3] - box[1], peer[3] - peer[1]
    return (abs(box[1] - peer[1]) >= .15 * width and abs(box[2] - peer[2]) <= .027 * width
            and max(heights) <= 1.25 * min(heights) and np.max(np.abs(paint - color)) <= 21
            and max(abs(panel[i] - other[i]) for i in (0, 2)) <= .032 * width)


def section_entry_actions(frame, overlay, index):
    """Read one click for each visible add link in a verified portrait section."""
    height, width = frame.shape[:2]
    if index is None or overlay is not None or width < 260 or not 1.31 * width <= height <= 3.12 * width:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    proposals = [(label, color, panel) for label, color in _accent_labels(frame, gray)
                 if (heading := _heading(gray, label)) is not None
                 and (panel := _section_panel(gray, label, heading)) is not None]
    cache = {}
    result = []
    for candidate in proposals:
        if not any(_similar(candidate, peer, width) for peer in proposals):
            continue
        (left, top, right, bottom), _, _ = candidate
        pad = max(3, round(.19 * (bottom - top)))
        area = (left - pad, top - pad, min(width, right + pad), bottom + pad)
        parsed = read_caption(frame, area, index, cache=cache, choice=True, join_radicals=True)
        if not parsed:
            continue
        text = parsed["text"]
        if not ((3 <= len(text) <= 8 and text.startswith(("新增", "添加", "新建", "创建", "增加")))
                or text.casefold().startswith("add ")):
            continue
        x, y, r, b = parsed["line_box"]
        result.append({"role": "add_item", "function": text, "preview_text": text, "box": list(area),
                       "click_point": [(x + r) // 2, (y + b) // 2]})
    return result if len(result) >= 2 and len({item["function"] for item in result}) == len(result) else []
