"""Recognize an isolated dismiss control on a continuous full-screen surface."""

import cv2
import numpy as np

from screen_back_control import _status_band
from screen_component_captions import _rectangular_button
from screen_dialog_close import _cross_footprint


def _cross(tile):
    height, width = tile.shape[:2]
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).astype(np.int16)
    paper = float(np.median(gray))
    if not (paper <= 76 or paper >= 207) or (abs(gray - paper) <= 12).mean() < .88:
        return None
    strong = gray - paper if paper <= 76 else paper - gray
    edge = 190 - paper if paper <= 76 else paper - 70
    if strong.max() < edge:
        return None
    measures = []
    for level in (.43, .56, .68):
        mask = np.uint8(strong >= max(62, strong.max() * level))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        parts = [(i, x, y, w, h, area) for i, (x, y, w, h, area) in enumerate(stats[1:], 1) if area >= .003 * width * height]
        if len(parts) != 1:
            continue
        identity, x, y, w, h, area = parts[0]
        if not (.27 * width <= w <= .69 * width and .27 * height <= h <= .69 * height
                and .84 <= w / h <= 1.19 and .17 * w * h <= area <= .51 * w * h
                and abs(x + w / 2 - width / 2) <= .105 * width
                and abs(y + h / 2 - height / 2) <= .105 * height):
            continue
        glyph = labels[y:y + h, x:x + w] == identity
        if _cross_footprint(glyph):
            measures.append((glyph.tobytes(), x, y, w, h))
    if len({key for key, *_ in measures}) < 2:
        return None
    boxes = np.asarray([box for _, *box in measures])
    if np.ptp(boxes, axis=0).max() > max(2, .07 * width):
        return None
    x, y, w, h = boxes[len(boxes) // 2]
    return int(x) + int(w) / 2, int(y) + int(h) / 2


def _matches_paint(samples, paint, tolerance):
    return (np.max(abs(samples.astype(np.int16) - paint), axis=2) <= tolerance).mean()


def _footer_matches(frame, gray, box, page):
    measured = _rectangular_button(gray, box)
    if measured is None:
        return False
    a, b, c, d = measured
    width, height = c - a, d - b
    image_height = frame.shape[0]
    edge = max(3, round(.15 * height))
    sides = (frame[b - edge:b - 2:2, a + edge:c - edge:5],
             frame[d + 2:min(d + edge, image_height):2, a + edge:c - edge:5])
    if any(not sample.size or _matches_paint(sample, page, 21) < .88 for sample in sides):
        return False
    join = frame[b + edge:d - edge, a + edge:c - edge]
    paint = np.median(join[::3, ::5], axis=(0, 1))
    if np.max(abs(paint - page)) < 68 or _matches_paint(join, paint, 22) < .87:
        return False
    contrast = cv2.cvtColor(join, cv2.COLOR_BGR2GRAY).astype(np.int16)
    contrast -= int(cv2.cvtColor(np.uint8(paint[None, None]), cv2.COLOR_BGR2GRAY)[0, 0])
    for sign in (1, -1):
        label = np.uint8(sign * contrast >= 53)
        x, y, w, h = cv2.boundingRect(label)
        if (.075 * width <= w <= .62 * width and .14 * height <= h <= .52 * height
                and abs(x + w / 2 - join.shape[1] / 2) <= .09 * width
                and abs(y + h / 2 - join.shape[0] / 2) <= .13 * height
                and .13 * w * h <= label.sum() <= .69 * w * h):
            return True
    return False


def fullscreen_close_action(frame, regions):
    """Use a real corner X and a distinct primary footer on the same page."""
    if regions.get("layer") != "page" or regions.get("regions") or regions.get("detected_close"):
        return None
    height, width = frame.shape[:2]
    if width < 320 or not 1.54 * width <= height <= 2.80 * width:
        return None
    headers, footers = [], []
    for item in regions["candidates"]:
        if item["visibility"] != "page":
            continue
        left, top, right, bottom = item["box"]
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            continue
        w, h = right - left, bottom - top
        x, y = (left + right) / 2, (top + bottom) / 2
        if (.042 * width <= min(w, h) <= max(w, h) <= .112 * width and .72 <= w / h <= 1.36
                and .048 * width <= min(x, width - x) <= .155 * width and .047 * height <= y <= .104 * height):
            headers.append((left, top, right, bottom))
        if (.68 * width <= w <= .965 * width and .065 * width <= h <= .156 * width
                and abs(x - width / 2) <= .045 * width and .855 * height <= y <= .951 * height):
            footers.append((left, top, right, bottom))
    if not headers or not footers:
        return None
    for left, top, right, bottom in headers:
        tile = frame[top:bottom, left:right]
        point = _cross(tile)
        if point is None:
            continue
        paint = np.median(tile[::2, ::2], axis=(0, 1))
        band = frame[top:bottom:3, round(.17 * width):round(.83 * width):4]
        canvas = frame[round(.145 * height):round(.847 * height):8]
        gutter = max(4, round(.021 * width))
        body = canvas[:, round(.11 * width):round(.89 * width):6]
        if (_matches_paint(band, paint, 19) < .97
                or min(_matches_paint(edge, paint, 23) for edge in (canvas[:, :gutter], canvas[:, -gutter:])) < .91
                or _matches_paint(body, paint, 45) >= .94):
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        box = left, top, right, bottom
        if not _status_band(gray, box) or sum(_footer_matches(frame, gray, footer, paint) for footer in footers) != 1:
            continue
        if any(other != box and _cross(frame[other[1]:other[3], other[0]:other[2]]) for other in headers):
            return None
        return {"role": "close", "function": "关闭", "box": list(box), "click_point": [left + point[0], top + point[1]]}
    return None
