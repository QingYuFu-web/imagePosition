"""Read verified actions on a photo selector's visible footer."""

import cv2
import numpy as np

from inventory_choice_glyphs import _tick
from screen_caption_transcript import read_caption
from screen_component_captions import _bands
from screen_dark_barrage_input import _round_fields
from screen_reply_draft_page import _read_word


_SEND = ("发送", "發送", "Send")
_PREVIEW = ("预览", "預覽", "Preview")
_ORIGINAL = ("原图", "原圖", "Original")
_MERGE = ("发送后合并展示", "發送後合併展示")
_OTHER = ("取消", "返回", "保存", "完成", "关闭", "分享", "搜索", "提交")


def _control(role, purpose, printed, box, point=None):
    left, top, right, bottom = box
    return {"role": role, "function": purpose, "preview_text": printed, "box": list(box),
            "click_point": point if point is not None else [(left + right) // 2, (top + bottom) // 2],
            "fixed_center": True}


def _prefix(frame, gray, window, words, rivals, index):
    left, top, right, bottom = window
    roi = gray[top:bottom, left:right]
    if min(roi.shape) < 9:
        return None
    threshold = max(164, min(201, round(float(np.median(roi)) + 53)))
    ink = roi >= threshold
    _, _, _, height = cv2.boundingRect(np.uint8(ink))
    spans = _bands(ink.any(axis=0))
    if len(spans) <= 2 or not all(.65 * height <= b - a <= 1.32 * height for a, b in spans[:2]):
        return None
    end, following = spans[1][1], spans[2][0]
    crop = left, top, left + (end + following) // 2, bottom
    if (reading := _read_word(gray, crop, words, index, rivals=rivals, threshold=threshold)) is not None:
        return reading[0]
    reading = read_caption(frame, crop, index, cache={})
    if (reading and reading["text"] in words and reading["support"] >= 2
            and reading["max_distance"] <= .16 and reading["min_margin"] >= .011):
        return reading["text"]
    return None


def _filled_check(gray, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    y, x = np.ogrid[:height, :width]
    inner = ((x + .5) / width - .5) ** 2 + ((y + .5) / height - .5) ** 2 <= .35 ** 2
    tile = gray[top:bottom, left:right].astype(np.int16)
    offset = tile - np.median(tile[inner])
    for strength in (offset, -offset):
        ink = np.uint8(inner & (strength >= 34))
        a, t, w, h = cv2.boundingRect(ink)
        if (.34 * width <= w <= .64 * width and .27 * height <= h <= .54 * height
                and _tick(ink[t:t + h, a:a + w] != 0, strength[t:t + h, a:a + w], minimum_fill=.16)):
            return True
    return False


def _check_labels(frame, gray, accent, index):
    width = gray.shape[1]
    outlines, _ = cv2.findContours(np.uint8(accent), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    choices = []
    for outline in outlines:
        left, top, w, h = cv2.boundingRect(outline)
        if (not .035 * width <= min(w, h) <= max(w, h) <= .071 * width or not .83 <= w / h <= 1.19
                or not .65 <= cv2.contourArea(outline) / (w * h) <= .85):
            continue
        marker = left, top, left + w, top + h
        if not _filled_check(gray, marker):
            continue
        a = left + w + round(.14 * h)
        t, b = max(0, top - round(.11 * h)), min(len(gray), top + round(1.12 * h))
        stop = min(width, a + round(5.80 * h))
        if a >= stop:
            continue
        x, y, span, depth = cv2.boundingRect(np.uint8(gray[t:b, a:stop] >= 164))
        if not .38 * h <= depth <= .92 * h or not .81 * h <= span <= 5.70 * h:
            continue
        pad = max(3, round(.22 * h))
        area = max(a, a + x - pad), max(t, t + y - pad), min(stop, a + x + span + pad), min(b, t + y + depth + pad)
        reading = _read_word(gray, area, (*_ORIGINAL, *_MERGE), index, rivals=_OTHER, threshold=174)
        if reading is None:
            text = read_caption(frame, area, index, cache={}, choice=True)
            if (text is None or text["text"] not in (*_ORIGINAL, *_MERGE) or text["support"] < 2
                    or text["max_distance"] > .16 or text["min_margin"] < .014):
                continue
            reading = text["text"], text["line_box"]
        printed, (x0, y0, x1, y1) = reading
        if not 0 < x0 - marker[2] <= .84 * h or abs(y0 + y1 - 2 * top - h) > .32 * h:
            continue
        purpose = "切换原图" if printed in _ORIGINAL else "切换发送后合并展示"
        choices.append(_control("checkbox_setting", purpose, printed, (left, min(top, y0), x1, max(top + h, y1)),
                                [left + w // 2, top + h // 2]))
    return choices


def photo_selection_footer(frame, footer_box, font_index):
    """Name only literal controls with independently measured footer affordances."""
    if font_index is None or footer_box is None:
        return []
    left, top, right, bottom = map(round, footer_box)
    width, height = right - left, bottom - top
    if (not (0 <= left < right <= frame.shape[1] and 0 <= top < bottom <= frame.shape[0])
            or width < max(260, .68 * frame.shape[1]) or not .16 * width <= height <= .44 * width):
        return []
    view = frame[top:bottom, left:right]
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    if (gray[::3, ::3] <= 98).mean() < .74:
        return []
    blue, green, red = cv2.split(view)
    accent = (green >= 96) & (cv2.subtract(green, red) >= 42) & (cv2.subtract(green, blue) >= 28)
    actions = _check_labels(view, gray, accent, font_index)
    pills = [(a, t, b, z) for a, t, b, z in _round_fields(accent, width, 1.95, 3.1)
             if .67 * width <= a < b <= .99 * width and .48 * height <= (t + z) / 2 <= .91 * height]
    if len(pills) == 1:
        pill, = pills
        a, t, b, z = pill
        h, w = z - t, b - a
        area = a + round(.09 * w), t + round(.09 * h), b - round(.09 * w), z - round(.09 * h)
        if printed := _prefix(view, gray, area, _SEND, (*_PREVIEW, *_OTHER), font_index):
            actions.append(_control("attachment_pick", "发送所选图片", printed, pill))
            preview_area = max(2, round(.012 * width)), area[1], round(.215 * width), area[3]
            if label := _prefix(view, gray, preview_area, _PREVIEW, (*_SEND, *_OTHER), font_index):
                l, y, _, v = preview_area
                tile = gray[y:v, l:preview_area[2]]
                x, j, span, rise = cv2.boundingRect(np.uint8(tile >= 172))
                pad = max(3, round(.29 * rise))
                actions.append(_control("preview", "预览所选图片", label,
                                        (l + x - pad, y + j - pad, l + x + span + pad, y + j + rise + pad)))
    for item in actions:
        item["box"] = [value + offset for value, offset in zip(item["box"], (left, top, left, top), strict=True)]
        item["click_point"][0] += left
        item["click_point"][1] += top
    return actions
