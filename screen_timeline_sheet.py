"""Read inactive time-mode cards from a verified foreground sheet."""

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_landscape_search_field import _available


_STATE_NAMES = {"当前时间线", "当前時間線"}
_MODE_NAMES = {
    "按月份选择", "按月份選擇", "按年份选择", "按年份選擇",
    "自定义范围", "自定義範圍", "按日期选择", "按日期選擇",
}
_CARD_NAMES = _STATE_NAMES | _MODE_NAMES
_SUBMIT_NAMES = {"确定", "確認", "确认", "確定"}


def _filled_card(frame, top, bottom, paper):
    width = frame.shape[1]
    mid = (top + bottom) // 2
    middle = frame[mid - 3:mid + 4]
    paint = np.median(middle[:, round(.29 * width):round(.72 * width)], axis=(0, 1))
    matched = (np.max(abs(middle - paint), axis=2) <= 6).mean(axis=0) >= .72
    edges = np.flatnonzero(matched)
    if edges.size == 0:
        return None
    left, right = int(edges[0]), int(edges[-1]) + 1
    if (not .907 * width <= right - left <= .982 * width or
            abs(left + right - width) > .019 * width or matched.mean() < .79
            or np.max(abs(paper - paint)) < 8):
        return None
    return (left, top, right, bottom), paint


def _centered_heading(frame, top, first, paper):
    width = frame.shape[1]
    l, r = round(.20 * width), round(.80 * width)
    crop = frame[top + round(.008 * width):first, l:r]
    ink = np.uint8(np.max(abs(crop - paper), axis=2) >= 70)
    x, y, w, h = cv2.boundingRect(ink)
    return (.19 * width <= w <= .57 * width and .022 * width <= h <= .052 * width
            and abs(l + x + w / 2 - width / 2) <= .031 * width and len(_bands(ink.any(axis=0))) >= 7)


def _header_text(frame, card, index, *, expanded):
    left, top, right, bottom = card
    width, height = right - left, bottom - top
    lead = left + round(.021 * width)
    limit = left + round(.39 * width)
    middle = top + round(.164 * height) if expanded else (top + bottom) // 2
    radius = round(.037 * width)
    result = read_caption(frame, (lead, max(top, middle - radius), limit, min(bottom, middle + radius)),
                          index, cache={}, choice=True)
    return result["text"] if result is not None else None


def _footer_text(frame, box, paint, index):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    center = (left + right) // 2
    a, b = center - round(.116 * width), center + round(.116 * width)
    tile = frame[top + round(.17 * height):bottom - round(.17 * height), a:b]
    ink = np.max(paint - tile, axis=2)
    peak = np.percentile(ink, 97)
    if not 18 <= peak <= 180:
        return None
    # The accent label can be lighter in contrast than the glyph reader's normal floor.
    normalized = np.uint8(255 - np.clip(ink * (190 / peak), 0, 240))
    reading_frame = np.full((tile.shape[0], frame.shape[1], 3), 255, dtype=np.uint8)
    reading_frame[:, :tile.shape[1]] = normalized[:, :, None]
    area = (0, 0, tile.shape[1], tile.shape[0])
    result = read_caption(reading_frame, area, index, cache={}, choice=True)
    return result["text"] if result is not None else None


def _action(box, name, role):
    left, top, right, bottom = box
    return {"role": role, "function": name, "preview_text": name, "box": list(box),
            "click_point": [(left + right) // 2, (top + bottom) // 2]}


def timeline_sheet_actions(frame, regions, font_index):
    """Export the collapsed mode alternatives and a confirmed footer action."""
    if font_index is None or regions.get("layer") != "bottom_sheet" or len(regions.get("regions", ())) != 1:
        return []
    height, width = frame.shape[:2]
    left, top, right, bottom = regions["regions"][0]
    if (not 0 <= left <= .015 * width or not .985 * width <= right <= width
            or not .21 * height <= top <= .59 * height or bottom != height):
        return []
    a, b = round(.16 * width), round(.86 * width)
    side = frame[top + round(.082 * width):bottom - round(.025 * width):6, ::max(2, round(.008 * width))]
    paper = np.median(side[:, :max(1, round(.024 * side.shape[1]))], axis=(0, 1))
    if np.mean(paper) < 211 or np.ptp(paper) > 25:
        return []
    backdrop = frame[top - round(.036 * width):top - round(.008 * width), a:b:7]
    if (np.max(paper - backdrop, axis=2) >= 57).mean() < .86:
        return []
    center = frame[top:bottom, a:b:7]
    change = np.max(abs(center - paper), axis=2)
    runs = _bands((change >= 8).mean(axis=1) >= .90)
    bands = [(top + a, top + b) for a, b in runs if .084 * width <= b - a <= .69 * width]
    if len(bands) not in (3, 4) or not _centered_heading(frame, top, bands[0][0], paper):
        return []
    measurements = [_filled_card(frame, a, b, paper) for a, b in bands]
    if any(item is None for item in measurements):
        return []
    boxes, colors = zip(*measurements, strict=True)
    modes = boxes[:3]
    heights = [box[3] - box[1] for box in modes]
    current = int(np.argmax(heights))
    collapsed = [j for j in range(len(modes)) if j != current]
    small, peer = sorted(heights[j] for j in collapsed)
    if (not .101 * width <= small <= peer <= .212 * width or peer > 1.13 * small
            or not 2.08 * peer <= heights[current] <= 4.0 * small
            or np.max(abs(colors[collapsed[1]] - colors[collapsed[0]])) > 9):
        return []
    active_text = _header_text(frame, modes[current], font_index, expanded=True)
    if active_text not in _CARD_NAMES:
        return []
    output = []
    for j in collapsed:
        if _available(modes[j], regions):
            name = _header_text(frame, modes[j], font_index, expanded=False)
            if name in _CARD_NAMES and name != active_text:
                output.append(_action(modes[j], name, "select_option"))
    if output and len(boxes) == 4:
        footer, footer_paint = boxes[3], colors[3]
        if (np.ptp(footer_paint) < 61 or footer[1] - modes[-1][3] < .108 * width
                or not .030 * width <= bottom - footer[3] <= .153 * width or not _available(footer, regions)):
            return output
        label = _footer_text(frame, footer, footer_paint, font_index)
        if label in _SUBMIT_NAMES:
            output.append(_action(footer, label, "confirm"))
    return output
