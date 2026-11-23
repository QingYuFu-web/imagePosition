"""Verify a labeled endpoint slider and related controls in a settings card."""

from collections import defaultdict

import cv2
import numpy as np

from screen_colored_settings_panel import _normalize, _printed, _word_shapes
from screen_component_captions import _bands


_CAPTIONS = {
    "algorithm intensity": "算法强度", "music volume": "音乐音量", "media volume": "媒体音量",
    "ring volume": "铃声音量", "effect intensity": "效果强度", "vibration intensity": "振动强度",
    "bass boost": "低音增强", "brightness": "亮度", "intensity": "强度", "sensitivity": "灵敏度",
    "volume": "音量", "opacity": "不透明度", "saturation": "饱和度", "compression": "压缩",
}
_NEARBY_WORDS = ("algorithm quality", "algorithm efficiency", "audio quality", "algorithm activity", "effects intensity")
_TITLES = ("Smart Dynamic Bass", "Dynamic Bass", "Bass Boost", "Adaptive Compression", "Graphic Equalizer",
           "Bass Dynamic Boost", "Dynamic Bass Control", "Settings")
_TOGGLES = {"Smart Dynamic Bass": "智能动态低音", "Dynamic Bass": "动态低音", "Bass Boost": "低音增强",
            "Adaptive Compression": "自适应压缩"}


def _bounds(component):
    x, y, w, h, _ = map(int, component)
    return x, y, x + w, y + h


def _holes(mask):
    contours, tree = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if tree is None:
        return []
    return [(cv2.boundingRect(piece), cv2.contourArea(piece)) for piece, parent in
            zip(contours, tree[0, :, 3], strict=True) if parent >= 0]


def _percent_value(mask):
    height, width = mask.shape
    readings = []
    for start, end in _bands(mask.any(axis=0)):
        if end != width or not .68 * height <= end - start <= 1.48 * height or start <= .60 * height:
            continue
        suffix = mask[:, start:end]
        w = end - start
        holes = [(x + a / 2, y + b / 2) for (x, y, a, b), area in _holes(suffix)
                 if .013 * height ** 2 <= area <= .12 * height ** 2]
        if not (any(x <= .42 * w and y <= .43 * height for x, y in holes)
                and any(x >= .54 * w and y >= .61 * height for x, y in holes)):
            continue
        digits = mask[:, :start]
        zeros = [(x + a / 2, y + b / 2, a) for (x, y, a, b), area in _holes(digits)
                 if .32 * height <= min(a, b) <= max(a, b) <= .91 * height and area >= .165 * height ** 2]
        zeros.sort()
        if len(zeros) == 1 and .69 * height <= start <= 1.55 * height:
            readings.append(0)
        if len(zeros) != 2 or not 2.30 * height <= start <= 3.85 * height or abs(zeros[0][1] - zeros[1][1]) > .18 * height:
            continue
        distance = zeros[1][0] - zeros[0][0]
        if not .85 * height <= distance <= 1.68 * height:
            continue
        split = round(zeros[0][0] - .48 * distance)
        if not .27 * height <= split <= .91 * height or abs(start - zeros[1][0] - .49 * distance) > .31 * height:
            continue
        stem = digits[:, :split]
        first = stem[round(.12 * height):round(.43 * height)]
        bottom = stem[round(.51 * height):round(.92 * height)]
        tail = max(1, round(.43 * split))
        if (first[:, :tail].mean() >= .39 and bottom[:, :tail].mean() <= .17
                and bottom[:, -tail:].mean() >= .71):
            readings.append(100)
    return readings[0] if len(readings) == 1 else None


def _read_percent(gray, box, scale):
    left, top, right, bottom = box
    tile = gray[top:bottom, left:right]
    if min(tile.shape) < 9:
        return None
    peak = float(np.percentile(tile, 98) - np.median(tile))
    if peak < 65:
        return None
    contrast = cv2.subtract(tile, round(float(np.median(tile))))
    readings = []
    for cut in (.37, .45, .53, .59):
        mask = np.uint8(contrast >= cut * peak)
        x, y, w, h = cv2.boundingRect(mask)
        if .27 * scale <= h <= .90 * scale and 1.15 * h <= w <= 5.0 * h:
            if (value := _percent_value(mask[y:y + h, x:x + w])) is not None:
                readings.append(value)
    if len(readings) >= 2 and len(set(readings)) == 1:
        return readings[0]
    return None


def _owner(gray, rail, thumb):
    left, top, right, bottom = rail
    h = bottom - top
    height, width = gray.shape
    y = thumb[3] + round(.35 * h)
    start = left + round(.14 * (right - left))
    if y < 0 or y + 4 >= height:
        return None
    paint = round(np.median(gray[y:y + 4, start:start + max(4, round(1.80 * h))]))
    if not 13 <= paint <= 100:
        return None
    contours, _ = cv2.findContours(cv2.inRange(gray, paint - 4, paint + 4), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    choices = []
    for contour in contours:
        l, t, w, d = cv2.boundingRect(contour)
        r, b = l + w, t + d
        if (not .72 * width <= w <= .96 * width or not 6.1 * h <= d <= 16.5 * h
                or not .25 * h <= left - l <= 2.05 * h or not .25 * h <= r - right <= 2.1 * h
                or not .27 * h <= b - thumb[3] <= 2.12 * h or top - t < 4.75 * h
                or not 0 < l < r < width or cv2.contourArea(contour) < .92 * w * d):
            continue
        pad = max(3, round(.40 * h))
        corners = (gray[t:t + pad, l:l + pad], gray[t:t + pad, r - pad:r],
                   gray[b - pad:b, l:l + pad], gray[b - pad:b, r - pad:r])
        if sum((abs(chip.astype(np.int16) - paint) > 5).mean() >= .29 for chip in corners) < 3:
            continue
        choices.append((l, t, r, b))
    return choices[0] if len(choices) == 1 else None


def _pigment(frame, box, mask):
    left, top, right, bottom = box
    return np.median(frame[top:bottom, left:right][mask[top:bottom, left:right] != 0], axis=0)


def _caption(gray, index, rail, thumb, card, *, maximum):
    left, top, right, bottom = rail
    span, size = right - left, bottom - top
    c0, _, c1, _ = card
    y0 = max(0, top - round(2.10 * size))
    y1 = min(top - 3, thumb[1] - max(2, round(.14 * size)))
    if y1 <= y0:
        return None
    label_start = max(c0 + 2, min(left, thumb[0]) - round(.48 * size))
    label_area = (label_start, y0, left + round(.66 * span), y1)
    label = _printed(gray, label_area, index, vocabulary=(*_CAPTIONS, *_NEARBY_WORDS))
    if label not in _CAPTIONS:
        return None
    numeric = (right - round(.24 * span), y0, min(c1 - 3, max(right, thumb[2]) + round(.35 * size)), y1)
    value = _read_percent(gray, numeric, size)
    if value != (100 if maximum else 0):
        return None
    return label, value


def _heading(gray, box, index):
    left, top, right, bottom = box
    tile = gray[top:bottom, left:right]
    if min(tile.shape) < 9:
        return None
    source_fonts = tuple(index.fonts or index.choice_latin_sources)
    if not source_fonts:
        return None
    folder = source_fonts[0].parent
    sources = tuple(dict.fromkeys((*source_fonts, *(folder / name for name in
                                    ("arialbd.ttf", "ariblk.ttf", "impact.ttf", "bahnschrift.ttf") if (folder / name).is_file()))))
    words, ratios, bank = _word_shapes(sources, _TITLES)
    if not words:
        return None
    ink = cv2.subtract(tile, round(float(np.median(tile))))
    peak = float(np.percentile(ink, 98))
    if peak < 70:
        return None
    readings = []
    for cut in (.30, .43, .56):
        x, y, w, h = cv2.boundingRect(np.uint8(ink >= cut * peak))
        if not .19 * len(tile) <= h <= .79 * len(tile) or not 5.0 * h <= w <= 24.5 * h:
            continue
        observed = _normalize(np.minimum(1, ink[y:y + h, x:x + w] / peak))
        weighted = np.einsum("ij,j->i", bank, observed) - .06 * np.abs(np.log(ratios * h / w))
        grouped = defaultdict(float)
        for word, score in zip(words, weighted, strict=True):
            grouped[word] = max(grouped[word], score)
        rank = sorted(grouped, key=grouped.__getitem__, reverse=True)
        if grouped[rank[0]] >= .86 and grouped[rank[0]] - grouped[rank[1]] >= .060:
            readings.append(rank[0])
    return readings[0] if len(readings) >= 2 and len(set(readings)) == 1 else None


def _toggle(frame, gray, mask, pieces, card, rail, accent, index):
    left, top, right, bottom = card
    h = rail[3] - rail[1]
    candidates = []
    for piece in pieces:
        x, y, w, d, count = map(int, piece)
        if (not 1.53 <= w / max(1, d) <= 2.25 or not 1.39 * h <= d <= 2.63 * h
                or not left + .61 * (right - left) <= x < x + w < right - .24 * h
                or not top + .46 * h <= y <= top + 2.63 * h or not .43 * w * d <= count <= .80 * w * d):
            continue
        if np.max(abs(_pigment(frame, (x, y, x + w, y + d), mask) - accent)) > 20:
            continue
        tile = mask[y:y + d, x:x + w]
        holes = [(box, area) for box, area in _holes(tile) if area >= .20 * d ** 2]
        if len(holes) != 1:
            continue
        (cx, cy, diameter, other), area = holes[0]
        if (not .60 * d <= min(diameter, other) <= max(diameter, other) <= .88 * d
                or not .83 <= diameter / other <= 1.17 or not .63 <= area / (diameter * other) <= .88
                or not .66 * w <= cx + diameter / 2 <= .77 * w or abs(cy + other / 2 - d / 2) > .12 * d):
            continue
        title = _heading(gray, (left + round(.47 * h), max(top, y - round(.37 * h)),
                                x - round(.75 * h), y + round(.45 * d)), index)
        if title in _TOGGLES:
            candidates.append({"role": "toggle_switch", "function": f"关闭{_TOGGLES[title]}", "setting_label": title,
                               "preview_text": title, "box": [x, y, x + w, y + d],
                               "click_point": [x + w // 2, y + d // 2], "setting_state": "on"})
    return candidates[0] if len(candidates) == 1 else None


def _accent_mask(frame):
    blue, green, red = cv2.split(frame)
    low = cv2.min(cv2.min(blue, green), red)
    high = cv2.max(cv2.max(blue, green), red)
    return cv2.bitwise_and(cv2.compare(cv2.subtract(high, low), 48, cv2.CMP_GE),
                           cv2.compare(high, 73, cv2.CMP_GE))


def range_card_actions(frame, overlay, index):
    """Return drag starts only when a card label and endpoint value are visible."""
    height, width = frame.shape[:2]
    if overlay is not None or index is None or width < 290 or not 1.49 * width <= height <= 2.74 * width:
        return []
    if width > 900:
        overview = cv2.resize(frame, (360, round(360 * height / width)), interpolation=cv2.INTER_AREA)
        _, _, coarse, _ = cv2.connectedComponentsWithStats(_accent_mask(overview))
        if not any(176 <= w <= 324 and 7 <= h <= 25 and area >= .72 * w * h for _, _, w, h, area in coarse[1:]):
            return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.uint8(_accent_mask(frame) != 0)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    pieces = stats[1:]
    actions = []
    for piece in pieces:
        x, y, w, h, area = map(int, piece)
        if (not .55 * width <= w <= .86 * width or not .026 * width <= h <= .058 * width
                or not 10.5 * h <= w <= 27 * h or area < .85 * w * h):
            continue
        rail = _bounds(piece)
        candidates = []
        rail_color = _pigment(frame, rail, mask)
        for other in pieces:
            a, b, d, e, count = map(int, other)
            if (not .13 * h <= d <= .55 * h or not 1.40 * h <= e <= 2.64 * h
                    or count < .60 * d * e or abs(b + e / 2 - y - h / 2) > .18 * h):
                continue
            maximum = .18 * h <= a - rail[2] <= 1.05 * h
            minimum = .18 * h <= rail[0] - a - d <= 1.05 * h
            if not (minimum or maximum):
                continue
            thumb = _bounds(other)
            thumb_color = _pigment(frame, thumb, mask)
            if maximum and np.max(abs(thumb_color - rail_color)) > 22:
                continue
            if minimum and float(np.max(thumb_color) - np.max(rail_color)) < 18:
                continue
            candidates.append((thumb, maximum, thumb_color))
        if len(candidates) != 1:
            continue
        thumb, maximum, accent = candidates[0]
        card = _owner(gray, rail, thumb)
        if card is None:
            continue
        if reading := _caption(gray, index, rail, thumb, card, maximum=maximum):
            name, value = reading
            label = _CAPTIONS[name]
            l, t, r, b = thumb
            actions.append({"role": "adjust_slider", "function": f"拖动{label}滑块", "setting_label": label,
                            "setting_display_label": name.capitalize(), "setting_value": f"{value}%", "slider_axis": "horizontal",
                            "box": [min(l, rail[0]), t, max(r, rail[2]), b],
                            "click_point": [(l + r) / 2, (t + b) / 2], "slider_thumb_box": list(thumb),
                            "slider_track_box": list(rail)})
        if switch := _toggle(frame, gray, mask, pieces, card, rail, accent, index):
            actions.append(switch)
    return actions
