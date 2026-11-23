"""Recognize a dark centered dialog with a vertical list of radio options."""

from dataclasses import dataclass

import cv2
import numpy as np

from inventory_choice_glyphs import find_choice_glyphs
from screen_caption_transcript import _shape_parts, read_caption
from screen_choice_row_geometry import _caption, choice_row_controls
from screen_dark_dialog_footer import _read_button
from screen_native_tab_bar import _distance


@dataclass(frozen=True)
class DarkRadioDialog:
    panel: tuple[int, int, int, int]
    actions: tuple[dict, ...]
    options_box: tuple[int, int, int, int]


def _face(frame):
    height, width = frame.shape[:2]
    step = max(4, round(.014 * width))
    center = frame[round(.33 * height):round(.71 * height):step,
                   round(.24 * width):round(.77 * width):step]
    paint = np.median(center, axis=(0, 1))
    if not 19 <= paint.mean() <= 103 or np.ptp(paint) > 34:
        return None
    ratio = min(1., 390 / width)
    small = cv2.resize(frame, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    mask = np.uint8(_distance(small, paint) <= 6)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = []
    for contour in contours:
        left, top, a, b = cv2.boundingRect(contour)
        right, bottom = left + a, top + b
        if (not .59 * w <= a <= .94 * w or not .32 * h <= b <= .69 * h
                or abs(left + right - w) > .043 * w or not .13 * h <= top <= .34 * h
                or not .60 * h <= bottom <= .89 * h or cv2.contourArea(contour) < .936 * a * b):
            continue
        pad = max(3, round(.014 * w))
        if min(left, w - right, top, h - bottom) <= 2 * pad:
            continue
        core = mask[top + pad:bottom - pad, left + pad:right - pad]
        if core.mean() < .86:
            continue
        edges = [
            small[top - 2 * pad:top - pad, left + pad:right - pad],
            small[bottom + pad:bottom + 2 * pad, left + pad:right - pad],
            small[top + 2 * pad:bottom - 2 * pad, left - 2 * pad:left - pad],
            small[top + 2 * pad:bottom - 2 * pad, right + pad:right + 2 * pad],
        ]
        if min((tile.mean(axis=2) <= paint.mean() - 11).mean() for tile in edges) < .76:
            continue
        edge = max(3, round(.034 * a))
        near = max(1, round(.006 * a))
        corners = (mask[top + near:top + edge, left + near:left + edge],
                   mask[top + near:top + edge, right - edge:right - near],
                   mask[bottom - edge:bottom - near, left + near:left + edge],
                   mask[bottom - edge:bottom - near, right - edge:right - near])
        if any(chip.mean() > .78 for chip in corners):
            continue
        faces.append((round(left * width / w), round(top * height / h),
                      round(right * width / w), round(bottom * height / h)))
    return (faces[0], paint) if len(faces) == 1 else None


def _title(gray, panel, first, index, frame, cache):
    left, top, right, _ = panel
    span = right - left
    x0, y0 = left + round(.041 * span), top + round(.024 * span)
    end = min(first - round(.028 * span), top + round(.158 * span))
    crop = gray[y0:end, x0:left + round(.74 * span)]
    if not crop.size:
        return False
    mask = np.uint8(crop >= np.median(crop) + 51)
    x, y, w, h = cv2.boundingRect(mask)
    if not .027 * span <= h <= .092 * span or not 1.56 * h <= w <= .68 * span:
        return False
    pad = max(2, round(.085 * h))
    area = (x0 + x - pad, y0 + y - pad, x0 + x + w + pad, y0 + y + h + pad)
    return read_caption(frame, area, index, cache=cache, choice=True, sentence_case=True) is not None


def _stem_word(frame, box, paint, index):
    """Resolve an upright lowercase l when neighboring option captions prove title case."""
    x0, y0, x1, y1 = box
    ink = _distance(frame[y0:y1, x0:x1], paint)
    peak = np.percentile(ink, 98)
    if peak < 73:
        return None
    readings = []
    for level in (.30, .43, .57):
        mask = np.uint8(ink >= level * peak)
        x, y, w, h = cv2.boundingRect(mask)
        if h < 10:
            continue
        parts = _shape_parts(mask[y:y + h, x:x + w], h, choice=True)
        if not 4 <= len(parts) <= 16 or any(b[0] - a[2] > .24 * h for a, b in zip(parts, parts[1:])):
            continue
        word, stems = [], 0
        for n, (_, _, _, _, glyph) in enumerate(parts):
            options = index.rank(glyph, alphabet="choice_latin", limit=6)
            if len(options) < 3:
                break
            best = options[0]
            if (0 < n < len(parts) - 1 and {value["text"] for value in options[:2]} == {"I", "l"}
                    and best["distance"] <= .085 and options[2]["distance"] - options[1]["distance"] >= .037
                    and glyph.shape[1] <= .19 * h):
                word.append("l")
                stems += 1
                continue
            name = best["text"]
            if not name.isascii() or not name.isalpha() or name.casefold() in {"i", "l"} and n == 0:
                break
            other = next((value["distance"] for value in options if value["text"].casefold() != name.casefold()), 1.)
            expected = name.upper() if n == 0 else name.lower()
            distance = next((value["distance"] for value in options if value["text"] == expected), 1.)
            if distance > .159 or distance - best["distance"] > .034 or other - distance < .021:
                break
            word.append(expected)
        if len(word) == len(parts) and stems == 1:
            readings.append("".join(word))
    return readings[0] if len(readings) >= 2 and len(set(readings)) == 1 else None


def _cancel(frame, panel, paint, last, index):
    left, top, right, bottom = panel
    span = right - left
    x0 = left + round(.55 * span)
    y0 = max(last + round(.043 * span), bottom - round(.140 * span))
    end = bottom - round(.028 * span)
    if end <= y0:
        return None
    crop = frame[y0:end, x0:right - round(.032 * span)]
    distances = _distance(crop, paint)
    ink = np.uint8(distances >= 41)
    x, y, w, h = cv2.boundingRect(ink)
    if (not .019 * span <= h <= .053 * span or not 1.45 * h <= w <= 5.4 * h
            or not .040 * span <= right - x0 - x - w <= .156 * span):
        return None
    ink_signals = crop[distances >= 66]
    if len(ink_signals) < 16 or np.ptp(np.median(ink_signals, axis=0)) < 30:
        return None
    box = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
    if _read_button(frame, box, paint, index) != ("cancel", "取消"):
        return None
    dx, dy = round(.027 * span), round(.020 * span)
    return {"role": "cancel", "function": "取消", "preview_text": "取消",
            "box": [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy],
            "click_point": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]}


def dark_radio_dialog(frame, font_index):
    """Return measured foreground and opt-out state when the panel, rings, and labels agree."""
    height, width = frame.shape[:2]
    if font_index is None or width < 270 or not 1.60 * width <= height <= 2.83 * width:
        return None
    surface = _face(frame)
    if surface is None:
        return None
    panel, paint = surface
    left, top, right, bottom = panel
    span = right - left
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    zone = (left + round(.035 * span), top + round(.128 * span),
            left + round(.179 * span), bottom - round(.126 * span))
    rings = sorted((glyph for glyph in find_choice_glyphs(gray, zone, (round(.034 * span), round(.076 * span)))
                    if glyph.kind == "radio" and glyph.confidence >= .78), key=lambda glyph: glyph.center[1])
    if not 4 <= len(rings) <= 13:
        return None
    xs, ys = zip(*(ring.center for ring in rings), strict=True)
    sizes = [ring.box[3] - ring.box[1] for ring in rings]
    steps = np.diff(ys)
    pitch = np.median(steps)
    if not .072 * span <= pitch <= .190 * span:
        return None
    intervals = np.rint(steps / pitch)
    if (np.ptp(xs) > .025 * span or np.ptp(sizes) > .19 * np.median(sizes)
            or max(abs(steps - intervals * pitch)) > .13 * pitch
            or max(intervals) > 2 or np.count_nonzero(intervals > 1) > 1):
        return None
    regions = {"layer": "dialog", "regions": [panel], "candidates": []}
    verified = [choice for choice in choice_row_controls(frame, regions) if choice.kind == "radio"]
    if len(verified) < max(3, len(rings) - 2):
        return None
    cache = {}
    if not _title(gray, panel, rings[0].box[1], font_index, frame, cache):
        return None
    rows = [_caption(gray, panel, ring, True) for ring in rings]
    if any(row is None for row in rows) or np.ptp([row.caption[0] for row in rows]) > .028 * span:
        return None
    labels = [read_caption(frame, row.caption, font_index, cache=cache, choice=True, sentence_case=True) for row in rows]
    simple = [label["text"] for label in labels if label]
    if len(simple) < 3 or len(set(simple)) != len(simple):
        return None
    cancel = _cancel(frame, panel, paint, rings[-1].box[3], font_index)
    options = (left, round(ys[0] - .49 * pitch), right, round(ys[-1] + .49 * pitch))
    if sum(ring.state == "on" for ring in rings) != 1 or any(ring.state not in ("on", "off") for ring in rings):
        return DarkRadioDialog(panel, (cancel,) if cancel is not None else (), options)
    title_case = sum(text[:1].isascii() and text[:1].isupper() and text[1:].islower() for text in simple) >= 3
    active_contrast = np.median([peer.contrast for peer in rings if peer.state == "off"])
    actions = []
    for ring, row, label in zip(rings, rows, labels, strict=True):
        if (ring.state != "off" or ring.contrast < .59 * active_contrast
                or any(choice.disabled and abs(choice.click_point[1] - ring.center[1]) <= .18 * pitch
                       for choice in verified)):
            continue
        name = label["text"] if label else _stem_word(frame, row.caption, paint, font_index) if title_case else None
        if name:
            _, y = ring.center
            actions.append({"role": "radio_setting", "function": f"选择{name}", "preview_text": name,
                            "box": [left + round(.045 * span), round(y - .43 * pitch),
                                    right - round(.045 * span), round(y + .43 * pitch)],
                            "click_point": [round((left + right) / 2), round(y)]})
    if len({a["function"] for a in actions}) != len(actions):
        return None
    if cancel is not None:
        actions.append(cancel)
    return DarkRadioDialog(panel, tuple(actions), options)
