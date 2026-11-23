"""Find separately addressable arrows on visually evidenced numeric fields."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

import cv2
import numpy as np

from inventory_list_rows import _Box, _labels
from inventory_value_select import _Line, _ink_strength, _inline_title, _looks_down, _overlap, _rectangles


_NUMBER = (
    r"[+\-−]?(?:(?:\d{1,3}(?:[ '\u202f]\d{3})+|\d+)(?:[.,]\d+)?|[.,]\d+)\s*"
    r"(?:%|px|pt|mm|cm|m|km|s|ms|[µμu]s|ns|min|h|fps|hz|khz|db|[kmg]i?b|°[cf]?|[×x]"
    r"|秒|毫秒|分钟|小時|小时|像素|毫米|厘米|點|点|倍)?"
)
_NUMERIC_TEXT = re.compile(_NUMBER, re.IGNORECASE)
_MERGED_NUMERIC = re.compile(rf"(?P<label>.*?[^\d\s.,+\-−])\s*(?P<value>{_NUMBER})", re.IGNORECASE)


@dataclass(frozen=True)
class _Arrow:
    box: _Box
    up: bool
    strength: float


def _numeric(text):
    return _NUMERIC_TEXT.fullmatch(unicodedata.normalize("NFKC", text).strip()) is not None


def _arrow_marks(distance, left, top, field, threshold):
    _, components, stats, _ = cv2.connectedComponentsWithStats(np.uint8(distance >= threshold), connectivity=8)
    h = field.height
    for index, (x, y, width, height, area) in enumerate(stats[1:], 1):
        if not (max(4, .145 * h) <= width <= .52 * h and max(3, .09 * h) <= height <= .32 * h
                and area >= 5 and 0 < x < x + width < distance.shape[1]
                and 0 <= y < y + height <= distance.shape[0]):
            continue
        glyph = components[y:y + height, x:x + width] == index
        down = _looks_down(glyph, relaxed=True)
        up = _looks_down(glyph[::-1], relaxed=True)
        box = _Box(left + x, top + y, left + x + width, top + y + height)
        dx = field.right - box.center[0]
        dy = (box.center[1] - field.top) / h
        if up == down or not .17 * h <= dx <= .61 * h or not (.14 <= dy <= .46 if up else .54 <= dy <= .86):
            continue
        strength = float(np.median(distance[y:y + height, x:x + width][glyph]))
        yield _Arrow(box, up, strength)


def _arrow_pair(gray, field):
    h = field.height
    left = max(field.left + round(.46 * field.width), field.right - round(1.00 * h))
    right = field.right - max(1, round(.055 * h))
    top, bottom = field.top + max(1, round(.10 * h)), field.bottom - max(1, round(.10 * h))
    chip = gray[top:bottom, left:right].astype(np.int16)
    if min(chip.shape) < 7:
        return None
    dark, light = np.quantile(chip, [.16, .84], axis=1)
    for threshold in (54, 82, 39, 30):
        pairs = []
        for distance in (light[:, None] - chip, chip - dark[:, None]):
            marks = tuple(_arrow_marks(distance, left, top, field, threshold))
            for up in (mark for mark in marks if mark.up):
                for down in (mark for mark in marks if not mark.up):
                    if (abs(up.box.center[0] - down.box.center[0]) <= max(1.25, .075 * h)
                            and max(1, .07 * h) <= down.box.top - up.box.bottom <= .47 * h
                            and .69 <= up.box.width / down.box.width <= 1.45
                            and .61 <= up.box.height / down.box.height <= 1.64):
                        pairs.append((up, down))
        if pairs:
            return pairs[0] if len(pairs) == 1 else None
    return None


def _value_matches(field, values):
    h = field.height
    return [line for line in values if field.left - .18 * h <= line.box.left < field.right - .54 * h
            and line.box.right <= field.right + .32 * h and .33 * h <= line.box.height <= 1.46 * h
            and abs(line.box.center[1] - field.center[1]) <= .25 * h]


def _title(gray, field, labels):
    # OCR quadrilaterals sometimes include padding beyond the very end of a short title.
    h = field.height
    candidates = []
    for line in labels:
        box = line.box
        if abs(box.center[1] - field.center[1]) > .23 * h:
            continue
        if field.left < box.right <= field.left + .40 * h and box.left <= field.left - .6 * h:
            line = _Line(_Box(box.left, box.top, field.left, box.bottom), line.text, line.score)
        candidates.append(line)
    return _inline_title(gray, field, candidates)


def _value_ink_strength(gray, field, value, arrows, title_strength, *, trim_outline=False):
    left, top = max(field.left + 2, value.box.left), max(field.top + 2, value.box.top)
    right = min(value.box.right, min(arrow.box.left for arrow in arrows) - 2)
    bottom = min(field.bottom - 2, value.box.bottom)
    if right - left < 5 or bottom - top < 4:
        return None
    chip = gray[top:bottom, left:right].astype(np.int16)
    contrast = np.abs(chip - np.median(chip))
    ink = contrast >= max(35, .39 * title_strength)
    if trim_outline:
        ink[ink.mean(axis=1) >= .78] = False
        ink[:, ink.mean(axis=0) >= .84] = False
    if np.count_nonzero(ink) < max(5, .048 * chip.size):
        return None
    return float(np.quantile(np.where(ink, contrast, 0), .94))


def _merged_field_text(gray, field, merged, arrows):
    h = field.height
    rail = min(arrow.box.left for arrow in arrows)
    candidates = [(line, parsed) for line, parsed in merged
                  if line.box.left <= field.left - .60 * h and field.left + .36 * h <= line.box.right < rail - 1
                  and .40 * h <= line.box.height <= 1.20 * h
                  and abs(line.box.center[1] - field.center[1]) <= .21 * h]
    if len(candidates) != 1:
        return None
    line, parsed = candidates[0]
    box = line.box
    margin = max(2, round(.24 * h))
    left, right = field.left - margin, field.left + margin + 1
    side = gray[field.top + margin:field.bottom - margin, left:right].astype(np.int16)
    edge_support = (np.abs(np.diff(side, axis=1)) >= 23).mean(axis=0)
    index = int(np.argmax(edge_support))
    if edge_support[index] < .60:
        return None
    boundary = left + index + 1
    # Require a continuous value-field edge between the OCR title and its numeric suffix.
    if boundary - box.left < .60 * h or box.right - boundary < .36 * h:
        return None
    title = _Line(_Box(box.left, box.top, boundary - 1, box.bottom), parsed["label"], line.score)
    value = _Line(_Box(boundary + 2, box.top, box.right, box.bottom), parsed["value"], line.score)
    return title, value


def _clear_value_lane(gray, field, value):
    h = field.height
    left = field.left + max(2, round(.14 * h))
    right = value.box.left - max(2, round(.12 * h))
    if right - left < h:
        return True
    margin = max(2, round(.23 * h))
    chip = gray[field.top + margin:field.bottom - margin, left:right].astype(np.int16)
    contrast = np.abs(chip - np.median(chip, axis=1)[:, None])
    # A right-aligned numeric value can have padding, but not intervening fields or labels.
    return np.count_nonzero(contrast > 32) <= max(5, .015 * chip.size)


def _enabled_arrows(gray, field, title, labels, arrows, title_strength):
    h = field.height
    arrow_strength = min(arrow.strength for arrow in arrows)
    if title_strength < 40 or arrow_strength < max(43, .37 * title_strength):
        return False
    nearby_strength = max((_ink_strength(gray, line) for line in labels
                           if .52 * h <= abs(title.box.center[1] - line.box.center[1]) <= 3.60 * h
                           and abs(title.box.left - line.box.left) <= 4.5 * h
                           and .62 * title.box.height <= line.box.height <= 1.38 * title.box.height),
                          default=title_strength)
    # Disabled controls may retain clear arrows while both the arrows and their label are dimmed.
    return title_strength >= .68 * nearby_strength or arrow_strength >= .73 * nearby_strength


def _action(field, arrow, split, label):
    box = arrow.box
    h = field.height
    pad_x, pad_y = max(1, round(.07 * h)), max(1, round(.045 * h))
    top = max(field.top + 1 if arrow.up else split, box.top - pad_y)
    bottom = min(split if arrow.up else field.bottom - 1, box.bottom + pad_y)
    bounds = (max(field.left + 1, box.left - pad_x), top, min(field.right - 1, box.right + pad_x), bottom)
    role, verb = ("increment_setting", "增加") if arrow.up else ("decrement_setting", "减少")
    return {"role": role, "setting_label": label, "function": f"{verb}{label}",
            "box": bounds, "center": box.center}


def suggest_stepper_actions(frame, nodes, *, overlay=None, evidence=None):
    """Return independent up and down clicks for verified, visibly named numeric fields."""
    height, width = frame.shape[:2]
    span = min(height, width)
    if span < 150 or not nodes:
        return []
    lines = (evidence.lines if evidence is not None else
             tuple(_Line(box, text, score) for box, text, score in _labels(nodes, width, height)))
    values, labels, merged = [], [], []
    for line in lines:
        if _numeric(line.text):
            values.append(line)
        else:
            labels.append(line)
            parsed = _MERGED_NUMERIC.fullmatch(unicodedata.normalize("NFKC", line.text).strip())
            if parsed is not None and any(char.isalpha() for char in parsed["label"]):
                merged.append((line, parsed))
    if not labels or not (values or merged):
        return []
    if evidence is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rectangles = _rectangles(gray)
    else:
        gray, rectangles = evidence.gray, evidence.rectangles
    regions = ((0, 0, width, height),) if overlay is None else overlay.regions
    found = []
    for field in rectangles:
        if not (max(14, .009 * span) <= field.height <= .10 * span
                and 1.55 <= field.width / field.height <= 23):
            continue
        region = next((r for r in regions if r[0] <= field.left < field.right <= r[2]
                       and r[1] <= field.top < field.bottom <= r[3]), None)
        if region is None or any(_overlap(field, accepted) for accepted, _ in found):
            continue
        matched = _value_matches(field, values)
        if len(matched) > 1 or (matched and not _clear_value_lane(gray, field, matched[0])):
            continue
        potential_merge = [(line, parsed) for line, parsed in merged
                           if line.box.left < field.left < line.box.right < field.right
                           and abs(line.box.center[1] - field.center[1]) <= .21 * field.height]
        if not matched and not potential_merge:
            continue
        arrows = _arrow_pair(gray, field)
        if arrows is None:
            continue
        local_labels = [line for line in labels if region[0] <= line.box.left < line.box.right <= region[2]
                        and region[1] <= line.box.top < line.box.bottom <= region[3]]
        if matched:
            value = matched[0]
            title = _title(gray, field, local_labels)
            if title is None:
                continue
        else:
            local_merged = [(line, parsed) for line, parsed in potential_merge if line in local_labels]
            separated = _merged_field_text(gray, field, local_merged, arrows)
            if separated is None:
                continue
            title, value = separated
        title_strength = _ink_strength(gray, title)
        if not _enabled_arrows(gray, field, title, local_labels, arrows, title_strength):
            continue
        value_strength = _value_ink_strength(gray, field, value, arrows, title_strength, trim_outline=not matched)
        if value_strength is None:
            continue
        label = title.text.strip(" :：")
        if label:
            split = (arrows[0].box.bottom + arrows[1].box.top) // 2
            # Keep the winning field during deduplication even when all three ink sources are muted.
            enabled = title_strength >= 96 or value_strength >= 106 or min(arrow.strength for arrow in arrows) >= 134
            actions = [_action(field, arrow, split, label) for arrow in arrows] if enabled else []
            found.append((field, actions))
    return [action for _, actions in sorted(found, key=lambda item: (item[0].top, item[0].left)) for action in actions]
