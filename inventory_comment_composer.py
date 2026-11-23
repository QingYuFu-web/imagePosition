"""Locate the empty comment field inside an active comments sheet."""

from itertools import combinations
import unicodedata

import cv2
import numpy as np

from inventory_back_evidence import _elements
from inventory_toolbar_actions import _spans
from screen_foreground import (COMMENT_SHEET_TITLES, _working_frame, account_comment_field,
                               is_account_comment_placeholder)


_PLACEHOLDERS = {"addacomment", "writeacomment", "添加评论", "写评论", "新增留言", "新增評論"}


def _text(value):
    value = unicodedata.normalize("NFKC", value).casefold().strip(" .…")
    return " ".join(value.split())


def _outline_rules(gray, first):
    height, width = gray.shape
    spans = []
    for y in range(first, height - max(1, round(.011 * width))):
        line = gray[y]
        for left, right in _spans((line >= 204) & (line <= 248)):
            if .10 * width <= left < right <= .985 * width and .55 * width <= right - left <= .88 * width:
                spans.append((left, y, right))
    return spans


def _fields(gray, first):
    width = gray.shape[1]
    for (l, t, r), (l2, b, r2) in combinations(_outline_rules(gray, first), 2):
        if .074 * width <= b - t <= .153 * width and max(abs(l - l2), abs(r - r2)) <= .037 * width:
            left, right = max(l, l2), min(r, r2)
            if gray[t + 2:b - 1, left:right].mean() >= 233:
                yield int(left), int(t), int(right), int(b)


def _read_placeholder(line_reader, variants, minimum):
    rows = line_reader.recognize_prepared_lines(variants)
    labels = ["".join(_text(row["text"]).split()) for row in rows if row["score"] >= minimum]
    if len(labels) == 2 and labels[0] == labels[1] and labels[0] in _PLACEHOLDERS:
        return labels[0], min(row["score"] for row in rows)
    return None


def _composer(box, center, width, height):
    sx, sy = width - 1, height - 1
    left, top, right, bottom = box
    x, y = center
    return {"role": "comment_input", "function": "输入评论", "anchor_id": None,
            "box_fraction": [left / sx, top / sy, right / sx, bottom / sy],
            "screen_fraction": [x / sx, y / sy]}


def _account_composer(frame, observation, layer):
    height, width = frame.shape[:2]
    points = {point["id"]: point for point in observation["keypoints"]}
    candidates = []
    for element in _elements(observation, width, height):
        if element.kind == "control_candidate":
            entries = points[element.id].get("text_evidence") or ()
        else:
            entries = ({"text": element.label, "confidence": element.confidence},)
        if any(row["confidence"] >= .95 and row.get("relation") != "associated_label"
               and is_account_comment_placeholder(row["text"]) for row in entries):
            candidates.append(element)
    if not candidates or (work := _working_frame(frame)) is None:
        return None
    h, w = work.shape[:2]
    scale = np.asarray([w / width, h / height])
    panel = np.asarray(layer["regions_fraction"][0]) * np.tile([(width - 1) * scale[0], (height - 1) * scale[1]], 2)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    measured = account_comment_field(gray, tuple(np.round(panel).astype(int)))
    if measured is None:
        return None
    left, top, right, bottom = measured
    pad = .023 * w
    matching = [element for element in candidates
                if all(lo - pad <= value <= hi + pad
                       for lo, hi, value in zip((left, top, left, top), (right, bottom, right, bottom),
                                                np.asarray(element.box) * np.tile(scale, 2), strict=True))]
    if len(matching) != 1:
        return None
    row = gray[top:bottom, left + round(.035 * w):right - round(.09 * w)]
    if row.size == 0 or (row <= 197).mean() < .017:
        return None
    box = np.asarray(measured) / np.tile(scale, 2)
    x, y = (left + .34 * (right - left)) / scale[0], (top + bottom) / (2 * scale[1])
    return _composer(box, (x, y), width, height)


def _viewer_shelf(frame, observation):
    """Locate the unboxed prompt on a separated dark input shelf."""
    height, width = frame.shape[:2]
    if not 1.55 * width <= height <= 2.65 * width:
        return None
    candidates = [item for item in _elements(observation, width, height)
                  if (item.kind == "text" and item.confidence >= .965
                      and "".join(_text(item.label).split()) in _PLACEHOLDERS
                      and .026 * width <= item.box[0] <= .19 * width and item.box[2] <= .62 * width
                      and .86 * height <= item.center[1] <= .96 * height
                      and .018 * width <= item.height <= .073 * width)]
    if len(candidates) != 1:
        return None
    anchor = candidates[0]
    x, y = anchor.center
    trim = round(.027 * width)
    upper, lower = round(y - .047 * width), min(height, round(y + .055 * width))
    before = round(y - .16 * width)
    cutoff = round(y - .10 * width)
    gray = cv2.cvtColor(frame[before:], cv2.COLOR_BGR2GRAY)
    shelf = gray[upper - before:lower - before, trim:width - trim]
    if ((shelf <= 21).mean() < .89 or np.median(shelf) > 9
            or (gray[:cutoff - before, trim:width - trim] >= 39).mean() < .16):
        return None
    probe = gray[anchor.box[1] - before:anchor.box[3] - before, anchor.box[0]:anchor.box[2]]
    if (probe >= 55).mean() < .035:
        return None
    # The lower navigation mark corroborates the boundary of the dark shelf.
    navigation = gray[max(lower, round(.95 * height)) - before:]
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(navigation >= 180), connectivity=8)
    bars = [(l, w) for l, _, w, h, area in stats[1:]
            if .25 * width <= w <= .56 * width and 1 <= h <= .025 * width
            and abs(l + w / 2 - width / 2) <= .065 * width and area >= .65 * w * h]
    if len(bars) != 1:
        return None
    return _composer((anchor.box[0], upper, anchor.box[2], lower), (x, y), width, height)


def suggest_comment_composer(frame, observation, *, line_reader=None):
    """Return the visible writing field with a confirmed empty placeholder."""
    layer = observation.get("active_layer") or {}
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None
    if not layer:
        return _viewer_shelf(frame, observation)
    if layer.get("kind") != "bottom_sheet":
        return None
    if (composer := _account_composer(frame, observation, layer)) is not None:
        return composer
    if line_reader is None:
        return None
    height, width = frame.shape[:2]
    _, top, _, bottom = layer["regions_fraction"][0]
    titles = [text for text in _elements(observation, width, height) if text.kind == "text"
              and text.confidence >= .92 and _text(text.label) in COMMENT_SHEET_TITLES
              and .33 * width <= text.center[0] <= .67 * width
              and top <= text.center[1] / (height - 1) <= top + .23 * (bottom - top)]
    if len(titles) != 1:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower = round((top + .78 * (bottom - top)) * height)
    fields = list(_fields(gray, lower))
    if not fields:
        return None
    left, upper, right, lower = fields[0]
    if any(max(abs(a - b) for a, b in zip((left, upper, right, lower), field, strict=True)) > .018 * width
           for field in fields):
        return None
    pad = max(2, round(.021 * width))
    # The straight border starts inward of the rounded end of the field.
    ink_left = max(0, left - round(.20 * (lower - upper)))
    clip = frame[upper + pad:lower - pad, ink_left:right - round(.17 * width)]
    if min(clip.shape[:2]) < 4:
        return None
    plane = cv2.cvtColor(clip, cv2.COLOR_BGR2GRAY)
    yy, xx = np.nonzero(plane < 201)
    if not xx.size:
        return None
    margin = max(2, round(.012 * width))
    crop = np.s_[max(0, yy.min() - margin):yy.max() + margin + 1,
                 max(0, xx.min() - margin):xx.max() + margin + 1]
    contrast = cv2.normalize(plane[crop], None, 0, 255, cv2.NORM_MINMAX)
    reading = _read_placeholder(line_reader, [clip[crop], cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)], 85)
    if reading is None:
        return None
    label, confidence = reading
    if confidence < 94:
        larger = cv2.resize(clip[crop], None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        ink = cv2.cvtColor(larger, cv2.COLOR_BGR2GRAY)
        span = cv2.normalize(ink, None, 0, 255, cv2.NORM_MINMAX)
        background = float(np.percentile(ink, 90))
        gentle = np.uint8(np.clip((ink.astype(np.float32) - (background - 85)) * (255 / 75), 0, 255))
        variants = [cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR) for plane in (span, gentle)]
        confirmed = _read_placeholder(line_reader, variants, 94)
        if confirmed is None:
            # At very small text sizes, the unmodified color and another
            # sampling grid may preserve different strokes from the same line.
            enlarged = cv2.resize(clip[crop], None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
            gray_enlarged = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
            adjusted = cv2.normalize(gray_enlarged, None, 0, 255, cv2.NORM_MINMAX)
            confirmed = _read_placeholder(line_reader, [larger, cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)], 94)
        if confirmed is None or confirmed[0] != label:
            return None
    # Keep the click in the typing area, away from the trailing media control.
    usable_right = right - round(.17 * width)
    return _composer((left, upper, usable_right, lower), ((left + usable_right) / 2, (upper + lower) / 2),
                     width, height)
