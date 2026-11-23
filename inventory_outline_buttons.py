"""Locate a dark bordered surface around a visible text label."""

from __future__ import annotations

import cv2
import numpy as np


def _rectangles(edges, origin, label):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x0, y0, x1, y1 = label
    text_w, text_h = x1 - x0, y1 - y0
    tx, ty = (x0 + x1) / 2, (y0 + y1) / 2
    boxes = set()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        l, t = x + origin[0], y + origin[1]
        r, b = l + w, t + h
        if (l < x0 < x1 < r and t < y0 < y1 < b and 1.4 <= w / h <= 12
                and .18 <= text_h / h <= .85 and .16 <= text_w / w <= .97
                and abs(tx - l - w / 2) <= .19 * w and abs(ty - t - h / 2) <= .16 * h
                and cv2.contourArea(contour) >= .84 * w * h):
            boxes.add((l, t, r, b))
    return sorted(boxes, key=lambda box: ((box[2] - box[0]) * (box[3] - box[1]), box))


def _axis_support(gray, outer, inner, *, horizontal):
    if not horizontal:
        gray = gray.T
        outer = (outer[1], outer[0], outer[3], outer[2])
        inner = (inner[1], inner[0], inner[3], inner[2])
    l, t, r, b = outer
    x0, x1 = l + round(.17 * (r - l)), r - round(.17 * (r - l))
    if x1 - x0 < 6:
        return False
    for first, second, direction in ((t, inner[1], 1), (b - 1, inner[3] - 1, -1)):
        stroke = abs(second - first)
        inset = max(2, round(.47 * stroke))
        sample = first + direction * inset
        before, after = first - direction * inset, second + direction * inset
        if min(before, sample, after) < 0 or max(before, sample, after) >= gray.shape[0]:
            return False
        line = gray[sample, x0:x1].astype(np.int16)
        surroundings = np.maximum(gray[before, x0:x1], gray[after, x0:x1])
        if np.mean(line - surroundings >= 10) < .78:
            return False
    return True


def _thin_or_filled_edge(gray, box, span, *, horizontal):
    if not horizontal:
        gray = gray.T
        box = (box[1], box[0], box[3], box[2])
    l, t, r, b = box
    x0, x1 = l + round(.20 * (r - l)), r - round(.20 * (r - l))
    if t <= 2 * span or b + 2 * span >= gray.shape[0] or x1 - x0 < 6:
        return False
    for edge, direction in ((t, 1), (b - 1, -1)):
        outside = gray[np.arange(edge - direction * 2 * span, edge, direction), x0:x1]
        inside = gray[np.arange(edge + direction * span, edge + direction * (2 * span + 1), direction), x0:x1]
        band = gray[edge - 1:edge + 2, x0:x1]
        outer_level, inner_level = np.median(outside, axis=0), np.median(inside, axis=0)
        step = inner_level - outer_level >= 10
        thin = band.max(axis=0) - np.maximum(outer_level, inner_level) >= 11
        if np.mean(step | thin) < .72:
            return False
    return True


def find_dark_outline(gray, label):
    """Return a four-sided surface box in source pixels and its edge style."""
    height, width = gray.shape
    left, top, right, bottom = label
    tw, th = right - left, bottom - top
    x0, x1 = max(0, left - tw), min(width, right + tw)
    y0, y1 = max(0, round(top - 2.1 * th)), min(height, round(bottom + 2.1 * th))
    label_crop = gray[top:bottom, left:right]
    if not label_crop.size:
        return None
    level, ink = np.percentile(label_crop, (40, 97))
    if level > 88 or ink - level < 53:
        return None
    edges = cv2.Canny(gray[y0:y1, x0:x1], 11, 28)
    boxes = _rectangles(edges, (x0, y0), label)
    for index, outer in enumerate(boxes):
        l, t, r, b = outer
        minimum = max(2, round(.023 * (b - t)))
        maximum = .22 * (b - t)
        for inner in boxes[:index]:
            offsets = (inner[0] - l, inner[1] - t, r - inner[2], b - inner[3])
            if min(offsets) < minimum or max(offsets) > maximum or max(offsets) > 3 * min(offsets):
                continue
            if (_axis_support(gray, outer, inner, horizontal=True)
                    and _axis_support(gray, outer, inner, horizontal=False)):
                return outer, "nested_outline"
    # A thin rim or a flat face may expose only one contour. In that case
    # require a brighter label and let the caller check the neighboring controls.
    if (ink - level) / (255 - level) >= .63:
        for box in boxes:
            span = max(2, round(.085 * (box[3] - box[1])))
            if (_thin_or_filled_edge(gray, box, span, horizontal=True)
                    and _thin_or_filled_edge(gray, box, span, horizontal=False)):
                return box, "single_surface"
    return None


def detached_leading_mark(gray, label):
    """Locate text after a visibly separated leading glyph within an OCR box."""
    left, top, right, bottom = label
    crop = gray[top:bottom, left:right]
    level = np.percentile(crop, 40)
    ink = crop >= level + .58 * (255 - level)
    edges = np.flatnonzero(np.diff(np.r_[False, ink.any(axis=0), False]))
    starts, ends = edges[::2], edges[1::2]
    if len(starts) < 4:
        return None
    rows = np.flatnonzero(ink.any(axis=1))
    ink_height = rows[-1] - rows[0] + 1
    prefix_height = np.count_nonzero(ink[:, starts[0]:ends[0]].any(axis=1))
    gap = starts[1] - ends[0]
    regular_spacing = np.median(starts[2:] - ends[1:-1])
    if (not .42 * ink_height <= ends[0] - starts[0] <= 1.20 * ink_height
            or prefix_height < .58 * ink_height or gap < max(.36 * ink_height, 1.9 * regular_spacing)):
        return None
    return (left + starts[1] - 1, top, right, bottom)
