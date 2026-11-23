"""Measure the visible border and state mark of a setting choice glyph."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor

import cv2
import numpy as np


@dataclass(frozen=True)
class ChoiceGlyph:
    box: tuple[int, int, int, int]
    kind: str
    state: str
    contrast: float
    confidence: float

    @property
    def center(self):
        left, top, right, bottom = self.box
        return (left + right) / 2, (top + bottom) / 2


@lru_cache(maxsize=96)
def _shape_masks(height, width):
    yy, xx = np.mgrid[:height, :width]
    dy, dx = np.abs(2 * (yy + .5) / height - 1), np.abs(2 * (xx + .5) / width - 1)
    square = np.maximum(dy, dx)
    round_ = np.hypot(dy, dx)
    return square, round_


def _border_tone(gray, box):
    left, top, right, bottom = box
    height, width = gray.shape
    gap = max(1, round(.13 * min(right - left, bottom - top)))
    surrounding = (
        gray[max(0, top - gap):top, left:right],
        gray[bottom:min(height, bottom + gap), left:right],
        gray[top:bottom, max(0, left - gap):left],
        gray[top:bottom, right:min(width, right + gap)],
    )
    tones = [float(np.median(strip)) for strip in surrounding if strip.size]
    if len(tones) >= 3 and np.ptp(tones) <= 24:
        return float(np.median(tones))
    return None


def _outline(contour, bounds):
    x, y, width, height = bounds
    if not .81 <= width / height <= 1.23:
        return None
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(canvas, [contour - (x, y)], -1, 1, cv2.FILLED)
    observed = canvas.astype(bool)
    square, round_ = _shape_masks(height, width)
    circle = round_ <= 1
    square_edges = square >= min(.86, 1 - 1 / min(width, height) - 1e-6)
    body_fill = observed.mean()
    edge_fill = float(observed[square_edges].mean())
    corners = (square >= .71) & (round_ >= .97)
    corner_fill = float(observed[corners].mean()) if corners.any() else .5
    circle_match = (observed & circle).sum() / (observed | circle).sum()
    if body_fill >= .86 and edge_fill >= .76 and corner_fill >= .66:
        return "checkbox", square, float(body_fill)
    if circle_match >= .83 and corner_fill <= .52 and .61 <= body_fill <= .86:
        return "radio", round_, float(circle_match)
    return None


def _mark_components(chip, core, surface):
    deviations = chip.astype(np.float32) - surface
    strong = float(np.quantile(np.abs(deviations[core]), .92))
    if strong < 18:
        return []
    components = []
    for direction in (-1, 1):
        mask = np.uint8(core & (direction * deviations >= max(13, .28 * strong)))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for index, (x, y, width, height, area) in enumerate(stats[1:], 1):
            if area >= max(3, .018 * chip.size):
                mark = labels[y:y + height, x:x + width] == index
                components.append((x, y, mark, direction, int(area)))
    return sorted(components, key=lambda item: item[4], reverse=True)


def _tick(mark, strength, *, minimum_fill):
    height, width = mark.shape
    if width < 4 or height < 3 or not minimum_fill <= mark.mean() <= .84:
        return False
    yy, xx = mark.nonzero()
    weights = strength[yy, xx]
    column_weights = np.bincount(xx, weights=weights, minlength=width)
    if (column_weights == 0).any():
        return False
    middle = np.bincount(xx, weights=yy * weights, minlength=width) / column_weights
    edge = max(1, round(.19 * width))
    if width - 2 * edge < 2:
        return False
    vertex = edge + int(np.argmax(middle[edge:width - edge]))
    valley = middle[vertex]
    return (.22 * width <= vertex + .5 <= .68 * width
            and valley - middle[:edge].mean() >= max(.55, .13 * height)
            and valley - middle[-edge:].mean() >= max(1.05, .24 * height)
            and abs(np.diff(middle)).max(initial=0) <= .55 * height + .5)


def _round_mark(mark):
    _, radius = _shape_masks(*mark.shape)
    circle = radius <= 1
    return (mark & circle).sum() >= .77 * (mark | circle).sum()


def _state(chip, radius, kind, boundary, background):
    height, width = chip.shape
    core = radius <= .78
    if not core.any():
        return None
    center = radius <= .57
    surface = float(np.median(chip[core]))
    interior = float(np.median(chip[center]))
    contrast = max(abs(boundary - background), abs(surface - background))
    if contrast < 16:
        return None
    marks = _mark_components(chip, core, surface)
    checkbox = kind == "checkbox"
    # Thin checkbox strokes can span the edge of a rounded pixel-width limit.
    tick_max_width = round(.82 * width) if checkbox else .82 * width
    tick_min_fill = .24 if checkbox else .27
    for x, y, mark, polarity, _ in marks:
        mark_height, mark_width = mark.shape
        cx, cy = x + mark_width / 2, y + mark_height / 2
        if not (.26 * width <= cx <= .74 * width and .25 * height <= cy <= .75 * height):
            continue
        strength = polarity * (chip[y:y + mark_height, x:x + mark_width].astype(float) - surface)
        if (.32 * width <= mark_width <= tick_max_width and .25 * height <= mark_height <= .79 * height
                and _tick(mark, strength, minimum_fill=tick_min_fill)):
            return "on", contrast, .94
        if checkbox and (.34 * width <= mark_width <= .77 * width
                         and .07 * height <= mark_height <= .24 * height
                         and mark_width >= 2.5 * mark_height and mark.mean() >= .67
                         and .35 * height <= cy <= .65 * height):
            return "mixed", contrast, .88
        yy, xx = mark.nonzero()
        if kind == "radio" and (.23 * width <= mark_width <= .63 * width
                                and .23 * height <= mark_height <= .63 * height
                                and .69 <= mark_width / mark_height <= 1.43 and _round_mark(mark)
                                and abs(xx.mean() + x + .5 - width / 2) <= .15 * width
                                and abs(yy.mean() + y + .5 - height / 2) <= .15 * height
                                and abs(boundary - surface) >= 12
                                and polarity * (boundary - surface) > 0):
            return "on", contrast, .92
    # A quiet middle is required before treating the absence of a mark as off.
    for x, y, mark, _, _ in marks:
        mark_height, mark_width = mark.shape
        central = radius[y:y + mark_height, x:x + mark_width] <= .48
        if (mark & central).sum() >= max(3, .03 * chip.size):
            return None
    quiet = (np.abs(chip[center].astype(float) - interior) <= 11).mean()
    if quiet < .84 or abs(surface - interior) > 16 or (kind == "radio" and abs(boundary - interior) < 16):
        return None
    return "off", contrast, .86 if abs(boundary - interior) >= 16 else .71


def _measure_glyph(gray, box, kind, radius):
    left, top, right, bottom = box
    background = _border_tone(gray, box)
    if background is None:
        return None
    chip = gray[top:bottom, left:right]
    rim = (radius >= .83) & (radius <= .99)
    if not rim.any():
        return None
    boundary = float(np.median(chip[rim]))
    return _state(chip, radius, kind, boundary, background)


def _overlaps(first, second):
    x0, y0 = np.maximum(first[:2], second[:2])
    x1, y1 = np.minimum(first[2:], second[2:])
    if x1 <= x0 or y1 <= y0:
        return False
    common = (x1 - x0) * (y1 - y0)
    smaller = min((box[2] - box[0]) * (box[3] - box[1]) for box in (first, second))
    return common >= .70 * smaller


def find_choice_glyphs(frame, roi, size_range=(7, 64)):
    """Find measured checkbox and radio shapes in a text-adjacent pixel ROI."""
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        return ()
    height, width = gray.shape
    left, top, right, bottom = map(round, roi)
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    minimum, maximum = size_range
    if min(right - left, bottom - top) < minimum:
        return ()
    chip = gray[top:bottom, left:right]
    differences = None
    if chip.dtype == np.uint8 and chip.size <= 2 ** 24:
        counts = cv2.calcHist([chip], [0], None, [256], [0, 256]).ravel()
        cumulative = np.cumsum(counts, dtype=np.int64)
        lo, hi = np.searchsorted(cumulative, ((chip.size - 1) // 2, chip.size // 2), side="right")
        tone = (int(lo) + int(hi)) / 2
    else:
        chip = chip.astype(np.int16)
        differences = np.abs(chip - np.median(chip))
    proposals = {}
    measurements = {}
    for threshold in (9, 12, 15, 17, 22, 28, 41, 61):
        if differences is None:
            low, high = floor(tone - threshold) + 1, ceil(tone + threshold) - 1
            mask = cv2.bitwise_not(cv2.inRange(chip, low, high))
        else:
            mask = np.uint8(differences >= threshold)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bounds = cv2.boundingRect(contour)
            x, y, w, h = bounds
            if not (minimum <= min(w, h) and max(w, h) <= maximum and .81 <= w / h <= 1.23):
                continue
            geometry = _outline(contour, bounds)
            if geometry is None:
                continue
            box = x + left, y + top, x + left + w, y + top + h
            kind, radius, shape_confidence = geometry
            key = box, kind
            if key not in measurements:
                # Distinct thresholds can describe the same shape over identical source pixels.
                measurements[key] = _measure_glyph(gray, box, kind, radius)
            evidence = measurements[key]
            if evidence is None:
                continue
            state, contrast, mark_confidence = evidence
            confidence = mark_confidence * shape_confidence
            if box not in proposals or confidence > proposals[box].confidence:
                proposals[box] = ChoiceGlyph(box, kind, state, contrast, confidence)
    accepted = []
    for glyph in sorted(proposals.values(), key=lambda item: item.confidence, reverse=True):
        if not any(_overlaps(existing.box, glyph.box) for existing in accepted):
            accepted.append(glyph)
    return tuple(sorted(accepted, key=lambda item: (item.box[1], item.box[0])))
