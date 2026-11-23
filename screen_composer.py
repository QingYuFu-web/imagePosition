"""Measure an outlined composer above a dimmed app surface."""

from __future__ import annotations

import math

import cv2
import numpy as np


def _level(image):
    return int(np.median(image))


def _flat(image, level):
    matching = np.abs(image.astype(np.int16) - level) <= 5
    return float(np.mean(matching.mean(axis=1) >= .91))


def _raised_seams(gray):
    height, width = gray.shape
    margin, shoulder = max(2, round(.067 * width)), max(3, round(.020 * width))
    section = gray[:, margin:width - margin]
    profile = np.median(section, axis=1)
    lo, hi = max(shoulder + 1, height - round(.37 * width)), height - round(.18 * width)
    for y in range(lo, hi):
        if profile[y] - profile[y - 1] < 9:
            continue
        above = section[y - shoulder:y]
        below = gray[y + 2:y + 2 + shoulder]
        # The field's horizontal stroke can span the middle of this band.
        inside = np.concatenate((below[:, 1:margin], below[:, -margin:-1]), axis=1)
        old, front = _level(above), _level(inside)
        if not 18 <= front <= 90 or not .24 <= old / front <= .76 or front - old < 11:
            continue
        if _flat(above, old) >= .90 and _flat(inside, front) >= .70:
            yield y, old, front


def _outlined_field(gray, top):
    height, width = gray.shape
    depth = min(height - top, round(.174 * width))
    for threshold in (143, 115):
        mask = np.uint8(gray[top:top + depth] >= threshold)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (.055 * width <= x <= .25 * width and .51 * width <= w <= .87 * width
                    and .082 * width <= h <= .154 * width and .008 * width <= y <= .055 * width
                    and x + w <= .95 * width and cv2.contourArea(contour) >= .81 * w * h):
                found.append((x, top + y, x + w, top + y + h))
        if found:
            return found[0] if len(found) == 1 else None
    return None


def _reaction_centers(frame, gray, lower, tone):
    height, width = frame.shape[:2]
    pad = max(2, round(.006 * width))
    start, stop = lower + pad, min(height, lower + round(.036 * width))
    edge = next((y for y in range(start, stop)
                 if np.mean(gray[y].astype(np.int16) - tone >= 9) >= .87), None)
    if edge is None:
        return []
    tray = frame[edge + pad:height - pad]
    if not .063 * width <= len(tray) <= .155 * width:
        return []
    color = (np.ptp(tray, axis=2) >= 45) & (tray.max(axis=2) >= 115)
    closed = cv2.morphologyEx(np.uint8(color), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    hits = sorted((x + w / 2, y + h / 2) for x, y, w, h, area in stats[1:]
                  if .028 * width <= min(w, h) <= max(w, h) <= .084 * width and area >= .34 * w * h)
    if not 4 <= len(hits) <= 10:
        return []
    xs, ys = np.asarray(hits).T
    gaps = np.diff(xs)
    pitch = float(np.median(gaps))
    if (np.ptp(ys) > .024 * width or np.max(abs(gaps - pitch)) > .19 * pitch
            or not .055 * width <= pitch <= .25 * width or xs[0] > .9 * pitch or width - xs[-1] > .9 * pitch):
        return []
    return hits


def _common_dim_ceiling(frame, top, expected):
    # Status/navigation chrome can be drawn above an application scrim.
    stride = max(1, frame.shape[1] // 450)
    covered = frame[round(.215 * frame.shape[0]):top - 2:stride, ::stride]
    if not covered.size:
        return False
    histograms = [np.bincount(section.ravel(), minlength=256) for section in np.array_split(covered, 3)]
    histogram = sum(histograms)
    cutoff = int(np.searchsorted(np.cumsum(histogram), .9993 * histogram.sum()))
    if not abs(expected - cutoff) <= 14 or cutoff < 43:
        return False
    upper = min(255, cutoff + 11)
    if histogram[upper + 1:].sum() > .00022 * histogram.sum():
        return False
    return sum(bins[max(0, cutoff - 9):cutoff + 3].sum() >= .0011 * bins.sum()
               for bins in histograms if bins.sum()) >= 2


def nested_composer_bounds(frame, working, gray):
    """Return the original-image bounds only with a distinct, dimmed underlay."""
    height, width = gray.shape
    if not 1.45 * width <= height <= 3.05 * width:
        return None
    native_h, native_w = frame.shape[:2]
    for y, old, front in _raised_seams(gray):
        field = _outlined_field(gray, y)
        if field is None or not _reaction_centers(working, gray, field[3], front):
            continue
        top = math.floor(y * native_h / height)
        if _common_dim_ceiling(frame, top, 255 * old / front):
            pad = math.ceil(native_h / height)
            start = max(0, top - pad)
            native_edge = frame[start:top + pad + 1, ::2, 1]
            changed = np.flatnonzero((native_edge >= old + max(7, round(.43 * (front - old)))).mean(axis=1) >= .90)
            return 0, start + int(changed[0]) if changed.size else top, native_w, native_h
    return None
