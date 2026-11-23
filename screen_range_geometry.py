"""Prove a horizontal rail and a raised thumb from the current pixels."""

from collections import defaultdict
from statistics import median

import cv2
import numpy as np

from inventory_list_rows import _Box
from inventory_slider_actions import _Slider, _spans


def _channel_contrast(first, second):
    blue, green, red = cv2.split(cv2.absdiff(first, second))
    return cv2.max(cv2.max(blue, green), red)


def _rail_seeds(gray, unit):
    height, width = gray.shape
    minimum = max(round(.15 * width), round(6.4 * unit))
    lanes = defaultdict(list)
    kernel = np.ones((1, max(3, round(1.38 * unit)) | 1), np.uint8)
    for offset in sorted({max(2, round(unit * factor)) for factor in (.26, .43, .65)}):
        body = gray[offset:height - offset]
        above = cv2.absdiff(body, gray[:height - 2 * offset])
        below = cv2.absdiff(body, gray[2 * offset:])
        line = cv2.compare(cv2.min(above, below), 10, cv2.CMP_GE)
        line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, kernel)
        changes = np.diff(np.pad(line != 0, ((0, 0), (1, 1))), axis=1)
        rows, columns = np.nonzero(changes)
        y, left, right = rows[::2], columns[::2], columns[1::2]
        select = (right - left >= minimum) & (left > 0) & (right < width)
        for row, start, stop in zip(y[select] + offset, left[select], right[select], strict=True):
            key = round(int(start) / unit), round(int(stop) / unit)
            lanes[key].append((int(row), int(start), int(stop)))
    gap = max(3, .32 * unit)
    for occurrences in lanes.values():
        rows = sorted(set(occurrences))
        breaks = [i for i, (first, second) in enumerate(zip(rows, rows[1:]), 1) if second[0] - first[0] >= gap]
        for start, stop in zip((0, *breaks), (*breaks, len(rows)), strict=True):
            if stop - start >= 2:
                yield tuple(int(median(axis)) for axis in zip(*rows[start:stop], strict=True))


def _protrusions(ink, unit):
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(ink))
    for x, y, w, h, area in stats[1:]:
        if .20 * unit <= w <= 1.85 * unit and h >= max(2, .13 * unit) and area >= .48 * w * h:
            yield _Box(int(x), int(y), int(x + w), int(y + h))


def _raised_rows(arm):
    width = arm.shape[1]
    pivot = width // 2
    counts = np.count_nonzero(arm >= 11, axis=1)
    result = counts > pivot
    if width % 2 == 0 and (tied := counts == pivot).any():
        center = np.partition(arm[tied], (pivot - 1, pivot), axis=1)[:, pivot - 1:pivot + 1]
        result[tied] = center.sum(axis=1) >= 22
    return result


def _measure(frame, y, left, right, unit):
    radius = round(1.36 * unit)
    if y < radius or y + radius >= frame.shape[0]:
        return
    chip = frame[y - radius:y + radius + 1, left:right]
    margin = max(2, round(.22 * unit))
    outer = np.concatenate((chip[:margin], chip[-margin:]))
    middle_values = np.partition(outer, (margin - 1, margin), axis=0)[margin - 1:margin + 1]
    background = (middle_values.sum(axis=0, dtype=np.uint16) // 2).astype(np.uint8)
    contrast = _channel_contrast(chip, cv2.repeat(background[None], len(chip), 1))
    span = right - left
    arms = (contrast[:, round(.10 * span):round(.31 * span)], contrast[:, round(.69 * span):round(.90 * span)])
    occupied = _raised_rows(arms[0]) & _raised_rows(arms[1])
    edges = next(((start, stop) for start, stop in _spans(occupied) if start <= radius < stop), None)
    if edges is None:
        return
    top, bottom = edges
    if (not 2 <= bottom - top <= .78 * unit or top < 3 or bottom + 3 > len(chip)
            or abs((top + bottom - 1) / 2 - radius) >= .27 * unit):
        return
    y0 = y - radius
    direction_proven = False
    for level in (34, 48):
        upper = list(_protrusions(contrast[:top] >= level, unit))
        lower = list(_protrusions(contrast[bottom:] >= level, unit))
        for high in upper:
            for low in lower:
                cy = (top + bottom - 1) / 2
                if (abs(high.center[0] - low.center[0]) > .18 * unit or high.bottom < top - 1 or low.top > 1
                        or abs((bottom + low.bottom + high.top - 1) / 2 - cy) > .17 * unit):
                    continue
                cx = (high.center[0] + low.center[0]) / 2
                full_height = bottom + low.bottom - high.top
                if not .67 * unit <= full_height <= 2.08 * unit or not .06 * span <= cx <= .94 * span:
                    continue
                middle = contrast[top:bottom, round(cx - .17 * unit):round(cx + .17 * unit)]
                if not middle.size or np.count_nonzero(middle >= level) < .70 * middle.size:
                    continue
                if not direction_proven:
                    row = chip[(top + bottom) // 2]
                    left_color = np.median(row[round(.13 * span):round(.33 * span)], axis=0)
                    right_color = np.median(row[round(.67 * span):round(.87 * span)], axis=0)
                    if max(abs(right_color - left_color)) < 19:
                        return
                    direction_proven = True
                half = round(max(.35 * full_height, high.width / 2, low.width / 2))
                thumb = _Box(left + round(cx) - half, y0 + high.top, left + round(cx) + half,
                             y0 + bottom + low.bottom)
                rail = _Box(left, y0 + top, right, y0 + bottom)
                yield _Slider(thumb, rail)


def _grouped_seeds(seeds):
    if not seeds:
        return []
    coords = np.float32(seeds)
    keep = []
    for start in range(0, len(seeds), 96):
        part = coords[start:start + 96, None, :]
        radius = np.maximum(part[..., 0], coords[:, 0])
        delta = np.abs(part[..., 1:] - coords[:, 1:])
        vertical = delta[..., 0]
        matched = ((vertical >= .50 * part[..., 0]) & (vertical <= 5.5 * radius)
                   & (np.maximum(delta[..., 1], delta[..., 2]) <= .85 * radius)).any(axis=1)
        keep.extend(seeds[start + int(i)] for i in np.flatnonzero(matched))
    return keep


def ranged_controls(frame, *, grouped=False):
    """Return measured candidates for caption and foreground validation."""
    height, width = frame.shape[:2]
    if min(width, height) < 150:
        return []
    scale = min(1., (660 if grouped else 900) / max(height, width))
    if scale < 1:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    units = (max(6., .020 * min(frame.shape[:2])), max(7., .035 * min(frame.shape[:2])))
    seeds = [(unit, y, left, right) for unit in units for y, left, right in _rail_seeds(gray, unit)]
    if grouped:
        seeds = _grouped_seeds(seeds)
    candidates = []
    for unit, y, left, right in seeds:
        for proposal in _measure(frame, y, left, right, unit):
            thumb = proposal.thumb
            known = next((i for i, item in enumerate(candidates)
                          if abs(item.thumb.center[0] - thumb.center[0]) <= .38 * thumb.width
                          and abs(item.thumb.center[1] - thumb.center[1]) <= .29 * thumb.height), None)
            if known is None:
                candidates.append(proposal)
            elif candidates[known].track.width < proposal.track.width:
                candidates[known] = proposal
    return [_Slider(*(_Box(*[round(value / scale) for value in box.xyxy()])
                       for box in (entry.thumb, entry.track))) for entry in candidates]
