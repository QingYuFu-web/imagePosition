"""Recover small missing cells of a visibly divided horizontal choice row."""

from itertools import combinations

import cv2
import numpy as np


def _peak(profile, start, stop):
    start = max(0, round(start))
    stop = min(profile.size, round(stop))
    if start >= stop:
        return None
    index = start + int(profile[start:stop].argmax())
    return index, profile[index]


def _size(box):
    left, top, right, bottom = box
    return right - left, bottom - top


def _center(box):
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def _caption(gray, box):
    left, top, right, bottom = box
    width, height = _size(box)
    inset_x, inset_y = round(.14 * width), round(.13 * height)
    field = gray[top + inset_y:bottom - inset_y, left + inset_x:right - inset_x].astype(np.int16)
    contrast = np.abs(field - np.median(field))
    peak = float(np.percentile(contrast, 95))
    if peak < 16:
        return False
    marks = np.uint8(contrast >= max(11, .54 * peak))
    x, y, span, depth = cv2.boundingRect(marks)
    return (.18 * width <= span <= .74 * width and .22 * height <= depth <= .72 * height
            and abs(x + span / 2 - field.shape[1] / 2) <= .15 * width
            and abs(y + depth / 2 - field.shape[0] / 2) <= .17 * height
            and .035 <= marks.mean() <= .48)


def _row(gray, first, last, width, height):
    x0, y0 = _center(first)
    x1, y1 = _center(last)
    left, right = first[0], last[2]
    center = (y0 + y1) / 2
    border = max(1, round(.027 * height))
    lo = max(0, round(center - .57 * height) - border)
    hi = min(gray.shape[0], round(center + .58 * height) + border)
    patch = gray[lo:hi, left:right].astype(np.int16)
    horizontal = np.pad(np.median(np.abs(patch[2 * border:] - patch[:-2 * border]), axis=1), border)
    top = _peak(horizontal, center - lo - .54 * height, center - lo - .20 * height)
    bottom = _peak(horizontal, center - lo + .20 * height, center - lo + .55 * height)
    if top is None or bottom is None or min(top[1], bottom[1]) < 8:
        return []
    a, b = top[0] + lo, bottom[0] + lo
    depth = b - a
    if not .43 * height <= depth <= .80 * height:
        return []
    cut = round(.16 * depth)
    field = gray[a + cut:b - cut, left:right].astype(np.int16)
    columns = np.pad(np.median(np.abs(field[:, 2 * border:] - field[:, :-2 * border]), axis=0), border)
    result = []
    for steps in range(2, 6):
        pitch = (x1 - x0) / steps
        if not .69 * width <= pitch <= .96 * width:
            continue
        slop = .092 * pitch
        predicted = [x0 + (index - .5) * pitch - left for index in range(steps + 2)]
        seams = [_peak(columns, x - slop, x + slop + 1) for x in predicted]
        if (any(seam is None or seam[1] < 9 for seam in seams)
                or min(seams[0][1], seams[1][1]) < 15):
            continue
        edges = [seam[0] + left for seam in seams]
        if any(not .82 * pitch <= right - left <= 1.18 * pitch for left, right in zip(edges, edges[1:])):
            continue
        cells = [(left, a, right, b) for left, right in zip(edges, edges[1:])]
        if all(_caption(gray, cell) for cell in cells):
            result.append((min(top[1], bottom[1]) + np.median([seam[1] for seam in seams]), cells[1:-1]))
    return result


def _covered(box, covered):
    x, y = _center(box)
    return any(a <= x < c and b <= y < d for a, b, c, d in covered)


def choice_context_boxes(upright, covered, blocked=None):
    """Return at most two evidenced cells between supplied crops on one row."""
    height, width = upright.shape[:2]
    if height < 240 or width < 1.40 * height:
        return []
    anchors = []
    for box in covered:
        w, h = _size(box)
        left, top, right, bottom = box
        if (0 <= left < right <= width and 0 <= top < bottom <= height
                and .05 * height <= h <= .16 * height and 1.55 * h <= w <= 3.80 * h):
            anchors.append((box, w, h, _center(box)))
    gray = None
    rows = []
    for (first, w0, h0, (x0, y0)), (last, w1, h1, (x1, y1)) in combinations(
            sorted(anchors, key=lambda item: item[3][0]), 2):
        w, h = (w0 + w1) / 2, (h0 + h1) / 2
        if (abs(y0 - y1) > .14 * h or abs(w0 - w1) > .23 * w or abs(h0 - h1) > .22 * h
                or not 1.55 * w <= x1 - x0 <= 4.90 * w):
            continue
        if gray is None:
            gray = cv2.cvtColor(upright, cv2.COLOR_BGR2GRAY)
        rows.extend(_row(gray, first, last, w, h))
    for _, cells in sorted(rows, key=lambda row: -row[0]):
        result = []
        for left, top, right, bottom in cells:
            if _covered((left, top, right, bottom), covered):
                continue
            pad = max(2, round(.08 * (bottom - top)))
            box = [max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad)]
            if blocked is None or not np.any(blocked[box[1]:box[3], box[0]:box[2]]):
                result.append(box)
        if result:
            return result[:2]
    return []
