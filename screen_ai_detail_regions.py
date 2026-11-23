"""Choose bounded evidence crops where the visible screen has uncovered detail."""

from math import floor, sqrt

import cv2
import numpy as np


def _edge_weights(page, frame, panel, covered, photo):
    left, top, right, bottom = panel
    width, height = right - left, bottom - top
    scale = min(1., 512 / max(width, height))
    size = max(1, round(width * scale)), max(1, round(height * scale))
    pixels = cv2.resize(frame[top:bottom, left:right], size, interpolation=cv2.INTER_AREA)
    sx, sy = size[0] / width, size[1] / height
    allowed = np.full(size[::-1], 1, np.uint8)
    if corners := page.get("screen_corners"):
        allowed.fill(0)
        polygon = np.rint((np.asarray(corners) - [left, top]) * [sx, sy]).astype(np.int32)
        cv2.fillConvexPoly(allowed, polygon, 1)
    if photo is not None and photo.blocked is not None:
        to_crop = np.float64([[sx, 0, -sx * left], [0, sy, -sy * top], [0, 0, 1]])
        blocked = cv2.warpPerspective(photo.blocked, to_crop @ photo.to_photo, size, flags=cv2.INTER_NEAREST)
        allowed[blocked != 0] = 0
    scoring = allowed.copy()
    for x0, y0, x1, y1 in covered:
        x0, x1 = max(0, floor((x0 - left) * sx)), min(size[0], round((x1 - left) * sx))
        y0, y1 = max(0, floor((y0 - top) * sy)), min(size[1], round((y1 - top) * sy))
        if x0 < x1 and y0 < y1:
            scoring[y0:y1, x0:x1] = 0
    pixels = cv2.GaussianBlur(pixels, (3, 3), 0)
    edges = np.zeros(size[::-1], np.uint8)
    edges[:, 1:-1] = cv2.absdiff(pixels[:, 2:], pixels[:, :-2]).max(axis=2)
    np.maximum(edges[1:-1], cv2.absdiff(pixels[2:], pixels[:-2]).max(axis=2), out=edges[1:-1])
    # Score compact concentrations of edges. A textured background alone
    # should not dominate the budget available for small visible details.
    edge_mask = (edges >= 20).astype(np.float32)
    edge_mask *= cv2.erode(scoring, np.ones((5, 5), np.uint8))
    local = cv2.boxFilter(edge_mask, -1, (13, 13))
    surrounding = cv2.boxFilter(edge_mask, -1, (41, 41))
    weights = edge_mask * (.20 + np.maximum(0., local - surrounding))
    return allowed, weights, (sx, sy)


def _positions(start, stop, span):
    end = stop - span
    stride = max(1, span // 5)
    return (*range(start, end, stride), end)


def _window_score(integral, left, top, right, bottom):
    return integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]


def detail_regions(page, frame, panels, covered, photo, *, budget, limit):
    """Return original-image rectangles only. No control labels or click points are inferred."""
    has_cards = bool(covered)
    excluded = [row["box"] for row in page["elements"] if row.get("disabled")]
    excluded.extend(sample["box"] for row in page["elements"]
                    if (sample := row.get("reviewed_sample")) is not None and sample["purpose"] is None)
    covered = [*covered, *excluded]
    candidates = []
    for panel in panels:
        left, top, right, bottom = panel
        width, height = right - left, bottom - top
        if min(width, height) < 8:
            continue
        allowed, weights, (sx, sy) = _edge_weights(page, frame, panel, covered, photo)
        detail = float(weights.sum())
        if detail < 2.:
            continue
        whole_budget = budget if len(panels) == 1 and not has_cards else budget // 2
        if width * height <= whole_budget and np.count_nonzero(allowed) >= .80 * allowed.size:
            candidates.append((2, detail / sqrt(allowed.size), panel))
            continue
        window_area = min(budget // limit, width * height)
        window_width = min(width, round(sqrt(1.75 * window_area)))
        window_height = min(height, window_area // max(1, window_width))
        if min(window_width, window_height) < 8:
            continue
        allowed_sum = cv2.integral(allowed)
        detail_sum = cv2.integral(weights)
        for y in _positions(top, bottom, window_height):
            y0, y1 = round((y - top) * sy), round((y + window_height - top) * sy)
            for x in _positions(left, right, window_width):
                x0, x1 = round((x - left) * sx), round((x + window_width - left) * sx)
                pixels = (x1 - x0) * (y1 - y0)
                if _window_score(allowed_sum, x0, y0, x1, y1) < .80 * pixels:
                    continue
                score = _window_score(detail_sum, x0, y0, x1, y1)
                if pixels and score >= max(2., .001 * pixels):
                    candidates.append((2, score / sqrt(pixels), [x, y, x + window_width, y + window_height]))
        if not has_cards:
            small_width = max(8, window_width // 2)
            small_height = max(8, window_height // 2)
            for y in (top, bottom - small_height):
                y0, y1 = round((y - top) * sy), round((y + small_height - top) * sy)
                for x in (left, right - small_width):
                    x0, x1 = round((x - left) * sx), round((x + small_width - left) * sx)
                    pixels = (x1 - x0) * (y1 - y0)
                    if pixels and _window_score(allowed_sum, x0, y0, x1, y1) >= .80 * pixels:
                        score = _window_score(detail_sum, x0, y0, x1, y1)
                        if score >= max(2., .001 * pixels):
                            candidates.append((int(y == top), score / sqrt(pixels),
                                               [x, y, x + small_width, y + small_height]))
    selected = []
    for _, _, box in sorted(candidates, key=lambda row: row[:2], reverse=True):
        left, top, right, bottom = box
        area = (right - left) * (bottom - top)
        if area > budget or any(max(0, min(right, b[2]) - max(left, b[0])) *
                                max(0, min(bottom, b[3]) - max(top, b[1])) > .16 * area for b in selected):
            continue
        selected.append(box)
        budget -= area
        if len(selected) == limit:
            break
    return selected
