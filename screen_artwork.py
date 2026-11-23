"""Identify picture tiles that share the same physical row and background."""

import cv2
import numpy as np


def _background(frame, bounds):
    left, top, right, bottom = bounds
    margin = max(2, round(.043 * (right - left)))
    if left < margin + 2 or right + margin + 2 > frame.shape[1]:
        return None
    y0, y1 = top + (bottom - top) // 5, bottom - (bottom - top) // 5
    gutters = (frame[y0:y1, left - margin - 2:left - 2], frame[y0:y1, right + 2:right + margin + 2])
    colors = [np.median(gutter, axis=(0, 1)).astype(np.int16) for gutter in gutters]
    if np.max(np.abs(colors[0] - colors[1])) > 9:
        return None
    if any(np.mean(_distance(gutter, color) <= 13) < .88 for gutter, color in zip(gutters, colors, strict=True)):
        return None
    return colors[0]


def _distance(patch, color):
    return np.abs(patch.astype(np.int16) - color).max(axis=2)


def _seamed(tile, shoulder):
    middle = slice(shoulder, tile.shape[1] - shoulder)
    slices = (tile[:shoulder, middle], tile[shoulder:2 * shoulder, middle],
              tile[-2 * shoulder:-shoulder, middle], tile[-shoulder:, middle])
    upper, start, end, lower = [np.mean(part >= 20) for part in slices]
    return max(upper, lower) <= .23 and min(start, end) >= .69


def aligned_artwork_tiles(frame, bounds):
    """Return an anchor and complete, similarly sized picture peers in one row."""
    left, top, right, bottom = bounds
    height, width = frame.shape[:2]
    span, depth = right - left, bottom - top
    if not (.18 * width <= span <= .48 * width and .78 <= span / max(1, depth) <= 1.27):
        return ()
    shoulder = max(3, round(.045 * depth))
    if top < shoulder or bottom + shoulder >= height or (color := _background(frame, bounds)) is None:
        return ()
    distance = _distance(frame[top - shoulder:bottom + shoulder], color)
    inner = distance[2 * shoulder:-2 * shoulder]
    occupied = np.uint8((inner >= 20).mean(axis=0) >= .68)[None, :]
    # A line of dark pixels in a cover does not split its physical footprint.
    join = np.ones((1, max(3, round(.024 * span) | 1)), np.uint8)
    linked = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, join)[0]
    run_bounds = np.flatnonzero(np.diff(np.r_[0, linked.astype(np.int8), 0]))
    tolerance = max(2, round(.033 * span))
    runs = list(zip(run_bounds[::2], run_bounds[1::2], strict=True))
    fits = [index for index, (a, b) in enumerate(runs) if max(abs(a - left), abs(b - right)) <= tolerance]
    if len(fits) != 1 or not _seamed(distance[:, left:right], shoulder):
        return ()
    anchor = fits[0]
    output = [bounds]
    strong = np.percentile(inner[:, left:right], 75)
    for index in (anchor - 1, anchor + 1):
        if not 0 <= index < len(runs):
            continue
        a, b = (int(value) for value in runs[index])
        gap = left - b if index < anchor else a - right
        if (a == 0 or b == width or not .82 * span <= b - a <= 1.16 * span
                or not .025 * span <= gap <= .23 * span
                or np.percentile(inner[:, a:b], 75) < .50 * strong
                or not _seamed(distance[:, a:b], shoulder)):
            continue
        output.append((a, top, b, bottom))
    return tuple(sorted(output)) if len(output) > 1 else ()


def _caption_start(frame, bounds, color):
    left, top, right, bottom = bounds
    depth = bottom - top
    start, stop = bottom + round(.04 * depth), min(frame.shape[0], bottom + round(.31 * depth))
    if stop - start < .16 * depth:
        return None
    ink = _distance(frame[start:stop, left:right], color) >= 45
    activity = np.flatnonzero(ink.mean(axis=1) >= .12)
    if .05 <= ink.mean() <= .42 and len(activity) >= max(3, round(.035 * depth)):
        return start + int(activity[0])
    return None


def is_captioned_artwork_row(frame, bounds):
    """Require a picture peer and separate, aligned captions on the page surface."""
    tiles = aligned_artwork_tiles(frame, bounds)
    if not tiles:
        return False
    color = _background(frame, bounds)
    first = _caption_start(frame, bounds, color)
    if first is None:
        return False
    row_tolerance = max(2, round(.052 * (bounds[3] - bounds[1])))
    return any((other_first := _caption_start(frame, peer, color)) is not None
               and abs(other_first - first) <= row_tolerance for peer in tiles if peer != bounds)
