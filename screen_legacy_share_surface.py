"""Find dark bottom sheets whose share and action rows use circular tiles."""

import math

import cv2
import numpy as np

from screen_comment_sheet_modal import _phrase
from screen_component_captions import _bands
from screen_foreground import BlockingOverlay, _working_view


def _content_bottom(gray):
    """Trim a quiet black bar below the controls in a letterboxed capture."""
    height, width = gray.shape
    depth = round(.28 * width)
    center = gray[height - depth:, round(.08 * width):round(.92 * width)]
    bands = _bands((center <= 18).mean(axis=1) >= .91)
    lead = round(.017 * width)
    for lo, hi in reversed(bands):
        if (lo >= lead and hi - lo >= .088 * width and depth - hi <= .050 * width
                and (center[lo - lead:lo] >= 31).mean() >= .67):
            return height - depth + lo
    return height


def _footer_labels(gray, content_bottom):
    width = gray.shape[1]
    start = content_bottom - round(.255 * width)
    stop = content_bottom - round(.015 * width)
    left, right = round(.41 * width), round(.59 * width)
    strip = gray[start:stop, left:right]
    rim_left = gray[start:stop, round(.32 * width):round(.405 * width)]
    rim_right = gray[start:stop, round(.595 * width):round(.655 * width)]
    paper = np.median(np.hstack((rim_left, rim_right)), axis=1).astype(np.int16)
    contrast = strip.astype(np.int16) - paper[:, None]
    low_ink, high_ink = contrast >= 26, contrast >= 36
    for lo, hi in reversed(_bands((low_ink.sum(axis=1) >= max(3, round(.011 * width))) & (paper <= 120))):
        if not .020 * width <= hi - lo <= .057 * width:
            continue
        x, _, span, _ = cv2.boundingRect(np.uint8(low_ink[lo:hi]))
        hx, hy, hw, hh = cv2.boundingRect(np.uint8(high_ink[lo:hi]))
        if not (.044 * width <= span <= .112 * width and 1.45 * (hi - lo) <= span <= 3.12 * (hi - lo)
                and abs(left + x + span / 2 - width / 2) <= .023 * width
                and abs(x - hx) <= .012 * width and hh >= .7 * (hi - lo) and hw >= .8 * span):
            continue
        yield (left + x, start + lo + hy, left + x + span, start + lo + hy + hh)


def _footer_edge(smooth, label):
    height, width = smooth.shape
    gap = max(3, round(.017 * width))
    left, right = round(.04 * width), round(.96 * width)
    lo = max(gap, label[1] - round(.102 * width))
    hi = min(height - gap, label[1] - round(.024 * width))
    offset = smooth[lo + gap:hi + gap, left:right] - smooth[lo - gap:hi - gap, left:right]
    if not len(offset):
        return None
    # The footer rises into a lighter shelf underneath the second icon row.
    common = (offset >= 8).mean(axis=1)
    central = (offset[:, round(.21 * width):round(.59 * width)] >= 7).mean(axis=1)
    candidates = np.flatnonzero((common >= .76) & (central >= .74))
    if not candidates.size:
        return None
    # Use the sharpest transition in the common band. Capping limits large
    # saturated stamps in archived screenshots to their actual local support.
    y = candidates + lo
    strength = np.clip(smooth[y + 1, left:right] - smooth[y - 1, left:right], 0, 22).mean(axis=1)
    return int(y[np.argmax(strength)])


def _roofs(gray, smooth, footer):
    height, width = gray.shape
    left, right = round(.04 * width), round(.96 * width)
    rim = smooth[:, left:right]
    spread = max(3, round(.016 * width))
    offset = rim[:-2 * spread] - rim[2 * spread:]
    abrupt = rim[spread - 1:-spread - 1] - rim[spread + 1:1 - spread]
    below = rim[2 * spread:]
    edges = ((offset >= 15).mean(axis=1) >= .70) & ((abrupt >= 13).mean(axis=1) >= .47)
    edges &= ((below >= 14) & (below <= 126)).mean(axis=1) >= .86
    scores = np.clip(abrupt, 0, 70).mean(axis=1)
    for lo, hi in _bands(edges):
        top = int(lo + spread + np.argmax(scores[lo:hi]))
        if not (.29 * height <= top <= .81 * height and .46 * width <= footer - top <= 1.18 * width):
            continue
        face = gray[top + spread:top + 3 * spread, left:right].astype(np.int16)
        if (np.abs(np.diff(face, axis=0)) <= 7).mean() >= .83:
            yield top


def _has_tile_caption(gray, x, y, radius, stride):
    height, width = gray.shape
    pad = round(.37 * stride)
    left, right = max(0, x - pad), min(width, x + pad)
    top = y + radius + max(1, round(.010 * width))
    bottom = min(height, y + radius + round(.080 * width))
    caption = gray[top:bottom, left:right]
    if not caption.size:
        return False
    delta = caption.astype(np.int16) - int(np.median(caption))
    return ((delta >= 18).sum(axis=1) >= max(3, round(.014 * width))).sum() >= max(3, round(.010 * width))


def _tile_rows(gray, top, footer):
    width = gray.shape[1]
    raster = cv2.GaussianBlur(gray[top:footer - round(.068 * width)], (3, 3), .55)
    circles = cv2.HoughCircles(raster, cv2.HOUGH_GRADIENT, 1.05, minDist=.10 * width,
                               param1=80, param2=20, minRadius=round(.041 * width),
                               maxRadius=round(.074 * width))
    if circles is None or len(circles[0]) < 7:
        return ()
    candidates = sorted((int(x), int(y) + top, int(r)) for x, y, r in np.rint(circles[0]))
    groups = {}
    for _, y, radius in candidates:
        row = tuple(circle for circle in candidates if abs(circle[1] - y) <= .019 * width
                    and .81 * radius <= circle[2] <= 1.23 * radius)
        if not 3 <= len(row) <= 7 or row in groups:
            continue
        centers, rows_y, radii = np.asarray(row).T
        gaps = np.diff(centers)
        stride = float(np.min(gaps))
        multiples = gaps / stride
        if not (.143 * width <= stride <= .237 * width and abs(multiples - np.rint(multiples)).max() <= .13
                and np.ptp(rows_y) <= .022 * width and centers[0] <= .25 * width):
            continue
        if sum(_has_tile_caption(gray, *circle, stride) for circle in row) < len(row) - 1:
            continue
        groups[row] = (float(np.mean(rows_y)), stride, float(np.median(radii)))

    pairs = []
    for row_a, (y_a, stride_a, radius_a) in groups.items():
        for row_b, (y_b, stride_b, radius_b) in groups.items():
            if (len(row_a) < 4 or not .070 * width <= y_a - top <= .55 * width
                    or not .19 * width <= y_b - y_a <= .36 * width
                    or not .11 * width <= footer - y_b <= .23 * width
                    or abs(stride_b - stride_a) > .015 * width or abs(radius_b - radius_a) > .014 * width):
                continue
            if sum(any(abs(x - other_x) <= .020 * width for other_x, _, _ in row_a) for x, _, _ in row_b) >= 3:
                pairs.append(((row_a, stride_a), (row_b, stride_b)))
    return pairs[0] if len(pairs) == 1 else ()


def _tiles(rows, width):
    result = []
    for row, stride in rows:
        for x, y, radius in row:
            pad = max(radius, round(.34 * stride))
            result.append((max(0, x - pad), y - pad, min(width, x + pad), y + pad))
    return result


def legacy_share_sheet_surface(frame, index=None, *, working_view=None):
    """Return the measured dark legacy panel, offering only observed tile locations."""
    _, gray = _working_view(frame) if working_view is None else working_view
    if gray is None:
        return None
    work_h, work_w = gray.shape
    if work_w < 260 or not 1.35 * work_w <= work_h <= 2.85 * work_w:
        return None
    content_bottom = _content_bottom(gray)
    labels = tuple(_footer_labels(gray, content_bottom))
    if not labels:
        return None
    smooth = cv2.blur(gray, (round(.058 * work_w) | 1, 3)).astype(np.int16)
    height, width = frame.shape[:2]
    scale_x, scale_y = width / work_w, height / work_h

    def source_box(bounds):
        left, top, right, bottom = bounds
        return (math.floor(left * scale_x), math.floor(top * scale_y),
                min(width, math.ceil(right * scale_x)), min(height, math.ceil(bottom * scale_y)))

    for label in labels:
        footer = _footer_edge(smooth, label)
        if footer is None:
            continue
        for top in _roofs(gray, smooth, footer):
            rows = _tile_rows(gray, top, footer)
            if not rows:
                continue
            tiles = _tiles(rows, work_w)
            left, y, right, bottom = label
            px, py = round(.025 * work_w), round(.022 * work_w)
            caption = (left - px, max(footer, y - py), right + px, min(content_bottom, bottom + py))
            close = None
            if index is not None:
                x0, y0, x1, y1 = source_box(caption)
                word = 255 - cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                if _phrase(word, (0, 0, x1 - x0, y1 - y0), ("取消", "Cancel"), index):
                    close = source_box(caption)
            if close is None:
                tiles.append(caption)
            return BlockingOverlay("bottom_sheet", source_box((0, top, work_w, content_bottom)), close,
                                   close_style="literal" if close is not None else None,
                                   share_icons=tuple(map(source_box, tiles)), include_coarse_proposals=True)
    return None
