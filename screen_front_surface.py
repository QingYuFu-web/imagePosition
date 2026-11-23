"""Recognize a bright foreground panel with an independently visible close mark."""

from math import ceil, floor

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_foreground import (BlockingOverlay, _light_panel_rim, _paired_sheet_corners,
                               _sheet_header_control, _working_view)


_ROWS, _COLUMNS = np.mgrid[0:32, 0:32]
_DIAGONALS = np.minimum(np.abs(_ROWS - _COLUMNS), np.abs(_ROWS + _COLUMNS - 31)) <= 4
_CORNERS = (slice(0, 11), slice(-11, None))


def _sheet_edge_support(gray, top):
    above = np.median(gray[top - 5:top - 1], axis=0)
    below = np.median(gray[top + 1:top + 6], axis=0)
    return (below >= 225) & (above <= 183) & (below - above >= 47)


def _partial_sheet_surfaces(gray, bright):
    """Follow a partly interrupted top edge into both continuous side edges."""
    height, width = gray.shape
    rail = max(3, round(.029 * width))
    left_rail = bright[:, :rail].mean(axis=1)
    right_rail = bright[:, -rail:].mean(axis=1)
    sample = gray[:, ::max(1, width // 154)].astype(np.int16)
    edge_votes = (sample[4:] - sample[:-4] >= 47).mean(axis=1)
    for start, stop in _bands(edge_votes >= .57):
        if not .32 * height <= start <= .83 * height:
            continue
        nearby_edges = []
        for top in range(start + 1, stop + 4):
            support = _sheet_edge_support(gray, top)
            fraction = support.mean()
            if not .64 <= fraction <= .885:
                continue
            longest = max((end - begin for begin, end in _bands(support)), default=0)
            if not .54 * width <= longest <= .87 * width:
                continue
            if not _paired_sheet_corners(bright, top, round(.078 * width)):
                continue
            straight_sides = slice(top + round(.052 * width), height - 2)
            if min(left_rail[straight_sides].mean(), right_rail[straight_sides].mean()) < .94:
                continue
            backdrop = gray[round(.064 * height):top - 6:2, ::2]
            if (backdrop <= 169).mean() < .925 or (backdrop > 209).mean() > .024:
                continue
            close = _sheet_header_control(gray, top, right_only=True)
            if close is not None and close[0] >= .885 * width:
                nearby_edges.append((top, fraction, close))
        if nearby_edges:
            top, _, close = max(nearby_edges, key=lambda edge: (edge[1], -edge[0]))
            yield (0, top, width, height), close


def _painted_dialog_surfaces(gray, bright):
    """Verify the perimeter of a light dialog containing artwork or dark ink."""
    height, width = gray.shape
    joined = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        left, top, span, depth = cv2.boundingRect(contour)
        right, bottom = left + span, top + depth
        if (not .61 * width <= span <= .89 * width or not .19 * height <= depth <= .55 * height
                or not .15 * height <= top < bottom <= .87 * height or abs(left + right - width) > .048 * width
                or cv2.contourArea(contour) < .94 * span * depth):
            continue
        bounds = left, top, right, bottom
        if (not .50 <= bright[top:bottom, left:right].mean() <= .75
                or not _light_panel_rim(gray, bounds, wide=False, attached=False)):
            continue
        backdrop = np.concatenate((gray[round(.065 * height):top - 3:3, ::2].ravel(),
                                   gray[bottom + 3:round(.976 * height):3, ::2].ravel(),
                                   gray[top:bottom:3, :max(0, left - 3)].ravel(),
                                   gray[top:bottom:3, min(width, right + 3):].ravel()))
        if np.median(backdrop) < 182 and (backdrop > 192).mean() <= .035:
            yield bounds


def _four_diagonal_arms(glyph):
    """Distinguish a close mark from diagonal strokes inside nearby text."""
    scaled = cv2.resize(np.float32(glyph), (32, 32), interpolation=cv2.INTER_AREA)
    ink = scaled.sum()
    arms = [scaled[rows, columns].sum() for rows in _CORNERS for columns in _CORNERS]
    return (scaled[_DIAGONALS].sum() >= .90 * ink and min(arms) >= .11 * ink
            and min(arms) >= .53 * max(arms) and scaled[12:20, 12:20].mean() >= .60)


def _dialog_footer_close(frame, bounds):
    left, _, right, bottom = bounds
    span = right - left
    x0, x1 = right - round(.23 * span), right - round(.034 * span)
    y0, y1 = bottom - round(.105 * span), bottom - round(.025 * span)
    tile = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if (tile >= 233).mean() < .78:
        return None
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(tile < 119), connectivity=8)
    matches = []
    pad = round(.007 * span)
    for label, (x, y, width, height, area) in enumerate(stats[1:], 1):
        if (not .017 * span <= min(width, height) <= max(width, height) <= .047 * span
                or not .76 <= width / height <= 1.28 or not .16 <= area / (width * height) <= .56
                or min(x, y, tile.shape[1] - x - width, tile.shape[0] - y - height) < 4):
            continue
        if _four_diagonal_arms(labels[y:y + height, x:x + width] == label):
            matches.append((x0 + int(x) - pad, y0 + int(y) - pad,
                            x0 + int(x + width) + pad, y0 + int(y + height) + pad))
    return matches[0] if len(matches) == 1 else None


def _project(box, width, height, small_width, small_height):
    left, top, right, bottom = box
    sx, sy = width / small_width, height / small_height
    return (floor(left * sx), floor(top * sy), min(width, ceil(right * sx)), min(height, ceil(bottom * sy)))


def detect_painted_front_surface(frame, *, prior=None, working_view=None):
    """Provide a fallback for an unscoped portrait frame with one proven foreground panel."""
    height, width = frame.shape[:2]
    if prior is not None or not 1.66 * width <= height <= 2.73 * width or width < 300:
        return None
    small, gray = _working_view(frame) if working_view is None else working_view
    if small is None:
        return None
    small_height, small_width = gray.shape
    projection = width, height, small_width, small_height
    blue, green, red = cv2.split(small)
    channel_span = cv2.subtract(cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red))
    bright = np.uint8((gray >= 228) & (channel_span <= 24))
    matches = [BlockingOverlay("bottom_sheet", _project(panel, *projection), _project(close, *projection))
               for panel, close in _partial_sheet_surfaces(gray, bright)]
    for panel in _painted_dialog_surfaces(gray, bright):
        bounds = _project(panel, *projection)
        if (close := _dialog_footer_close(frame, bounds)) is not None:
            matches.append(BlockingOverlay("dialog", bounds, close))
    return matches[0] if len(matches) == 1 else None


def detect_photo_sheet_with_partial_top(frame, *, working_view=None):
    """Recognize a light sheet whose top edge is interrupted by content."""
    height, width = frame.shape[:2]
    if not 1.6 * width <= height <= 2.73 * width or width < 300:
        return None
    small, gray = _working_view(frame) if working_view is None else working_view
    if small is None:
        return None
    h, w = gray.shape
    inset = max(4, round(.03 * w))
    anchor = slice(round(.32 * w), w - 2 * inset)
    rails = (slice(inset, 2 * inset), slice(w - 2 * inset, w - inset))
    step = max(3, round(.013 * w))
    signed = gray[step:].astype(np.int16) - gray[:-step]
    votes = (signed >= 46).mean(axis=1)
    matches = []
    for begin, end in _bands((votes >= .59) & (votes <= .88)):
        if not .25 * h <= begin <= .85 * h:
            continue
        possible = []
        for top in range(begin + 1, min(end + step, h - round(.16 * w))):
            above = np.median(gray[top - step:top - 1], axis=0)
            below = np.median(gray[top + 1:top + step + 1], axis=0)
            edge = (above <= 180) & (below >= 211) & (below - above >= 47)
            if not .61 <= edge.mean() <= .89 or edge[anchor].mean() < .90:
                continue
            # Ignore the header artwork and the full-width colored footer action.
            body = gray[top + round(.18 * w):h - round(.11 * w)]
            if body.shape[0] < .19 * w or any((body[:, rail] >= 215).mean() < .84 for rail in rails):
                continue
            backdrop = gray[round(.066 * h):top - step:2, inset:w - inset:2]
            if np.median(backdrop) > 124 or (backdrop > 203).mean() > .045:
                continue
            upper = gray[round(.066 * h):top - step]
            if any((upper[:, rail] <= 174).mean() < .94 for rail in rails):
                continue
            close = _sheet_header_control(gray, top, right_only=True)
            if close is None or (close[0] + close[2]) / 2 < .88 * w:
                continue
            possible.append((edge.mean(), -top, close))
        if possible:
            _, negative_top, (left, y, right, bottom) = max(possible)
            top = -negative_top
            pad = max(2, round(.008 * w))
            panel = _project((0, top, w, h), width, height, w, h)
            glyph = _project((left - pad, max(top, y - pad), min(w, right + pad), bottom + pad), width, height, w, h)
            matches.append(BlockingOverlay("bottom_sheet", panel, glyph))
    return matches[0] if len(matches) == 1 else None
