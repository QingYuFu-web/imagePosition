"""Measure a local media tray from its divider, tile seams and footer."""

from itertools import combinations, pairwise
from math import ceil, floor

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_dark_barrage_input import _round_fields


def _dividers(interior, width):
    height = interior.shape[0]
    white = (interior >= 237).mean(axis=1)
    spacing = max(3, round(.007 * width))
    shelf = round(.115 * height)
    for top, end in _bands(white >= .84):
        if not round(.62 * height) <= top <= round(.83 * height) or end - top < spacing:
            continue
        previous = interior[top - 3 * spacing:top - spacing]
        backdrop = float(np.median(previous))
        if not 132 <= backdrop <= 204 or np.median(interior[top]) - backdrop < 50:
            continue
        plate = interior[top - shelf:top - spacing:3]
        if (np.abs(plate.astype(np.int16) - backdrop) <= 6).mean() >= .88:
            yield top, white


def _column_seams(strip, width, row_pitch):
    bands = _bands((strip >= 237).mean(axis=0) >= .84)
    narrow = [(l, r) for l, r in bands if .12 * width <= l < r <= .91 * width and r - l <= .018 * width]
    if not 2 <= len(narrow) <= 12:
        return None
    centers = np.array([(left + right) / 2 for left, right in narrow])
    thickness = round(np.median([right - left for left, right in narrow]))
    layouts = []
    for first, last in combinations(centers, 2):
        for spaces in range(1, 5):
            pitch = (last - first) / spaces
            if not .15 * width <= pitch <= .25 * width or not .87 * pitch <= row_pitch <= 1.23 * pitch:
                continue
            offsets = range(ceil((.12 * width - first) / pitch), floor((.91 * width - first) / pitch) + 1)
            gutters = []
            supported = error = 0
            for step in offsets:
                center = first + pitch * step
                nearest = int(np.argmin(abs(centers - center)))
                delta = abs(centers[nearest] - center)
                if delta <= .054 * pitch:
                    gutters.append(narrow[nearest])
                    supported += 1
                    error += delta
                elif any(l + 1 <= center <= r - 1 and r - l > .018 * width for l, r in bands):
                    # Blank neighboring cells can join two gutters into one
                    # white band. Only restore a seam inside that visible band.
                    left = round(center - thickness / 2)
                    gutters.append((left, left + thickness))
            if 3 <= len(gutters) <= 6 and supported >= 2:
                strides = [b[0] + b[1] - a[0] - a[1] for a, b in pairwise(gutters)]
                if all(abs(gap / 2 - pitch) <= .086 * pitch for gap in strides):
                    layouts.append(((-supported, -len(gutters), error, abs(row_pitch - pitch)),
                                    pitch, gutters))
    if not layouts:
        return None
    _, pitch, gutters = min(layouts, key=lambda layout: layout[0])
    return pitch, gutters


def _tile_grid(gray, top, white):
    height, width = gray.shape
    strip = gray[top + round(.04 * width):top + round(.38 * width)]
    if strip.shape[0] < .28 * width:
        return None
    seams = _bands(white[top:min(height, top + round(.58 * width))] >= .86)
    middle = [(a, b) for a, b in seams if .13 * width <= a <= .308 * width and b - a <= .033 * width]
    if len(middle) != 1 or (columns := _column_seams(strip, width, middle[0][0])) is None:
        return None
    pitch, gutters = columns
    first_top, first_bottom = middle[0]
    header = [top + b for a, b in seams if a <= .025 * pitch and 1 <= b - a <= .13 * pitch]
    ends = [top + a for a, b in seams if 1.81 * pitch <= a <= 2.27 * pitch and b - a >= .075 * pitch]
    if len(header) != 1 or len(ends) != 1:
        return None

    # Use the continuous line above the search footer to locate the raster
    # edges. A camera image can still have a narrow strip of glass here.
    footer = gray[ends[0] + 1:ends[0] + round(.025 * width)]
    rails = [(left, right) for left, right in _bands((footer >= 237).mean(axis=0) >= .84)
             if left <= .06 * width and right >= .94 * width]
    if len(rails) != 1:
        return None
    rail_left, rail_right = rails[0]
    left = max(rail_left, round(gutters[0][1] - pitch))
    right = min(rail_right, round(gutters[-1][0] + pitch))
    cells = [(left, gutters[0][0]),
             *((left_end, right_start) for (_, left_end), (right_start, _) in pairwise(gutters)),
             (gutters[-1][1], right)]
    trim = round(.065 * pitch)
    tiles = []
    for a, b in ((header[0], top + first_top), (top + first_bottom, ends[0])):
        occupied = []
        for l, r in cells:
            if r - l < .36 * pitch:
                continue
            chip = gray[a + trim:b - trim:3, l + trim:r - trim:3]
            if chip.size and (chip < 231).mean() >= .12 and chip.std() >= 19:
                occupied.append((l, a, r, b))
        # A grid remains identifiable with empty cells, provided the two
        # horizontal rows each contain independently observed artwork.
        if len(occupied) < 2:
            return None
        tiles.extend(occupied)
    return ends[0], tuple(tiles)


def _footer_end(gray, footer_top):
    height, width = gray.shape
    footer = gray[footer_top:]
    fields = [(left, top + footer_top, right, bottom + footer_top)
              for left, top, right, bottom in _round_fields((footer >= 225) & (footer <= 248), width, 8.5, 15)
              if .01 * width <= left <= .05 * width and .72 * width <= right <= .89 * width]
    if len(fields) != 1:
        return None
    _, top, _, bottom = fields[0]
    if not .84 * height <= (top + bottom) / 2 <= .985 * height:
        return None
    tail = gray[bottom:, round(.07 * width):round(.93 * width)]
    dark = np.flatnonzero((tail >= 237).mean(axis=1) < .85)
    end = bottom + int(dark[0]) if dark.size else height
    if .009 * width <= end - bottom <= .09 * width:
        return end
    return None


def media_picker_layout(frame):
    """Return the tray and row-major visible thumbnail bounds, or None."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        return None
    height, width = frame.shape[:2]
    if width < 260 or not 1.55 * width <= height <= 2.75 * width:
        return None
    display_width = min(640, width)
    display_height = round(height * display_width / width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if width > display_width:
        gray = cv2.resize(gray, (display_width, display_height), interpolation=cv2.INTER_AREA)
    interior = gray[:, round(.07 * display_width):round(.93 * display_width)]
    found = []
    for top, white in _dividers(interior, display_width):
        if (grid := _tile_grid(gray, top, white)) is None:
            continue
        footer_top, tiles = grid
        if (bottom := _footer_end(gray, footer_top)) is not None:
            sx, sy = width / display_width, height / display_height
            tray = (0, ceil(top * sy), width, floor(bottom * sy))
            tiles = tuple((ceil(left * sx), ceil(top * sy), floor(right * sx), floor(bottom * sy))
                          for left, top, right, bottom in tiles)
            found.append((tray, tiles))
    return found[0] if len(found) == 1 else None


def media_picker_surface(frame):
    """Return half-open bounds of a visibly evidenced local tray, or None."""
    layout = media_picker_layout(frame)
    return layout[0] if layout is not None else None
