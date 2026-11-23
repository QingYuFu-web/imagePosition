"""Read captions from verified file tabs with a repeated leading icon lane."""

from statistics import median

import cv2
import numpy as np

from inventory_list_rows import _Box


def _ink_bounds(ink, left, right):
    rows = np.flatnonzero(ink[:, left:right].any(axis=1))
    return int(rows[0]), int(rows[-1] + 1)


def _title_ink(frame, cell, right):
    height = cell.height
    x0 = cell.left + max(2, round(.10 * height))
    x1 = right - max(2, round(.10 * height))
    y0 = cell.top + max(2, round(.22 * height))
    y1 = cell.bottom - max(2, round(.22 * height))
    if x1 - x0 < 1.3 * height or y1 <= y0:
        return None
    chip = frame[y0:y1, x0:x1].astype(np.int16)
    ink = np.abs(chip - np.median(chip, axis=(0, 1))).max(axis=2) >= 36
    columns = np.flatnonzero(ink.sum(axis=0) >= 2)
    breaks = np.flatnonzero(np.diff(columns) > max(4, round(.11 * height)))
    if not breaks.size:
        return None
    icon_left, icon_right = int(columns[0]), int(columns[breaks[0]] + 1)
    title_left, title_right = int(columns[breaks[0] + 1]), int(columns[-1] + 1)
    icon_top, icon_bottom = _ink_bounds(ink, icon_left, icon_right)
    title_top, title_bottom = _ink_bounds(ink, title_left, title_right)
    icon_offset = (x0 + icon_left - cell.left) / height
    if (not .17 <= icon_offset <= .66
            or not .13 * height <= icon_right - icon_left <= .58 * height
            or not .17 * height <= icon_bottom - icon_top <= .57 * height
            or not .11 * height <= title_left - icon_right <= .42 * height
            or title_right - title_left < .82 * height
            or not .19 * height <= title_bottom - title_top <= .47 * height):
        return None
    title = _Box(x0 + title_left, y0 + title_top, x0 + title_right, y0 + title_bottom)
    return icon_offset, title


def _title_views(frame, title, cell, right):
    pad = max(2, round(.043 * cell.height))
    left = max(cell.left, title.left - pad)
    end = min(right, title.right + pad)
    top = max(cell.top, title.top - pad)
    bottom = min(cell.bottom, title.bottom + pad)
    color = frame[top:bottom, left:end]
    luminance = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.int16)
    ground = np.median(luminance)
    delta = luminance - ground if ground <= 127 else ground - luminance
    contrast = np.clip(1.75 * delta, 0, 255).astype(np.uint8)
    return color, cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)


def read_tab_title_boxes(frame, cells, titles, line_reader, *, right_edges=None):
    """Read physically isolated titles from color and high-contrast crops."""
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    results = [None] * len(cells)
    pending = [(index, title) for index, title in enumerate(titles) if title is not None]
    if recognize is None or not pending:
        return results
    limits = [cell.right for cell in cells] if right_edges is None else right_edges
    crops = [crop for index, title in pending for crop in _title_views(frame, title, cells[index], limits[index])]
    try:
        readings = recognize(crops)
    except (RuntimeError, ValueError):
        return results
    if len(readings) != len(crops):
        return results
    for (index, title), first, second in zip(pending, readings[::2], readings[1::2], strict=True):
        name = first["text"]
        if (name == second["text"] and min(first["score"], second["score"]) >= 90
                and 2 <= len(name) <= 140 and sum(character.isalnum() for character in name) >= 2):
            results[index] = name, title
    return results


def read_file_tab_titles(frame, cells, close_mark, line_reader):
    """Return corroborated (caption, ink box) per cell, or None for unreadable cells."""
    results = [None] * len(cells)
    if getattr(line_reader, "recognize_prepared_lines", None) is None or close_mark is None or len(cells) < 3:
        return results
    marks = [index for index, cell in enumerate(cells)
             if cell.left < close_mark.left < close_mark.right <= cell.right
             and cell.top < close_mark.top < close_mark.bottom < cell.bottom]
    if len(marks) != 1:
        return results
    close_cell = marks[0]
    limits = [close_mark.left if index == close_cell else cell.right for index, cell in enumerate(cells)]
    candidates = [_title_ink(frame, cell, right) for cell, right in zip(cells, limits, strict=True)]
    offsets = [candidate[0] for candidate in candidates if candidate is not None]
    if len(offsets) < 3:
        return results
    common = median(offsets)
    titles = [candidate[1] if candidate is not None and abs(candidate[0] - common) <= .10 else None
              for candidate in candidates]
    if sum(title is not None for title in titles) < 3:
        return results
    return read_tab_title_boxes(frame, cells, titles, line_reader, right_edges=limits)
