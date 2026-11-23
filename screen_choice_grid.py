"""Find closed, repeatedly spaced option cells and isolate their captions."""

from statistics import median

import cv2
import numpy as np

from screen_component_captions import CaptionControl, _bands, _contrast, _rectangular_button, _tight


def _inside(box, bounds):
    left, top, right, bottom = box
    x0, y0, x1, y1 = bounds
    return x0 <= left < right <= x1 and y0 <= top < bottom <= y1


def _x(box):
    return (box[0] + box[2]) / 2


def _y(box):
    return (box[1] + box[3]) / 2


def _size(box):
    return box[2] - box[0], box[3] - box[1]


def _matching_columns(row, columns, width):
    sites = tuple(min(range(len(columns)), key=lambda i: abs(_x(box) - columns[i])) for _, box in row)
    return (len(sites) == len(set(sites)) and
            all(abs(_x(box) - columns[i]) <= .075 * width for i, (_, box) in zip(sites, row, strict=True)))


def _lattices(cells):
    seen = set()
    for _, anchor in cells:
        w, h = _size(anchor)
        sizes = [cell for cell in cells if .90 * w <= _size(cell[1])[0] <= 1.10 * w and
                 .86 * h <= _size(cell[1])[1] <= 1.14 * h]
        rows = []
        for item in sorted(sizes, key=lambda cell: _y(cell[1])):
            if rows and _y(item[1]) - _y(rows[-1][0][1]) <= .13 * h:
                rows[-1].append(item)
            else:
                rows.append([item])
        rows = [sorted(row, key=lambda cell: _x(cell[1])) for row in rows]
        complete = max(rows, key=len)
        if not 2 <= len(complete) <= 6:
            continue
        columns = [_x(box) for _, box in complete]
        gaps = np.diff(columns)
        if min(gaps) < 1.025 * w or max(gaps) > 1.36 * w or np.ptp(gaps) > .12 * w:
            continue
        if any(a[2] >= b[0] for (_, a), (_, b) in zip(complete[:-1], complete[1:], strict=True)):
            continue
        compatible = [row for row in rows if _matching_columns(row, columns, w)]
        runs = []
        for row in compatible:
            if runs and 1.06 * h <= _y(row[0][1]) - _y(runs[-1][-1][0][1]) <= 1.80 * h:
                runs[-1].append(row)
            else:
                runs.append([row])
        for run in runs:
            if sum(len(row) == len(columns) for row in run) < 2:
                continue
            step = np.diff([_y(row[0][1]) for row in run])
            if np.ptp(step) > .20 * h:
                continue
            group = [cell for row in run for cell in row]
            key = tuple(sorted(item for item, _ in group))
            if key not in seen:
                seen.add(key)
                yield group


def _caption_and_style(frame, box):
    left, top, right, bottom = box
    w, h = _size(box)
    x0, x1 = left + round(.13 * w), right - round(.13 * w)
    y0, y1 = top + round(.12 * h), bottom - round(.12 * h)
    distances, paint = _contrast(frame[y0:y1, x0:x1])
    if (distances <= 13).mean() < .69:
        return None
    active = distances >= 36
    rows = _bands(active.sum(axis=1) >= max(2, round(.009 * w)))
    rows = [(start, stop) for start, stop in rows if stop - start >= max(4, round(.045 * h))]
    if not 1 <= len(rows) <= 2:
        return None
    bands = [_tight(active[start:stop], x0, y0 + start) for start, stop in rows]
    primary = bands[0]
    tw, th = _size(primary)
    if (not .12 * h <= th <= .45 * h or not .10 * h <= tw <= .74 * w or
            abs(_x(primary) - _x(box)) > .10 * w):
        return None
    if len(bands) == 1 and abs(_y(primary) - _y(box)) > .16 * h:
        return None
    if len(bands) == 2 and (not .11 * h <= _y(bands[1]) - _y(primary) <= .45 * h or
                            abs(_x(bands[1]) - _x(box)) > .10 * w or
                            not _y(primary) < _y(box) < _y(bands[1])):
        return None
    # Include the faint outer pixels without admitting the secondary price or subtitle.
    lx, ty, rx, by = primary
    pad = max(2, round(.075 * th))
    caption = (max(x0, lx - pad), max(y0, ty - pad), min(x1, rx + pad), min(y1, by + pad))
    ink = distances[ty - y0:by - y0, lx - x0:rx - x0]
    pixels = frame[ty:by, lx:rx]
    peak = float(np.percentile(ink, 96))
    text_ink = np.median(pixels[ink >= peak * .82], axis=0)
    core = frame[max(0, top - 2):top + max(4, round(.047 * h)), x0:x1].astype(np.int16)
    rim = float(np.max(np.median(np.max(np.abs(core - paint), axis=2), axis=1)))
    return caption, len(bands), paint, peak, rim, text_ink


def _is_keypad(group, candidates, frame_height, frame_width):
    """Check a bottom key field and its separate spanning action keys."""
    boxes = [box for _, box in group]
    cell_width = median(_size(box)[0] for box in boxes)
    cell_height = median(_size(box)[1] for box in boxes)
    left, top = min(box[0] for box in boxes), min(box[1] for box in boxes)
    right, bottom = max(box[2] for box in boxes), max(box[3] for box in boxes)
    if (not .60 * frame_height <= top or bottom < .875 * frame_height or right - left < .88 * frame_width
            or (right - left) / cell_width < 3.85 or (bottom - top) / cell_height < 2.85):
        return False
    peers = [row["box"] for row in candidates if top - .18 * cell_height <= row["box"][1] <= bottom
             and row["box"][3] <= bottom + .30 * cell_height]
    wide = any(1.72 * cell_width <= w <= 3.35 * cell_width and .75 * cell_height <= h <= 1.28 * cell_height
               for w, h in map(_size, peers))
    tall = any(.72 * cell_width <= w <= 1.28 * cell_width and 1.70 * cell_height <= h <= 2.75 * cell_height
               for w, h in map(_size, peers))
    return wide and tall


def grid_choice_controls(frame, regions):
    """Return one control per recognizable option surface in the active layer."""
    height, width = frame.shape[:2]
    layer = regions["layer"]
    if layer not in {"page", "bottom_sheet", "dialog"}:
        return []
    if layer != "page" and not regions["regions"]:
        return []
    bounds = (0, 0, width, height) if layer == "page" else regions["regions"][0]
    scope_width = bounds[2] - bounds[0]
    available = []
    for item in regions["candidates"]:
        box = item["box"]
        w, h = _size(box)
        if (_inside(box, bounds) and item["visibility"] in {"page", "foreground"} and
                .105 * scope_width <= w <= .46 * scope_width and .030 * scope_width <= h <= .17 * scope_width and
                1.04 <= w / h <= 5.0):
            available.append(item)
    if len(available) < 4:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cells = []
    for item in available:
        box = _rectangular_button(gray, item["box"])
        if box is not None and _inside(box, bounds):
            cells.append((item["id"], box))
    output = []
    used = set()
    for group in sorted(_lattices(cells), key=len, reverse=True):
        if (any(item in used for item, _ in group) or
                (layer == "page" and _is_keypad(group, regions["candidates"], height, width))):
            continue
        members = [(item, box, _caption_and_style(frame, box)) for item, box in group]
        supported = [details for _, _, details in members if details is not None]
        if len(supported) < max(4, .75 * len(group)):
            continue
        band_count = round(median(details[1] for details in supported))
        consistent = [details for details in supported if details[1] == band_count]
        if len(consistent) < .75 * len(group):
            continue
        ref_paint = np.median([details[2] for details in consistent], axis=0)
        ref_peak = median(details[3] for details in consistent)
        ref_rim = median(details[4] for details in consistent)
        ref_ink = np.median([details[5] for details in consistent], axis=0)
        for item, box, details in members:
            if details is None or details[1] != band_count:
                continue
            caption, _, paint, peak, rim, text_ink = details
            disabled = peak < max(64, .67 * ref_peak)
            selected = (np.max(abs(paint - ref_paint)) > 20 or rim > 1.50 * ref_rim + 24 or
                        np.max(abs(text_ink - ref_ink)) > 53)
            output.append(CaptionControl(item, box, caption, "select", bool(selected), bool(disabled)))
            used.add(item)
    return output
