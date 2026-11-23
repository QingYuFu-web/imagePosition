"""Recover a vertically repeated caption rail from independently visible pixels."""

from itertools import combinations

import cv2
import numpy as np

from screen_wide_panel_actions import _runs, _selected_row


def _compressed_edge(chroma, width):
    """Measure a diffuse, unique chroma step across almost the whole pane."""
    height = len(chroma)
    field = chroma[round(.075 * height):round(.935 * height):3]
    lo, hi = round(.061 * width), round(.185 * width)
    found = []
    for reach in (max(4, round(.0047 * width)), max(6, round(.0070 * width))):
        delta = field[:, lo:hi] - field[:, lo + reach:hi + reach]
        support = (delta >= 8).mean(axis=0)
        scores = np.median(delta, axis=0) + 10 * support
        valid = (support >= .88) & (scores >= 19)
        if not valid.any():
            return None
        peak = scores[valid].max()
        bands = _runs(valid & (scores >= .88 * peak))
        if len(bands) != 1 or bands[0][1] - bands[0][0] > reach + 3:
            return None
        start, stop = bands[0]
        found.append(lo + (int(start) + int(stop) - 1) / 2 + reach / 2)
    if abs(found[0] - found[1]) > max(2.5, .004 * width):
        return None
    return round(sum(found) / len(found))


def _caption_bands(gray, rail):
    """Keep centered, tightly bounded strokes across the visible rail."""
    left, right = round(.105 * rail), round(.895 * rail)
    field = gray[:, left:right]
    baseline = np.median(field, axis=1)
    foreground = np.maximum(67, baseline[:, None] + 27)
    mask = field >= foreground
    runs = _runs(mask.sum(axis=1) >= max(3, round(.053 * rail)))
    groups = []
    max_gap = max(1, round(.035 * rail))
    for top, bottom in runs:
        top, bottom = int(top), int(bottom)
        if groups and top - groups[-1][1] <= max_gap:
            groups[-1] = groups[-1][0], bottom
        else:
            groups.append((top, bottom))
    labels = []
    for top, bottom in groups:
        patch = np.uint8(mask[top:bottom])
        x, y, w, h = cv2.boundingRect(patch)
        if (not .072 * rail <= h <= .212 * rail or not 1.24 * h <= w <= .747 * rail
                or abs(left + x + w / 2 - rail / 2) > .081 * rail
                or not .14 <= patch[y:y + h, x:x + w].mean() <= .74):
            continue
        labels.append((top + y + h / 2, h, (left + x, top + y, left + x + w, top + y + h)))
    return labels


def _repeated_centers(labels, rail, height):
    """Require a unique chain with contiguous, equally spaced printed centers."""
    centers = np.asarray([center for center, _, _ in labels])
    chains = {}
    for first, second in combinations(range(len(centers)), 2):
        origin = centers[first]
        step = centers[second] - origin
        if not .385 * rail <= step <= .675 * rail:
            continue
        raw = (centers - origin) / step
        slots = np.rint(raw).astype(int)
        near = abs(raw - slots) <= .073
        for a, b in _runs(np.isin(np.arange(slots.min(), slots.max() + 1), slots[near])):
            found = np.flatnonzero(near & (slots >= slots.min() + a) & (slots < slots.min() + b))
            if len(found) != b - a or not 5 <= len(found) <= 10:
                continue
            group = tuple(map(int, found))
            samples = centers[found]
            pitch, start = np.polyfit(np.arange(len(found)), samples, 1)
            tolerance = max(1.6, .058 * pitch)
            if abs(samples - (start + pitch * np.arange(len(found)))).max() > tolerance:
                continue
            top = start - pitch / 2
            bottom = start + (len(found) - .5) * pitch
            preceding = round(top / pitch)
            if not 0 <= preceding <= 2 or abs(top - preceding * pitch) > .091 * pitch or bottom > height - .02 * pitch:
                continue
            chains[group] = top, pitch, preceding
    if not chains:
        return None
    largest = max(map(len, chains))
    fits = [(key, fit) for key, fit in chains.items() if len(key) == largest]
    if len(fits) != 1:
        return None
    key, (top, pitch, preceding) = fits[0]
    found_heights = [labels[i][1] for i in key]
    if max(found_heights) - min(found_heights) > .36 * np.median(found_heights):
        return None
    rows = [(max(0, round(top + i * pitch)), round(top + (i + 1) * pitch)) for i in range(-preceding, len(key))]
    return rows, preceding


def _physical_rules(gray, chroma, rail, rows):
    """Independently witness painted horizontal dividers within the cell chain."""
    gap = max(2, round(.030 * rail))
    wiggle = max(2, round(.083 * (rows[0][1] - rows[0][0])))
    left, right = round(.12 * rail), round(.90 * rail)
    count = 0
    for _, y in rows[:-1]:
        a, b = y - wiggle, y + wiggle + 1
        witnesses = []
        for plane in (gray, chroma):
            for reach in (max(2, gap - 1), gap + 1):
                before, after = plane[a - reach:b - reach, left:right], plane[a + reach:b + reach, left:right]
                middle = plane[a:b, left:right]
                crest = (middle - np.maximum(before, after) >= 2).mean(axis=1)
                step = np.maximum((after - before >= 6).mean(axis=1), (before - after >= 6).mean(axis=1))
                witnesses.append(bool(np.any((crest >= .60) | (step >= .87))))
        count += any(witnesses)
    return count >= max(3, round(.56 * (len(rows) - 1)))


def rail_grid(gray, chroma, pane_width):
    """Return a measured rail width, visible caption rows and selected index."""
    if (gray.ndim != 2 or gray.shape != chroma.shape or min(gray.shape) < 90
            or gray.shape[1] < round(.20 * pane_width)):
        return None
    rail = _compressed_edge(chroma, pane_width)
    if rail is None:
        return None
    labels = _caption_bands(gray, rail)
    grid = _repeated_centers(labels, rail, len(gray))
    if grid is None:
        return None
    rows, leading = grid
    selected = _selected_row(chroma, rail, rows)
    if selected is None or selected < leading - 1 or not _physical_rules(gray, chroma, rail, rows):
        return None
    # A header preceding the printed categories is not a selectable cell.
    start = min(selected, leading)
    return rail, rows[start:], selected - start


def rail_caption_boxes(gray, rail, rows):
    """Return one tight measured caption crop per row, where uniquely visible."""
    captions = _caption_bands(gray, rail)
    boxes = []
    for top, bottom in rows:
        pitch = bottom - top
        found = [(box, height) for center, height, box in captions
                 if abs(center - (top + bottom) / 2) <= .094 * pitch]
        if len(found) != 1:
            boxes.append(None)
            continue
        (left, a, right, b), height = found[0]
        pad = max(2, round(.22 * height))
        boxes.append((max(0, left - pad), max(top, a - pad), min(rail, right + pad), min(bottom, b + pad)))
    return boxes
