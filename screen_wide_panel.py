"""Locate a wide tinted overlay with a visibly attached close button."""

from itertools import product

import cv2
import numpy as np


def _runs(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def _horizontal_rims(frame, gap):
    height, width = frame.shape[:2]
    strip = frame[:, round(.046 * width):round(.84 * width):4].astype(np.int16)
    blue = strip[:, :, 0] - strip[:, :, 2]
    sides = (blue[gap:-gap] - np.maximum(blue[:-2 * gap], blue[2 * gap:])) >= 13
    rows = np.flatnonzero(sides.mean(axis=1) >= .77) + gap
    if not rows.size:
        return
    supported = strip[rows]
    tones = np.median(supported, axis=1)
    even = (abs(supported - tones[:, None]).max(axis=2) <= 9).mean(axis=1) >= .82
    strong = np.zeros(height, bool)
    strong[rows] = even & (tones[:, 0] - tones[:, 2] >= 32)
    rims = [(int(a), int(b), np.median(tones[(rows >= a) & (rows < b)], axis=0))
            for a, b in _runs(strong) if b - a <= .014 * height]
    top = [row for row in rims if .016 * height <= row[0] <= .15 * height]
    bottom = [row for row in rims if .85 * height <= row[1] <= .985 * height]
    for first, last in product(top, bottom):
        if (.75 * height <= last[1] - first[0] <= .962 * height
                and abs(first[0] - (height - last[1])) <= .063 * height
                and max(abs(first[2] - last[2])) <= 16):
            yield first[0], last[1]


def _side_rim(frame, top, bottom, gap):
    width = frame.shape[1]
    reach = round(.038 * width)
    pad = max(4, round(.065 * (bottom - top)))
    patch = frame[top + pad:bottom - pad:3, :reach + gap].astype(np.int16)
    colors = np.median(patch, axis=0)
    hue = patch[:, :, 0] - patch[:, :, 2]
    contrast = hue[:, :-gap] - hue[:, gap:]
    even = (abs(patch - colors[None]).max(axis=2) <= 12).mean(axis=0) >= .72
    found = np.flatnonzero(even[:-gap] & ((contrast >= 12).mean(axis=0) >= .74)
                          & (colors[:-gap, 0] - colors[:-gap, 2] >= 30))
    return int(found[0]) if found.size else None


def _tinted(frame, gray, top, bottom, gap):
    width = frame.shape[1]
    columns = slice(round(.082 * width), round(.82 * width), 3)
    for rim, direction in ((top, 1), (bottom - 1, -1)):
        before, after = rim - direction * gap, rim + direction * gap
        exterior, interior = gray[before, columns], gray[after, columns]
        colors = frame[[after, before], columns].astype(np.int16)
        hue_delta = np.diff(colors[:, :, 2] - colors[:, :, 0], axis=0)[0]
        if ((hue_delta >= 12).mean() < .74 or exterior.std() < 8 or interior.std() < .9
                or interior.std() >= .76 * exterior.std() or np.corrcoef(exterior, interior)[0, 1] < .32
                or np.percentile(exterior, 94) > 105):
            return False
        if direction == -1 and (exterior - interior >= 4).mean() < .66:
            return False
    return True


def _diagonal_cross(mask):
    height, width = mask.shape
    y, x = np.ogrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    diagonal = abs(abs(x) - abs(y)) <= .49
    outer_axes = ((abs(x) >= .59) & (abs(y) <= .23)) | ((abs(y) >= .59) & (abs(x) <= .23))
    if not mask[height // 2, width // 2] or mask[outer_axes].mean() > .09 or mask[diagonal].sum() < .95 * mask.sum():
        return False
    copies = (mask[::-1, :], mask[:, ::-1], mask[::-1, ::-1])
    return all((mask & mirror).sum() >= .73 * (mask | mirror).sum() for mirror in copies)


def _quiet_tab(frame, glyph, right):
    left, top, box_right, bottom = glyph
    size = bottom - top
    guard, lane, trim = (max(2, round(factor * size)) for factor in (.23, .91, .15))
    y0, y1 = top + trim, bottom - trim
    other = min(right, box_right + lane)
    if other <= box_right + guard or left < lane or bottom + lane >= frame.shape[0]:
        return False
    halves = tuple(frame[y0:y1, a:b].astype(np.int16) for a, b in
                   ((left - lane, left - guard), (box_right + guard, other)))
    fills = [np.median(patch, axis=(0, 1)) for patch in halves]
    if (max(abs(fills[0] - fills[1])) > 9 or min(color[0] - color[2] for color in fills) < 28
            or any((abs(patch - color).max(axis=2) <= 12).mean() < .83 for patch, color in zip(halves, fills))):
        return False
    below = frame[bottom + round(.51 * size):bottom + lane, left - lane:other].astype(np.int16)
    return (abs(below - fills[0]).max(axis=2) >= 19).mean() >= .74


def _close(frame, gray, top, right):
    height, width = gray.shape
    x0, y0 = round(.84 * width), top + max(1, round(.005 * height))
    clip = gray[y0:top + round(.12 * height), x0:right]
    matches = []
    for threshold in (135, 155, 175, 195, 210):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(clip >= threshold), connectivity=8)
        for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if (not .025 * height <= min(w, h) <= max(w, h) <= .073 * height or not .83 <= w / h <= 1.20
                    or not .27 <= area / (w * h) <= .63
                    or not .017 * height <= y0 + y + h / 2 - top <= .08 * height):
                continue
            if not _diagonal_cross(labels[y:y + h, x:x + w] == i):
                continue
            box = tuple(map(int, (x0 + x, y0 + y, x0 + x + w, y0 + y + h)))
            if _quiet_tab(frame, box, right):
                matches.append((threshold, box))
    stable = [(threshold, box) for threshold, box in matches if sum(
        other != threshold and max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= max(2, .10 * (box[3] - box[1]))
        for other, peer in matches) >= 2]
    if not stable:
        return None
    _, witness = min(stable, key=lambda item: abs(item[0] - 175))
    if any(max(abs(a - b) for a, b in zip(witness, box, strict=True)) > max(3, .15 * (witness[3] - witness[1]))
           for _, box in stable):
        return None
    return witness


def _soft_seams(strip, start, direction, gap, height):
    # A constant plane still has a visible boundary after chroma blur.
    # The exterior has to remain structured on the other side.
    shade = np.median(strip, axis=1)
    even = (abs(strip - shade[:, None]).max(axis=2) <= 13).mean(axis=1)
    high = even[1:] if direction > 0 else even[:-1]
    low = even[:-1] if direction > 0 else even[1:]
    for i in np.flatnonzero((high >= .82) & (high - low >= .17)) + 1:
        a, b = max(0, i - gap), min(len(strip) - 1, i + gap)
        if direction > 0:
            exterior, interior = strip[a], strip[b]
        else:
            exterior, interior = strip[b], strip[a]
        lift = (interior[:, 0] - interior[:, 2]) - (exterior[:, 0] - exterior[:, 2])
        y = start + int(i)
        if (lift >= 12).mean() >= .73 and .016 * height <= y <= .985 * height:
            yield y


def _soft_side(frame, top, bottom, gap):
    span = bottom - top
    pad = max(gap + 1, round(.072 * span))
    band = frame[top + pad:bottom - pad:4, :gap + 5].astype(np.int16)
    if not band.size:
        return False
    edge = band[:, 0]
    nearer = np.median(band[:, gap + 1:gap + 5], axis=1)
    median_color = np.median(edge, axis=0)
    delta = edge - nearer
    brightness_delta = delta @ (.114, .587, .299)
    return (median_color[0] - median_color[2] >= 28 and (abs(edge - median_color).max(axis=1) <= 14).mean() >= .74
            and (brightness_delta >= 12).mean() >= .77 and (delta[:, 0] >= 16).mean() >= .75)


def _soft_wide_panel_bounds(frame, gap):
    """Verify a full-width panel with diffuse borders and its visible close."""
    height, width = frame.shape[:2]
    first, last = round(.153 * height), round(.847 * height)
    if not _soft_side(frame, first, last, gap) or not _soft_side(frame[:, ::-1], first, last, gap):
        return None
    horizon = round(.153 * height) + gap
    lower = height - horizon
    columns = slice(round(.08 * width), round(.82 * width), 5)
    head = frame[:horizon, columns].astype(np.int16)
    tail = frame[lower:, columns].astype(np.int16)
    tops = list(_soft_seams(head, 0, 1, gap, height))
    ends = list(_soft_seams(tail, lower, -1, gap, height))
    if not tops or not ends:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    results = []
    for top, bottom in product(tops, ends):
        if (not .75 * height <= bottom - top <= .962 * height
                or abs(top - (height - bottom)) > .063 * height
                or not _soft_side(frame, top, bottom, gap)
                or not _soft_side(frame[:, ::-1], top, bottom, gap)
                or not _tinted(frame, gray, top, bottom, gap)):
            continue
        if (close := _close(frame, gray, top, width)) is not None:
            results.append(((0, top, width, bottom), close))
    return results[0] if len(results) == 1 else None


def wide_panel_bounds(frame):
    """Return a wide panel and its close glyph in original half-open pixels."""
    height, width = frame.shape[:2]
    if height < 200 or width < 1.42 * height:
        return None
    gap = max(3, round(.006 * height))
    rims = list(_horizontal_rims(frame, gap))
    if not rims:
        return _soft_wide_panel_bounds(frame, gap)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    results = []
    for top, bottom in rims:
        if not _tinted(frame, gray, top, bottom, gap):
            continue
        left = _side_rim(frame, top, bottom, gap)
        inset = _side_rim(frame[:, ::-1], top, bottom, gap)
        if left is None or inset is None or left + inset > .063 * width:
            continue
        right = width - inset
        if (close := _close(frame, gray, top, right)) is not None:
            results.append(((left, top, right, bottom), close))
    if results:
        return results[0] if len(results) == 1 else None
    return _soft_wide_panel_bounds(frame, gap)
