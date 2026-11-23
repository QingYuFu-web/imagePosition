"""Locate a full-height side drawer with a visible collapse handle."""

import cv2
import numpy as np


def _runs(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def _seams(gray):
    height, width = gray.shape
    left, right = round(.21 * width), round(.79 * width)
    stride = max(2, height // 200)
    strip = gray[::stride, left - 1:right].astype(np.int16)
    edges = strip[:, :-1] - strip[:, 1:]
    y1, y2, y3, y4 = (round(ratio * height / stride) for ratio in (.035, .33, .67, .965))
    top, bottom = edges[y1:y2], edges[y3:y4]
    for direction in (1, -1):
        upper, lower = direction * top, direction * bottom
        support = np.minimum((upper >= 16).mean(axis=0), (lower >= 16).mean(axis=0))
        for start, stop in _runs(support >= .76):
            if stop - start > max(4, .012 * height):
                continue
            indices = range(start, stop)
            seam = max(indices, key=lambda index: (support[index], np.median(upper[:, index]) + np.median(lower[:, index])))
            yield (left + seam, direction)


def _handle(gray, seam):
    height, width = gray.shape
    column = gray[:, seam - 1].astype(np.int16) - gray[:, seam]
    for top, bottom in _runs(column < 13):
        if (not .14 * height <= bottom - top <= .35 * height
                or not .23 * height <= top < bottom <= .77 * height
                or not .36 * height <= (top + bottom) / 2 <= .64 * height
                or min((column[:top] >= 16).mean(), (column[bottom:] >= 16).mean()) < .75):
            continue
        d = bottom - top
        first, last = seam + round(.025 * height), min(width, seam + round(.094 * height))
        y0, y1 = top + round(.25 * d), bottom - round(.25 * d)
        strip = gray[::2, first - 1:last].astype(np.int16)
        edges = strip[:, :-1] - strip[:, 1:]
        center = edges[y0 // 2:y1 // 2]
        witnessed = (center >= 15).mean(axis=0)
        if witnessed.max(initial=0) < .76:
            continue
        index = int(np.argmax(witnessed))
        edge = first + index
        extra = edges[:, index]
        if max((extra[:top // 2] >= 15).mean(), (extra[(bottom + 1) // 2:] >= 15).mean()) > .42:
            continue
        pad = max(2, round(.16 * (edge - seam)))
        inner = np.median(gray[y0:y1:2, seam - pad:seam], axis=1)
        protrusion = np.median(gray[y0:y1:2, seam:seam + pad], axis=1)
        if (abs(inner - protrusion) <= 13).mean() >= .72:
            yield (seam, int(top), edge, int(bottom))


def _dimmed_backdrop(gray, handle):
    height, width = gray.shape
    seam, _, outer, _ = handle
    y0, y1 = round(.037 * height), round(.966 * height)
    pad = round(.045 * height)
    backdrop = gray[y0:y1:4, outer + pad:width - pad:4]
    panel = gray[y0:y1:4, pad:seam - pad:4]
    if min(backdrop.size, panel.size) == 0:
        return False
    dark_floor, dark_scale, dark_peak = np.percentile(backdrop, (8, 90, 98))
    front_scale, front_peak = np.percentile(panel, (88, 98))
    return (dark_scale <= 85 and dark_peak <= 119 and dark_scale - dark_floor >= 14
            and front_scale - dark_scale >= 26 and front_peak - dark_peak >= 60)


def _left_chevron(mask):
    h, w = mask.shape
    y, x = np.nonzero(mask)
    center = np.bincount(y, weights=x, minlength=h) / np.bincount(y, minlength=h) / max(1, w - 1)
    shape = np.abs(np.linspace(-1., 1., h))
    half = h // 2
    inner = (shape[:half] >= .17) & (shape[:half] <= .76)
    asymmetry = abs(center[:half] - center[-half:][::-1])[inner]
    tips = max(2, round(.18 * h))
    return (np.max(asymmetry, initial=0) <= .20
            and .10 <= center[round(.41 * h):round(.59 * h)].mean() <= .34
            and all(.61 <= band.mean() <= .92 for band in (center[:tips], center[-tips:]))
            and np.sqrt(np.mean((center - (.74 * shape + .125)) ** 2)) <= .098)


def _icon(gray, handle):
    left, top, right, bottom = handle
    width, height = right - left, bottom - top
    pad = round(.11 * height)
    x0, y0 = left + 1, top + pad
    chip = gray[y0:bottom - pad, x0:right - 1]
    fill, peak = np.percentile(chip, (50, 99.55))
    if peak - fill < 72:
        return None
    matches = []
    for contrast in (.53, .61, .69, .77, .84):
        mask = np.uint8(chip >= fill + contrast * (peak - fill))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if (not max(13, .111 * height) <= h <= .34 * height or not .39 <= w / h <= .88
                    or not .11 <= area / (w * h) <= .47 or not .28 * width <= w <= .82 * width
                    or abs(y + h / 2 + pad - height / 2) > .17 * height
                    or not (0 < x < x + w < chip.shape[1])):
                continue
            if _left_chevron(labels[y:y + h, x:x + w] == i):
                matches.append((contrast, (x0 + x, y0 + y, x0 + x + w, y0 + y + h)))
    accepted = [(level, box) for level, box in matches if sum(
        other != level and max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= max(3, .11 * (box[3] - box[1]))
        for other, peer in matches) >= 2]
    if not accepted:
        return None
    _, coordinates = min(accepted, key=lambda row: abs(row[0] - .69))
    if any(max(abs(a - b) for a, b in zip(coordinates, box, strict=True)) > max(4, .18 * width) for _, box in accepted):
        return None
    return tuple(map(int, coordinates))


def side_drawer_bounds(frame):
    """Return the drawer panel and its inward-pointing collapse icon in original pixels."""
    height, width = frame.shape[:2]
    if height < 200 or width < 1.38 * height:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = []
    for seam, direction in _seams(gray):
        view = gray if direction == 1 else cv2.flip(gray, 1)
        split = seam if direction == 1 else width - seam
        for handle in _handle(view, split):
            if not _dimmed_backdrop(view, handle) or (icon := _icon(view, handle)) is None:
                continue
            if direction == 1:
                found.append(((0, 0, seam, height), icon))
            else:
                left, top, right, bottom = icon
                found.append(((seam, 0, width, height), (width - right, top, width - left, bottom)))
    return found[0] if len(found) == 1 else None
