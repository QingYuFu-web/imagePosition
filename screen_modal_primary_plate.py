"""Read the action on a filled, outlined plate owned by an existing modal."""

import re

import cv2
import numpy as np

from screen_candidate_glyphs import _overlap
from screen_caption_transcript import _bands, read_caption


def _filled_shapes(frame, panel):
    left, top, right, bottom = panel
    width, height = right - left, bottom - top
    x, y = left + round(.095 * width), top + round(.735 * height)
    end_x, end_y = right - round(.095 * width), bottom - round(.025 * height)
    clip = frame[y:end_y, x:end_x]
    saturation = np.ptp(clip, axis=2)
    floor, local_high, peak = np.percentile(saturation, (50, 75, 97.5))
    if peak - floor < 57:
        return []
    reports = []
    for ratio in (.23, .40, .58):
        edge = max(local_high + 10, floor + ratio * (peak - floor))
        mask = np.uint8(saturation >= edge)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for k, (a, b, w, h, area) in enumerate(stats[1:], 1):
            if (not .154 * width <= w <= .505 * width or not .050 * height <= h <= .155 * height
                    or not 1.72 <= w / h <= 4.38 or not .615 <= area / (w * h) <= .925
                    or abs(x + a + w / 2 - (left + right) / 2) > .112 * width
                    or not .031 * height <= bottom - (y + b + h) <= .135 * height):
                continue
            shape = labels[b:b + h, a:a + w] == k
            if _rounded_fill(shape):
                box = (x + int(a), y + int(b), x + int(a + w), y + int(b + h))
                reports.append((ratio, box))
    stable = [box for ratio, box in reports if sum(other != ratio and _overlap(box, peer) >= .79
                                                  for other, peer in reports) >= 2]
    if stable and all(_overlap(stable[0], box) >= .78 for box in stable):
        return [stable[len(stable) // 2]]
    return []


def _rounded_fill(mask):
    height, width = mask.shape
    reach = max(3, round(.15 * height))
    top = mask[:reach].mean(axis=0)
    bottom = mask[-reach:].mean(axis=0)
    for profile in (top, bottom):
        extrema = np.flatnonzero(profile >= .49)
        if (not extrema.size or not .058 * height <= min(extrema[0], width - extrema[-1] - 1)
                or max(extrema[0], width - extrema[-1] - 1) > .41 * height):
            return False
        if profile[round(.24 * width):round(.76 * width)].mean() < .67:
            return False
    return True


def _dark_rim(plane, anchor, first, last, direction, size):
    extent = max(5, round(.18 * size))
    inside = max(2, round(.064 * size))
    axis = anchor + direction * np.arange(-extent, inside + 1)
    if min(axis) < 0 or max(axis) >= plane.shape[0]:
        return None
    strips = plane[axis, first:last].astype(np.int16)
    unpainted = np.median(strips[:max(2, round(.24 * extent))], axis=0)
    output = (unpainted - strips >= 29).mean(axis=1)
    painted = np.median(strips[-2:], axis=0)
    for lo, hi in _bands(output >= .65):
        if (lo <= .28 * extent or hi >= len(output) or not .020 * size <= hi - lo <= .142 * size
                or (painted - np.median(strips[lo:hi], axis=0) >= 28).mean() < .70):
            continue
        return int(axis[lo])
    return None


def _plate_outline(gray, fill):
    left, top, right, bottom = fill
    w, h = right - left, bottom - top
    a, b = left + round(.22 * w), right - round(.21 * w)
    u, v = top + round(.23 * h), bottom - round(.19 * h)
    l = _dark_rim(gray.T, left, u, v, 1, h)
    r = _dark_rim(gray.T, right - 1, u, v, -1, h)
    t = _dark_rim(gray, top, a, b, 1, h)
    d = _dark_rim(gray, bottom - 1, a, b, -1, h)
    if None in (l, t, r, d):
        return None
    return l, t, r + 1, d + 1


def _ink_bounds(mask, offset):
    x, y, w, h = cv2.boundingRect(np.uint8(mask))
    return (offset[0] + x, offset[1] + y, offset[0] + x + w, offset[1] + y + h) if w and h else None


def _label_plausible(text):
    return (2 <= len(text) <= 18 and sum(char.isalpha() for char in text) >= 2
            and not re.search(r"\d+[.:]\d+", text) and sum(ch.isdigit() for ch in text) <= 2)


def _plate_caption(frame, gray, box, index):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    a, b = left + round(.11 * w), right - round(.095 * w)
    y, end = top + round(.135 * h), bottom - round(.136 * h)
    tile = gray[y:end, a:b]
    background = float(np.percentile(tile, 64))
    cache = {}
    readings = {}
    for level in (49, 62, 73):
        ink = np.uint8(tile <= background - level)
        extents = _ink_bounds(ink, (a, y))
        if not extents:
            continue
        # A separate compact icon may share the plate. Only the caption names it.
        columns = _bands(ink.any(axis=0))
        gaps = [(first - previous, previous, first) for (_, previous), (first, _) in
                zip(columns[:-1], columns[1:], strict=True)]
        gaps.sort(reverse=True)
        if gaps and .125 * h <= gaps[0][0] <= .64 * h and (len(gaps) == 1 or gaps[0][0] > 1.57 * gaps[1][0]):
            split = (gaps[0][1] + gaps[0][2]) // 2
            leading = ink[:, :split]
            glyph = _ink_bounds(leading, (a, y))
            count, _, area, _ = cv2.connectedComponentsWithStats(leading, connectivity=8)
            if glyph and count > 1 and area[1:, -1].max() >= .67 * leading.sum():
                gx, gy, r, end_glyph = glyph
                if (.45 * h <= end_glyph - gy <= .84 * h and .41 <= (r - gx) / (end_glyph - gy) <= 1.44
                        and gx - left <= .34 * w):
                    extents = _ink_bounds(ink[:, split:], (a + split, y))
                    if not extents:
                        continue
        x, t, X, Y = extents
        if (not .30 * h <= Y - t <= .73 * h or not .64 <= (X - x) / (Y - t) <= 9.7
                or abs((t + Y) / 2 - (top + bottom) / 2) > .117 * h):
            continue
        pad = max(3, round(.14 * (Y - t)))
        reading = read_caption(frame, (max(a, x - pad), max(y, t - pad), min(b, X + pad), min(end, Y + pad)),
                               index, cache=cache, choice=True)
        if reading and _label_plausible(reading["text"]):
            readings.setdefault(reading["text"], set()).add(ink.tobytes())
    matching = [text for text, cuts in readings.items() if len(cuts) >= 2]
    return matching[0] if len(matching) == 1 else None


def modal_primary_plate_actions(frame, panel, index):
    """Publish one literal footer action when the parent owns the panel."""
    frame_height, frame_width = frame.shape[:2]
    if index is None or panel is None or len(panel) != 4:
        return []
    l, t, r, b = map(int, panel)
    width, height = r - l, b - t
    if (not 0 <= l < r <= frame_width or not 0 <= t < b <= frame_height or width < 168
            or not .60 * width <= height <= 1.95 * width or not .41 * frame_width <= width <= .94 * frame_width):
        return []
    bounds = _filled_shapes(frame, (l, t, r, b))
    if len(bounds) != 1:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    box = _plate_outline(gray, bounds[0])
    if not box:
        return []
    left, top, right, bottom = box
    if (not l <= left < right <= r or not t <= top < bottom <= b
            or not (caption := _plate_caption(frame, gray, box, index))):
        return []
    return [{"role": "modal_primary_action", "function": caption, "preview_text": caption, "box": list(box),
             "click_point": [(left + right) // 2, (top + bottom) // 2]}]
