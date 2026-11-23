"""Identify an outlined paper-plane share button on a video reaction rail."""

import cv2
import numpy as np

from inventory_compact_video_rail import _footer_top, _marks
from inventory_filled_video_rail import _stable_locations


_SAMPLE_X, _SAMPLE_Y = np.meshgrid(np.linspace(-.68, .68, 31, dtype=np.float32),
                                   np.linspace(-.68, .68, 31, dtype=np.float32))


def _areas(points, reach):
    areas = [np.flatnonzero((np.abs(_SAMPLE_X - x) <= reach) & (np.abs(_SAMPLE_Y - y) <= reach))
             for x, y in points]
    return [(area, round(.70 * (len(area) - 1))) for area in areas]


_REPOST_STROKES = _areas(((-.48, .01), (.46, .02), (-.22, -.36), (.22, .43), (.11, -.41), (-.18, .39)), .14)
_REPOST_GAPS = _areas(((0, 0), (.30, -.28), (-.31, .26)), .14)
_TILE_EDGES = _areas(((0, -.49), (0, .49), (-.47, 0), (.47, 0)), .14)
_TILE_CORE = _areas(((0, 0),), .14)

_PLANE_X, _PLANE_Y = np.meshgrid(np.linspace(0, 1, 29), np.linspace(0, 1, 29))
_PLANE_SEAM = (_PLANE_X >= .39) & (_PLANE_X <= .70) & (abs(_PLANE_Y - (.62 - .57 * _PLANE_X)) <= .052)
_UPPER_POCKET = (_PLANE_X >= .25) & (_PLANE_X <= .54) & (_PLANE_Y >= .165) & (_PLANE_Y <= .235)
_LOWER_POCKET = (_PLANE_X >= .48) & (_PLANE_X <= .65) & (_PLANE_Y >= .51) & (_PLANE_Y <= .69)


def _strength(probe, areas):
    return [int(np.partition(probe.ravel()[area], rank)[rank]) for area, rank in areas]


def _footer_edge(light, *, join_fragments=False):
    height, width = light.shape
    start, stop = round(.89 * height), round(.976 * height)
    left, right = round(.06 * width), round(.94 * width)
    strip = light[start:stop, left:right]
    size = round(.052 * width) | 1
    ground = cv2.morphologyEx(strip, cv2.MORPH_OPEN, np.ones((size, size), dtype=np.uint8))
    mask = np.uint8((strip >= 140) & (cv2.subtract(strip, ground) >= 42))
    edge = _navigation_row(mask, ground, start, left, height, width)
    if edge is not None or not join_fragments:
        return edge
    # A people or camera icon can be drawn with several separate strokes.
    kernel = np.ones((round(.019 * width) | 1, round(.023 * width) | 1), np.uint8)
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return _navigation_row(joined, ground, start, left, height, width)


def _navigation_row(mask, ground, start, left, height, width):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glyphs = sorted((x + w / 2 + left, y + h / 2 + start, h)
                    for x, y, w, h in map(cv2.boundingRect, contours)
                    if .029 * width <= w <= .093 * width and .029 * width <= h <= .092 * width
                    and np.median(ground[y:y + h, x:x + w]) <= 106)
    for _, reference_y, _ in glyphs:
        row = [glyph for glyph in glyphs if abs(glyph[1] - reference_y) <= .013 * width]
        if len(row) != 5:
            continue
        xs, ys, sizes = map(np.asarray, zip(*row, strict=True))
        gaps = np.diff(xs)
        if (ys.min() >= .91 * height and xs[0] <= .21 * width and xs[-1] >= .79 * width
                and gaps.min() >= .135 * width and gaps.max() <= .213 * width
                and gaps.max() - gaps.min() <= .041 * width):
            return round(np.median(ys) - 1.25 * np.median(sizes))
    return None


def _plane(mark, light):
    box, mask = mark.box, mark.mask
    h, w = mask.shape
    if not (.88 <= w / h <= 1.34 and .52 <= mask.mean() <= .76):
        return False
    top, tip = mask[:max(2, round(.25 * h))], np.flatnonzero(mask[-1])
    if (np.median(top.sum(axis=1)) < .83 * w or not 0 < tip.size <= .34 * w
            or not .36 * w <= tip.mean() + .5 <= .66 * w):
        return False
    chip = light[box.top:box.bottom, box.left:box.right]
    normalized = cv2.resize(chip, (29, 29), interpolation=cv2.INTER_LINEAR).astype(np.int16)
    seam = np.percentile(normalized[_PLANE_SEAM], 45)
    return bool(seam - max(np.median(normalized[_UPPER_POCKET]), np.median(normalized[_LOWER_POCKET])) >= 36)


def _heart_outline(mark, light):
    mask = mark.mask
    h, w = mask.shape
    if not (.90 <= w / h <= 1.41 and .63 <= mask.mean() <= .83):
        return False
    alignment = (mask & mask[:, ::-1]).sum() / (mask | mask[:, ::-1]).sum()
    center = w // 2
    top = mask[:max(1, round(.20 * h))]
    tip = np.flatnonzero(mask[-1])
    if not (alignment >= .76 and 0 < tip.size <= .31 * w and abs((tip.mean() + .5) / w - .5) <= .14):
        return False
    lobe_size = np.minimum(top[:, :center].sum(axis=1), top[:, center + 1:].sum(axis=1))
    if np.any((top[:, center] == 0) & (lobe_size >= .22 * w)):
        return True
    # At this scale one pixel can change how the two lobes round. Confirm their brightness separately.
    box = mark.box
    for y in np.flatnonzero((top[:, center] == 0) & (lobe_size >= round(.22 * w))):
        row = light[box.top + y, box.left:box.right]
        left = np.median(row[round(.13 * w):round(.42 * w)])
        right = np.median(row[round(.58 * w):round(.87 * w)])
        if min(left, right) - row[center] >= 23:
            return True
    return False


def _repost_edges(profile):
    strokes = _strength(profile, _REPOST_STROKES)
    behind = _strength(profile, _REPOST_GAPS)
    return min(strokes) >= 44 and min(strokes) - max(behind) >= 13


def _repost_on_light_background(profile):
    strokes = _strength(profile, _REPOST_STROKES)
    behind = _strength(profile, _REPOST_GAPS)
    return min(strokes) >= 36 and np.median(strokes) >= 48 and min(strokes) - max(behind) >= 18


def _framed_artwork(profile):
    edges = _strength(profile, _TILE_EDGES)
    inside, = _strength(profile, _TILE_CORE)
    return min(edges) >= 36 and np.median(edges) - inside >= 16


def _sees_peer(light, x, y, span, extent, drift, check):
    search = max(1, round(drift))
    reach = round(search + extent + span)
    left, top = max(0, round(x - reach)), max(0, round(y - reach))
    inset = light[top:round(y + reach) + 1, left:round(x + reach) + 1]
    size = round(span) | 1
    neighborhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    relief = cv2.subtract(inset, cv2.morphologyEx(inset, cv2.MORPH_OPEN, neighborhood))
    for center_y in (round(y) + offset for offset in range(-search, search + 1)):
        for size_ratio in (1, .93, 1.08):
            scale = extent * size_ratio
            chip = cv2.remap(relief, _SAMPLE_X * scale + (x - left),
                             _SAMPLE_Y * scale + (center_y - top), cv2.INTER_LINEAR)
            if check(chip):
                return True
    return False


def _sees_more_button(light, x, y, span, pitch):
    radius_x, radius_y = round(.67 * span), round(.17 * pitch)
    left, top = round(x) - radius_x, round(y) - radius_y
    patch = light[top:top + 2 * radius_y + 1, left:left + 2 * radius_x + 1]
    size = max(3, round(.40 * span) | 1)
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    relief = cv2.subtract(patch, cv2.morphologyEx(patch, cv2.MORPH_OPEN, opening))
    ink = np.uint8((patch >= 142) & (relief >= 49))
    _, _, stats, centers = cv2.connectedComponentsWithStats(ink)
    dots = sorted((px, py) for (bx, by, w, h, area), (px, py) in zip(stats[1:], centers[1:], strict=True)
                  if 1 <= area <= .051 * span ** 2 and max(w, h) <= .25 * span
                  and min(bx, by) > 0 and bx + w < patch.shape[1] and by + h < patch.shape[0])
    if len(dots) != 3:
        return False
    xs, ys = np.asarray(dots).T
    gaps = np.diff(xs)
    return (abs(xs[1] - radius_x) <= .12 * span and np.ptp(ys) <= max(1, .095 * span)
            and min(gaps) >= .21 * span and max(gaps) <= .42 * span and np.ptp(gaps) <= .12 * span)


def _anchored_marks(marks, shape):
    # One clear threshold can retain the correct silhouette when another joins a background edge.
    accepted = []
    for mark in marks:
        if not shape(mark) or any(
            max(abs(a - b) for a, b in zip(mark.box.center, other.box.center, strict=True)) <= .24 * mark.box.height
            for other in accepted
        ):
            continue
        if any(peer.level != mark.level and max(abs(a - b) for a, b in zip(peer.box.xyxy(), mark.box.xyxy(), strict=True))
               <= max(1, .16 * mark.box.height) for peer in marks):
            accepted.append(mark)
    return accepted


def _rail_with_artwork(marks, light, footer):
    planes = _stable_locations(marks, lambda mark: _plane(mark, light))
    hearts = _stable_locations(marks, lambda mark: _heart_outline(mark, light))
    candidates = []
    for plane in planes:
        x, y = plane.box.center
        span = plane.box.height
        for heart in hearts:
            hx, hy = heart.box.center
            pitch = (y - hy) / 3
            if (abs(hx - x) > .22 * span or not 2.25 * span <= pitch <= 3.62 * span
                    or not 1.47 * pitch <= footer - y <= 2.12 * pitch):
                continue
            if not _sees_peer(light, x, y - pitch, span, span, .12 * pitch, _repost_edges):
                continue
            if _sees_peer(light, x, y + pitch, span, 1.50 * span, .12 * pitch, _framed_artwork):
                candidates.append(plane)
                break
    return candidates


def _rail_with_more(marks, light, footer):
    planes = _anchored_marks(marks, lambda mark: _plane(mark, light))
    hearts = _anchored_marks(marks, lambda mark: _heart_outline(mark, light))
    candidates = []
    for plane in planes:
        x, y = plane.box.center
        span = plane.box.height
        for heart in hearts:
            hx, hy = heart.box.center
            pitch = (y - hy) / 3
            if (abs(hx - x) > .22 * span or not 2.25 * span <= pitch <= 3.62 * span
                    or not 1.55 * pitch <= footer - y <= 2.28 * pitch):
                continue
            if (not _sees_more_button(light, x, y + .96 * pitch, span, pitch)
                    or not _sees_peer(light, x, y - pitch, span, span, .12 * pitch, _repost_on_light_background)):
                continue
            if _sees_peer(light, x, y + 1.59 * pitch, span, 1.40 * span, .13 * pitch, _framed_artwork):
                candidates.append(plane)
                break
    return candidates


def find_paper_plane_share_box(frame, observation):
    """Return a plane on a reaction rail, with an independent navigation row below."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or observation.get("active_layer"):
        return None
    source_height, source_width = frame.shape[:2]
    if not (source_width >= 150 and 1.78 * source_width <= source_height <= 2.55 * source_width):
        return None
    if source_width > 540:
        # Keep the outlined strokes within the detector's working pixel scale.
        width = 360
        height = round(source_height * width / source_width)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    height, width = frame.shape[:2]
    light = cv2.min(cv2.min(frame[..., 0], frame[..., 1]), frame[..., 2])
    footer = _footer_edge(light)
    candidates = _rail_with_artwork(list(_marks(frame, footer)), light, footer) if footer is not None else []
    if not candidates:
        nav_edge = _footer_edge(light, join_fragments=True)
        visible_edge = _footer_top(frame) if nav_edge is not None else None
        if visible_edge is None or abs(nav_edge - visible_edge) > .045 * width:
            return None
        candidates = _rail_with_more(list(_marks(frame, visible_edge)), light, visible_edge)
    if len(candidates) != 1:
        return None
    left, top, right, bottom = candidates[0].box.xyxy()
    return (round(left * source_width / width), round(top * source_height / height),
            round(right * source_width / width), round(bottom * source_height / height))
