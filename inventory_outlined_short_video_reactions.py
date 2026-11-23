"""Find small speech bubbles with independently visible short-video rail peers."""

from dataclasses import dataclass

import cv2
import numpy as np

from inventory_compact_video_rail import (
    _curved_arrow, _footer_top, _marks, _neighbor, _speech, find_compact_video_share_box,
)
from inventory_filled_video_rail import _stable_locations
from inventory_list_rows import _Box
from inventory_video_edge_shapes import (
    _arc_peer, _framed_square, _outlined_arrow, _same_extent, find_outlined_video_share_box,
)
from screen_foreground import detect_blocking_overlay


@dataclass(frozen=True)
class ShortVideoPeer:
    """Measured rail geometry. The comment coordinates constrain search when bubble is absent."""

    bubble: _Box | None
    pitch: float
    share_box: _Box | None
    footer_y: int
    comment_x: float
    comment_y: float
    span: float


_CIRCLE_ANGLES = np.float32(np.arange(48) * np.pi / 24)
_CIRCLE_X = np.cos(_CIRCLE_ANGLES)
_CIRCLE_Y = np.sin(_CIRCLE_ANGLES)


def _circle_competitor(light, x, y, span, pitch):
    """Reject an apparently broken arc if another nearby fit closes the ring."""
    left = max(0, round(x) - round(1.5 * span))
    top = max(0, round(y) - round(1.75 * span))
    roi = light[top:round(y) + round(1.75 * span) + 1, left:round(x) + round(1.5 * span) + 1]
    dx, dy = round(.24 * span), round(.20 * pitch)
    centers_x, centers_y, radii = np.meshgrid(
        np.linspace(round(x) - dx, round(x) + dx, min(2 * dx + 1, 19), dtype=np.float32) - left,
        np.linspace(round(y) - dy, round(y) + dy, min(2 * dy + 1, 19), dtype=np.float32) - top,
        np.arange(.27 * span, .69 * span, max(.40, .024 * span), dtype=np.float32),
        indexing="ij",
    )
    centers_x = centers_x.reshape(-1, 1)
    centers_y = centers_y.reshape(-1, 1)
    radii = radii.reshape(-1, 1)

    def sample(radius):
        return cv2.remap(roi, centers_x + radius * _CIRCLE_X, centers_y + radius * _CIRCLE_Y,
                         cv2.INTER_LINEAR).astype(np.float32)

    peak = np.maximum.reduce([sample(radii + delta) for delta in (-.28, 0, .28)])
    inner = np.minimum(sample(np.maximum(.20, radii - .92)), sample(np.maximum(.20, radii - 2.05)))
    outer = np.minimum(sample(radii + .92), sample(radii + 2.05))
    contrast = peak - .50 * (inner + outer) - .14 * np.abs(inner - outer)
    brightness = np.median(peak.reshape(-1, 8, 6), axis=2).min(axis=1)
    stroke = np.median(contrast.reshape(-1, 8, 6), axis=2).min(axis=1)
    closed = ((brightness >= 185) & (stroke >= 64)) | ((brightness >= 224) & (stroke >= 30))
    return bool(closed.any())


def _split_curls(mark, light):
    """Two opposed curls can form a remix glyph without a crossed center."""
    left, top, right, bottom = mark.box.xyxy()
    chip = light[top:bottom, left:right]
    height, width = chip.shape
    if min(height, width) < 6 or not (.78 <= width / height <= 1.35 and .30 <= mark.mask.mean() <= .61):
        return False
    left_side = slice(round(.10 * width), round(.36 * width))
    right_side = slice(round(.64 * width), round(.90 * width))
    seam = slice(round(.42 * width), round(.60 * width))
    upper = slice(round(.20 * height), round(.38 * height))
    lower = slice(-round(.38 * height), -round(.20 * height))
    middle = slice(round(.39 * height), round(.62 * height))

    def strength(rows, columns):
        return np.percentile(chip[rows, columns], 65)

    upper_left, upper_right = strength(upper, left_side), strength(upper, right_side)
    lower_left, lower_right = strength(lower, left_side), strength(lower, right_side)
    middle_sides = min(strength(middle, left_side), strength(middle, right_side))
    return bool(min(upper_left, lower_right, middle_sides) >= 204
                and min(upper_left - upper_right, lower_right - lower_left,
                        middle_sides - strength(middle, seam)) >= 29)


def _verified_remix(light, curls, x, y, span, pitch):
    split = any(abs(x - mark.box.center[0]) <= .29 * span
                and abs(y - mark.box.center[1]) <= .19 * pitch
                and .60 * span <= mark.box.height <= 1.35 * span for mark in curls)
    found = split or any(_arc_peer(light, x, y, size * span, pitch) for size in (1, .90))
    return found and not _circle_competitor(light, x, y, span, pitch)


def _same_peer(first, second):
    drift = max(1, .24 * min(first.span, second.span))
    return (abs(first.comment_x - second.comment_x) <= drift
            and abs(first.comment_y - second.comment_y) <= drift
            and abs(first.pitch - second.pitch) <= .14 * first.pitch)


def _unique(peers):
    if not peers or any(not _same_peer(peers[0], other) for other in peers[1:]):
        return None
    return peers[0]


def _confirmed_share(frame, observation, share, line_reader):
    if share is None:
        return None
    fractions = share.get("screen_fraction")
    if fractions is None or len(fractions) != 2:
        return None
    height, width = frame.shape[:2]
    x, y = fractions[0] * (width - 1), fractions[1] * (height - 1)
    compact = find_compact_video_share_box(frame, observation, line_reader=line_reader)
    if compact is not None:
        candidate = _Box(*compact)
        if candidate.left <= x < candidate.right and candidate.top <= y < candidate.bottom:
            return candidate
    outline = find_outlined_video_share_box(frame, observation)
    if outline is not None:
        candidate = _Box(*outline)
        if candidate.left <= x < candidate.right and candidate.top <= y < candidate.bottom:
            return candidate
    return None


def _share_backed_bubbles(bubbles, marks, share, footer):
    sx, sy = share.center
    for bubble in bubbles:
        bx, by = bubble.box.center
        span = (bubble.box.height + share.height) / 2
        pitch = sy - by
        if not (1.72 * span <= pitch <= 4.20 * span and abs(sx - bx) <= .24 * span
                and .61 * span <= share.height <= 1.38 * bubble.box.height):
            continue
        axis = (sx + bx) / 2
        # A third visible mark distinguishes the corroborated bubble from another
        # otherwise plausible shape on the same column.
        if (any(_neighbor(mark, bubble, span, pitch, axis, -1) for mark in marks)
                or any(abs(mark.box.center[0] - axis) <= .32 * span
                       and abs(mark.box.center[1] - (sy + 2 * pitch)) <= .43 * pitch
                       and 1.08 * span <= mark.box.height <= 1.90 * span for mark in marks)):
            yield ShortVideoPeer(bubble.box, pitch, share, footer, bx, by, bubble.box.height)


def _aligned_artwork(box, tile, footer, gaps):
    x, y = box.center
    tx, ty = tile.box.center
    pitch = (ty - y) / gaps
    span = box.height
    if (2.22 * span <= pitch <= 3.90 * span and 1.16 * span <= tile.box.height <= 1.88 * span
            and abs(x - tx) <= .29 * span and .34 * pitch <= footer - ty <= .87 * pitch):
        return (x + tx) / 2, pitch
    return None


def _independent_peers(light, marks, bubbles, footer):
    artwork = _stable_locations(marks, lambda mark: _framed_square(mark, light))
    if not artwork:
        return []
    curls = _stable_locations(marks, lambda mark: _split_curls(mark, light))
    output = []
    for bubble in bubbles:
        bx, by = bubble.box.center
        for tile in artwork:
            geometry = _aligned_artwork(bubble.box, tile, footer, 3)
            if geometry is None:
                continue
            axis, pitch = geometry
            if _verified_remix(light, curls, axis, by + 2 * pitch, bubble.box.height, pitch):
                output.append(ShortVideoPeer(bubble.box, pitch, None, footer, bx, by, bubble.box.height))
                break
    if output:
        return output

    # A genuinely observed Share, Remix, and artwork can retain the rail context
    # with Comment erased. The projected location is only a search constraint.
    arrows = {mark.box for mark in _stable_locations(marks, lambda mark: _curved_arrow(mark.mask))}
    arrows.update(mark.box for mark in marks
                  if (score := _outlined_arrow(mark, light)) >= 48
                  or (score >= 24 and any(_same_extent(mark, other) for other in marks)))
    for arrow in sorted(arrows, key=lambda box: box.xyxy()):
        ax, ay = arrow.center
        for tile in artwork:
            geometry = _aligned_artwork(arrow, tile, footer, 2)
            if geometry is None:
                continue
            axis, pitch = geometry
            if _verified_remix(light, curls, axis, ay + pitch, arrow.height, pitch):
                output.append(ShortVideoPeer(None, pitch, arrow, footer, ax, ay - pitch, arrow.height))
                break
    return output


def find_outlined_short_video_peer(frame, observation, *, share=None, line_reader=None):
    """Return uniquely supported rail context and any actually measured speech box."""
    if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
            or observation.get("active_layer") or observation.get("mode") == "home"):
        return None
    height, width = frame.shape[:2]
    if not (128 <= width <= 420 and 1.60 * width <= height <= 2.65 * width):
        return None
    footer = _footer_top(frame)
    if footer is None:
        return None
    marks = list(_marks(frame, footer))
    bubbles = _stable_locations(marks, lambda mark: _speech(mark.mask))
    measured_share = _confirmed_share(frame, observation, share, line_reader) if bubbles else None
    peer = _unique(list(_share_backed_bubbles(bubbles, marks, measured_share, footer))) if measured_share else None
    if peer is None:
        light = frame.min(axis=2)
        peer = _unique(_independent_peers(light, marks, bubbles, footer))
    if peer is None or detect_blocking_overlay(frame) is not None:
        return None
    return peer
