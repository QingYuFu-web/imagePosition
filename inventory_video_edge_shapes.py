"""Verify outlined video-rail icons against cluttered backgrounds."""

import cv2
import numpy as np

from inventory_compact_video_rail import _curved_arrow, _footer_top, _marks, _one_position, _speech, _stable


_SAMPLE_AXIS = np.linspace(-.68, .68, 31, dtype=np.float32)
_SAMPLE_X, _SAMPLE_Y = np.meshgrid(_SAMPLE_AXIS, _SAMPLE_AXIS)


def _patch(light, box):
    return light[box.top:box.bottom, box.left:box.right]


def _outlined_arrow(mark, light):
    silhouette = mark.mask
    if not _curved_arrow(silhouette):
        # One missing diagonal sample can open an otherwise readable thin outline.
        if mark.box.height > 12 or not .24 <= silhouette.mean() <= .43:
            return 0
        closed = cv2.morphologyEx(np.pad(silhouette, 2), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            return 0
        silhouette = cv2.drawContours(np.zeros_like(closed), contours, 0, 1, cv2.FILLED)[2:-2, 2:-2]
        if not _curved_arrow(silhouette):
            return 0
    stroke = cv2.distanceTransform(np.pad(silhouette, 1), cv2.DIST_L2, cv2.DIST_MASK_5)[1:-1, 1:-1]
    thickness = max(1.1, .145 * mark.box.height)
    rim = (stroke > 0) & (stroke <= thickness)
    center = stroke > thickness
    if np.count_nonzero(center) < 3:
        return 0
    region = _patch(light, mark.box)
    return np.median(region[rim]) - np.median(region[center])


def _framed_square(mark, light):
    box = mark.box
    if not (.81 <= box.width / box.height <= 1.23 and mark.mask.mean() >= .84):
        return False
    region = _patch(light, box)
    col, row = round(.23 * box.width), round(.23 * box.height)
    inside = np.percentile(region[row:-row, col:-col], 60)
    sides = (region[:row, col:-col].max(axis=0), region[-row:, col:-col].max(axis=0),
             region[row:-row, :col].max(axis=1), region[row:-row, -col:].max(axis=1))
    return all(np.percentile(side, 25) - inside >= 26 for side in sides)


def _area(x, y, reach):
    indexes = np.flatnonzero((np.abs(_SAMPLE_X - x) <= reach) & (np.abs(_SAMPLE_Y - y) <= reach))
    return indexes, round(.70 * (len(indexes) - 1))


_ARC_SIDES = tuple(_area(x, y, .14) for x in (-.45, .45) for y in (-.25, .25))
_ARC_CAPS = (_area(0, -.51, .14), _area(0, .51, .14))
_ARC_BREAKS = (_area(-.48, 0, .14), _area(.48, 0, .14))
_CROSS_ARMS = tuple(_area(x, y, .11) for x, y in ((-.16, 0), (.16, 0), (0, -.16), (0, .16)))


def _arc_profile(response):
    def strength(areas):
        return [int(np.partition(response.ravel()[indexes], rank)[rank]) for indexes, rank in areas]

    # Opposed arcs leave lower-brightness side gaps around a separate center cross.
    caps, sides = strength(_ARC_CAPS), strength(_ARC_SIDES)
    if min(caps) < 34 or min(sides) < 29:
        return False
    gaps = max(strength(_ARC_BREAKS))
    arms = strength(_CROSS_ARMS)
    return bool(min(arms) >= 24 and np.median(caps + sides) >= gaps + 12
                and np.median(arms) >= gaps + 9)


def _arc_peer(light, x, y, span, pitch):
    dx, dy = max(1, round(.09 * span)), max(1, round(.20 * pitch))
    reach = round(max(dx, dy) + 1.8 * span)
    left, top = max(0, round(x) - reach), max(0, round(y) - reach)
    region = light[top:round(y) + reach + 1, left:round(x) + reach + 1]
    size = round(span) | 1
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    response = cv2.subtract(region, cv2.morphologyEx(region, cv2.MORPH_OPEN, opening))
    for extent in (span, 1.12 * span):
        scaled_x, scaled_y = _SAMPLE_X * extent, _SAMPLE_Y * extent
        for center_y in range(round(y) - dy, round(y) + dy + 1):
            rows = scaled_y + (center_y - top)
            for offset in (0, -dx, dx):
                profile = cv2.remap(response, scaled_x + (x - left + offset), rows, cv2.INTER_LINEAR)
                if _arc_profile(profile):
                    return True
    return False


def _same_extent(mark, peer):
    return (peer.level != mark.level
            and max(abs(a - b) for a, b in zip(mark.box.xyxy(), peer.box.xyxy(), strict=True))
            <= max(1, .16 * mark.box.height))


def find_outlined_video_share_box(frame, observation):
    """Return a share box when the edge and three independent rail controls agree."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or observation.get("active_layer"):
        return None
    height, width = frame.shape[:2]
    if not (128 <= width <= 420 and 1.60 * width <= height <= 2.65 * width):
        return None
    footer = _footer_top(frame)
    if footer is None:
        return None
    marks = list(_marks(frame, footer))
    bubbles = _stable(marks, _speech)
    if not bubbles:
        return None
    light = cv2.min(cv2.min(frame[..., 0], frame[..., 1]), frame[..., 2])
    arrows = []
    for mark in marks:
        edge = _outlined_arrow(mark, light)
        if edge >= 48 or (edge >= 24 and any(_same_extent(mark, peer) for peer in marks)):
            arrows.append(mark)
    if not arrows:
        return None
    frames = [mark for mark in marks if _framed_square(mark, light)
              and any(_same_extent(mark, peer) for peer in marks)]
    candidates = []
    for arrow in arrows:
        ax, ay = arrow.box.center
        if any(np.hypot(ax - found.box.center[0], ay - found.box.center[1]) <= .20 * arrow.box.height
               for found in candidates):
            continue
        for bubble in bubbles:
            bx, by = bubble.box.center
            span = (arrow.box.height + bubble.box.height) / 2
            if not (2.1 * span <= ay - by <= 4.2 * span and abs(ax - bx) <= .22 * span):
                continue
            for tile in frames:
                tx, ty = tile.box.center
                pitch = (ty - by) / 3
                if (not 1.12 * span <= tile.box.height <= 1.82 * span
                        or abs(tx - ax) > .25 * span
                        or not .90 * pitch <= ay - by <= 1.23 * pitch):
                    continue
                if _arc_peer(light, (ax + bx + tx) / 3, (ay + ty) / 2, span, pitch):
                    candidates.append(arrow)
                    break
            if candidates and candidates[-1] is arrow:
                break
    selected = _one_position(candidates)
    return selected.box.xyxy() if selected else None
