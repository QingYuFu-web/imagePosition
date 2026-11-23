"""Locate a hollow speech bubble on a corroborated video reaction rail."""

import cv2
import numpy as np

from inventory_compact_video_rail import _Mark, _footer_top, _marks
from inventory_filled_video_rail import _stable_locations
from inventory_list_rows import _Box
from inventory_paper_plane_share import (
    _footer_edge, _framed_artwork, _heart_outline, _plane, _repost_edges,
    _repost_on_light_background, _sees_peer,
)


def _speech_outline(contour, hole, bounds):
    x, y, w, h = bounds
    if not (.84 <= w / h <= 1.19 and .65 <= cv2.contourArea(contour) / (w * h) <= .84
            and .45 <= cv2.contourArea(hole) / (w * h) <= .77):
        return None
    solid = np.zeros((h, w), np.uint8)
    cv2.drawContours(solid, [contour], 0, 1, cv2.FILLED, offset=(-x, -y))
    # The callout tail extends the lower right edge below the circular arc.
    edge = solid[-max(2, round(.15 * h)):]
    side = max(2, round(.18 * w))
    if edge[:, -side:].mean() < .37 or edge[:, :side].mean() > .14:
        return None
    return solid


def _bubble_marks(gray, heart):
    height, width = gray.shape
    x, y = heart.box.center
    span = heart.box.height
    left = max(0, round(x - .93 * span))
    right = min(width, round(x + .93 * span))
    top, bottom = round(y + 1.36 * span), min(height, round(y + 3.72 * span))
    patch = gray[top:bottom, left:right]
    close = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for level, cut in enumerate((201, 212, 222, 231, 240)):
        ink = cv2.morphologyEx(np.uint8(patch >= cut), cv2.MORPH_CLOSE, close)
        contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for i, contour in enumerate(contours):
            _, _, child, parent = hierarchy[0, i]
            if child < 0 or parent >= 0 or hierarchy[0, child, 0] >= 0:
                continue
            lx, ty, w, h = cv2.boundingRect(contour)
            if (not .74 * span <= h <= 1.36 * span or lx < 1 or ty < 1
                    or lx + w >= patch.shape[1] or ty + h >= patch.shape[0]):
                continue
            silhouette = _speech_outline(contour, contours[child], (lx, ty, w, h))
            if silhouette is None:
                continue
            box = _Box(left + lx, top + ty, left + lx + w, top + ty + h)
            cx, cy = box.center
            if abs(cx - x) <= .24 * span and 1.90 * span <= cy - y <= 3.15 * span:
                yield _Mark(box, silhouette, level)


def _rail_support(light, footer, bubble, heart, planes):
    x, y = bubble.box.center
    pitch = y - heart.box.center[1]
    span = (bubble.box.height + heart.box.height) / 2
    if not 3.61 * pitch <= footer - y <= 4.45 * pitch:
        return False
    plane = any(abs(mark.box.center[0] - x) <= .24 * span
                and abs(mark.box.center[1] - y - 2 * pitch) <= .17 * pitch for mark in planes)
    if not plane and not any(_sees_peer(light, x, y + pitch, span, span, .17 * pitch, check)
                             for check in (_repost_edges, _repost_on_light_background)):
        return False
    return _sees_peer(light, x, footer - .54 * pitch, span, 1.42 * span, .20 * pitch, _framed_artwork)


def suggest_outlined_comment(frame, observation):
    """Return one visible viewer comment button in original frame coordinates."""
    if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
            or observation.get("active_layer") or observation.get("mode") == "home"):
        return None
    source_height, source_width = frame.shape[:2]
    if not (source_width >= 150 and 1.78 * source_width <= source_height <= 2.55 * source_width):
        return None
    if source_width > 384:
        width = 384
        height = round(source_height * width / source_width)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    height, width = frame.shape[:2]
    light = frame.min(axis=2)
    footer = _footer_edge(light) or _footer_top(frame)
    if footer is None:
        return None
    marks = list(_marks(frame, footer))
    hearts = [mark for mark in marks if _heart_outline(mark, light)]
    if not hearts:
        return None
    planes = _stable_locations(marks, lambda mark: _plane(mark, light))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates = []
    for heart in hearts:
        bubbles = _stable_locations(list(_bubble_marks(gray, heart)), lambda mark: True)
        for bubble in bubbles:
            if _rail_support(light, footer, bubble, heart, planes):
                if not any(np.linalg.norm(np.subtract(bubble.box.center, peer.center)) <= .22 * bubble.box.height
                           for peer in candidates):
                    candidates.append(bubble.box)
    if len(candidates) != 1:
        return None
    left, top, right, bottom = candidates[0].xyxy()
    scale_x = source_width / (width * (source_width - 1))
    scale_y = source_height / (height * (source_height - 1))
    return {"anchor_id": None, "role": "comments", "function": "打开视频评论",
            "box_fraction": [left * scale_x, top * scale_y, right * scale_x, bottom * scale_y],
            "screen_fraction": [(left + right) / 2 * scale_x, (top + bottom) / 2 * scale_y]}
