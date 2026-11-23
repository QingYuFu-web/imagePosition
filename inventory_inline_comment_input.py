"""Locate a light, filled inline comment field on an ordinary page."""

from __future__ import annotations

import cv2
import numpy as np

from inventory_comment_composer import _PLACEHOLDERS, _text
from inventory_list_rows import _Box, _labels


def _surface_contour(frame, label):
    height, width = frame.shape[:2]
    em = label.height
    y = round(label.center[1])
    dy = max(2, round(.17 * em))
    sample = frame[y - dy:y + dy, label.right + round(.35 * em):label.right + round(1.65 * em)]
    if not sample.size:
        return
    fill = np.median(sample, axis=(0, 1))
    if (fill.min() < 176 or np.ptp(fill) > 19
            or (np.max(np.abs(sample - fill), axis=2) <= 4).mean() < .94):
        return
    origin = max(0, min(label.top - round(.08 * width), height - round(.17 * width)))
    low = tuple(float(v) for v in np.maximum(0, fill - 3))
    high = tuple(float(v) for v in np.minimum(255, fill + 3))
    mask = cv2.inRange(frame[origin:], low, high)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, local_y, w, h = cv2.boundingRect(contour)
        box = _Box(x, origin + local_y, x + w, origin + local_y + h)
        if (not .61 * width <= w <= .92 * width or not .062 * width <= h <= .13 * width
                or not .01 * width <= x <= .096 * width or not .003 * width <= height - box.bottom <= .055 * width
                or cv2.contourArea(contour) < .95 * w * h
                or not .14 * h <= label.left - box.left <= .86 * h or label.right - x > .59 * w
                or not .25 * h <= em <= .65 * h or abs(label.center[1] - box.center[1]) > .145 * h
                or not box.top < label.top < label.bottom < box.bottom):
            continue
        surface = mask[local_y:local_y + h, x:x + w] != 0
        if _rounded_corners(surface) and _surrounding_page(frame, box, fill) and _visible_placeholder(frame, label, fill):
            yield box


def _rounded_corners(surface):
    height, width = surface.shape
    reach = max(3, round(.09 * height))
    center_x, center_y = width // 2, height // 2
    corners = (surface[:reach, :reach], surface[:reach, -reach:][:, ::-1],
               surface[-reach:, :reach][::-1], surface[-reach:, -reach:][::-1, ::-1])
    edge_support = reach - max(1, reach // 7)
    return (all(not corner[0, 0] and corner[-1, -1] and .10 <= corner.mean() <= .94 for corner in corners)
            and all(edge.sum() >= edge_support for edge in (surface[:reach, center_x], surface[-reach:, center_x],
                    surface[center_y, :reach], surface[center_y, -reach:])))


def _surrounding_page(frame, box, fill):
    pad = max(2, round(.075 * box.height))
    skip = max(2, round(.025 * box.height))
    y0, y1 = box.top + 2 * pad, box.bottom - 2 * pad
    x0, x1 = box.left + 2 * pad, box.right - 2 * pad
    if not (pad + skip <= box.left and box.right + pad + skip <= frame.shape[1]
            and pad + skip <= box.top and box.bottom + pad + skip <= frame.shape[0]):
        return False
    surround = (frame[y0:y1, box.left - pad - skip:box.left - skip],
                frame[y0:y1, box.right + skip:box.right + pad + skip],
                frame[box.top - pad - skip:box.top - skip, x0:x1],
                frame[box.bottom + skip:box.bottom + pad + skip, x0:x1])
    paper = np.median(np.asarray([np.median(part, axis=(0, 1)) for part in surround]), axis=0)
    # Distinct page color on four sides supports an actual input surface.
    return (paper.min() - fill.min() >= 7 and np.max(paper - fill) <= 52 and np.ptp(paper) <= 19
            and all((np.max(np.abs(part - paper), axis=2) <= 5).mean() >= .91 for part in surround))


def _visible_placeholder(frame, label, fill):
    ink = frame[label.top:label.bottom, label.left:label.right]
    return bool(ink.size and (np.max(fill - ink, axis=2) >= 59).mean() >= .07)


def suggest_inline_comment_input(frame_bgr, nodes, *, overlay=None):
    """Return the editable portion of a visibly bounded page-level comment field."""
    if (overlay is not None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
            or frame_bgr.dtype != np.uint8):
        return []
    height, width = frame_bgr.shape[:2]
    if width < 180 or not 1.49 * width <= height <= 2.75 * width:
        return []
    labels = [box for box, text, score in _labels(nodes, width, height) if score >= 95
              and "".join(_text(text).split()) in _PLACEHOLDERS
              and .023 * width <= box.left <= .21 * width and .020 * width <= box.height <= .073 * width
              and .91 * height <= box.top < box.bottom <= height - .012 * width]
    fields = [field for label in labels for field in _surface_contour(frame_bgr, label)]
    if len(fields) != 1:
        return []
    field = fields[0]
    # A trailing affordance can share the textbox's painted surface.
    # Use the leading typing area without assuming what the trailing glyph does.
    box = _Box(field.left, field.top, field.right - round(1.15 * field.height), field.bottom)
    return [{"role": "comment_input", "function": "输入评论", "preview_text": "Comment",
             "preview_above": True, "box": box.xyxy(), "center": box.center}]
