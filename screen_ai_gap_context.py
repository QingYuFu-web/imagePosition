"""Find small supplemental crops with visible form controls missed by existing crops."""

from math import ceil, floor

import cv2
import numpy as np

from screen_ai_choice_context import choice_context_boxes
from screen_wide_panel_checks import setting_check_layout


def _mapped_box(matrix, box):
    left, top, right, bottom = box
    corners = np.float64([[[left, top], [right, top], [right, bottom], [left, bottom]]])
    points = cv2.perspectiveTransform(corners, matrix)[0].round(5)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return [floor(x0), floor(y0), ceil(x1), ceil(y1)]


def _checkbox_context_boxes(view, coverage):
    _, checks = setting_check_layout(view, denoise=True)
    for check in checks:
        label, button, cue = check.label, check.box, check.cue
        left = min(label[0], button[0])
        top = min(label[1], button[1], cue[1])
        right = max(button[2], cue[2])
        bottom = max(label[3], button[3], cue[3])
        if any(x0 <= left and y0 <= top and right <= x1 and bottom <= y1 for x0, y0, x1, y1 in coverage):
            continue
        pad = max(3, round(.13 * check.height))
        yield [left - pad, top - pad, right + pad, bottom + pad]


def propose_gap_context(page, frame, coverage, photo):
    """Return original-photo boxes, without assigning purposes or action points."""
    layer = page["layer"]
    if layer["kind"] not in ("page", "wide_panel") or layer.get("floating_boxes"):
        return []
    width, height = photo.size if photo is not None else frame.shape[1::-1]
    if width < 1.42 * height or height < 360:
        return []
    scale = min(1., 1600 / width)
    view_width, view_height = round(width * scale), round(height * scale)
    to_view = np.diag([view_width / width, view_height / height, 1.])
    blocked = None
    if photo is not None:
        to_view = to_view @ photo.to_screen
        view = cv2.warpPerspective(frame, to_view, (view_width, view_height))
        if photo.blocked is not None:
            blocked = cv2.resize(photo.blocked, (view_width, view_height), interpolation=cv2.INTER_NEAREST)
            view[blocked != 0] = 32
    else:
        view = cv2.resize(frame, (view_width, view_height), interpolation=cv2.INTER_AREA) if scale < 1 else frame
    covered = [_mapped_box(to_view, box) for box in coverage]
    to_photo = np.linalg.inv(to_view)
    windows = [*_checkbox_context_boxes(view, covered), *choice_context_boxes(view, covered, blocked)]
    result = []
    for left, top, right, bottom in windows:
        if not (0 <= left < right <= view_width and 0 <= top < bottom <= view_height):
            continue
        if blocked is not None and blocked[top:bottom, left:right].any():
            continue
        result.append(_mapped_box(to_photo, (left, top, right, bottom)))
    return result
