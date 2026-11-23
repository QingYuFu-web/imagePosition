"""Package visible grouped drawer rows without requiring a local text transcript."""

import cv2
import numpy as np

from screen_ai_gap_context import _mapped_box
from screen_native_drawer_outline import _drawer_outline, _grouped_layout
from screen_page_handoff import _intersection
from screen_portrait_drawer import _drawer_bounds, _seam_positions
from screen_user_examples import _coverage


def _measured_drawer(frame):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    matches = []
    for reverse in (False, True):
        view, tone = (frame[:, ::-1], gray[:, ::-1]) if reverse else (frame, gray)
        for seam in _seam_positions(tone):
            if (outline := _drawer_outline(view, tone, seam, reverse)) is None:
                continue
            top, body, bottom, paint = outline
            box = (width - seam, top, width, bottom) if reverse else (0, top, seam, bottom)
            matches.append((box, body, paint))
    if len(matches) == 1:
        return matches[0]
    if not matches and (ordinary := _drawer_bounds(frame)) is not None:
        return ordinary[:3]
    return None


def _row_ink(drawer, body, row, pitch):
    caption, detail, icon_left, center = row
    pad = max(3, round(.10 * pitch))
    left = max(drawer[0], icon_left - pad)
    top = max(body, min(caption[1] - pad, round(center - .37 * pitch)))
    right = min(drawer[2], max(caption[2], detail[2] if detail else 0) + pad)
    bottom = min(drawer[3], max(caption[3], detail[3] if detail else 0, round(center + .37 * pitch)) + pad)
    return left, top, right, bottom


def drawer_context_boxes(page, frame, panels, covered, photo, *, budget, limit):
    """Return small original-photo crops of independently measured, uncovered drawer rows."""
    layer = page["layer"]
    if layer["kind"] != "side_drawer" or layer.get("floating_boxes") or budget <= 0 or limit <= 0:
        return []
    size = frame.shape[1::-1] if photo is None else photo.size
    width, height = size
    if width < 240 or not 1.44 * width <= height <= 2.80 * width:
        return []
    blocked = None if photo is None else photo.blocked
    if photo is not None:
        frame = cv2.warpPerspective(frame, photo.to_screen, size)
        if blocked is not None:
            frame[blocked != 0] = 32
    measured = _measured_drawer(frame)
    if measured is None:
        return []
    drawer, body, paint = measured
    owner = list(drawer) if photo is None else _mapped_box(photo.to_photo, drawer)
    matching = [panel for panel in panels if min(_coverage(owner, panel), _coverage(panel, owner)) >= .88]
    if len(matching) != 1 or (layout := _grouped_layout(frame, drawer, body, paint)) is None:
        return []
    panel, = matching
    rows, pitch, selected_row = layout
    excluded = [item["box"] for item in page["elements"] if item.get("disabled")]
    excluded.extend(sample["box"] for item in page["elements"]
                    if (sample := item.get("reviewed_sample")) is not None and sample["purpose"] is None)
    stamps = []
    for index, raw in enumerate(rows):
        box = _row_ink(drawer, body, raw, pitch)
        left, top, right, bottom = box
        original = list(box) if photo is None else _mapped_box(photo.to_photo, box)
        invalid = (index == selected_row or _intersection(original, panel) != original
                   or (blocked is not None and blocked[top:bottom, left:right].any())
                   or any(_intersection(original, area) for area in excluded))
        stamps.append((original, invalid, any(_coverage(original, area) >= .72 for area in covered)))

    windows = []
    for start, (origin, unavailable, seen) in enumerate(stamps):
        if unavailable or seen:
            continue
        bounds = origin.copy()
        missing = nearby = 0
        for offset, (box, invalid, covered_row) in enumerate(stamps[start:], start):
            if invalid or (offset > start and rows[offset][3] - rows[offset - 1][3] > 1.32 * pitch):
                break
            bounds[0], bounds[1] = min(bounds[0], box[0]), min(bounds[1], box[1])
            bounds[2], bounds[3] = max(bounds[2], box[2]), max(bounds[3], box[3])
            area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
            if area > budget or any(_intersection(bounds, mask) for mask in excluded):
                break
            if covered_row:
                nearby += 1
                if nearby > 1:
                    break
                continue
            missing += 1
            windows.append((-missing, start, area, bounds.copy()))

    selected = []
    for _, _, area, box in sorted(windows):
        if area <= budget and not any(_intersection(box, prior) for prior in selected):
            selected.append(box)
            budget -= area
            if len(selected) == limit:
                break
    return selected
