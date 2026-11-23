"""Name unselected chips in a visually verified phone filter row."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from inventory_list_rows import _Box, _labels


@dataclass(frozen=True)
class _Chip:
    box: _Box
    fill: tuple[int, int, int]
    text_polarity: int


def _rail_color(lane):
    shoulder = max(2, round(.073 * lane.shape[0]))
    rails = (lane[:shoulder, ::3], lane[-shoulder:, ::3])
    tones = [np.median(rail.reshape(-1, 3), axis=0) for rail in rails]
    if np.max(np.abs(tones[0] - tones[1])) > 7:
        return None
    base = np.mean(tones, axis=0)
    if all((np.max(np.abs(rail - base), axis=2) <= 10).mean() >= .87 for rail in rails):
        return base
    return None


def _capsule(contour, box):
    left, top, right, bottom = box.xyxy()
    h = box.height
    if not .76 * box.width * h <= cv2.contourArea(contour) <= .974 * box.width * h:
        return False
    corners = [(x, y) for x in (left + .13 * h, right - .13 * h)
               for y in (top + .13 * h, bottom - .13 * h)]
    return all(cv2.pointPolygonTest(contour, corner, False) < 0 for corner in corners)


def _chip_face(lane, box):
    offset_x, offset_y = round(.27 * box.height), round(.20 * box.height)
    inside = lane[box.top + offset_y:box.bottom - offset_y, box.left + offset_x:box.right - offset_x]
    color = np.median(inside.reshape(-1, 3), axis=0)
    if (np.max(np.abs(inside - color), axis=2) <= 8).mean() < .65:
        return None
    gray = cv2.cvtColor(inside, cv2.COLOR_BGR2GRAY).astype(np.int16)
    ink = gray - round(.114 * color[0] + .587 * color[1] + .299 * color[2])
    high, low = (ink >= 57).mean(), (ink <= -57).mean()
    if max(high, low) < .039 or min(high, low) > .027:
        return None
    return tuple(map(round, color)), 1 if high > low else -1


def _scan_lane(frame, choices):
    height, width = frame.shape[:2]
    font = float(np.median([box.height for box, _ in choices]))
    center = float(np.median([box.center[1] for box, _ in choices]))
    start, stop = max(0, round(center - 1.56 * font)), min(height, round(center + 1.56 * font))
    lane = frame[start:stop]
    if (base := _rail_color(lane)) is None:
        return []
    low = tuple(float(value) for value in np.maximum(0, base - 13))
    high = tuple(float(value) for value in np.minimum(255, base + 13))
    foreground = 255 - cv2.inRange(lane, low, high)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chips = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = _Box(x, y, x + w, y + h)
        if (not 1.56 * font <= h <= 2.50 * font or not 1.29 <= w / h <= 5.8 or w > .42 * width
                or abs(start + box.center[1] - center) > .30 * font or not _capsule(contour, box)):
            continue
        if (face := _chip_face(lane, box)) is not None:
            fill, polarity = face
            chips.append(_Chip(_Box(x, y + start, x + w, y + h + start), fill, polarity))
    return sorted(chips, key=lambda chip: chip.box.left)


def _separate_groups(chips):
    groups = []
    for chip in chips:
        if groups:
            previous = groups[-1][-1].box
            size = min(previous.height, chip.box.height)
            if (.075 * size <= chip.box.left - previous.right <= .66 * size
                    and max(previous.height, chip.box.height) <= 1.16 * size
                    and abs(previous.center[1] - chip.box.center[1]) <= .11 * size):
                groups[-1].append(chip)
                continue
        groups.append([chip])
    return [group for group in groups if len(group) >= 3]


def _tone_difference(first, other):
    return max(abs(a - b) for a, b in zip(first.fill, other.fill, strict=True))


def _unselected(group):
    interpretations = []
    for active in group:
        peers = [chip for chip in group if chip is not active]
        anchor = peers[0]
        if (all(chip.text_polarity == anchor.text_polarity and _tone_difference(anchor, chip) <= 8 for chip in peers)
                and active.text_polarity == -anchor.text_polarity
                and all(_tone_difference(active, chip) >= 39 for chip in peers)):
            interpretations.append(peers)
    return interpretations[0] if len(interpretations) == 1 else []


def _caption(chip, choices):
    box = chip.box
    h = box.height
    candidates = [label for text, label in choices
                  if abs(text.center[0] - box.center[0]) <= .13 * h and abs(text.center[1] - box.center[1]) <= .145 * h
                  and .33 * h <= text.height <= .69 * h and box.left + .19 * h <= text.left < text.right <= box.right - .19 * h]
    return candidates[0] if len(candidates) == 1 else None


def suggest_top_filter_chips(frame_bgr, nodes, *, overlay=None):
    """Return independent unselected chip actions with a proven selected peer."""
    if (overlay is not None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
            or frame_bgr.dtype != np.uint8):
        return []
    height, width = frame_bgr.shape[:2]
    if width < 180 or not 1.48 * width <= height <= 2.75 * width:
        return []
    labels = [(box, label) for box, label, score in _labels(nodes, width, height) if score >= 95
              and .047 * height <= box.center[1] <= .235 * height and .020 * width <= box.height <= .068 * width
              and 2 <= len(label) <= 28 and sum(char.isalpha() for char in label) >= 2]
    row_candidates = set()
    found = {}
    for reference, _ in labels:
        row = tuple((box, label) for box, label in labels if abs(box.center[1] - reference.center[1]) <= .22 * reference.height
                    and .74 * reference.height <= box.height <= 1.29 * reference.height)
        if len(row) < 2 or row in row_candidates:
            continue
        row_candidates.add(row)
        for group in _separate_groups(_scan_lane(frame_bgr, row)):
            peers = _unselected(group)
            items = [(chip.box, label) for chip in peers if (label := _caption(chip, row)) is not None]
            if len(items) < 2 or len({label.casefold() for _, label in items}) != len(items):
                continue
            for box, label in items:
                found[box] = {"role": "select_filter", "function": f"切换筛选到「{label}」",
                              "preview_text": label if label.isascii() else "Filter", "box": box.xyxy(), "center": box.center}
    return sorted(found.values(), key=lambda row: (row["center"][1], row["center"][0]))
