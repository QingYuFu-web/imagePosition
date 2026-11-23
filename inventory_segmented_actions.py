"""Read selections from a visually connected, labeled segmented control."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

import cv2
import numpy as np

from inventory_list_rows import _Box, _labels
from inventory_value_select import _Line, _ink_strength, _overlap


@dataclass(frozen=True)
class _Group:
    rail: _Box
    slots: tuple[_Box, ...]
    labels: tuple[_Line | None, ...]
    selected: int


def _rounded_shape(contour, bounds):
    x, y, w, h = bounds
    if cv2.contourArea(contour) < .77 * w * h:
        return False
    silhouette = np.zeros((h, w), np.uint8)
    cv2.drawContours(silhouette, [contour - (x, y)], -1, 1, cv2.FILLED)
    corner = max(2, round(.17 * h))
    corners = (silhouette[:corner, :corner], silhouette[:corner, -corner:],
               silhouette[-corner:, :corner], silhouette[-corner:, -corner:])
    return sum(chip.mean() <= .70 for chip in corners) >= 3


def _rounded_rails(gray):
    height, width = gray.shape
    dx = cv2.absdiff(gray[:, 1:], gray[:, :-1])
    dy = cv2.absdiff(gray[1:, :], gray[:-1, :])
    kernel = np.ones((3, 3), np.uint8)
    boxes = set()
    for threshold in (4, 9, 18):
        edges = np.zeros_like(gray)
        edges[:, 1:] = dx >= threshold
        edges[1:, :] |= dy >= threshold
        for mask in (edges, cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)):
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                bounds = cv2.boundingRect(contour)
                x, y, w, h = bounds
                if not (max(16, .014 * min(height, width)) <= h <= min(140, .26 * min(height, width))
                        and 2.45 <= w / h <= 18 and 0 < x < x + w < width and 0 < y < y + h < height
                        and _rounded_shape(contour, bounds)):
                    continue
                boxes.add(_Box(x, y, x + w, y + h))
    return sorted(boxes, key=lambda box: (-box.width * box.height, box.top, box.left))


def _rail_strength(gray, rail):
    h = rail.height
    inset = round(.46 * h)
    left, right = rail.left + inset, rail.right - inset
    margin = max(2, round(.05 * h))
    strengths = []
    for y in (rail.top, rail.bottom):
        patch = gray[max(0, y - margin):min(gray.shape[0], y + margin + 1), left:right].astype(np.int16)
        strengths.append(float(np.median(np.abs(np.diff(patch, axis=0)), axis=1).max()))
    return min(strengths)


def _place_labels(rail, lines):
    h = rail.height
    aligned = [line for line in lines if rail.left < line.box.left < line.box.right < rail.right
               and abs(line.box.center[1] - rail.center[1]) <= .23 * h and .19 * h <= line.box.height <= .81 * h]
    if len(aligned) < 2:
        return
    for count in range(max(2, len(aligned)), min(7, len(aligned) + 2) + 1):
        pitch = rail.width / count
        bound = [round(rail.left + index * pitch) for index in range(count + 1)]
        if pitch < .90 * h:
            continue
        slots = tuple(_Box(left, rail.top, right, rail.bottom) for left, right in zip(bound, bound[1:]))
        names = [None] * count
        for line in aligned:
            index = int((line.box.center[0] - rail.left) / pitch)
            slot = slots[index]
            if (names[index] is not None or abs(line.box.center[0] - slot.center[0]) > .21 * pitch
                    or not slot.left < line.box.left < line.box.right < slot.right):
                break
            names[index] = line
        else:
            yield slots, tuple(names)


def _paint(frame, slot):
    h, w = slot.height, slot.width
    left, right = slot.left + round(.25 * w), slot.right - round(.25 * w)
    upper = frame[slot.top + round(.20 * h):slot.top + round(.32 * h), left:right]
    lower = frame[slot.bottom - round(.32 * h):slot.bottom - round(.20 * h), left:right]
    above, below = np.median(upper, axis=(0, 1)), np.median(lower, axis=(0, 1))
    if np.max(np.abs(above - below)) > 11:
        return None
    paint = (above + below) / 2
    if any((np.max(np.abs(band.astype(np.int16) - paint), axis=2) <= 12).mean() < .68 for band in (upper, lower)):
        return None
    return paint


def _surface_transition(profile, tone, expected, inward, height):
    depth = max(2, round(.09 * height))
    radius = round(.30 * height)
    start = max(depth + 2, expected - radius)
    stop = min(len(profile) - depth - 2, expected + radius)
    evidence = []
    for edge in range(start, stop + 1):
        if inward > 0:
            inside, outside = profile[edge:edge + depth], profile[edge - depth - 1:edge - 1]
            boundary = profile[edge - 1]
        else:
            inside, outside = profile[edge - depth:edge], profile[edge + 1:edge + depth + 1]
            boundary = profile[edge]
        contrast = abs(np.median(outside) - tone)
        if (np.abs(inside - tone) <= 2).all() and contrast >= 3 and abs(boundary - tone) >= 2:
            evidence.append((abs(expected - edge), -contrast, edge))
    return min(evidence)[2] if evidence else None


def _luminance_wing(gray, plate, wing, direction, neutral):
    h, w = plate.height, plate.width
    tone = round(float(np.dot(neutral, (.114, .587, .299))))
    collar = max(3, round(.40 * h))
    left, right = max(0, min(plate.left, wing.left) - collar), min(gray.shape[1], max(plate.right, wing.right) + collar)
    top, bottom = max(0, plate.top - collar), min(gray.shape[0], plate.bottom + collar)
    chip = gray[top:bottom, left:right]
    if direction < 0:
        chip = chip[:, ::-1]
    seam = plate.right - left if direction > 0 else right - plate.left
    outer = seam + w
    bands = (chip[:, seam + round(a * w):seam + round(b * w)] for a, b in ((.18, .34), (.60, .78)))
    profiles = [np.median(band, axis=1) for band in bands]
    tops = [_surface_transition(profile, tone, plate.top - top, 1, h) for profile in profiles]
    bottoms = [_surface_transition(profile, tone, plate.bottom - top, -1, h) for profile in profiles]
    if any(edge is None for edge in tops + bottoms) or max(np.ptp(tops), np.ptp(bottoms)) > max(2, .09 * h):
        return None
    y0, y1 = round(np.median(tops)), round(np.median(bottoms))
    depth = y1 - y0
    if not .78 * h <= depth <= 1.15 * h:
        return None
    follows = seam + round(.65 * w)
    caps = [chip[y0 + round(a * depth):y0 + round(b * depth), follows:]
            for a, b in ((.10, .22), (.34, .66), (.79, .91))]
    edges = [_surface_transition(np.median(band, axis=0), tone, outer - follows, -1, h) for band in caps]
    if any(edge is None for edge in edges):
        return None
    upper, middle, lower = edges
    if not all(max(2, .07 * depth) <= middle - end <= .36 * depth for end in (upper, lower)):
        return None
    # The neutral fill must also appear under both rounded corners of the accent plate.
    x0, x1 = seam - max(3, round(.16 * depth)), seam - 1
    inset, band = max(1, round(.025 * depth)), max(3, round(.12 * depth))
    joints = (chip[y0 + inset:y0 + band, x0:x1], chip[y1 - band:y1 - inset, x0:x1])
    for joint in joints:
        distance = np.abs(joint.astype(np.int16) - tone)
        if np.median(distance) > 3 or (distance <= 2).mean(axis=1).max() < .68:
            return None
    end = follows + middle
    return (_Box(plate.left, top + y0, left + end, top + y1) if direction > 0 else
            _Box(right - end, top + y0, plate.right, top + y1))


def _accent_rails(frame, gray, rails, lines):
    """Check for a shared neutral surface behind a saturated selected plate."""
    height, width = frame.shape[:2]
    for plate in rails:
        h, w = plate.height, plate.width
        if not 2.45 <= w / h <= 8:
            continue
        cx, cy = plate.center
        aligned = [line for line in lines if abs(line.box.center[1] - cy) <= .16 * h]
        if sum(abs(line.box.center[0] - cx) <= .19 * w for line in aligned) != 1:
            continue
        others = [line for line in aligned if .83 * w <= abs(line.box.center[0] - cx) <= 1.17 * w]
        if len(others) != 1:
            continue
        active = _paint(frame, plate)
        if active is None or np.ptp(active) < 68:
            continue
        direction = 1 if others[0].box.center[0] > cx else -1
        start, end = (plate.right, plate.right + w) if direction > 0 else (plate.left - w, plate.left)
        if not 0 <= start < end <= width:
            continue
        wing = _Box(start, plate.top, end, plate.bottom)
        neutral = _paint(frame, wing)
        if neutral is None or np.ptp(neutral) > 24:
            continue
        edge = plate.right if direction > 0 else plate.left
        overlap_x = round(edge - direction * .13 * h)
        band = np.array([frame[plate.top + round(.13 * h), overlap_x], frame[plate.bottom - round(.15 * h), overlap_x]])
        measured = []
        if (np.max(np.abs(band - neutral), axis=1) <= 2).all():
            pad = max(2, round(.22 * h))
            left, right = max(0, min(plate.left, start) - pad), min(width, max(plate.right, end) + pad)
            top, bottom = max(0, plate.top - pad), min(height, plate.bottom + pad)
            crop = frame[top:bottom, left:right].astype(np.int16)
            mask = np.uint8((np.max(np.abs(crop - neutral), axis=2) <= 2) | (np.max(np.abs(crop - active), axis=2) <= 12))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                bounds = cv2.boundingRect(contour)
                x, y, bw, bh = bounds
                if (.87 * h <= bh <= 1.15 * h and 1.85 * w <= bw <= 2.15 * w and 0 < x < x + bw < crop.shape[1]
                        and 0 < y < y + bh < crop.shape[0] and _rounded_shape(contour, bounds)):
                    measured.append(_Box(left + x, top + y, left + x + bw, top + y + bh))
        if measured:
            yield from measured
        else:
            # Chroma bleed can break an otherwise continuous low-contrast wing after compression.
            backed = _luminance_wing(gray, plate, wing, direction, neutral)
            if backed is not None:
                yield backed


def _divider(gray, rail, x):
    h = rail.height
    pad = max(2, round(.23 * h))
    sample = gray[rail.top + round(.19 * h):rail.bottom - round(.19 * h), x - pad:x + pad + 1].astype(np.int16)
    return bool(((np.abs(np.diff(sample, axis=1)) >= 6).mean(axis=0) >= .41).any())


def _selected(frame, gray, rail, slots):
    colors = [_paint(frame, slot) for slot in slots]
    if any(color is None for color in colors):
        return None
    candidates = []
    for index, active in enumerate(colors):
        inactive = np.array([color for i, color in enumerate(colors) if i != index])
        if np.max(np.ptp(inactive, axis=0)) > 8 or np.min(np.max(np.abs(inactive - active), axis=1)) < 10:
            continue
        slot = slots[index]
        h, w = slot.height, slot.width
        x0, x1 = slot.left + round(.27 * w), slot.right - round(.27 * w)
        margin = max(2, round(.13 * h))
        edge = np.concatenate((frame[slot.top:slot.top + margin, x0:x1], frame[slot.bottom - margin:slot.bottom, x0:x1]))
        rim = np.median(edge, axis=(0, 1))
        inset = np.max(np.abs(rim - inactive[0])) <= 8 and np.max(np.abs(rim - active)) >= 10
        emphatic_color = np.ptp(active) >= 61 and np.max(np.ptp(inactive, axis=1)) <= 26
        if not (inset or emphatic_color):
            outside = np.concatenate((frame[max(0, rail.top - margin):rail.top, x0:x1],
                                      frame[rail.bottom:min(frame.shape[0], rail.bottom + margin), x0:x1]))
            background = np.median(outside, axis=(0, 1))
            if np.max(np.abs(inactive - background)) > 9 or np.max(np.abs(active - background)) < 10:
                continue
        # A boundary between neutral neighbors must be drawn. The selected plate can supply
        # its own boundaries in an inset control.
        seams = [_divider(gray, rail, after.left) for after in slots[1:]]
        complete = all(seam or (inset or emphatic_color) and i in (index - 1, index)
                       for i, seam in enumerate(seams))
        if complete and (len(slots) >= 3 or inset or emphatic_color):
            candidates.append(index)
    return candidates[0] if len(candidates) == 1 else None


def _label_patch(frame, gray, slot):
    h, w = slot.height, slot.width
    left, right = slot.left + round(.19 * w), slot.right - round(.19 * w)
    top, bottom = slot.top + round(.23 * h), slot.bottom - round(.23 * h)
    chip = gray[top:bottom, left:right].astype(np.int16)
    ink = np.abs(chip - np.median(chip)) >= 43
    rows, cols = np.where(ink)
    if len(rows) < 7 or np.ptp(rows) < .15 * h or np.ptp(cols) > .55 * w:
        return None
    margin = max(2, round(.09 * h))
    tight = _Box(max(left, left + int(cols.min()) - margin), max(top, top + int(rows.min()) - margin),
                 min(right, left + int(cols.max()) + margin + 1), min(bottom, top + int(rows.max()) + margin + 1))
    return frame[tight.top:tight.bottom, tight.left:tight.right].copy() if min(tight.height, tight.width) >= 6 else None


def _key(label):
    return "".join(char for char in unicodedata.normalize("NFKC", label).casefold() if char.isalnum())


def _matches_weak_label(weak, slot, text):
    key = _key(text)
    return len(key) == 1 and any(key == _key(label) and slot.left < box.left < box.right < slot.right
                                 and abs(box.center[1] - slot.center[1]) <= .23 * slot.height for box, label, _ in weak)


def suggest_segmented_actions(frame, nodes, *, overlay=None, line_reader=None):
    """Return one bounded point per visible option in a shared single-selection rail."""
    height, width = frame.shape[:2]
    if min(height, width) < 150 or len(nodes) < 2:
        return []
    lines = [_Line(box, text, score) for box, text, score in _labels(nodes, width, height) if _key(text)]
    if len(lines) < 2:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    regions = ((0, 0, width, height),) if overlay is None else overlay.regions
    candidates = []
    rails = _rounded_rails(gray)
    rails = sorted(set(rails).union(_accent_rails(frame, gray, rails, lines)),
                   key=lambda box: (-_rail_strength(gray, box), -box.width * box.height, box.top, box.left))
    for rail in rails:
        if not any(x0 <= rail.left < rail.right <= x1 and y0 <= rail.top < rail.bottom <= y1
                   for x0, y0, x1, y1 in regions):
            continue
        if any(_overlap(rail, group.rail) for group in candidates):
            continue
        matches = []
        for slots, labels in _place_labels(rail, lines):
            selected = _selected(frame, gray, rail, slots)
            if selected is not None:
                matches.append(_Group(rail, slots, labels, selected))
        if len(matches) == 1:
            candidates.append(matches[0])
    reads, crops = [], []
    for i, group in enumerate(candidates):
        for j, (slot, line) in enumerate(zip(group.slots, group.labels, strict=True)):
            if line is None:
                crop = _label_patch(frame, gray, slot)
                if crop is not None:
                    reads.append((i, j))
                    crops.append(crop)
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    recovered = {}
    if crops and recognize is not None:
        weak = tuple(_labels((dict(node, score=84) for node in nodes if 70 <= (node.get("score") or 0) < 84),
                             width, height))
        try:
            for (i, j), result in zip(reads, recognize(crops)):
                text = result.get("text") or ""
                score = result.get("score", 0)
                if _key(text) and (score >= 84 or score >= 74 and _matches_weak_label(weak, candidates[i].slots[j], text)):
                    recovered[i, j] = text
        except (RuntimeError, ValueError):
            pass
    actions = []
    for i, group in enumerate(candidates):
        labels = [line.text if line is not None else recovered.get((i, j)) for j, line in enumerate(group.labels)]
        if not all(labels) or len({_key(label) for label in labels}) != len(labels):
            continue
        for j, (slot, label, line) in enumerate(zip(group.slots, labels, group.labels, strict=True)):
            if line is not None and _ink_strength(gray, line) < 44:
                continue
            on = j == group.selected
            title = label.strip()
            inset_x, inset_y = max(2, round(.16 * slot.width)), max(2, round(.17 * slot.height))
            action = {"role": "select_segment", "setting_label": title, "setting_state": "on" if on else "off",
                      "function": f"选择{title}（{'当前已选中' if on else '当前未选中'}）",
                      "box": (slot.left + inset_x, slot.top + inset_y, slot.right - inset_x, slot.bottom - inset_y),
                      "center": slot.center}
            actions.append(action)
    return actions
