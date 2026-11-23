"""Suggest list rows and their overflow controls from repeated visual cues."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np


@dataclass(frozen=True)
class _Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @property
    def center(self):
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2

    def xyxy(self):
        return self.left, self.top, self.right, self.bottom


@dataclass(frozen=True)
class _Mark:
    box: _Box
    fill: float
    mask: np.ndarray


def _labels(nodes, width, height, *, max_height_fraction=.11):
    span = min(width, height)
    for node in nodes:
        label = str(node.get("content") or "").strip()
        score = node.get("score") or 0
        if (node.get("kind") != "text" or not 1 <= len(label) <= 128 or score < 84
                or not any(char.isalnum() for char in label)):
            continue
        box = _Box(*map(round, node["box"]))
        if (0 <= box.left < box.right <= width and 0 <= box.top < box.bottom <= height
                and max(8, .007 * span) <= box.height <= max_height_fraction * span):
            yield box, label, score


def _runs(lines):
    """Find equally spaced lines sharing a left edge and text scale."""
    result = {}
    lanes = set()
    for anchor, _, _ in lines:
        lane = [index for index, (box, _, _) in enumerate(lines)
                if .72 <= box.height / anchor.height <= 1.38
                and abs(box.left - anchor.left) <= .32 * anchor.height]
        lane.sort(key=lambda index: lines[index][0].center[1])
        key = tuple(lane)
        if len(lane) < 5 or key in lanes:
            continue
        lanes.add(key)
        for start in range(len(lane) - 4):
            first_y = lines[lane[start]][0].center[1]
            for after in range(start + 1, len(lane) - 3):
                pitch = lines[lane[after]][0].center[1] - first_y
                if pitch < 1.7 * anchor.height:
                    continue
                if pitch > 7 * anchor.height:
                    break
                selected = [lane[start], lane[after]]
                index = after + 1
                while index < len(lane):
                    expected = first_y + len(selected) * pitch
                    y = lines[lane[index]][0].center[1]
                    if y < expected - .12 * pitch:
                        index += 1
                    elif y > expected + .12 * pitch:
                        break
                    else:
                        selected.append(lane[index])
                        index += 1
                if len(selected) >= 5:
                    result[tuple(selected)] = [lines[index] for index in selected]
    return sorted(result.values(), key=len, reverse=True)


def _marks(gray, box):
    chip = gray[box.top:box.bottom, box.left:box.right]
    if min(chip.shape) < 4:
        return []
    rim = np.concatenate((chip[0], chip[-1], chip[:, 0], chip[:, -1])).astype(np.int16)
    background = np.median(rim)
    noise = np.median(np.abs(rim - background))
    foreground = np.uint8(np.abs(chip.astype(np.int16) - background) >= max(24, 4 * noise + 8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    marks = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if min(width, height) < 3 or x == 0 or y == 0 or x + width >= chip.shape[1] or y + height >= chip.shape[0]:
            continue
        marks.append(_Mark(_Box(box.left + x, box.top + y, box.left + x + width, box.top + y + height),
                           area / (width * height), components[y:y + height, x:x + width] == index))
    return marks


def _leading(gray, label):
    h = label.height
    cy = label.center[1]
    box = _Box(max(0, round(label.left - 4.75 * h)), max(0, round(cy - .88 * h)),
               max(0, round(label.left - .42 * h)), min(gray.shape[0], round(cy + .88 * h)))
    for mark in _marks(gray, box):
        shape = mark.box
        if (.42 * h <= shape.width <= 1.85 * h and .43 * h <= shape.height <= 1.55 * h
                and .12 <= mark.fill <= .93 and .55 <= shape.width / shape.height <= 1.9
                and abs(shape.center[1] - cy) <= .42 * h):
            silhouette = cv2.resize(mark.mask.astype(np.uint8), (24, 24), interpolation=cv2.INTER_NEAREST) != 0
            yield shape, silhouette


def _repeat_leading(leading, label_height):
    best = []
    for reference, shape in (item for row in leading for item in row):
        aligned = []
        for row in leading:
            peers = []
            for box, candidate in row:
                if (abs(box.center[0] - reference.center[0]) > .28 * label_height
                        or not .74 <= box.width / reference.width <= 1.35
                        or not .74 <= box.height / reference.height <= 1.35):
                    continue
                score = np.count_nonzero(shape & candidate) / np.count_nonzero(shape | candidate)
                if score >= .70:
                    peers.append((score, box))
            aligned.append(max(peers, key=lambda peer: peer[0])[1] if peers else None)
        if sum(box is not None for box in aligned) > sum(box is not None for box in best):
            best = aligned
    return best


def _overflow(gray, label):
    """Measure strokes of a vertical-ellipsis glyph independently of the row glyph."""
    h = label.height
    cy = label.center[1]
    band = _Box(min(gray.shape[1], round(label.right + .28 * h)), max(0, round(cy - .85 * h)),
                gray.shape[1], min(gray.shape[0], round(cy + .85 * h)))
    dots = [mark.box for mark in _marks(gray, band)
            if .10 * h <= mark.box.width <= .42 * h and .10 * h <= mark.box.height <= .42 * h
            and .68 <= mark.box.width / mark.box.height <= 1.43 and .56 <= mark.fill <= .98]
    for middle in dots:
        x, y = middle.center
        size = (middle.width + middle.height) / 2
        if abs(y - cy) > .40 * h:
            continue
        lane = [box for box in dots if abs(box.center[0] - x) <= .34 * size
                and .72 <= (box.width + box.height) / (2 * size) <= 1.39]
        upper = [box for box in lane if 1.08 * size <= y - box.center[1] <= 2.45 * size]
        lower = [box for box in lane if 1.08 * size <= box.center[1] - y <= 2.45 * size]
        if len(upper) != 1 or len(lower) != 1:
            continue
        a, c = upper[0], lower[0]
        if abs((y - a.center[1]) - (c.center[1] - y)) > .34 * size:
            continue
        yield (x, y), _Box(min(a.left, middle.left, c.left), a.top,
                           max(a.right, middle.right, c.right), c.bottom)


def _repeat_overflow(candidates, h):
    best = []
    for (x, _), _ in (item for row in candidates for item in row):
        aligned = [min((item for item in row if abs(item[0][0] - x) <= .31 * h),
                       key=lambda item: abs(item[0][0] - x), default=None) for row in candidates]
        if sum(item is not None for item in aligned) > sum(item is not None for item in best):
            best = aligned
    return best


def _list_actions(gray, lines):
    h = median(box.height for box, _, _ in lines)
    glyphs = _repeat_leading([list(_leading(gray, box)) for box, _, _ in lines], h)
    if not glyphs or any(glyph is None for glyph in glyphs):
        return []
    options = _repeat_overflow([list(_overflow(gray, box)) for box, _, _ in lines], h)
    if sum(option is not None for option in options) < max(4, .72 * len(lines)):
        return []
    present = [option for option in options if option is not None]
    option_x = median(center[0] for center, _ in present)
    option_size = median(box.height for _, box in present)
    pitch = median(np.diff([box.center[1] for box, _, _ in lines]))
    row_left = max(0, round(median(glyph.left for glyph in glyphs) - .5 * h))
    row_right = round(option_x - .88 * option_size - max(.16 * pitch, .45 * h))
    if row_right <= max(box.right for box, _, _ in lines):
        return []
    image_h, image_w = gray.shape
    actions = []
    for index, ((label, text, _), glyph, option) in enumerate(zip(lines, glyphs, options, strict=True)):
        if option is None and index not in (0, len(lines) - 1):
            continue
        half_row = min(.34 * pitch, max(1.18 * glyph.height, 1.15 * label.height))
        _, y = glyph.center
        end = row_right if option else min(row_right, round(option_x - max(.60 * pitch, 2.4 * h)))
        if end <= label.right:
            continue
        actions.append({"role": "open_list_item", "function": f"打开{text}", "center": label.center,
                        "box": (row_left, max(0, round(y - half_row)), end, min(image_h, round(y + half_row)))})
        if option is None:
            continue
        (x, y), dots = option
        half_w = max(.88 * dots.height, .72 * h)
        half_y = min(.30 * pitch, max(1.10 * dots.height, .88 * h))
        actions.append({"role": "row_more", "function": f"{text}的更多选项", "center": (x, y),
                        "box": (max(end + 1, round(x - half_w)), max(0, round(y - half_y)),
                                min(image_w, round(x + half_w)), min(image_h, round(y + half_y)))})
    return actions


def _suggest_in(frame_bgr, nodes):
    height, width = frame_bgr.shape[:2]
    runs = _runs(list(_labels(nodes, width, height)))
    if not runs:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    found = []
    occupied = []
    for lines in runs:
        top, bottom = lines[0][0].top, lines[-1][0].bottom
        if any(top < end and bottom > start for start, end in occupied):
            continue
        actions = _list_actions(gray, lines)
        if actions:
            occupied.append((top, bottom))
            found.extend(actions)
    return found


def suggest_list_rows(frame_bgr, nodes, *, overlay=None) -> list[dict]:
    """Return visible rows of a regular labeled list and individually visible menus.

    Requires repeated matching leading glyphs and a separate column of actual
    vertical ellipses. Region bounds are conservative approximations and click
    anchors come from visible labels or the middle ellipsis dot. With an active
    overlay, all evidence and proposed regions must be inside its main panel.
    """
    if overlay is None:
        return _suggest_in(frame_bgr, nodes)
    height, width = frame_bgr.shape[:2]
    left, top, right, bottom = overlay.panel
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return []
    local_nodes = []
    for node in nodes:
        if node.get("kind") != "text":
            continue
        x0, y0, x1, y1 = node["box"]
        if left <= x0 < x1 <= right and top <= y0 < y1 <= bottom:
            local_nodes.append({**node, "box": (x0 - left, y0 - top, x1 - left, y1 - top)})
    result = []
    for action in _suggest_in(frame_bgr[top:bottom, left:right], local_nodes):
        x0, y0, x1, y1 = action["box"]
        cx, cy = action["center"]
        result.append({**action, "box": (x0 + left, y0 + top, x1 + left, y1 + top),
                       "center": (cx + left, cy + top)})
    return result
