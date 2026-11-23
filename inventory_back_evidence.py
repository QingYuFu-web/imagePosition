"""Recognize a leading Back chevron when the surrounding toolbar supports it."""

from __future__ import annotations

from dataclasses import dataclass
import re

import cv2
import numpy as np


_CLOCK = re.compile(r"(?:[01]?\d|2[0-3]):[0-5]\d")
_SEARCH_LABELS = {"搜索", "搜寻", "search"}


@dataclass(frozen=True)
class _Element:
    kind: str
    box: tuple[int, int, int, int]
    label: str
    confidence: float
    id: str

    @property
    def center(self):
        left, top, right, bottom = self.box
        return ((left + right) / 2, (top + bottom) / 2)

    @property
    def width(self):
        return self.box[2] - self.box[0]

    @property
    def height(self):
        return self.box[3] - self.box[1]


def _elements(observation, width, height):
    for point in observation.get("keypoints", ()):
        outline = point.get("outline_fraction")
        if not outline or point.get("type") not in ("text", "control_candidate"):
            continue
        xx, yy = zip(*outline)
        left, right = (round(x * (width - 1)) for x in (min(xx), max(xx)))
        top, bottom = (round(y * (height - 1)) for y in (min(yy), max(yy)))
        if 0 <= left < right < width and 0 <= top < bottom < height:
            yield _Element(point["type"], (left, top, right, bottom), point.get("label", ""),
                           point.get("confidence") or 0, point["id"])


def _chevrons(mask, center_y, min_height, max_height):
    """Find thin components whose x coordinate has a left-pointing vertex."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask), connectivity=8)
    for identifier, (left, top, width, height, area) in enumerate(stats[1:], start=1):
        if not (min_height <= height <= max_height and .38 <= width / height <= .74
                and .04 <= area / (width * height) <= .27
                and abs(center_y - (top + height / 2)) <= .24 * height):
            continue
        body = labels[top:top + height, left:left + width] == identifier
        y, x = body.nonzero()
        line_x = np.bincount(y, weights=x) / np.bincount(y)
        expected_x = np.abs(np.linspace(-1, 1, height))
        error = np.sqrt(np.mean((line_x / max(1, width - 1) - expected_x) ** 2))
        if error <= .105:
            yield (left, top, left + width - 1, top + height - 1)


def is_shaft_back_arrow(glyph):
    """Verify a left-pointing arrow with a horizontal shaft and two diagonal arms."""
    height, width = glyph.shape
    if height < 10 or not .83 <= width / height <= 1.23 or not .16 <= glyph.mean() <= .42:
        return False
    full = np.flatnonzero(glyph.sum(axis=1) >= .79 * width)
    if (not full.size or np.any(np.diff(full) != 1) or len(full) > .24 * height
            or abs(full.mean() + .5 - height / 2) > .11 * height):
        return False
    depth = round(.19 * height)
    arms = [np.nonzero(band)[1] for band in (glyph[:depth], glyph[-depth:])]
    if any(not arm.size or not .28 * width <= np.median(arm) <= .56 * width or arm.max() > .65 * width for arm in arms):
        return False
    shoulder = glyph[round(.22 * height):round(.36 * height)]
    return bool(shoulder.any() and np.median(np.nonzero(shoulder)[1]) <= .34 * width)


def suggest_wide_toolbar_back(frame, nodes):
    """Find a left chevron in the leading segment of a wide segmented toolbar."""
    height, width = frame.shape[:2]
    if width <= height:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    matches = []
    for node in nodes:
        left, top, right, bottom = node["box"]
        x, y = node["center"]
        w, h = right - left, bottom - top
        if not (node["kind"] in {"icon", "container"} and 0 <= x <= .28 * width and 0 <= y <= .19 * height
                and .020 * height <= h <= .115 * height and .38 <= w / h <= .74):
            continue
        gap = max(4, round(h * .18))
        pad_left, pad_top = max(0, left - gap), max(0, top - gap)
        first = right + gap
        edge = min(width, right + round(2.65 * h))
        strip = gray[top:bottom, first:edge]
        if strip.shape[1] < h:
            continue
        fill = np.median(strip)
        if (np.abs(strip - fill) <= 12).mean() < .90:
            continue
        separator_end = min(width, right + round(4.85 * h))
        following = gray[top:bottom, edge:separator_end]
        if following.shape[1] < 5:
            continue
        contrasting = (np.abs(following - fill) >= 24).mean(axis=0) >= .80
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(contrasting[None, :]))
        if not any(2 <= run <= max(3, .18 * h) for run in stats[1:, cv2.CC_STAT_WIDTH]):
            continue
        roi = gray[pad_top:min(height, bottom + gap), pad_left:min(width, right + gap)]
        marks = _chevrons(np.abs(roi - fill) > 52, y - pad_top, .80 * h, 1.24 * h)
        for x0, y0, x1, y1 in marks:
            matches.append((x0 + pad_left, y0 + pad_top, x1 + pad_left, y1 + pad_top))
    if len(matches) == 1:
        return matches[0]
    return None


def _bare_marks(gray, labels, clock):
    height, width = gray.shape
    for label in labels:
        x, y = label.center
        # A centered title or the leading text in a labeled search toolbar.
        is_title = .37 * width <= x <= .63 * width and label.width <= .33 * width
        is_input = .07 * width <= label.box[0] <= .24 * width and any(
            peer.label.strip().casefold() in _SEARCH_LABELS and peer.center[0] > .72 * width
            and abs(peer.center[1] - y) <= .5 * label.height for peer in labels)
        if not (is_title or is_input):
            continue
        right = round(min(.18 * width, 3.35 * clock.height, label.box[0] - .10 * label.height))
        top = max(0, round(y - 1.22 * label.height))
        bottom = min(height, round(y + 1.22 * label.height))
        if right <= 2 or bottom - top <= 2:
            continue
        roi = gray[top:bottom, :right]
        lower, upper = np.percentile(roi, (10, 90))
        if upper < 170 or upper - lower > 31:
            continue
        for x1, y1, x2, y2 in _chevrons(roi < upper - 53, y - top, .72 * label.height, 1.55 * label.height):
            if x1 > .25 * clock.height and x2 + .21 * clock.height < right:
                yield (x1, y1 + top, x2, y2 + top), None


def _round_marks(gray, controls, clock, roles):
    width = gray.shape[1]
    plain = [item for item in controls if item.label in ("", "未命名控件")
             and roles.get(item.id) in (None, "unknown", "back")
             and .79 <= item.width / item.height <= 1.24
             and 1.30 * clock.height <= item.height <= 2.38 * clock.height]
    for button in plain:
        x, y = button.center
        if x > .16 * width or not any(peer.center[0] > .72 * width and
                                     abs(peer.center[1] - y) <= .20 * button.height for peer in plain):
            continue
        left, top, right, bottom = button.box
        roi = gray[top:bottom + 1, left:right + 1]
        yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
        radii = np.hypot(yy - (y - top), xx - (x - left)) / min(button.width, button.height)
        inside = roi[(.34 <= radii) & (radii <= .43)]
        outside = roi[radii >= .62]
        if not inside.size or not outside.size:
            continue
        lo, mid, hi = np.percentile(inside, (8, 50, 92))
        if hi - lo > 22 or np.median(outside) - mid < 36:
            continue
        mask = (roi > mid + 55) & (radii < .43)
        for x1, y1, x2, y2 in _chevrons(mask, y - top, .49 * button.height, .80 * button.height):
            if abs((x1 + x2) / 2 + left - x) <= .15 * button.width:
                yield (x1 + left, y1 + top, x2 + left, y2 + top), button


def suggest_back_control(frame_bgr, observation: dict) -> dict | None:
    """Use the same screen frame as the existing observation, without a model.

    Requires an OCR-observed status clock and a structural cue in its next
    row. Modal frames and ambiguous leading marks yield no suggestion. The
    result contains normalized box and center, with ``anchor_id`` when a
    detected circular control already represents this symbol.
    """
    if observation.get("active_layer") or frame_bgr.ndim != 3:
        return None
    height, width = frame_bgr.shape[:2]
    if height <= width or width < 200:
        return None
    elements = list(_elements(observation, width, height))
    clock = min((item for item in elements if item.kind == "text" and item.confidence >= .85
                 and item.center[1] <= .085 * height and _CLOCK.fullmatch(item.label.strip())),
                key=lambda item: item.center[1], default=None)
    if clock is None:
        return None
    status_bottom = clock.box[3]
    row = [item for item in elements if status_bottom + .5 * clock.height <= item.center[1]
           <= status_bottom + 3.25 * clock.height]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    labels = [item for item in row if item.kind == "text" and item.confidence >= .88
              and len(item.label.strip()) >= 2 and not _CLOCK.fullmatch(item.label.strip())]
    controls = [item for item in row if item.kind == "control_candidate"]
    roles = {item["id"]: item.get("role") for item in observation.get("screen_context", {}).get("controls", ())}
    marks = dict(_bare_marks(gray, labels, clock))
    marks.update(_round_marks(gray, controls, clock, roles))
    if len(marks) != 1:
        return None
    mark, anchor = next(iter(marks.items()))
    left, top, right, bottom = anchor.box if anchor else mark
    x, y = (left + right) / 2, (top + bottom) / 2
    return {
        "role": "back", "function": "返回上一页",
        "box_fraction": [left / (width - 1), top / (height - 1), right / (width - 1), bottom / (height - 1)],
        "screen_fraction": [x / (width - 1), y / (height - 1)],
        "anchor_id": anchor.id if anchor else None,
    }
