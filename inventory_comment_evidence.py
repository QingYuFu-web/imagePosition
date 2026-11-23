"""Find a textless comment button on a corroborated video reaction rail."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re
import unicodedata

import cv2
import numpy as np


_COUNT = re.compile(r"\d+(?:[.,]\d+)*(?:[万亿kKmM])?\+?")
_MEDIA = re.compile(r"视频|直播|短剧|\b(?:videos?|shorts|reels|live)\b", re.IGNORECASE)
_AUTHOR = re.compile(r"@[\w.\-]{2,32}")


@dataclass(frozen=True)
class _Control:
    id: str
    box: tuple[int, int, int, int]

    @property
    def width(self):
        return self.box[2] - self.box[0]

    @property
    def height(self):
        return self.box[3] - self.box[1]

    @property
    def center(self):
        l, t, r, b = self.box
        return (l + r) / 2, (t + b) / 2


def _reliable_text(point, minimum):
    if point["type"] == "text" and (point.get("confidence") or 0) >= minimum:
        yield unicodedata.normalize("NFKC", point["label"]).strip()
    for fragment in point.get("text_evidence", ()):
        if (fragment.get("confidence") or 0) >= minimum and fragment.get("text"):
            yield unicodedata.normalize("NFKC", fragment["text"]).strip()


def _counted_controls(points, width, height, roles):
    for point in points:
        if (point.get("type") != "control_candidate" or point.get("interactivity") != "candidate"
                or roles.get(point["id"]) not in (None, "unknown", "like", "comments", "share", "favorite")):
            continue
        evidence = tuple(_reliable_text(point, .82))
        if len(evidence) != 1 or not _COUNT.fullmatch(evidence[0]) or not point.get("outline_fraction"):
            continue
        x, y = zip(*point["outline_fraction"])
        l, r = [round(v * (width - 1)) for v in (min(x), max(x))]
        t, b = [round(v * (height - 1)) for v in (min(y), max(y))]
        if not (0 <= l < r < width and 0 <= t < b < height):
            continue
        control = _Control(point["id"], (l, t, r, b))
        center_x, _ = control.center
        if (.042 * width <= control.width <= .26 * width and .43 <= control.width / control.height <= 1.6
                and min(center_x, width - center_x) <= .24 * width):
            yield control


def _same_rail(first, middle, last):
    xs, ys = zip(*(control.center for control in (first, middle, last)))
    row_height = min(control.height for control in (first, middle, last))
    column_width = min(control.width for control in (first, middle, last))
    top_gap, bottom_gap = ys[1] - ys[0], ys[2] - ys[1]
    return (max(xs) - min(xs) <= .23 * column_width and .70 * row_height <= min(top_gap, bottom_gap)
            and max(top_gap, bottom_gap) <= 1.5 * min(top_gap, bottom_gap)
            and max(top_gap, bottom_gap) <= 2.4 * row_height)


def _glyphs(gray, control):
    left, top, right, _ = control.box
    # The visual symbol belongs above its separate OCR-backed count.
    roi = gray[top:top + round(.65 * control.height), left:right + 1]
    threshold = max(171, float(np.median(roi)) + 47)
    contours, hierarchy = cv2.findContours(np.uint8(roi >= threshold), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return
    parents = hierarchy[0][:, 3]
    for i, contour in enumerate(contours):
        if parents[i] >= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if (w < .35 * control.width or h < .32 * roi.shape[0] or abs(x + w / 2 - control.width / 2) > .21 * control.width
                or x <= 0 or y <= 0 or x + w >= roi.shape[1] or y + h >= roi.shape[0]):
            continue
        area = cv2.contourArea(contour)
        if area >= .30 * w * h:
            yield (x, y, w, h), area, contour, [sub for sub, owner in zip(contours, parents) if owner == i]


def _speech_bubble(box, area, outline, holes):
    x, y, w, h = box
    if not (.79 <= w / h <= 1.48 and .62 <= area / (w * h) <= .93):
        return False
    dots = []
    for hole in holes:
        hx, hy, hw, hh = cv2.boundingRect(hole)
        dot_area = cv2.contourArea(hole)
        if .010 <= dot_area / area <= .072 and .71 <= hw / hh <= 1.40 and dot_area >= .53 * hw * hh:
            dots.append(((hx + hw / 2 - x) / w, (hy + hh / 2 - y) / h))
    if len(dots) != 3:
        return False
    dx, dy = zip(*sorted(dots))
    if (not all(.15 <= b - a <= .34 for a, b in zip(dx, dx[1:]))
            or not (.39 <= dx[1] <= .61 and .34 <= min(dy) <= max(dy) <= .64)
            or max(dy) - min(dy) > .12):
        return False
    pixels = outline.reshape(-1, 2)
    lower_edge = (pixels[pixels[:, 1] >= y + .90 * h, 0] - x) / w
    return lower_edge.size and min(lower_edge.min(), 1 - lower_edge.max()) <= .18


def suggest_comment_control(frame_bgr, observation: dict) -> dict | None:
    """Return one evidenced comment candidate on the same frame as observation.

    Requires a filled light three-dot speech bubble with a tail, peers with
    visible glyphs and numeric counters, and a separately observed video
    context. Outlined, uncounted, or partially obscured styles may abstain.
    """
    if observation.get("active_layer") or frame_bgr.ndim != 3:
        return None
    height, width = frame_bgr.shape[:2]
    if height <= width or width < 200:
        return None
    points = observation.get("keypoints", ())
    if not any(_MEDIA.search(text) for point in points for text in _reliable_text(point, .88)):
        return None
    authors = [point["screen_fraction"] for point in points if point.get("screen_fraction")
               and any(_AUTHOR.fullmatch(text) for text in _reliable_text(point, .88))]
    if not authors:
        return None
    roles = {item["id"]: item.get("role") for item in observation.get("screen_context", {}).get("controls", ())}
    counted = sorted(_counted_controls(points, width, height, roles), key=lambda control: control.center[1])
    if len(counted) < 3:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    glyphs = {control.id: list(_glyphs(gray, control)) for control in counted}
    matches = {}
    for first, middle, last in combinations(counted, 3):
        if (roles.get(middle.id) not in (None, "unknown", "comments")
                or not (_same_rail(first, middle, last) and glyphs[first.id] and glyphs[last.id])):
            continue
        left, top, _, _ = middle.box
        if not any(abs(ax * width - middle.center[0]) >= 2.4 * middle.width
                   and first.box[1] <= ay * height <= last.box[3] + .6 * last.height for ax, ay in authors):
            continue
        speech = [box for box, area, outline, holes in glyphs[middle.id] if _speech_bubble(box, area, outline, holes)]
        if len(speech) == 1:
            x, y, w, h = speech[0]
            scale = width - 1, height - 1
            matches[middle.id] = {
                "anchor_id": middle.id, "role": "comments", "function": "打开视频评论",
                "box_fraction": [(left + x) / scale[0], (top + y) / scale[1],
                                 (left + x + w - 1) / scale[0], (top + y + h - 1) / scale[1]],
                "screen_fraction": [(left + x + (w - 1) / 2) / scale[0], (top + y + (h - 1) / 2) / scale[1]],
            }
    return next(iter(matches.values())) if len(matches) == 1 else None
