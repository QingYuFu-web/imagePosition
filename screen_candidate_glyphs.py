"""Measure common glyphs inside proposed UI regions using screenshot pixels."""

import cv2
import numpy as np

from inventory_back_evidence import _chevrons, is_shaft_back_arrow
from inventory_comment_evidence import _speech_bubble
from inventory_share_glyph import is_share_glyph
from inventory_share_video import _curved_share
from inventory_toolbar_actions import _holes, _is_search


_LIGHT_LEVELS = (176, 192, 208, 222, 235, 244, 250)


def _overlap(first, second):
    x0, y0, x1, y1 = first
    a0, b0, a1, b1 = second
    area = max(0, min(x1, a1) - max(x0, a0)) * max(0, min(y1, b1) - max(y0, b0))
    return area / ((x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - area)


def _search(mask):
    holes = _holes(mask)
    if _is_search(mask, holes):
        return True
    height, width = mask.shape
    if len(holes) != 1 or not .08 <= mask.mean() <= .29:
        return False
    _, _, hole_width, hole_height = cv2.boundingRect(holes[0])
    if (not .84 <= hole_width / hole_height <= 1.18
            or not .42 < cv2.contourArea(holes[0]) / (height * width) <= .59):
        return False
    # Test a thin lens stroke at comparable thickness to a regular toolbar icon.
    radius = max(1, round(.07 * min(height, width)))
    diameter = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    thick = cv2.dilate(np.pad(mask, radius), kernel)
    return _is_search(thick, _holes(thick))


def _centered_bubble(mask, area, bounds, holes):
    """Verify a filled three-dot speech bubble with a centered bottom tail."""
    x, y, width, height = bounds
    if len(holes) != 3 or not (.87 <= width / height <= 1.30 and .71 <= area / (width * height) <= .92):
        return False
    points = []
    for hole in holes:
        left, top, w, h = cv2.boundingRect(hole)
        pixels = cv2.contourArea(hole)
        if not (.70 <= w / h <= 1.43 and .01 <= pixels / area <= .06 and pixels >= .52 * w * h):
            return False
        points.append(((left + w / 2 - x) / width, (top + h / 2 - y) / height))
    xs, ys = zip(*sorted(points))
    toe = np.nonzero(mask[round(.90 * height):])[1]
    return (all(.19 <= b - a <= .33 for a, b in zip(xs, xs[1:])) and .41 <= xs[1] <= .59
            and .35 <= min(ys) <= max(ys) <= .62 and max(ys) - min(ys) <= .09
            and toe.size and .30 * width <= toe.min() <= .47 * width
            and .53 * width <= toe.max() <= .70 * width
            and np.median(mask[round(.24 * height):round(.66 * height)].sum(axis=1)) >= .80 * width)


def _glyphs(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blue, green, red = cv2.split(crop)
    maximum = cv2.max(cv2.max(blue, green), red)
    minimum = cv2.min(cv2.min(blue, green), red)
    neutral = cv2.subtract(maximum, minimum) <= 38
    polarities = (gray, 255 - gray) if np.mean(neutral & (gray >= 175)) >= .55 else (gray,)
    for polarity, luminance in enumerate(polarities):
        previous_count = 0
        components = ()
        for level in _LIGHT_LEVELS:
            ink = np.uint8(neutral & (luminance >= level))
            count = cv2.countNonZero(ink)
            # Raising the threshold can only remove pixels. An unchanged count
            # means the mask and its measured shapes are identical.
            if count != previous_count:
                components = tuple(_components(ink)) if count else ()
                previous_count = count
            for role, box in components:
                yield role, (polarity, level), box


def _components(ink):
    crop_height, crop_width = ink.shape
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return
    parents = hierarchy[0][:, 3]
    for index, outline in enumerate(contours):
        if parents[index] >= 0:
            continue
        x, y, width, height = bounds = cv2.boundingRect(outline)
        if (not (0 < x < x + width < crop_width and 0 < y < y + height < crop_height)
                or not .24 * crop_width <= width <= .96 * crop_width
                or not .22 * crop_height <= height <= .98 * crop_height):
            continue
        fill = cv2.drawContours(np.zeros((height, width), np.uint8), [outline], 0, 1,
                                cv2.FILLED, offset=(-x, -y))
        mask = fill & ink[y:y + height, x:x + width]
        area = cv2.contourArea(outline)
        holes = [contour for contour, parent in zip(contours, parents, strict=True) if parent == index]
        box = x, y, x + width, y + height
        if is_shaft_back_arrow(mask) or any(_chevrons(mask, height / 2, height, height)):
            yield "back", box
        if _search(mask):
            yield "search", box
        if _speech_bubble(bounds, area, outline, holes) or _centered_bubble(fill, area, bounds, holes):
            yield "comments", box
        if _curved_share((None, bounds, area, outline, holes)):
            yield "share", box


def _share_carrier(gray, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if not .79 <= width / height <= 1.26:
        return None
    if is_share_glyph(gray, box):
        return box
    crop = gray[top:bottom, left:right]
    surface = float(np.percentile(crop, 85))
    if surface < 172:
        return None
    matches = []
    for lift in (42, 61, 81):
        contours, _ = cv2.findContours(np.uint8(crop <= surface - lift), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (not (0 < x < x + w < width and 0 < y < y + h < height)
                    or not .66 * width <= w <= .97 * width or not .66 * height <= h <= .97 * height
                    or not .84 <= w / h <= 1.17):
                continue
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, True)
            if not .65 <= area / (w * h) <= .87 or 4 * np.pi * area < .78 * length * length:
                continue
            part = left + x, top + y, left + x + w, top + y + h
            if is_share_glyph(gray, part):
                matches.append((lift, part))
    stable = [part for lift, part in matches if any(other != lift and _overlap(part, peer) >= .80 for other, peer in matches)]
    if stable and all(_overlap(stable[0], part) >= .80 for part in stable):
        return stable[len(stable) // 2]
    return None


class CandidateGlyphs:
    """Share pixel measurements across recognizers analyzing the same frame."""

    def __init__(self, frame):
        self._frame = frame
        self._gray = None
        self._measured = {}

    def _measure(self, bounds):
        left, top, right, bottom = bounds
        height, width = self._frame.shape[:2]
        short = min(height, width)
        w, h = right - left, bottom - top
        if (not 0 <= left < right <= width or not 0 <= top < bottom <= height
                or not max(12, .015 * short) <= min(w, h) <= max(w, h) <= .24 * short):
            return None
        marks = list(_glyphs(self._frame[top:bottom, left:right]))
        stable = [(role, box) for role, (polarity, level), box in marks if any(
            other_role == role and other_polarity == polarity and other_level != level and _overlap(box, other_box) >= .76
            for other_role, (other_polarity, other_level), other_box in marks)]
        if self._gray is None:
            self._gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
        share = _share_carrier(self._gray, bounds)
        if share is not None and all(role == "share" for role, _ in stable):
            return "share", share
        if stable:
            role, box = stable[len(stable) // 2]
            if all(other_role == role and _overlap(box, other_box) >= .76 for other_role, other_box in stable):
                x0, y0, x1, y1 = box
                return role, (left + x0, top + y0, left + x1, top + y1)
        return None

    def __call__(self, candidates):
        results = []
        for row in candidates:
            box = tuple(row["box"])
            if box not in self._measured:
                self._measured[box] = self._measure(box)
            if (mark := self._measured[box]) is not None:
                role, bounds = mark
                results.append({"id": row["id"], "role": role, "box": list(bounds)})
        return results


def candidate_glyphs(frame, candidates):
    """Return stable glyph boxes in original pixels with their candidate IDs."""
    return CandidateGlyphs(frame)(candidates)
