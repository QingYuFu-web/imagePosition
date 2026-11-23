"""Verify a wide search field when only the field received a proposal."""

import cv2
import numpy as np

from screen_back_control import _center, _status_band, _title_run
from screen_candidate_glyphs import _overlap
from screen_search_owners import _bounds, _field_text, _surface


def _enclosed_surface(frame, box, glyph, paper):
    """Recover the true surface edges inside the proposal and its padded rim."""
    l, t, r, b = box
    span, depth = r - l, b - t
    border = round(.33 * depth)
    height, width = frame.shape[:2]
    left, top = max(0, l - border), max(0, t - border)
    patch = frame[top:min(height, b + border), left:min(width, r + border)]
    gx, gy = _center(glyph)
    groups = []
    for level in (3, 5):
        ink = cv2.inRange(patch, np.maximum(0, paper - level).astype(np.uint8),
                          np.minimum(255, paper + level).astype(np.uint8))
        outlines, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fields = []
        for contour in outlines:
            x, y, w, h = cv2.boundingRect(contour)
            if (not .82 * span <= w <= 1.08 * span or not .65 * depth <= h <= 1.21 * depth
                    or not 0 < x < x + w < patch.shape[1] or not 0 < y < y + h < patch.shape[0]
                    or not .927 <= cv2.contourArea(contour) / (w * h) <= .994
                    or not .24 * h <= gx - (left + x) <= 1.03 * h
                    or abs(gy - (top + y + h / 2)) > .16 * h):
                continue
            corner = max(3, round(.11 * h))
            if any(chip.mean() > 244 for chip in (ink[y:y + corner, x:x + corner],
                                                 ink[y:y + corner, x + w - corner:x + w])):
                continue
            fields.append((left + x, top + y, left + x + w, top + y + h))
        if len(fields) != 1:
            return None
        groups.extend(fields)
    if _overlap(*groups) < .93:
        return None
    x, y, r, b = groups[-1]
    h = b - y
    side = max(3, round(.09 * h))
    # A colored or monochrome plate still needs independent paint changes
    # outside the top and bottom boundaries, away from the rounded corners.
    edges = (frame[y - side:y, x + h:r - h:4], frame[b:b + side, x + h:r - h:4])
    if any(not strip.size or (np.max(abs(strip.astype(np.int16) - paper), axis=2) >= 8).mean() < .77
           for strip in edges):
        return None
    return groups[-1]


def _header_owns_field(frame, field):
    height, width = frame.shape[:2]
    a, t, r, b = field
    depth = b - t
    x, y = round(.023 * width), round(t - 1.52 * depth)
    header = frame[y:round(t - .13 * depth), x:round(.77 * width)]
    if not header.size:
        return False
    gray = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY)
    fill = np.median(gray)
    contrast = abs(gray.astype(np.int16) - fill)
    if (contrast <= 13).mean() < .87:
        return False
    run = _title_run(gray, fill, .73 * depth, minimum=4, high=1.66)
    if run is None:
        return False
    left, right = run
    _, upper, _, size = cv2.boundingRect(np.uint8(contrast[:, left:right] >= 43))
    start = x + left
    end = x + right
    if (not .134 * width <= end - start <= .66 * width or not .49 * depth <= size <= 1.16 * depth
            or not .15 * depth <= t - y - upper - size <= .73 * depth
            or (abs(start - a) > .27 * depth and abs((start + end) / 2 - width / 2) > .11 * width)):
        return False
    title = (start, y + upper, end, y + upper + size)
    if not _status_band(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), title):
        return False
    cap = frame[round(.067 * height):title[1] - round(.11 * depth),
                round(.052 * width):round(.95 * width):3].astype(np.int16)
    if not cap.size:
        return False
    surrounding = np.median(header[::3, ::5], axis=(0, 1))
    delta = np.max(abs(cap - surrounding), axis=2)
    return (delta <= 13).mean() >= .92 and (delta >= 20).mean(axis=1).max() <= .52


def wide_search_fields(frame, regions, index, *, diagnostic=False):
    """Run the shared magnifier and caption reader on top/heading-owned fields."""
    height, width = frame.shape[:2]
    if (index is None or regions.get("layer") != "page" or regions.get("regions")
            or width < 240 or not 1.45 * width <= height <= 2.83 * width):
        return []
    boxes = [tuple(map(int, row["box"])) for row in regions["candidates"] if row.get("visibility") == "page"
             and 0 <= row["box"][0] < row["box"][2] <= width and 0 <= row["box"][1] < row["box"][3] <= height]
    results, cache = [], {}
    for box in boxes:
        left, top, right, bottom = box
        depth = bottom - top
        if (not .52 * width <= right - left <= .995 * width or not .052 * width <= depth <= .171 * width
                or not .033 * height <= top <= .26 * height or min(left, width - right) < .007 * width):
            continue
        seed = (left + round(.07 * depth), top + round(.04 * depth),
                left + round(1.10 * depth), bottom - round(.065 * depth))
        glyph, paper = _bounds(frame, seed, width)
        if not glyph:
            continue
        if top <= .145 * height:
            field = _surface(frame, glyph, paper)
            if field is None or _overlap(field, box) < .58:
                continue
            witness = "top_field"
        else:
            field = _enclosed_surface(frame, box, glyph, paper)
            if field is None or not _header_owns_field(frame, field):
                continue
            witness = "titled_field"
        verified = _field_text(frame, field, glyph, paper, boxes, index, cache)
        if verified is None:
            continue
        owner, caption = verified
        location = _center(owner)
        if any(a <= x < r and t <= y < b for a, t, r, b in regions.get("floating_controls", ())
               for x, y in (location, _center(glyph))):
            continue
        if any(_overlap(owner, row["box"]) >= .89 for row in results):
            continue
        result = {"role": "search", "function": "搜索", "box": list(owner), "click_point": list(location)}
        if diagnostic:
            result.update({"witness": witness, "caption": caption, "glyph": list(glyph), "proposal": list(box)})
        results.append(result)
    return results if len(results) == 1 else []
