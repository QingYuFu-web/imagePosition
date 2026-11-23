"""Verify Search captions within measured input surfaces or visible drawer rows."""

import cv2
import numpy as np

from screen_back_control import _center, _shape_masks, _status_band
from screen_candidate_glyphs import _overlap
from screen_caption_transcript import _bands, read_caption
from screen_component_captions import _contrast
from screen_search_owners import _SEARCH, _bounds, _peer_mark, _surface
from screen_wide_search_fields import _enclosed_surface, _header_owns_field


def _boxes(regions, width, height):
    visible = "page" if regions["layer"] == "page" else "foreground"
    for item in regions["candidates"]:
        a, t, r, b = map(int, item["box"])
        if item.get("visibility") == visible and 0 <= a < r <= width and 0 <= t < b <= height:
            yield a, t, r, b


def _read(frame, bounds, index, cache):
    return read_caption(frame, bounds, index, cache=cache, choice=True, join_letters=True,
                        join_radicals=True, sentence_case=True)


def _outlined_surface(frame, box, glyph, paint):
    left, top, right, bottom = box
    depth, width = bottom - top, right - left
    border = round(.24 * depth)
    a, t = max(0, left - border), max(0, top - border)
    roi = frame[t:min(frame.shape[0], bottom + border), a:min(frame.shape[1], right + border)]
    distance = np.max(abs(roi.astype(np.int16) - paint), axis=2)
    findings = []
    gx, gy = _center(glyph)
    for level in (13, 21):
        ink = np.uint8(distance >= level)
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enclosed = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (not .86 * width <= w <= 1.10 * width or not .70 * depth <= h <= 1.22 * depth
                    or cv2.contourArea(contour) < .91 * w * h or not 0 < x < x + w < roi.shape[1]
                    or not 0 < y < y + h < roi.shape[0]
                    or not .23 * h <= gx - (a + x) <= 1.03 * h or abs(gy - (t + y + h / 2)) > .13 * h):
                continue
            pad, margin = max(3, round(.064 * h)), round(.15 * h)
            strips = (ink[y:y + pad, x + h:x + w - h], ink[y + h - pad:y + h, x + h:x + w - h],
                      ink[y + margin:y + h - margin, x:x + pad],
                      ink[y + margin:y + h - margin, x + w - pad:x + w])
            if any(not strip.size or (strip.max(axis=0 if i < 2 else 1) != 0).mean() < .93
                   for i, strip in enumerate(strips)):
                continue
            paper = distance[y + pad:y + h - pad, x + pad:x + w - pad]
            if (paper <= 8).mean() < .91:
                continue
            enclosed.append((a + x, t + y, a + x + w, t + y + h))
        if len(enclosed) != 1:
            return None
        findings.extend(enclosed)
    return findings[0] if _overlap(*findings) >= .97 else None


def _side_owner(frame, box, *, drawer=False):
    """A flush, uninterrupted seam with a substantially darker exterior."""
    height, width = frame.shape[:2]
    a, top, r, _ = box
    if drawer:
        columns = range(max(4, r - round(.025 * width)), min(width - 4, r + round(.025 * width)))
        inward = -1
    else:
        columns = range(max(4, a - round(.095 * width)), a - round(.016 * width))
        inward = 1
    lo = round(min(top / height + .042, .36) * height)
    end = round(.91 * height)
    if lo >= end or len(columns) == 0:
        return None
    # Sample luminance from the same rows on both sides of a narrow seam.
    swatch = cv2.cvtColor(frame[lo:end:9, columns.start - 3:columns.stop + 3].copy(), cv2.COLOR_BGR2GRAY)
    votes = inward * (swatch[:, 3:-3].astype(np.int16) - swatch[:, 2:-4])
    coverage = (votes >= 47).mean(axis=0)
    if coverage.max(initial=0) < .90:
        return None
    seam = columns.start + int(np.argmax(coverage))
    if not .035 * width <= seam <= .87 * width:
        return None
    stride = max(4, round(.017 * width))
    low, high = sorted((seam + inward * stride, seam + inward * 3 * stride))
    outer_a, outer_b = sorted((seam - inward * stride, seam - inward * 3 * stride))
    inside = frame[lo:end:12, max(0, low):min(width, high):3]
    outside = frame[lo:end:12, max(0, outer_a):min(width, outer_b):3]
    if not inside.size or not outside.size:
        return None
    paint = np.median(inside.reshape(-1, 3), axis=0)
    if paint.mean() < 182 or (np.max(abs(inside - paint), axis=2) <= 10).mean() < .86:
        return None
    if np.percentile(outside, 91) > paint.mean() - 47:
        return None
    return seam


def _caret_caption(frame, area, paper, depth, index, cache):
    """Read the surviving five glyphs when a colored caret covers the first."""
    left, top, right, bottom = area
    if np.ptp(paper) > 15:
        return None
    end = min(right, left + round(3.55 * depth))
    patch = frame[top:bottom, left:end]
    colored = np.uint8((np.ptp(patch, axis=2) >= 42) & (np.max(abs(patch - paper), axis=2) > 62))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(colored, connectivity=8)
    lines = [(i, (int(a), int(t), int(w), int(h))) for i, (a, t, w, h, area_pixels) in enumerate(stats[1:], 1)
             if 0 <= a < .62 * depth and .37 * depth <= h <= .62 * depth and 2 <= w <= .13 * h
             and area_pixels >= .88 * w * h and .077 * patch.shape[0] <= t <= .34 * patch.shape[0]]
    if len(lines) != 1:
        return None
    identity, (a, t, w, h) = lines[0]
    if colored.sum() - np.sum(labels == identity) > .08 * w * h:
        return None
    clean = patch.copy()
    clean[labels == identity] = paper
    dark, _ = _contrast(clean)
    ink = dark >= max(34, .38 * float(np.percentile(dark, 99)))
    spans = _bands(ink.any(axis=0))
    if len(spans) < 6:
        return None
    first, first_end = spans[0]
    if not (a + .55 * w <= first <= a + w + max(2, .05 * h)
            and .24 * h <= first_end - first <= .57 * h):
        return None
    first_height = cv2.boundingRect(np.uint8(ink[:, first:first_end]))[3]
    if not .50 * h <= first_height <= .81 * h:
        return None
    reading = _read(frame, (left + first_end, top, end, bottom), index, cache)
    if not reading or reading["text"].casefold() != "earch" or reading["support"] < 3:
        return None
    # The first visibly clipped glyph touches the caret. Only its suffix is
    # transcribed, and the independent magnifier supplies the meaning.
    _, t1, r1, b1 = reading["line_box"]
    return {**reading, "line_box": [left + first, t1, r1, b1]}


def _caption(frame, field, glyph, paper, index, cache, *, width_factor=.65, caret=False, inset=.04):
    a, t, r, b = field
    depth = b - t
    _, gy = _center(glyph)
    x0 = glyph[2] + max(2, round(inset * depth))
    y0, y1 = round(gy - .30 * depth), round(gy + .32 * depth)
    x1 = r - round(.10 * depth)
    if y0 < t or y1 >= b or x1 - x0 < 1.60 * depth:
        return None
    area = (x0, y0, x1, y1)
    # Restrict percentile estimation to the actual leading word. A wide
    # span of blank paint otherwise hides sparse placeholder text entirely.
    tail = min(x1, x0 + round(3.7 * depth))
    local = frame[y0:y1, x0:tail]
    off = np.max(abs(local - paper), axis=2)
    bands = _bands((off >= 32).sum(axis=0) >= 1)
    window = None
    if bands:
        for previous, following in zip(bands, bands[1:]):
            if (previous[1] - bands[0][0]) >= .90 * depth and following[0] - previous[1] >= .12 * depth:
                window = previous[1] + min(round(.06 * depth), (following[0] - previous[1]) // 2)
                break
    head = min(x1, x0 + window) if window else tail
    reading = _read(frame, (x0, y0, head, y1), index, cache)
    if not reading or not _SEARCH.match(reading["text"].casefold().strip()):
        reading = _read(frame, area, index, cache)
    if not reading or not _SEARCH.match(reading["text"].casefold().strip()):
        reading = _caret_caption(frame, area, paper, depth, index, cache) if caret else None
    if not reading:
        return None
    l, top, _, bottom = reading["line_box"]
    if (not 0 <= l - x0 <= width_factor * depth or not .18 * depth <= bottom - top <= .59 * depth
            or abs((top + bottom) / 2 - gy) > .13 * depth):
        return None
    return reading


def _peer(frame, field, glyph, paper, boxes):
    """Prove a separate same-height control immediately outside a deep field."""
    _, top, right, bottom = field
    depth = bottom - top
    _, y = _center(glyph)
    for box in boxes:
        left, t, r, b = box
        width, height = r - left, b - t
        if (-.05 * depth <= left - right <= .47 * depth
                and .40 * depth <= min(width, height) <= max(width, height) <= 1.32 * depth
                and abs((b + t) / 2 - y) < .16 * depth):
            peer_paper = np.median(frame[t:b:2, left:r:2].reshape(-1, 3), axis=0)
            if np.max(abs(peer_paper - paper)) <= 27 and _peer_mark(frame, box, peer_paper):
                return True
    return False


def _drawer_peers(box, seam, boxes):
    left, top, right, bottom = box
    height = bottom - top
    candidates = [peer for peer in boxes if peer[1] >= bottom - .05 * height and abs(peer[0] - left) <= .07 * height
                  and abs(peer[2] - seam) <= .20 * height and .78 * height <= peer[3] - peer[1] <= 1.22 * height]
    return sum(peer[1] - top < 4.5 * height for peer in candidates) >= 2


def owned_search(frame, regions, index, *, diagnostic=False):
    """Find one visibly owned search component using proposals and local pixels."""
    height, width = frame.shape[:2]
    layer = regions.get("layer")
    if (index is None or layer not in {"page", "side_drawer"}
            or not 1.45 * width <= height <= 2.83 * width or width < 240):
        return []
    boxes = list(_boxes(regions, width, height))
    cache = {}
    results = []
    for box in boxes:
        left, top, right, bottom = box
        span, depth = right - left, bottom - top
        if (not .52 * width <= span <= .995 * width or not .049 * width <= depth <= .171 * width
                or not .033 * height <= top <= .31 * height):
            continue
        seed = (left + round(.07 * depth), top + round(.04 * depth),
                left + round(1.10 * depth), bottom - round(.065 * depth))
        glyph, paper = _bounds(frame, seed, width)
        kind, field = None, None
        if glyph and left <= .008 * width:
            seam = _side_owner(frame, box, drawer=True)
            if seam and abs(seam - right) <= .18 * depth and _drawer_peers(box, seam, boxes):
                field, kind = (left, top, seam, bottom), "drawer_row"
        elif glyph and layer == "page":
            if top <= .145 * height:
                field = _surface(frame, glyph, paper)
                if field and _overlap(field, box) >= .58:
                    kind = "top_field"
            if kind is None and left >= .041 * width:
                outlined = _outlined_surface(frame, box, glyph, paper)
                if outlined and _side_owner(frame, outlined):
                    field, kind = outlined, "outlined_field"
            if kind is None and .145 * height < top:
                surface = _enclosed_surface(frame, box, glyph, paper)
                if surface and (_header_owns_field(frame, surface) or _peer(frame, surface, glyph, paper, boxes)):
                    field, kind = surface, "grouped_field"
        elif layer == "page" and top <= .115 * height and left <= .060 * width and right >= .94 * width:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marks = _shape_masks(gray, seed, width)
            if marks and all(_overlap(marks[0], mark) >= .82 for mark in marks[1:]) and _status_band(gray, marks[0]):
                glyph = marks[0]
                surface = _surface(frame, glyph, paper)
                if surface and _overlap(surface, box) >= .70:
                    field, kind = surface, "focused_header"
        if kind is None:
            continue
        reading = _caption(frame, field, glyph, paper, index, cache, caret=kind == "top_field",
                           width_factor=1.14 if kind == "drawer_row" else .68,
                           inset=.13 if kind == "focused_header" else .04)
        if reading is None:
            continue
        owner = tuple(map(int, field if kind != "focused_header" else (reading["line_box"][0], field[1], *field[2:])))
        point = list(_center(owner))
        if regions.get("regions") and not any(a <= point[0] < r and t <= point[1] < b
                                              for a, t, r, b in regions["regions"]):
            continue
        if any(a <= x < r and t <= y < b for x, y in (point, _center(glyph))
               for a, t, r, b in regions.get("floating_controls", ())):
            continue
        if any(_overlap(owner, row["box"]) >= .86 for row in results):
            continue
        element = {"role": "search", "function": "搜索", "box": list(owner), "click_point": point}
        if diagnostic:
            element.update({"witness": kind, "caption": reading["text"], "caption_box": reading["line_box"],
                            "glyph": list(glyph), "proposal": list(box)})
        results.append(element)
    return results if len(results) == 1 else []
