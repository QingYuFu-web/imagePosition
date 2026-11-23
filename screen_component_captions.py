"""Find caption-bearing controls from shared geometry and visible selection."""

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np


@dataclass(frozen=True)
class CaptionControl:
    region_id: int | None
    box: tuple
    caption_area: tuple
    purpose: str
    selected: bool = False
    disabled: bool = False
    click_point: tuple | None = None


def _tight(mask, left, top):
    x, y, w, h = cv2.boundingRect(np.uint8(mask))
    return (left + x, top + y, left + x + w, top + y + h) if w and h else None


def _range(box):
    left, top, right, bottom = box
    return right - left, bottom - top


def _bands(rows):
    return [tuple(map(int, limits)) for limits in np.flatnonzero(np.diff(np.r_[False, rows, False])).reshape(-1, 2)]


def _contrast(frame):
    patch = frame if frame.dtype == np.uint8 else frame.astype(np.int16)
    paint = np.median(patch[::2, ::3], axis=(0, 1))
    if patch.shape[2] != 3:
        return np.max(np.abs(patch - paint), axis=2), paint
    blue = np.abs(patch[..., 0] - paint[0])
    green = np.abs(patch[..., 1] - paint[1])
    red = np.abs(patch[..., 2] - paint[2])
    return np.maximum(np.maximum(blue, green), red), paint


def _outlier(samples, *, same_max, gap):
    values = np.asarray(samples, dtype=np.float32)
    distances = np.linalg.norm(values[:, None] - values, axis=2)
    candidates = []
    for i, row in enumerate(distances):
        others = np.arange(len(row)) != i
        if min(row[others]) >= gap and distances[np.ix_(others, others)].max() <= same_max:
            candidates.append(i)
    return candidates[0] if len(candidates) == 1 else None


def _captions_at(masks, contrasts, patches, band, spans, top, width):
    start, stop = band
    h = stop - start
    icon_start, icon_end = max(0, start - round(.090 * width)), max(0, start - round(.008 * width))
    separators = sum(mask.mean(axis=1) >= .64 for mask in masks) >= len(masks) - 1
    breaks = np.flatnonzero(separators[icon_start:icon_end])
    if breaks.size:
        icon_start += int(breaks[-1]) + 1
    proposals = []
    for ink, (left, right), distances, patch in zip(masks, spans, contrasts, patches, strict=True):
        caption = _tight(ink[start:stop], left, top + start)
        icon = _tight(ink[icon_start:icon_end], left, top + icon_start)
        if caption is None or icon is None:
            return []
        cw, ch = _range(caption)
        iw, ih = _range(icon)
        if (not .63 * h <= ch <= h or not .011 * width <= cw <= .163 * width
                or not .020 * width <= iw <= .104 * width or not .018 * width <= ih <= .078 * width
                or abs((caption[0] + caption[2]) / 2 - (icon[0] + icon[2]) / 2) > .044 * width
                or abs((caption[0] + caption[2] - left - right) / 2) > .033 * width):
            return []
        lx, ty, rx, by = caption
        values = distances[ty - top:by - top, lx - left:rx - left]
        foreground = values > 22
        level = np.percentile(values[foreground], 76)
        color = np.median(patch[ty - top:by - top, lx - left:rx - left][foreground & (values >= level)], axis=0)
        proposals.append((caption, icon, color))
    if np.ptp([icon[1] for _, icon, _ in proposals]) > .023 * width:
        return []
    return proposals


def _footer(frame, candidates):
    height, width = frame.shape[:2]
    bottom = [row for row in candidates if
              height - .22 * width <= row["center"][1] <= height - .022 * width and
              .074 * width <= _range(row["box"])[1] <= .196 * width and
              .078 * width <= _range(row["box"])[0] <= .28 * width]
    for count in range(6, 2, -1):
        edges = [round(i * width / count) for i in range(count + 1)]
        near = [[row for row in bottom if abs(row["center"][0] - (a + b) / 2) <= .038 * width]
                for a, b in zip(edges[:-1], edges[1:], strict=True)]
        if any(not rows for rows in near):
            continue
        anchors = [min(rows, key=lambda row: abs(row["center"][0] - (a + b) / 2))
                   for rows, a, b in zip(near, edges[:-1], edges[1:], strict=True)]
        tops = [row["box"][1] for row in anchors]
        if len({row["id"] for row in anchors}) != count or np.ptp(tops) > .026 * width:
            continue
        top = max(0, round(np.median(tops) - .010 * width))
        end = min(height, round(np.median(tops) + .146 * width))
        inset = round(.011 * width)
        spans = [(a + inset, b - inset) for a, b in zip(edges[:-1], edges[1:], strict=True)]
        patches = [frame[top:end, a:b] for a, b in spans]
        contrasts = [_contrast(patch)[0] for patch in patches]
        masks = [value >= 23 for value in contrasts]
        if any(mask.mean() >= .35 for mask in masks):
            continue
        active_rows = sum(mask.sum(axis=1) >= max(3, round(.004 * width)) for mask in masks) >= count - 1
        for start, stop in reversed(_bands(active_rows)):
            if not .008 * width <= stop - start <= .034 * width or start < .061 * width:
                continue
            margin = max(2, round(.004 * width))
            band = max(0, start - margin), min(end - top, stop + margin)
            labels = _captions_at(masks, contrasts, patches, band, spans, top, width)
            if not labels:
                continue
            selected = _outlier([color for _, _, color in labels], same_max=25, gap=39)
            if selected is None:
                continue
            result = []
            for i, (row, (caption, icon, _)) in enumerate(zip(anchors, labels, strict=True)):
                left, right = edges[i:i + 2]
                box = (left, max(top, icon[1] - round(.009 * width)), right,
                       min(height, caption[3] + round(.009 * width)))
                result.append(CaptionControl(row["id"], box, caption, "switch", i == selected))
            return result
    return []


def _painted_chip(frame, box):
    left, top, right, bottom = box
    w, h = _range(box)
    strip = frame[top:bottom, left + round(.20 * w):right - round(.20 * w)]
    width = strip.shape[1]
    if not width:
        return None
    paint = np.median(strip[round(.10 * h):round(.25 * h)], axis=(0, 1))
    gap = max(3, round(.26 * h))
    band = frame[top - gap:bottom + gap, left + round(.28 * w):right - round(.28 * w)]
    level = np.max(abs(band.astype(np.int16) - paint), axis=2)
    tolerance = 3 if np.std(paint) < 8 else 7
    rows = (level <= tolerance).mean(axis=1) >= .85
    fills = [span for span in _bands(rows) if span[1] - span[0] >= .09 * h]
    if (not fills or fills[0][0] <= 0 or fills[-1][1] >= len(rows)
            or not .83 * h <= fills[-1][1] - fills[0][0] <= 1.39 * h):
        return None
    return paint


def _chip_rail(frame, candidates):
    height, width = frame.shape[:2]
    anchors = [row for row in candidates if
               .045 * height <= row["center"][1] <= .25 * height and
               .095 * width <= _range(row["box"])[0] <= .29 * width and
               .050 * width <= _range(row["box"])[1] <= .112 * width]
    visited = set()
    for anchor in anchors:
        _, h = _range(anchor["box"])
        y = anchor["center"][1]
        group = sorted((row for row in anchors if
                        abs(row["center"][1] - y) <= .13 * h and .78 * h <= _range(row["box"])[1] <= 1.26 * h),
                       key=lambda row: row["box"][0])
        key = tuple(row["id"] for row in group)
        if key in visited or not 4 <= len(group) <= 7:
            continue
        visited.add(key)
        boxes = [row["box"] for row in group]
        if (boxes[-1][2] - boxes[0][0] < .72 * width or
                any(not -.12 * h <= b[0] - a[2] <= .40 * h for a, b in zip(boxes[:-1], boxes[1:], strict=True))):
            continue
        paints = [_painted_chip(frame, box) for box in boxes]
        if any(paint is None for paint in paints):
            continue
        selected = _outlier(paints, same_max=9, gap=15)
        if selected is None:
            continue
        return [CaptionControl(row["id"], tuple(box), _caption_center(box), "switch", i == selected)
                for i, (row, box) in enumerate(zip(group, boxes, strict=True))]
    return []


def _caption_center(box):
    left, top, right, bottom = box
    w, h = _range(box)
    return left + round(.06 * w), top + round(.20 * h), right - round(.06 * w), bottom - round(.20 * h)


def _rectangular_button(gray, box):
    left, top, right, bottom = box
    w, h = _range(box)
    pad = max(3, round(.18 * h))
    x, y = max(0, left - pad), max(0, top - pad)
    patch = gray[y:bottom + pad, x:right + pad]
    edges = cv2.Canny(patch, 17, 40)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        dx, dy, rw, rh = cv2.boundingRect(contour)
        if (.88 * w <= rw <= 1.09 * w and .82 * h <= rh <= 1.20 * h
                and abs(x + dx + rw / 2 - (left + right) / 2) <= .10 * h
                and abs(y + dy + rh / 2 - (top + bottom) / 2) <= .15 * h
                and cv2.contourArea(contour) >= .86 * rw * rh):
            return x + dx, y + dy, x + dx + rw, y + dy + rh
    return None


def _sheet_footer(frame, regions, candidates):
    if not regions["regions"]:
        return []
    left, top, right, bottom = regions["regions"][0]
    width = right - left
    peers = [row for row in candidates if
             bottom - .17 * width <= row["center"][1] <= bottom - .019 * width and
             .36 * width <= _range(row["box"])[0] <= .50 * width and
             .064 * width <= _range(row["box"])[1] <= .132 * width and
             left <= row["box"][0] < row["box"][2] <= right and top <= row["box"][1] < row["box"][3] <= bottom]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for first, second in combinations(sorted(peers, key=lambda row: row["box"][0]), 2):
        a, b = first["box"], second["box"]
        if not (.006 * width <= b[0] - a[2] <= .063 * width and
                max(abs(a[1] - b[1]), abs(a[3] - b[3])) <= .015 * width):
            continue
        measured = [_rectangular_button(gray, box) for box in (a, b)]
        if not all(box is not None and left <= box[0] < box[2] <= right
                   and top <= box[1] < box[3] <= bottom for box in measured):
            continue
        tones = [np.median(gray[y0:y1, x0:x1]) for x0, y0, x1, y1 in map(_caption_center, measured)]
        if abs(tones[0] - tones[1]) < 42:
            continue
        return [CaptionControl(row["id"], box, _caption_center(box), "press")
                for row, box in zip((first, second), measured, strict=True)]
    return _primary_footer(frame, gray, (left, top, right, bottom), candidates)


def _primary_footer(frame, gray, region, candidates):
    left, top, right, bottom = region
    width = right - left
    active = []
    for row in candidates:
        x, y = row["center"]
        w, h = _range(row["box"])
        if not (.78 * width <= w <= .97 * width and .067 * width <= h <= .138 * width and
                bottom - .18 * width <= y <= bottom - .035 * width and
                abs(x - (left + right) / 2) <= .06 * width):
            continue
        box = _rectangular_button(gray, row["box"])
        if box is None or not (left <= box[0] < box[2] <= right and top <= box[1] < box[3] <= bottom):
            continue
        x0, y0, x1, y1 = _caption_center(box)
        _, color = _contrast(frame[y0:y1, x0:x1])
        pad = max(3, round(.025 * width))
        outer = frame[max(top, box[1] - pad):box[1] - 2, x0:x1]
        if outer.size and np.ptp(color) >= 65 and np.max(abs(np.median(outer, axis=(0, 1)) - color)) >= 70:
            active.append(CaptionControl(row["id"], box, (x0, y0, x1, y1), "press"))
    return active if len(active) == 1 else []


def caption_controls(frame, regions):
    if regions["layer"] == "page":
        return _footer(frame, regions["candidates"]) + _chip_rail(frame, regions["candidates"])
    if regions["layer"] == "bottom_sheet":
        return _sheet_footer(frame, regions, [row for row in regions["candidates"] if row["visibility"] == "foreground"])
    return []
