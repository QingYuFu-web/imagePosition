"""Verify a curved share arrow on a filled, individually counted video rail."""

from itertools import product

import cv2
import numpy as np

from inventory_compact_video_rail import _Mark, _curved_arrow
from inventory_compact_video_text import read_compact_video_caption
from inventory_list_rows import _Box
from inventory_toolbar_actions import _spans


def _footer(frame):
    height, width = frame.shape[:2]
    start, stop = round(.84 * height), round(.98 * height)
    margin = round(.055 * width)
    gray = cv2.cvtColor(frame[start:stop], cv2.COLOR_BGR2GRAY)
    quiet = (gray[:, margin:width - margin] < 50).mean(axis=1) >= .70
    closed = cv2.morphologyEx(np.uint8(quiet)[:, None], cv2.MORPH_CLOSE,
                              np.ones((2 * max(1, round(.002 * height)) + 1, 1), np.uint8))[:, 0]
    for first, last in reversed(_spans(closed > 0)):
        y = int(first + start)
        if not (.86 * height <= y <= .953 * height and last - first >= .035 * height
                and last >= stop - start - round(.012 * height) and quiet[first:last].mean() >= .83):
            continue
        extent = height - y
        strip = gray[first + round(.04 * extent):first + round(.51 * extent)].astype(np.int16)
        if not strip.size:
            continue
        profile = ((strip - np.median(strip, axis=1, keepdims=True)) > 38).sum(axis=0) >= max(1, round(.05 * extent))
        width_kernel = 2 * max(1, round(.01 * width)) + 1
        connected = cv2.morphologyEx(np.uint8(profile)[None, :], cv2.MORPH_CLOSE, np.ones((1, width_kernel), np.uint8))[0]
        centers = [(left + right) / 2 for left, right in _spans(connected > 0)
                   if .016 * width <= right - left <= .16 * width and .015 * width <= left < right <= .985 * width]
        if len(centers) == 5 and .045 * width <= centers[0] <= .17 * width and .83 * width <= centers[-1] <= .955 * width:
            gaps = np.diff(centers)
            if gaps.min() >= .15 * width and gaps.max() <= .25 * width:
                return y
    return None


def _rail_marks(frame, footer):
    height, width = frame.shape[:2]
    left, right = round(.81 * width), round(.978 * width)
    top = round(.38 * height)
    bottom = footer - round(.022 * height)
    size = round(.073 * width) | 1
    halo = size - 1
    ox, oy = max(0, left - halo), max(0, top - halo)
    plane = frame[oy:bottom + halo, ox:right + halo].min(axis=2)
    background = cv2.morphologyEx(plane, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    roi = (slice(top - oy, bottom - oy), slice(left - ox, right - ox))
    bright, gain = plane[roi], cv2.subtract(plane[roi], background[roi])
    radius = max(1, round(.004 * width))
    close = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * radius + 1, 2 * radius + 1))
    for level, (floor, lift) in enumerate(((182, 26), (198, 32), (208, 34))):
        mask = np.uint8((bright >= floor) & (gain >= lift))
        connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (not .049 * width <= h <= .115 * width or not .052 * width <= w <= .129 * width
                    or not .74 <= w / h <= 1.55 or not .84 * width <= left + x + w / 2 <= .952 * width
                    or min(x, y) == 0 or x + w >= mask.shape[1] or y + h >= mask.shape[0]):
                continue
            silhouette = np.zeros((h, w), np.uint8)
            cv2.drawContours(silhouette, [contour], 0, 1, cv2.FILLED, offset=(-x, -y))
            if .34 <= silhouette.mean() <= .95 and (silhouette & mask[y:y + h, x:x + w]).sum() >= .33 * w * h:
                yield _Mark(_Box(left + x, top + y, left + x + w, top + y + h), silhouette, level)


def _heart(mask):
    height, width = mask.shape
    overlap = (mask & mask[:, ::-1]).sum() / (mask | mask[:, ::-1]).sum()
    if not .90 <= width / height <= 1.40 or overlap < .72 or not .62 <= mask.mean() <= .86:
        return False
    center = slice(round(.43 * width), round(.57 * width))
    lobes = mask[:max(2, round(.20 * height))]
    divided = ((lobes[:, :round(.42 * width)].sum(axis=1) >= .18 * width)
               & (lobes[:, round(.58 * width):].sum(axis=1) >= .18 * width)
               & (lobes[:, center].sum(axis=1) <= .05 * width))
    tip = np.flatnonzero(mask[-1])
    return bool(divided.any() and np.median(mask[round(.25 * height):round(.55 * height)].sum(axis=1)) >= .86 * width
                and 0 < tip.size <= .43 * width and abs(tip.mean() + .5 - width / 2) <= .18 * width)


def _bookmark(mask):
    height, width = mask.shape
    if not .73 <= width / height <= 1.12 or not .75 <= mask.mean() <= .94:
        return False
    center = slice(round(.35 * width), round(.65 * width))
    near_bottom = mask[round(.79 * height):]
    left, right = near_bottom[:, :round(.23 * width)], near_bottom[:, round(.77 * width):]
    return bool(np.median(mask[round(.22 * height):round(.63 * height)].sum(axis=1)) >= .92 * width
                and near_bottom[-2:, center].mean() <= .14 and min(left.mean(), right.mean()) >= .55)


def _bubble(frame, mark):
    box, mask = mark.box, mark.mask
    height, width = mask.shape
    if not .83 <= width / height <= 1.22 or not .67 <= mask.mean() <= .91:
        return False
    chip = frame[box.top:box.bottom, box.left:box.right].min(axis=2).astype(np.int16)
    core = np.zeros_like(mask)
    core[round(.28 * height):round(.63 * height), round(.12 * width):round(.90 * width)] = 1
    core &= cv2.erode(mask, np.ones((3, 3), np.uint8))
    if not core.any():
        return False
    ground = np.median(chip[core != 0])
    dots = np.uint8((ground - chip >= 27) & (core != 0))
    _, _, stats, centers = cv2.connectedComponentsWithStats(dots)
    points = sorted((x / width, y / height) for (_, _, w, h, area), (x, y) in zip(stats[1:], centers[1:], strict=True)
                    if max(2, .005 * width * height) <= area <= .066 * width * height
                    and .55 <= w / h <= 1.75)
    if len(points) != 3:
        return False
    xs, ys = zip(*points)
    return max(ys) - min(ys) <= .095 and .40 <= xs[1] <= .61 and all(.15 <= step <= .29 for step in np.diff(xs))


def _stable_locations(marks, verify):
    accepted = [mark for mark in marks if verify(mark)]
    stable = []
    for mark in accepted:
        if any(max(abs(a - b) for a, b in zip(mark.box.center, other.box.center, strict=True)) <= .24 * mark.box.height
               for other in stable):
            continue
        if any(peer.level != mark.level and max(abs(a - b) for a, b in zip(peer.box.xyxy(), mark.box.xyxy(), strict=True))
               <= max(1, .16 * mark.box.height) for peer in accepted):
            stable.append(mark)
    return stable


def _aligned(rail, footer):
    boxes = [mark.box for mark in rail]
    xs, ys = zip(*(box.center for box in boxes))
    span = float(np.median([box.height for box in boxes]))
    gaps = np.diff(ys)
    pitch = float(np.median(gaps))
    return (max(xs) - min(xs) <= .24 * span and min(box.height for box in boxes) >= .68 * span
            and max(box.height for box in boxes) <= 1.35 * span and 1.76 * span <= pitch <= 3.28 * span
            and max(abs(gaps - pitch)) <= .17 * pitch and .74 * pitch <= footer - ys[-1] <= 2.5 * pitch)


def find_filled_video_rail(frame, observation, *, line_reader=None):
    """Return the proven heart, speech bubble, bookmark, and share glyphs."""
    if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or line_reader is None
            or observation.get("active_layer")):
        return None
    height, width = frame.shape[:2]
    if not (150 <= width <= 1024 and 1.68 * width <= height <= 2.60 * width):
        return None
    footer = _footer(frame)
    if footer is None:
        return None
    marks = list(_rail_marks(frame, footer))
    peers = (_stable_locations(marks, lambda m: _heart(m.mask)), _stable_locations(marks, lambda m: _bubble(frame, m)),
             _stable_locations(marks, lambda m: _bookmark(m.mask)), _stable_locations(marks, lambda m: _curved_arrow(m.mask)))
    rails = [rail for rail in product(*peers) if _aligned(rail, footer)]
    if len(rails) != 1:
        return None
    _, bubble, bookmark, share = rails[0]
    counts = 0
    for mark, following in ((share, None), (bubble, bookmark), (bookmark, share)):
        caption = read_compact_video_caption(frame, mark.box, line_reader, following_box=following.box if following else None)
        if caption is not None and caption.is_count:
            counts += 1
            if counts == 2:
                return rails[0]
    return None


def find_filled_video_share_box(frame, observation, *, line_reader=None):
    rail = find_filled_video_rail(frame, observation, line_reader=line_reader)
    return rail[-1].box.xyxy() if rail is not None else None
