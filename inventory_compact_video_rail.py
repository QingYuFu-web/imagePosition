"""Identify a small video share glyph with its independently visible rail."""

from dataclasses import dataclass

import cv2
import numpy as np

from inventory_compact_video_text import read_compact_video_caption
from inventory_list_rows import _Box
from inventory_toolbar_actions import _spans


@dataclass(frozen=True)
class _Mark:
    box: _Box
    mask: np.ndarray
    level: int


def _footer_top(frame):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    margin = round(.04 * width)
    dark = (gray[:, margin:width - margin] <= 47).mean(axis=1) >= .70
    for top, end in reversed(_spans(dark)):
        if not (.83 * height <= top <= .955 * height and end >= round(.96 * height)
                and .038 * height <= end - top <= .15 * height):
            continue
        first_band = gray[top:top + round(.67 * (end - top))]
        columns = (first_band >= 121).sum(axis=0) >= max(2, round(.045 * (end - top)))
        centers = []
        for left, right in _spans(columns):
            if not max(1, round(.008 * width)) <= right - left <= .145 * width:
                continue
            center = (left + right) / 2
            if not centers or center - centers[-1] >= .072 * width:
                centers.append(center)
        if len(centers) >= 4 and centers[-1] - centers[0] >= .59 * width:
            return int(top)
    return None


def _marks(frame, footer):
    height, width = frame.shape[:2]
    size = round(.107 * width) | 1
    neighborhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    radius = max(1, round(.005 * width))
    close = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * radius + 1, 2 * radius + 1))
    left, top = round(.78 * width), round(.36 * height)
    right, bottom = round(.981 * width), footer - max(2, round(.016 * height))
    if right <= left or bottom <= top:
        return
    # Both passes of the opening stay within this halo around the rail.
    halo = size - 1
    padded_left, padded_top = max(0, left - halo), max(0, top - halo)
    light = frame[padded_top:bottom + halo, padded_left:right + halo].min(axis=2)
    ground = cv2.morphologyEx(light, cv2.MORPH_OPEN, neighborhood)
    rail_rows = slice(top - padded_top, bottom - padded_top)
    rail_cols = slice(left - padded_left, right - padded_left)
    bright = light[rail_rows, rail_cols]
    delta = cv2.subtract(bright, ground[rail_rows, rail_cols])
    for level, (brightness, lift) in enumerate(((117, 24), (126, 32), (145, 42), (158, 49))):
        ink = np.uint8((bright >= brightness) & (delta >= lift))
        connected = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, close)
        outlines, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for outline in outlines:
            x, y, w, h = cv2.boundingRect(outline)
            center = x + left + w / 2
            if (not (.035 * width <= h <= .116 * width and .038 * width <= w <= .125 * width
                     and .66 <= w / h <= 1.64 and .83 * width <= center <= .961 * width)
                    or not (0 < x < x + w < ink.shape[1] and 0 < y < y + h < ink.shape[0])):
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [outline], 0, 1, cv2.FILLED, offset=(-x, -y))
            if np.count_nonzero(mask & ink[y:y + h, x:x + w]) >= .22 * w * h:
                yield _Mark(_Box(left + x, top + y, left + x + w, top + y + h), mask, level)


def _speech(mask):
    height, width = mask.shape
    if not .84 <= width / height <= 1.60 or not .67 <= mask.mean() <= .94:
        return False
    top = mask[round(.12 * height):round(.64 * height)]
    toe = np.flatnonzero(mask[-1])
    return bool(np.median(top.sum(axis=1)) >= .83 * width
                and .62 * width <= mask[0].sum()
                and 0 < toe.size <= .46 * width
                and abs(toe.mean() / width - .5) >= .09)


def _curved_arrow(mask):
    height, width = mask.shape
    if not .82 <= width / height <= 1.57 or not .34 <= mask.mean() <= .72:
        return False
    left = np.nonzero(mask[:, :max(1, round(.20 * width))])[0]
    right = np.nonzero(mask[:, round(.81 * width):])[0]
    tail_height, tip_height = np.median(left) / height, np.median(right) / height
    if not (.51 <= tail_height <= .89 and .31 <= tip_height <= .62
            and tail_height - tip_height >= .08):
        return False
    above, below = np.flatnonzero(mask[0]), np.flatnonzero(mask[-1])
    if not (above.size <= .56 * width and below.size <= .50 * width
            and .23 * width <= above.mean() <= .84 * width
            and .23 * width <= below.mean() <= .84 * width):
        return False
    columns = mask.sum(axis=0)
    peaks = np.flatnonzero(columns >= .86 * columns.max())
    return bool(.34 * width <= np.median(peaks) <= .80 * width
                and np.median(columns[round(.81 * width):]) <= .66 * columns.max())


def _stable(marks, predicate):
    possible = [mark for mark in marks if predicate(mark.mask)]
    return [mark for mark in possible if any(
        peer.level != mark.level
        and max(abs(a - b) for a, b in zip(mark.box.xyxy(), peer.box.xyxy(), strict=True))
        <= max(1, .19 * mark.box.height)
        for peer in possible)]


def _neighbor(mark, anchor, span, pitch, center_x, offset):
    x, y = mark.box.center
    return (.61 * span <= mark.box.height <= 1.58 * span and .63 <= mark.mask.shape[1] / mark.mask.shape[0] <= 1.58
            and .35 <= mark.mask.mean() <= .94 and abs(x - center_x) <= .29 * span
            and abs((y - anchor.box.center[1]) / pitch - offset) <= .37)


def _one_position(marks):
    if not marks:
        return None
    first = min(marks, key=lambda mark: mark.level)
    if any(max(abs(a - b) for a, b in zip(first.box.center, mark.box.center, strict=True)) > .31 * first.box.height
           for mark in marks):
        return None
    return first


def find_compact_video_share_box(frame, observation, *, line_reader=None):
    """Return one half-open share box backed by stable glyph and peer geometry."""
    if frame.ndim != 3 or frame.shape[2] != 3 or observation.get("active_layer"):
        return None
    height, width = frame.shape[:2]
    if not (128 <= width <= 420 and 1.60 * width <= height <= 2.65 * width):
        return None
    footer = _footer_top(frame)
    if footer is None:
        return None
    marks = list(_marks(frame, footer))
    bubbles = _stable(marks, _speech)
    arrows = _stable(marks, _curved_arrow)
    matches = []
    labelled = []
    for arrow in arrows:
        x, y = arrow.box.center
        for comment in bubbles:
            cx, cy = comment.box.center
            span = (arrow.box.height + comment.box.height) / 2
            pitch = y - cy
            if not (1.72 * span <= pitch <= 4.05 * span and abs(x - cx) <= .24 * span
                    and .61 * span <= arrow.box.height <= 1.38 * comment.box.height):
                continue
            axis = (x + cx) / 2
            if not any(_neighbor(mark, comment, span, pitch, axis, -1) for mark in marks):
                continue
            if any(_neighbor(mark, arrow, span, pitch, axis, 1) for mark in marks):
                matches.append(arrow)
                break
            if line_reader is not None and any(_neighbor(mark, comment, span, pitch, axis, -2) for mark in marks):
                labelled.append(arrow)
                break
    if matches:
        selected = _one_position(matches)
        return selected.box.xyxy() if selected is not None else None
    selected = _one_position(labelled)
    if selected is None:
        return None
    box = selected.box
    padding = max(1, round(.10 * box.height))
    caption = read_compact_video_caption(
        frame, (box.left - padding, box.top - padding, box.right + padding, box.bottom + padding), line_reader)
    if caption and not caption.is_count and caption.text.casefold() in {"share", "分享"}:
        return box.xyxy()
    return None
