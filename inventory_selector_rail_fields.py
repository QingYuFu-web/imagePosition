"""Recover selector enclosures from directional edges around a nearby verified field."""

from dataclasses import dataclass

import cv2
import numpy as np

from inventory_list_rows import _Box
from inventory_toolbar_actions import _spans


@dataclass(frozen=True)
class _Rail:
    left: int
    y: int
    right: int
    coverage: float


def _horizontal_rails(delta, height):
    # Follow a bright rim along one axis. Opposite signed edges can straddle a
    # two-pixel highlight, even when the rim does not join at the corners.
    pulse = (np.maximum(delta[:-3], delta[1:-2]) >= 12) & (np.minimum(delta[2:-1], delta[3:]) <= -12)
    width = max(3, 2 * round(.13 * height) + 1)
    connected = cv2.morphologyEx(pulse.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, width), np.uint8))
    for index, row in enumerate(connected):
        for left, right in _spans(row != 0):
            if left < 2 or right >= row.size - 1 or right - left < 2.32 * height:
                continue
            coverage = float(pulse[index, left:right].mean())
            if coverage >= .72:
                yield _Rail(int(left), index + 1, int(right), coverage)


def _upright(delta, top, bottom, start, stop):
    start, stop = max(1, start), min(stop, delta.shape[1] - 1)
    if start >= stop or bottom - top < 5:
        return None
    strip = delta[top:bottom, start:stop]
    support = np.maximum((strip >= 13).mean(axis=0), (strip <= -13).mean(axis=0))
    best = int(np.argmax(support))
    return (start + best, float(support[best])) if support[best] >= .75 else None


def _same_field(first, second):
    h = min(first.height, second.height)
    return (max(abs(first.top - second.top), abs(first.bottom - second.bottom)) <= .26 * h
            and max(abs(first.left - second.left), abs(first.right - second.right)) <= .26 * h)


def _nearby_boxes(gray, anchor):
    h = anchor.height
    y0 = max(0, anchor.top - round(3.0 * h))
    y1 = min(gray.shape[0], anchor.bottom + round(3.0 * h))
    x0 = max(0, anchor.left - round(.62 * h))
    x1 = min(gray.shape[1], anchor.right + max(round(2.5 * h), round(.34 * anchor.width)))
    patch = gray[y0:y1, x0:x1].astype(np.int16)
    dx = np.diff(patch, axis=1, prepend=patch[:, :1])
    dy = np.diff(patch, axis=0, prepend=patch[:1])
    rails = list(_horizontal_rails(dy, h))
    for index, top in enumerate(rails):
        if abs(top.left + x0 - anchor.left) > .56 * h:
            continue
        for bottom in rails[index + 1:]:
            gap = bottom.y - top.y
            if gap > 1.25 * h:
                break
            if (gap < .68 * h or abs(top.left - bottom.left) > .42 * gap
                    or abs(top.right - bottom.right) > .42 * gap):
                continue
            inset = max(2, round(.14 * gap))
            search = round(.36 * gap)
            inside = max(2, round(.18 * gap))
            start, stop = top.y + inset, bottom.y - inset
            left = _upright(dx, start, stop, min(top.left, bottom.left) - search,
                            max(top.left, bottom.left) + inside)
            right = _upright(dx, start, stop, min(top.right, bottom.right) - inside,
                             max(top.right, bottom.right) + search)
            if left is None or right is None:
                continue
            box = _Box(x0 + left[0] - 1, y0 + top.y - 1, x0 + right[0] + 1, y0 + bottom.y + 2)
            if (abs(box.left - anchor.left) > .26 * h or not .76 <= box.height / h <= 1.30
                    or not 2.7 * box.height <= box.width <= 1.65 * anchor.width
                    or abs(box.center[1] - anchor.center[1]) > 3.2 * h):
                continue
            yield box, min(top.coverage, bottom.coverage, left[1], right[1])


def selector_rail_fields(gray, strict_fields):
    """Yield full-frame _Box candidates, without inferring an arrow or field name."""
    anchors = {}
    maximum_height = .11 * min(gray.shape)
    for field, mark in strict_fields.items():
        if 11 <= field.height <= maximum_height and field.width >= 2.7 * field.height:
            anchors.setdefault(mark, field)
    candidates = {}
    for anchor in anchors.values():
        for field, score in _nearby_boxes(gray, anchor):
            if not any(_same_field(field, known) for known in strict_fields):
                candidates[field] = max(score, candidates.get(field, 0))
    found = []
    for field in sorted(candidates, key=lambda item: (-candidates[item], item.top, item.left, item.height, item.width)):
        if not any(_same_field(field, previous) for previous in found):
            found.append(field)
            yield field
