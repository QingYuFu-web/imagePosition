"""Recognize neutral bright short-video strokes across different video backgrounds."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from statistics import mean

import cv2
import numpy as np

from inventory_compact_video_rail import _Mark, _curved_arrow, _footer_top, _speech
from inventory_list_rows import _Box
from inventory_outlined_short_video_reactions import _split_curls, _verified_remix
from inventory_small_reaction_glyphs import _thumb_shape, _wrist
from screen_foreground import detect_blocking_overlay


_LIGHT_LEVELS = (237, 242, 244, 246, 250)
_ROLES = ("like", "dislike", "comments", "share")
_THUMB_FAMILIES = {
    "like": ("filled-like", "outlined-like"),
    "dislike": ("filled-dislike", "outlined-dislike"),
}
_FUNCTIONS = {"like": "点赞", "comments": "打开视频评论", "share": "打开分享菜单"}


@dataclass(frozen=True)
class _Stroke:
    mark: _Mark
    first_hole: int


def _turned(box):
    return _Box(-box.right, -box.bottom, -box.left, -box.top)


def _palm_hole(contours, links, stroke):
    child = stroke.first_hole
    width, height = stroke.mark.box.width, stroke.mark.box.height
    while child >= 0:
        contour = contours[child]
        _, _, w, h = cv2.boundingRect(contour)
        if w >= .40 * width and h >= .41 * height and cv2.contourArea(contour) >= .22 * width * height:
            return True
        child = links[child][0]
    return False


def _thumbs(stroke, pieces, contours, links):
    mark = stroke.mark
    box = mark.box
    aspect = box.width / box.height
    if not .56 < aspect < 1.08:
        return
    for role, shape, turn in (("like", mark.mask, False), ("dislike", np.rot90(mark.mask, 2), True)):
        if not _thumb_shape(shape):
            continue
        if aspect >= .76 and _palm_hole(contours, links, stroke):
            yield f"outlined-{role}"
            continue
        main = _turned(box) if turn else box
        cuffs = [piece.mark for piece in pieces if _wrist(main, _turned(piece.mark.box) if turn else piece.mark.box)
                 and piece.mark.mask.mean() >= .68]
        if len(cuffs) == 1:
            yield f"filled-{role}"


def _pieces(light, level, left, top, right):
    width = light.shape[1]
    ink = np.uint8(light[top:, left:right] >= level)
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], (), ()
    links = hierarchy[0]
    result = []
    for contour, (_, _, first_hole, parent) in zip(contours, links, strict=True):
        if parent >= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not (.01 * width <= w <= .15 * width and max(5, .025 * width) <= h <= .14 * width):
            continue
        mask = cv2.drawContours(np.zeros((h, w), np.uint8), [contour], 0, 1, cv2.FILLED, offset=(-x, -y))
        result.append(_Stroke(_Mark(_Box(left + x, top + y, left + x + w, top + y + h), mask, level), first_hole))
    return result, contours, links


def _measure(light, level, boundaries):
    pieces, contours, links = _pieces(light, level, *boundaries)
    for stroke in pieces:
        mark = stroke.mark
        width, height = mark.box.width, mark.box.height
        if width < .033 * light.shape[1] or not .49 <= width / height <= 1.65:
            continue
        if _speech(mark.mask):
            yield "comments", mark
        if _curved_arrow(mark.mask):
            yield "share", mark
        for role in _thumbs(stroke, pieces, contours, links):
            yield role, mark
        if _split_curls(mark, light):
            yield "remix", mark


def _same_stroke(first, second):
    overlap = (max(0, min(first.right, second.right) - max(first.left, second.left))
               * max(0, min(first.bottom, second.bottom) - max(first.top, second.top)))
    return overlap >= .72 * (first.width * first.height + second.width * second.height - overlap)


def _stable_strokes(marks):
    stable = []
    for mark in marks:
        if (any(peer.level != mark.level and _same_stroke(mark.box, peer.box) for peer in marks)
                and all(not _same_stroke(mark.box, prior.box) for prior in stable)):
            stable.append(mark)
    return stable


def _aligned(positions, boxes):
    first_y, last_y = boxes[0].center[1], boxes[-1].center[1]
    start = positions[0]
    pitch = (last_y - first_y) / (positions[-1] - start)
    span = mean(box.height for box in boxes)
    if not 1.70 * span <= pitch <= 4.10 * span:
        return None
    axis = mean(box.center[0] for box in boxes)
    if any(abs(x - axis) > .27 * span or abs(y - (first_y + (position - start) * pitch)) > .19 * pitch
           for position, (x, y) in zip(positions, (box.center for box in boxes), strict=True)):
        return None
    # A lower glyph or genuine footer can corroborate this column.
    remix_y = first_y + (len(_ROLES) - start) * pitch
    return axis, remix_y, span, pitch


def _unique_rail(working, light, stable):
    roles = {
        role: [mark.box for family in _THUMB_FAMILIES.get(role, (role,)) for mark in stable.get(family, ())]
        for role in _ROLES
    }
    if sum(bool(boxes) for boxes in roles.values()) < 3:
        return None
    footer = _footer_top(working)
    curls = stable.get("remix", ())
    for count in (4, 3):
        confirmed = []
        for positions in combinations(range(len(_ROLES)), count):
            for boxes in product(*(roles[_ROLES[position]] for position in positions)):
                layout = _aligned(positions, boxes)
                if layout is None:
                    continue
                axis, remix_y, span, pitch = layout
                if not .5 * span <= remix_y <= light.shape[0] - .5 * span:
                    continue
                close_footer = footer is not None and .70 * pitch < footer - remix_y <= 2.5 * pitch
                if close_footer or _verified_remix(light, curls, axis, remix_y, span, pitch):
                    confirmed.append({_ROLES[i]: box for i, box in zip(positions, boxes, strict=True)})
                    if len(confirmed) > 1:
                        return None
        if confirmed:
            return confirmed[0]
    return None


def suggest_bright_short_video_strokes(frame, observation):
    """Return measured video controls on a uniquely corroborated bright column."""
    if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
            or observation.get("active_layer") or observation.get("mode") == "home"):
        return []
    height, width = frame.shape[:2]
    if width < 128 or not 1.60 * width <= height <= 2.65 * width:
        return []
    working = frame
    if width > 400:
        working = cv2.resize(frame, (400, round(height * 400 / width)), interpolation=cv2.INTER_AREA)
    height, width = working.shape[:2]
    light = cv2.min(cv2.min(working[..., 0], working[..., 1]), working[..., 2])
    bounds = (round(.77 * width), round(.33 * height), round(.99 * width))
    measured = defaultdict(list)
    for cutoff in _LIGHT_LEVELS:
        for role, mark in _measure(light, cutoff, bounds):
            measured[role].append(mark)
    stable = {role: _stable_strokes(marks) for role, marks in measured.items()}
    rail = _unique_rail(working, light, stable)
    if rail is None or detect_blocking_overlay(frame) is not None:
        return []
    sx, sy = width - 1, height - 1
    return [{"role": role, "function": _FUNCTIONS[role], "anchor_id": None,
             "box_fraction": [box.left / sx, box.top / sy, box.right / sx, box.bottom / sy],
             "screen_fraction": [box.center[0] / sx, box.center[1] / sy]}
            for role, box in rail.items() if role in _FUNCTIONS]
