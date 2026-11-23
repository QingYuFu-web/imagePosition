"""Ground typed video controls with independent numeric captions below their column."""

from collections import defaultdict
from itertools import combinations
from statistics import mean

import cv2
import numpy as np

from inventory_bright_short_video_strokes import _LIGHT_LEVELS, _pieces, _stable_strokes, _thumbs
from inventory_compact_video_rail import _curved_arrow, _marks, _speech
from inventory_compact_video_text import _COUNT, _normalized
from inventory_filled_video_rail import _stable_locations
from inventory_list_rows import _Box
from inventory_toolbar_actions import _spans
from screen_foreground import detect_blocking_overlay


_FUNCTIONS = {"like": "点赞", "comments": "打开视频评论", "share": "打开分享菜单"}
_CAPTION_LEVELS = (174, 198, 220, 235)


def _left_foot_arrow(mask):
    """Verify a rightward share arrow with a lower-left terminal stroke."""
    height, width = mask.shape
    if not (.82 <= width / height <= 1.57 and .34 <= mask.mean() <= .72):
        return False
    tail = np.nonzero(mask[:, :max(1, round(.20 * width))])[0]
    tip = np.nonzero(mask[:, round(.81 * width):])[0]
    if not tail.size or not tip.size:
        return False
    tail_y, tip_y = np.median(tail) / height, np.median(tip) / height
    if not (.63 <= tail_y <= .90 and .31 <= tip_y <= .62 and tail_y - tip_y >= .15):
        return False
    top, bottom = np.flatnonzero(mask[0]), np.flatnonzero(mask[-1])
    if not (0 < top.size <= .37 * width and 0 < bottom.size <= .30 * width
            and .32 * width <= top.mean() <= .74 * width and bottom.mean() <= .21 * width):
        return False
    columns = mask.sum(axis=0)
    peak = columns.max()
    broad = np.flatnonzero(columns >= .86 * peak)
    return bool(.32 * width <= np.median(broad) <= .80 * width
                and np.median(columns[round(.81 * width):]) <= .71 * peak)


def _same_location(first, second):
    span = max(first.height, second.height)
    fx, fy = first.center
    sx, sy = second.center
    return abs(fx - sx) <= .26 * span and abs(fy - sy) <= .23 * span


def _bright_types(light):
    height, width = light.shape
    crop = (round(.77 * width), round(.34 * height), round(.99 * width))
    roles = defaultdict(list)
    for level in _LIGHT_LEVELS:
        pieces, contours, links = _pieces(light, level, *crop)
        for stroke in pieces:
            mark = stroke.mark
            if mark.box.width >= .033 * width and _speech(mark.mask):
                roles["comments"].append(mark)
            # A physically matched wrist can corroborate a narrow thumb body.
            for family in _thumbs(stroke, pieces, contours, links):
                if family in {"filled-like", "outlined-like"}:
                    roles[family].append(mark)
    return {role: [mark.box for mark in _stable_strokes(pool)] for role, pool in roles.items()}


def _measured_types(light, marks):
    bright = _bright_types(light)
    return {
        "like": [*bright.get("filled-like", ()), *bright.get("outlined-like", ())],
        "comments": [mark.box for mark in _stable_locations(marks, lambda mark: _speech(mark.mask))]
                    + bright.get("comments", []),
        "share": [mark.box for mark in _stable_locations(
            marks, lambda mark: _curved_arrow(mark.mask) or _left_foot_arrow(mark.mask))],
    }


def _positions(marks):
    positions = []
    for mark in sorted(marks, key=lambda item: item.level, reverse=True):
        if all(not _same_location(mark.box, other) for other in positions):
            positions.append(mark.box)
    return sorted(positions, key=lambda box: box.center[1])


def _columns(positions):
    for first, middle, last in combinations(positions, 3):
        boxes = first, middle, last
        first_x, first_y = first.center
        middle_x, middle_y = middle.center
        last_x, last_y = last.center
        span = mean(box.height for box in boxes)
        pitch = (last_y - first_y) / 2
        if (1.50 * span <= pitch <= 4.65 * span
                and max(first_x, middle_x, last_x) - min(first_x, middle_x, last_x) <= .33 * span
                and abs(middle_y - first_y - pitch) <= .16 * pitch
                and min(box.height for box in boxes) >= .73 * span):
            yield boxes, pitch


def _typed_boxes(boxes, types):
    matches = {}
    for anchor in boxes:
        roles = [(role, next(box for box in measured if _same_location(anchor, box)))
                 for role, measured in types.items() if any(_same_location(anchor, box) for box in measured)]
        if len(roles) != 1:
            continue
        role, box = roles[0]
        if role in matches:
            return {}
        matches[role] = box
    return matches


def _upward_hand(mask):
    height, width = mask.shape
    top = mask[:max(2, round(.28 * height))]
    raised = np.flatnonzero(top.any(axis=0))
    palm = mask[round(.48 * height):round(.83 * height)]
    return bool(.57 <= mask.mean() <= .80 and .14 * width <= np.median(top.sum(axis=1)) <= .47 * width
                and .28 * width <= raised.mean() <= .66 * width
                and np.median(palm.sum(axis=1)) >= .82 * width)


def _outline_thumb(light, anchor):
    margin = max(2, round(.13 * anchor.height))
    left, top = max(0, anchor.left - margin), max(0, anchor.top - margin)
    patch = light[top:anchor.bottom + margin, left:anchor.right + margin]
    found = []
    for level in (185, 201, 217, 225):
        contours, hierarchy = cv2.findContours(
            np.uint8(patch >= level), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        links = hierarchy[0]
        for outline, (_, _, hole, parent) in zip(contours, links, strict=True):
            if parent >= 0 or hole < 0:
                continue
            x, y, width, height = cv2.boundingRect(outline)
            box = _Box(left + x, top + y, left + x + width, top + y + height)
            if not (.80 <= width / height <= 1.30 and .76 * anchor.height <= height <= 1.22 * anchor.height
                    and _same_location(box, anchor)
                    and 0 < x < x + width < patch.shape[1] and 0 < y < y + height < patch.shape[0]):
                continue
            cuff = palm = False
            while hole >= 0:
                inside = contours[hole]
                hx, hy, hw, hh = cv2.boundingRect(inside)
                cx = (hx - x + hw / 2) / width
                cy = (hy - y + hh / 2) / height
                area = cv2.contourArea(inside) / (width * height)
                cuff |= (.08 <= cx <= .30 and .51 <= cy <= .82 and .12 <= hw / width <= .34
                         and .36 <= hh / height <= .72 and .055 <= area <= .18)
                palm |= (.50 <= cx <= .84 and hy - y <= .19 * height and .43 <= hw / width <= .86
                         and .64 <= hh / height <= .98 and .24 <= area <= .49)
                hole = links[hole][0]
            if cuff and palm:
                mask = cv2.drawContours(np.zeros((height, width), np.uint8), [outline], 0, 1,
                                        cv2.FILLED, offset=(-x, -y))
                if _upward_hand(mask):
                    found.append((box, level))
    stable = [box for box, level in found if any(
        level != other and _same_location(box, peer) for peer, other in found)]
    if not stable or not all(_same_location(stable[0], box) for box in stable[1:]):
        return None
    return stable[len(stable) // 2]


def _number_band(light, glyph, next_glyph, pitch):
    height, width = light.shape
    x, y = glyph.center
    reach = 1.10 * glyph.width
    left, right = max(0, round(x - reach)), min(width, round(x + reach))
    top = glyph.bottom + 1
    bottom = min(height, round(y + .85 * pitch))
    if next_glyph is not None:
        bottom = min(bottom, next_glyph.top - round(.09 * pitch))
    if bottom - top < 8:
        return None
    patch = light[top:bottom, left:right]
    bands = []
    for level in _CAPTION_LEVELS:
        ink = patch >= level
        for start, stop in _spans(ink.sum(axis=1) >= 2):
            if not .23 * glyph.height <= stop - start <= .83 * glyph.height:
                continue
            cols = np.flatnonzero(ink[start:stop].sum(axis=0) >= 2)
            if not cols.size:
                continue
            first, last = int(cols[0]), int(cols[-1]) + 1
            if (.39 * glyph.width <= last - first <= 1.8 * glyph.width
                    and abs((first + last) / 2 + left - x) <= .26 * glyph.width):
                box = _Box(left + first, top + int(start), left + last, top + int(stop))
                bands.append((box, level))
    stable = [box for box, level in bands if any(
        other != level and abs(box.center[1] - peer.center[1]) <= max(1, .23 * box.height)
        and abs(box.center[0] - peer.center[0]) <= .25 * glyph.width for peer, other in bands)]
    if not stable or max(row.center[1] for row in stable) - min(row.center[1] for row in stable) > .25 * glyph.height:
        return None
    widest = max(stable, key=lambda box: (box.width, box.height))
    return _Box(max(left, widest.left - 2), max(top, widest.top - 2),
                min(right, widest.right + 2), min(bottom, widest.bottom + 2))


def _verify_numbers(frame, boxes, recognize):
    crops = []
    for box in boxes:
        patch = frame[box.top:box.bottom, box.left:box.right]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.int16)
        contrast = np.clip(2.2 * (gray - np.median(gray)), 0, 255).astype(np.uint8)
        for variant in (patch, cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)):
            crops.append(cv2.resize(variant, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
    try:
        readings = recognize(crops)
    except (RuntimeError, ValueError, cv2.error):
        return False
    if len(readings) != len(crops):
        return False
    for first, second in zip(readings[::2], readings[1::2], strict=True):
        value = _normalized(first["text"])
        if (not _COUNT.fullmatch(value) or value.casefold() != _normalized(second["text"]).casefold()
                or min(first["score"], second["score"]) < 91):
            return False
    return True


def suggest_counted_reaction_column(frame, observation, *, line_reader=None):
    """Return only typed controls on one column with three measured numeric captions."""
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    if (not callable(recognize) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
            or observation.get("active_layer") or observation.get("mode") == "home"):
        return []
    height, width = frame.shape[:2]
    if width < 148 or not 1.60 * width <= height <= 2.65 * width:
        return []
    working = frame
    if width > 400:
        working = cv2.resize(frame, (400, round(height * 400 / width)), interpolation=cv2.INTER_AREA)
    height, width = working.shape[:2]
    marks = list(_marks(working, height - 1))
    columns = list(_columns(_positions(marks)))
    if not columns:
        return []
    light = cv2.min(cv2.min(working[..., 0], working[..., 1]), working[..., 2])
    types = _measured_types(light, marks)
    confirmed = []
    for boxes, pitch in columns:
        typed = _typed_boxes(boxes, types)
        if len(typed) < 2:
            continue
        captions = [_number_band(light, glyph, after, pitch)
                    for glyph, after in zip(boxes, (*boxes[1:], None), strict=True)]
        if all(caption is not None for caption in captions):
            confirmed.append((boxes, typed, captions))
    if len(confirmed) != 1 or detect_blocking_overlay(frame) is not None:
        return []
    boxes, typed, captions = confirmed[0]
    if not _verify_numbers(working, captions, recognize):
        return []
    if "like" not in typed:
        for anchor in boxes:
            if not any(_same_location(anchor, box) for box in typed.values()):
                if (hand := _outline_thumb(light, anchor)) is not None:
                    typed["like"] = hand
    sx, sy = width - 1, height - 1
    return [{"role": role, "function": _FUNCTIONS[role], "anchor_id": None,
             "box_fraction": [box.left / sx, box.top / sy, box.right / sx, box.bottom / sy],
             "screen_fraction": [box.center[0] / sx, box.center[1] / sy]}
            for role, box in typed.items()]
