"""Find independently typed engagement controls with an aligned row of visible counts."""

from dataclasses import dataclass
from itertools import combinations
from statistics import mean

import cv2
import numpy as np

from inventory_compact_video_rail import _speech
from inventory_compact_video_text import _COUNT, _normalized
from inventory_list_rows import _Box
from inventory_small_reaction_glyphs import _heart_shape
from inventory_toolbar_actions import _spans
from screen_foreground import detect_blocking_overlay


_LEVELS = (37, 49, 64, 26)
_LABELS = {"reply": "回复", "repost": "转发", "like": "点赞", "favorite": "收藏", "share": "打开分享菜单"}
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


@dataclass(frozen=True)
class _Glyph:
    role: str
    box: _Box
    color: np.ndarray
    level: int


def _star(mask, outline):
    ys, xs = np.nonzero(mask)
    origin = np.array([xs.mean(), ys.mean()])
    points = cv2.approxPolyDP(outline, .025 * cv2.arcLength(outline, True), True).reshape(-1, 2)
    if len(points) != 10:
        return False
    radii = np.linalg.norm(points - points.min(axis=0) - origin, axis=1)
    outward = int(radii.argmax()) % 2
    tips, valleys = radii[outward::2], radii[1 - outward::2]
    return bool(min(tips) >= 1.32 * max(valleys) and min(tips) >= .69 * max(tips))


def _network(box, holes):
    if len(holes) != 3:
        return False
    points = []
    width, height = box.width, box.height
    for hole in holes:
        x, y, w, h = cv2.boundingRect(hole)
        if not (.12 * width <= w <= .32 * width and .12 * height <= h <= .34 * height
                and .61 <= w / h <= 1.53):
            return False
        points.append(((x + w / 2 - box.left) / width, (y + h / 2 - box.top) / height))
    left, upper, lower = sorted(points)
    return (.11 <= left[0] <= .34 and .36 <= left[1] <= .64 and .61 <= upper[0] <= .92
            and .61 <= lower[0] <= .92 and .54 <= abs(upper[1] - lower[1]) <= .84)


def _left_reply(mask):
    height, width = mask.shape
    if not (1.14 <= width / height <= 1.72 and .245 <= mask.mean() <= .53):
        return False
    top = np.nonzero(mask[:max(1, round(.24 * height))])[1]
    foot = np.nonzero(mask[round(.75 * height):])[1]
    tip = np.nonzero(mask[:, 0])[0]
    bar = mask[round(.24 * height):round(.64 * height)].sum(axis=1)
    return bool(top.size and foot.size and tip.size and .17 * width <= top.mean() <= .48 * width
                and foot.mean() >= .70 * width and .18 * height <= tip.mean() <= .53 * height
                and max(bar) >= .54 * width)


def _right_loop(mask):
    height, width = mask.shape
    if not (1.55 <= width / height <= 2.42 and .24 <= mask.mean() <= .55):
        return False
    top = np.nonzero(mask[:max(1, round(.22 * height))])[1]
    foot = np.nonzero(mask[round(.75 * height):])[1]
    middle = mask[round(.20 * height):round(.65 * height)].sum(axis=1)
    return bool(top.size and foot.size and top.mean() >= .62 * width and foot.mean() <= .33 * width
                and top.mean() - foot.mean() >= .34 * width and max(middle) >= .68 * width)


def _reposts(top_parts, bottom_parts):
    for top, upper in top_parts:
        for bottom, lower in bottom_parts:
            span = (top.height + bottom.height) / 2
            x, y = top.center
            lower_x, lower_y = bottom.center
            if not (abs(x - lower_x) <= .09 * (top.width + bottom.width)
                    and .78 * span <= lower_y - y <= 1.38 * span and .82 <= top.width / bottom.width <= 1.24):
                continue
            turned = cv2.resize(lower[::-1, ::-1], upper.shape[::-1], interpolation=cv2.INTER_NEAREST)
            if (turned & upper).sum() >= .54 * (turned | upper).sum():
                yield _Box(min(top.left, bottom.left), top.top, max(top.right, bottom.right), bottom.bottom)


def _ink_color(frame, relief, box, level):
    region = (slice(box.top, box.bottom), slice(box.left, box.right))
    ink = relief[region] >= max(level, 49)
    return np.median(frame[region][ink], axis=0) if ink.any() else None


def _similar_ink(first, second):
    difference = first - second
    return np.ptp(difference) <= 14 and abs(difference.mean()) <= 64


def _contour_roles(relief, level):
    limit = min(relief.shape)
    ink = np.uint8(relief >= level)
    # CCOMP needs only components that can pass the exact glyph-size test.
    # Removing ineligible components first avoids tracing unrelated contours.
    if cv2.countNonZero(ink) > .32 * ink.size:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        widths, heights = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
        eligible = (7 <= heights) & (heights <= .20 * limit) & (7 <= widths) & (widths <= .22 * limit)
        eligible[0] = False
        if not eligible.any():
            return
        ink = np.uint8(eligible[labels])
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return
    links = hierarchy[0]
    top_parts, bottom_parts = [], []
    for outline, (_, _, hole, parent) in zip(contours, links, strict=True):
        if parent >= 0:
            continue
        x, y, w, h = cv2.boundingRect(outline)
        if not (7 <= h <= .20 * limit and 7 <= w <= .22 * limit):
            continue
        box = _Box(x, y, x + w, y + h)
        mask = cv2.drawContours(np.zeros((h, w), np.uint8), [outline], 0, 1, cv2.FILLED, offset=(-x, -y))
        if hole < 0:
            if _right_loop(mask):
                top_parts.append((box, mask))
            if _right_loop(mask[::-1, ::-1]):
                bottom_parts.append((box, mask))
            if _left_reply(mask):
                yield "reply", box
            continue
        holes = []
        while hole >= 0:
            holes.append(contours[hole])
            hole = links[hole][0]
        main = max(holes, key=cv2.contourArea)
        _, _, inside_width, inside_height = cv2.boundingRect(main)
        if inside_width >= .40 * w and inside_height >= .43 * h:
            if _speech(mask):
                yield "reply", box
            elif _heart_shape(relief[y:y + h, x:x + w], outline, main):
                yield "like", box
            elif .80 <= w / h <= 1.30 and _star(mask, outline):
                yield "favorite", box
        if .69 <= w / h <= 1.25 and _network(box, holes):
            yield "share", box
    for box in _reposts(top_parts, bottom_parts):
        yield "repost", box


def _same_box(first, second):
    return max(abs(a - b) for a, b in zip(first.xyxy(), second.xyxy(), strict=True)) <= max(
        1, .18 * max(first.height, second.height))


def _stable_glyphs(frame, relief):
    observed = []
    for level in _LEVELS:
        for role, box in _contour_roles(relief, level):
            color = _ink_color(frame, relief, box, level)
            if color is not None:
                observed.append(_Glyph(role, box, color, level))
    stable = []
    for glyph in observed:
        if (any(glyph.role == peer.role and glyph.level != peer.level and _same_box(glyph.box, peer.box)
                for peer in observed)
                and all(glyph.role != peer.role or not _same_box(glyph.box, peer.box) for peer in stable)):
            stable.append(glyph)
    return stable


def _columns(ink, gap):
    spans = []
    for left, right in _spans(ink.sum(axis=0) >= 2):
        if spans and left - spans[-1][1] <= gap:
            spans[-1] = spans[-1][0], right
        else:
            spans.append((left, right))
    return spans


def _count_box(frame, relief, glyph):
    box = glyph.box
    height, width = relief.shape
    y = round(box.center[1])
    left = box.right + max(1, round(.10 * box.width))
    right = min(width, box.right + round(2.55 * box.width))
    top, bottom = max(0, round(y - .56 * box.height)), min(height, round(y + .56 * box.height))
    area = relief[top:bottom, left:right]
    if not area.size:
        return None
    candidates = []
    for level in _LEVELS[:3]:
        ink = area >= level
        for start, stop in _columns(ink, max(2, round(.23 * box.height))):
            if start > 1.03 * box.width or not .24 * box.width <= stop - start <= 1.72 * box.width:
                continue
            rows = np.nonzero(ink[:, start:stop])[0]
            first, last = int(rows.min()), int(rows.max()) + 1
            if not .45 * box.height <= last - first <= .99 * box.height or abs(top + (first + last) / 2 - y) > .24 * box.height:
                continue
            text_box = _Box(left + int(start), top + first, left + int(stop), top + last)
            pigment = _ink_color(frame, relief, text_box, level)
            if pigment is not None and _similar_ink(pigment, glyph.color):
                candidates.append((text_box, level))
    stable = [box for box, level in candidates if any(
        other != level and _same_box(box, peer) for peer, other in candidates)]
    if not stable or not all(_same_box(stable[0], box) for box in stable):
        return None
    return max(stable, key=lambda box: box.width)


def _rows(frame, relief, glyphs):
    counted = [(glyph, box) for glyph in glyphs if glyph.role != "share"
               and (box := _count_box(frame, relief, glyph)) is not None]
    ordered = sorted(counted, key=lambda item: item[0].box.center[0])
    for row in combinations(ordered, 3):
        icons, captions = zip(*row)
        roles = {icon.role for icon in icons}
        if len(roles) != 3 or "repost" not in roles:
            continue
        xx, yy = zip(*(icon.box.center for icon in icons))
        span = mean(icon.box.height for icon in icons)
        pitch = (xx[-1] - xx[0]) / 2
        if (.10 * frame.shape[1] <= pitch <= .34 * frame.shape[1] and abs(xx[1] - xx[0] - pitch) <= .09 * pitch
                and max(yy) - min(yy) <= .25 * span and min(icon.box.height for icon in icons) >= .65 * span
                and all(_similar_ink(icons[0].color, icon.color) for icon in icons[1:])):
            yield icons, captions, span, pitch


def _read_counts(frame, boxes, recognize):
    crops = []
    for box in boxes:
        chip = frame[max(0, box.top - 2):box.bottom + 2, max(0, box.left - 2):box.right + 2]
        gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY).astype(np.int16)
        contrast = np.clip(2.2 * (np.median(gray) - gray), 0, 255).astype(np.uint8)
        for variant in (chip, cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)):
            crops.append(cv2.resize(variant, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
    try:
        readings = recognize(crops)
    except (RuntimeError, ValueError, cv2.error):
        return False
    return len(readings) == len(crops) and all(
        _COUNT.fullmatch(_normalized(first["text"])) and _normalized(first["text"]) == _normalized(second["text"])
        and min(first["score"], second["score"]) >= 91
        for first, second in zip(readings[::2], readings[1::2], strict=True))


def suggest_unlabeled_engagement_row(frame, observation, *, line_reader=None):
    """Return measured controls after three aligned numeric captions corroborate their row."""
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    if (not callable(recognize) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
            or observation.get("active_layer") or observation.get("mode") == "home"):
        return []
    height, width = frame.shape[:2]
    if min(height, width) < 90 or width < 180:
        return []
    working = frame
    if max(height, width) > 1500:
        ratio = 1500 / max(height, width)
        working = cv2.resize(frame, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
    height, width = working.shape[:2]
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    relief = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _KERNEL)
    glyphs = _stable_glyphs(working, relief)
    rows = list(_rows(working, relief, glyphs))
    if not rows or detect_blocking_overlay(frame) is not None:
        return []
    accepted = {}
    for icons, captions, span, pitch in rows:
        if not _read_counts(working, captions, recognize):
            continue
        accepted.update({(icon.role, icon.box): icon for icon in icons})
        last = icons[-1]
        x, y = last.box.center
        shares = [glyph for glyph in glyphs if glyph.role == "share" and abs(glyph.box.center[0] - x - pitch) <= .11 * pitch
                  and abs(glyph.box.center[1] - y) <= .27 * span and _similar_ink(last.color, glyph.color)]
        if len(shares) == 1:
            accepted[shares[0].role, shares[0].box] = shares[0]
    sx, sy = width - 1, height - 1
    return [{"role": role, "function": _LABELS[role], "anchor_id": None,
             "box_fraction": [box.left / sx, box.top / sy, box.right / sx, box.bottom / sy],
             "screen_fraction": [box.center[0] / sx, box.center[1] / sy]}
            for role, box in accepted]
