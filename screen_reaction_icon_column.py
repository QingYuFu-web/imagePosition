"""Confirm a comment bubble from independently typed controls in the same rail."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import ceil

import cv2
import numpy as np

from inventory_compact_video_rail import _Mark
from inventory_counted_reaction_column import _left_foot_arrow, _number_band
from inventory_filled_video_rail import _bookmark, _heart
from inventory_list_rows import _Box
from inventory_paper_plane_share import _plane
from inventory_share_video import _curved_share
from inventory_small_reaction_glyphs import _heart_shape
from inventory_toolbar_actions import _spans
from inventory_unlabeled_engagement_row import _star
from screen_colored_settings_panel import _printed
from screen_font_glyphs import FontGlyphIndex
from screen_video_controls import _foot, _pale_lower_pane, _side_candidates, _speech


@dataclass(frozen=True)
class _Witness:
    role: str
    box: _Box
    level: int
    evidence: int


def _near(first, second):
    tolerance = max(2, round(.18 * min(first.height, second.height)))
    return max(abs(a - b) for a, b in zip(first.xyxy(), second.xyxy(), strict=True)) <= tolerance


def _stable(witnesses):
    found = []
    for best in sorted(witnesses, key=lambda entry: (entry.evidence, -entry.level), reverse=True):
        if any(row.role == best.role and _near(row.box, best.box) for row in found):
            continue
        if any(row.role == best.role and row.level != best.level and _near(row.box, best.box) for row in witnesses):
            found.append(best)
    return found


def _types(plane, mask, level, unit):
    height, width = mask.shape
    contours, links = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if links is None:
        return
    for i, outline in enumerate(contours):
        if links[0, i, 3] >= 0:
            continue
        x, y, w, h = bounds = cv2.boundingRect(outline)
        if (not max(7, .019 * unit) <= min(w, h) <= max(w, h) <= .13 * unit or not .67 <= w / h <= 1.60
                or not (1 <= x < x + w <= width - 1 and 1 <= y < y + h <= height - 1)):
            continue
        area = cv2.contourArea(outline)
        if not .27 * w * h <= area <= .95 * w * h:
            continue
        holes = []
        child = links[0, i, 2]
        while child >= 0:
            if cv2.contourArea(contours[child]) >= .009 * w * h:
                holes.append(contours[child])
            child = links[0, child, 0]
        solid = cv2.drawContours(np.zeros((h, w), np.uint8), [outline], -1, 1, cv2.FILLED, offset=(-x, -y))
        box = _Box(x, y, x + w, y + h)
        marked = _Mark(box, solid, level)
        strongest = max(holes, key=cv2.contourArea) if holes else None
        if (len(holes) <= 1 and (_heart(solid) or (strongest is not None and
                _heart_shape(plane[y:y + h, x:x + w], outline, strongest)))):
            yield _Witness("like", box, level, 2)
        if evidence := _speech(solid, bounds, area, holes):
            yield _Witness("comments", box, level, evidence)
        if len(holes) <= 1 and (_bookmark(solid) or _star(solid, outline)):
            yield _Witness("favorite", box, level, 2)
        if len(holes) <= 1 and (_curved_share((None, bounds, area, outline, holes)) or _left_foot_arrow(solid)
                                or _plane(marked, plane)):
            yield _Witness("share", box, level, 2)


def _scan(plane, unit):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, round(.117 * unit) | 1),) * 2)
    relief = cv2.morphologyEx(plane, cv2.MORPH_TOPHAT, kernel)
    found = []
    previous = -1
    cached = ()
    for i, (brightness, contrast) in enumerate(((140, 28), (158, 36), (181, 40), (199, 50), (216, 59), (232, 68))):
        mask = cv2.bitwise_and(cv2.compare(plane, brightness, cv2.CMP_GE), cv2.compare(relief, contrast, cv2.CMP_GE))
        count = cv2.countNonZero(mask)
        if count != previous:
            cached = tuple(_types(plane, mask, i, unit)) if count else ()
            previous = count
        found.extend(_Witness(item.role, item.box, i, item.evidence) for item in cached)
    return _stable(found)


def _selected_likes(blue, green, red, unit):
    amount = cv2.subtract(red, cv2.max(blue, green))
    found = []
    for i, (brightness, lift) in enumerate(((175, 53), (194, 66), (219, 83))):
        mask = cv2.bitwise_and(cv2.compare(red, brightness, cv2.CMP_GE), cv2.compare(amount, lift, cv2.CMP_GE))
        found.extend(mark for mark in _types(red, mask, i, unit) if mark.role == "like")
    return _stable(found)


@lru_cache(maxsize=4)
def _digit_index(fonts):
    return FontGlyphIndex(fonts=fonts, charset="0123456789KMkmWw+.万亿萬億")


def _number(plane, band, index):
    if band is None:
        return False
    left, top, right, bottom = band.xyxy()
    chip = plane[top:bottom, left:right]
    if min(chip.shape) < 5:
        return False
    floor, high = map(float, np.percentile(chip, (48, 97)))
    if high - floor < 44:
        return False
    bank = _digit_index(index.fonts)
    readings = []
    for fraction in (.50, .64, .76):
        cut = chip >= floor + fraction * (high - floor)
        x, y, w, h = cv2.boundingRect(np.uint8(cut))
        if h < 6 or not .27 <= w / h <= 5.2:
            continue
        cut = cut[y:y + h, x:x + w]
        parts = list(_spans(cut.any(axis=0)))
        if not 1 <= len(parts) <= 8:
            continue
        literal = []
        for k, (a, b) in enumerate(parts):
            glyph = cut[:, a:b]
            gx, gy, gw, gh = cv2.boundingRect(np.uint8(glyph))
            if .07 * h <= gw <= .35 * h and .07 * h <= gh <= .25 * h and gy >= .67 * h:
                literal.append(".")
                continue
            if not .58 * h <= gh <= h or not .16 * h <= gw <= .99 * h:
                break
            tail = k == len(parts) - 1 and len(parts) > 1 and .66 * h <= gw
            alternatives = bank.rank(glyph, alphabet="all" if tail else "latin", limit=2)
            if (len(alternatives) < 2 or alternatives[0]["distance"] > .205 or
                    alternatives[1]["distance"] - alternatives[0]["distance"] < .015):
                break
            literal.append(alternatives[0]["text"].lower())
        if len(literal) == len(parts):
            label = "".join(literal).removesuffix("+").rstrip("kmw万亿萬億")
            if label and label[0].isdigit() and label[-1].isdigit() and label.replace(".", "", 1).isdigit():
                readings.append(label)
    return len(readings) >= 2 and len(set(readings)) == 1


def _rail_options(witnesses, plane, index, footer):
    liked = [item for item in witnesses if item.role == "like"]
    bubbles = [item for item in witnesses if item.role == "comments"]
    trailing = [item for item in witnesses if item.role in {"share", "favorite"}]
    numbers = {}
    for like, bubble, peer in product(liked, bubbles, trailing):
        first, middle, last = like.box, bubble.box, peer.box
        x, y = middle.center
        pitch = y - first.center[1]
        second = last.center[1] - y
        span = middle.height
        if not (1.16 * span <= pitch <= 4.7 * span and min(abs(second / pitch - step) for step in (1, 2)) <= .22
                and all(abs(box.center[0] - x) <= .34 * span and .69 * span <= box.height <= 1.46 * span for box in (first, last))):
            continue
        foot = footer is not None and .42 * pitch <= footer - last.center[1] <= 4.3 * pitch
        second_role = any(row.role != peer.role and abs(row.box.center[0] - x) <= .34 * span and
                          .69 * span <= row.box.height <= 1.46 * span and
                          min(abs((row.box.center[1] - y) / pitch - step) for step in (1, 2)) <= .22 and
                          abs(row.box.center[1] - last.center[1]) >= .69 * pitch for row in trailing)
        if foot and second_role and bubble.evidence >= 2:
            yield (bubble.evidence, 2, foot), middle, last if peer.role == "share" else None
            continue
        count = 0
        for box, after in ((first, middle), (middle, last), (last, None)):
            key = box, after
            if key not in numbers:
                numbers[key] = _number(plane, _number_band(plane, box, after, pitch), index)
            count += numbers[key]
        if count >= 2 or (count >= 1 and bubble.evidence >= 2 and foot):
            yield (bubble.evidence, count, foot), middle, last if peer.role == "share" else None


def _labeled_shares(marks, plane, index):
    liked = [item for item in marks if item.role == "like"]
    for item in marks:
        share = item.box
        if item.role != "share" or not any(
                abs(like.box.center[0] - share.center[0]) <= .37 * share.width
                and 2.45 * share.height <= share.center[1] - like.box.center[1] <= 8.9 * share.height
                and .67 * share.height <= like.box.height <= 1.55 * share.height for like in liked):
            continue
        x, _ = share.center
        span = share.height
        caption = (max(0, round(x - .93 * span)), min(len(plane), share.bottom + round(.34 * span)),
                   min(plane.shape[1], round(x + .93 * span)), min(len(plane), share.bottom + round(1.26 * span)))
        word = _printed(plane, caption, index, vocabulary=("分享", "转发", "评论", "留言", "发送", "转帖"),
                        minimum_score=.80, minimum_margin=.026)
        if word in {"分享", "转发"}:
            yield share


def _central_post_footer(frame):
    """Recognize a raised plus plate between four separately inked navigation items."""
    height, width = frame.shape[:2]
    start = round(.914 * height)
    gray = cv2.cvtColor(frame[start:], cv2.COLOR_BGR2GRAY)
    left, right = round(.36 * width), round(.64 * width)
    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.compare(gray[:, left:right], 218, cv2.CMP_GE))
    good = []
    for x, y, w, h, count in stats[1:]:
        center = x + left + w / 2
        if (not .061 * width <= w <= .151 * width or not .039 * width <= h <= .098 * width
                or not .455 * width <= center <= .545 * width or not 1.12 <= w / h <= 1.99 or count < .68 * w * h):
            continue
        a, b = left + x + round(.24 * w), left + x + w - round(.24 * w)
        t, d = y + round(.20 * h), y + h - round(.20 * h)
        roi = gray[t:d, a:b]
        l, u, wide, high = cv2.boundingRect(cv2.compare(roi, 116, cv2.CMP_LE))
        if not (.28 * h <= min(wide, high) <= max(wide, high) <= .64 * h and .78 <= wide / high <= 1.28):
            continue
        shape = roi[u:u + high, l:l + wide] <= 116
        if (not (.27 <= shape.mean() <= .58 and shape.sum(axis=1).max() >= .79 * wide
                 and shape.sum(axis=0).max() >= .79 * high)):
            continue
        patches = 0
        for offset in (-2, -1, 1, 2):
            cx = center + offset * .200 * width
            zone = gray[t:d, max(0, round(cx - .071 * width)):min(width, round(cx + .071 * width))]
            if min(zone.shape) < 4:
                continue
            tone = float(np.median(zone))
            _, _, ww, hh = cv2.boundingRect(np.uint8(zone >= tone + 43))
            if (.23 * h <= hh <= .76 * h and .53 * h <= ww <= 1.90 * h and .8 <= ww / hh <= 3.4):
                patches += 1
        if patches == 4:
            good.append(start + y - round(.3 * h))
    return good[0] if len(good) == 1 else None


def reaction_icon_column(frame, regions, index):
    """Locate a visible speech glyph even when a proposal detector missed its box."""
    h, w = frame.shape[:2]
    if (index is None or regions["layer"] != "page" or not 1.59 * w <= h <= 2.77 * w or w < 160
            or regions.get("floating_controls") or _pale_lower_pane(frame)):
        return []
    if "candidates" in regions:
        anchors = list(_side_candidates(regions, w, h))
        if not any(abs(left["center"][0] - right["center"][0]) <= .032 * w
                   and .077 * h <= abs(left["center"][1] - right["center"][1]) <= .43 * h
                   and min(left["center"][0], right["center"][0], w - max(left["center"][0], right["center"][0])) <= .16 * w
                   for i, left in enumerate(anchors) for right in anchors[i + 1:]):
            return []
    scale = min(1.0, 510 / w)
    overview = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
    height, width = overview.shape[:2]
    low, high = round(.32 * height), round(.995 * height)
    matches = []
    share_boxes = []
    for left, right in ((round(.76 * width), width - 1), (0, round(.24 * width))):
        area = overview[low:high, left:right]
        blue, green, red = cv2.split(area)
        light = cv2.min(cv2.min(blue, green), red)
        dark = cv2.bitwise_not(cv2.max(cv2.max(blue, green), red))
        ranked = []
        selected = _selected_likes(blue, green, red, width)
        for polarity, plane in enumerate((light, dark)):
            group = _scan(plane, width)
            if polarity == 0:
                group.extend(selected)
            share_boxes.extend((left, low, item) for item in _labeled_shares(group, plane, index))
            if {row.role for row in group} >= {"like", "comments"} and any(row.role in {"share", "favorite"} for row in group):
                ranked.append((group, plane))
        if not ranked:
            continue
        small, footer = _foot(overview)
        footer = round(footer * height / small.shape[0]) if footer is not None else _central_post_footer(overview)
        y_foot = footer - low if footer is not None else None
        for group, plane in ranked:
            for rank, box, share in _rail_options(group, plane, index, y_foot):
                matches.append((rank, _Box(left + box.left, low + box.top, left + box.right, low + box.bottom)))
                if share is not None:
                    share_boxes.append((left, low, share))

    def result(role, function, candidates):
        if candidates and all(_near(candidates[0], other) for other in candidates[1:]):
            left, top, right, bottom = candidates[0].xyxy()
            box = [int(left * w / width), int(top * h / height), ceil(right * w / width), ceil(bottom * h / height)]
            return [{"role": role, "function": function, "box": box,
                     "click_point": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]}]
        return []

    comments = [box for _, box in sorted(matches, key=lambda pair: pair[0], reverse=True)]
    shares = [_Box(left + box.left, top + box.top, left + box.right, top + box.bottom)
              for left, top, box in share_boxes]
    return result("comments", "打开评论区", comments) + result("share", "打开分享", shares)
