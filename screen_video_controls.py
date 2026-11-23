"""Read conventional video controls from grounded glyphs on a reaction rail."""

from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from inventory_compact_video_rail import _footer_top
from inventory_counted_reaction_column import _number_band
from inventory_filled_video_rail import _bookmark
from inventory_list_rows import _Box
from inventory_outlined_comment import suggest_outlined_comment
from inventory_outlined_short_video_reactions import find_outlined_short_video_peer
from inventory_paper_plane_share import find_paper_plane_share_box
from inventory_share_video import _curved_share
from inventory_video_edge_shapes import find_outlined_video_share_box
from screen_box_actions import _video_rail
from screen_candidate_glyphs import CandidateGlyphs


_ROLES = {"comments": ("打开评论区", "Comments"), "share": ("打开分享", "Share")}
_LEVELS = (205, 218, 226, 233, 238, 243, 247)


@dataclass(frozen=True)
class _Glyph:
    role: str
    box: _Box
    anchor: int | None
    level: int
    evidence: int = 1


def _result(glyph):
    function, preview = _ROLES[glyph.role]
    return {"role": glyph.role, "function": function, "box": [int(value) for value in glyph.box.xyxy()],
            "click_point": [round(value) for value in glyph.box.center], "preview_text": preview}


def _counted_rail(output, candidates, width):
    """Locate the repeated slots shared by the already recognized glyphs."""
    bubble, arrow = (_Box(*row["box"]) for row in output)
    bx, by = bubble.center
    ax, ay = arrow.center
    span = (bubble.height + arrow.height) / 2
    if abs(bx - ax) > min(.032 * width, .44 * span) or ay - by < 1.52 * span:
        return None
    lane = [(row, _Box(*row["box"])) for row in candidates if row["visibility"] == "page"
            and abs(row["center"][0] - (ax + bx) / 2) <= min(.035 * width, .52 * span)]
    lifts = []
    for glyph in (bubble, arrow):
        x, y = glyph.center
        owners = [(abs(box.top - glyph.top), row["center"][1] - y) for row, box in lane
                  if not row.get("disabled") and box.left <= x < box.right and box.top <= y < box.bottom
                  and abs(box.top - glyph.top) <= .45 * span]
        if not owners:
            return None
        lifts.append(min(owners, key=lambda item: item[0])[1])
    if abs(lifts[0] - lifts[1]) > .45 * span:
        return None
    lift = sum(lifts) / 2
    options = []
    for steps in (1, 2):
        pitch = (ay - by) / steps
        if not 1.42 * span <= pitch <= 4.60 * span:
            continue
        offsets = [(row["center"][1] - lift - by) / pitch for row, _ in lane]
        slots = {round(offset) for offset in offsets if abs(offset - round(offset)) <= .24}
        slots &= set(range(-3, 6))
        if len(slots) >= 3 and set(range(steps + 1)) <= slots:
            options.append((len(slots), pitch))
    if not options or (len(options) == 2 and options[0][0] == options[1][0]):
        return None
    _, pitch = max(options)
    return (bubble, arrow), [box for _, box in lane], pitch


def _attach_count_context(output, rail, light, floating):
    """Keep adjacent count pixels with their glyph without expanding its tap area."""
    glyphs, proposals, pitch = rail
    for action, glyph in zip(output, glyphs, strict=True):
        _, gy = glyph.center
        followers = [box for box in (*glyphs, *proposals) if .45 * pitch <= box.center[1] - gy <= 1.43 * pitch]
        following = min(followers, key=lambda box: box.top, default=None)
        count = _number_band(light, glyph, following, pitch)
        if count is None or count.top - glyph.bottom > .76 * glyph.height:
            continue
        context = [min(glyph.left, count.left), glyph.top, max(glyph.right, count.right), count.bottom]
        left, top, right, bottom = context
        if any(max(left, x0) < min(right, x1) and max(top, y0) < min(bottom, y1) for x0, y0, x1, y1 in floating):
            continue
        action.update(engagement_count_box=list(count.xyxy()), context_box=context)


def _groups(flags):
    return np.flatnonzero(np.diff(np.r_[False, flags, False])).reshape(-1, 2).tolist()


def _pale_lower_pane(frame):
    height, width = frame.shape[:2]
    sample = frame[::4, round(.06 * width):round(.94 * width):6]
    light = cv2.min(cv2.min(sample[..., 0], sample[..., 1]), sample[..., 2])
    rows = (light >= 225).mean(axis=1)
    link = np.ones((max(3, round(.040 * len(rows)) | 1), 1), np.uint8)
    stable = cv2.morphologyEx(np.uint8(rows >= .75)[:, None], cv2.MORPH_CLOSE, link)[:, 0]
    for top, bottom in _groups(stable):
        if not .27 * len(rows) <= top <= .77 * len(rows) or bottom - top < .10 * len(rows):
            continue
        before = rows[max(0, top - round(.06 * len(rows))):top]
        continuing = rows[top:min(len(rows), top + round(.28 * len(rows)))]
        if np.median(before) <= .38 and (continuing >= .68).mean() >= .81:
            return True
    return False


def _side_candidates(regions, width, height):
    for row in regions["candidates"]:
        x, y = row["center"]
        left, top, right, bottom = row["box"]
        w, h = right - left, bottom - top
        if (row["visibility"] in {"page", "layer_uncertain"}
                and (x >= .79 * width or x <= .21 * width) and .31 * height <= y <= .965 * height
                and .025 * width <= min(w, h) <= max(w, h) <= .225 * width):
            yield row


def _speech(mask, bounds, area, holes):
    left, top, w, h = bounds
    if not .84 <= w / h <= 1.42 or not .60 <= area / (w * h) <= .89:
        return 0
    mid = mask[round(.20 * h):round(.68 * h)]
    tip = mask[-max(1, round(.13 * h)):]
    _, body_x = np.nonzero(mid)
    _, toe_x = np.nonzero(tip)
    if (not body_x.size or not toe_x.size or np.median(mid.sum(axis=1)) < .74 * w
            or not 0 < mask[-1].sum() <= .22 * w):
        return 0
    centers = []
    features = [hole for hole in holes if cv2.contourArea(hole) >= .008 * w * h]
    if len(features) == 3:
        for hole in features:
            x, y, ww, hh = cv2.boundingRect(hole)
            if not (.67 <= ww / hh <= 1.52 and .014 <= cv2.contourArea(hole) / (w * h) <= .068):
                return 0
            centers.append(((x + ww / 2 - left) / w, (y + hh / 2 - top) / h))
        centers.sort()
        dotted = (max(y for _, y in centers) - min(y for _, y in centers) <= .11
                  and all(.15 <= b[0] - a[0] <= .31 for a, b in zip(centers, centers[1:]))
                  and .35 <= centers[1][0] <= .65 and .30 <= centers[1][1] <= .64)
        return 3 if dotted else 0
    if abs(float(toe_x.mean() - body_x.mean())) < .105 * w:
        return 0
    if not features:
        return 1
    if len(features) != 1:
        return 0
    _, _, hole_w, hole_h = cv2.boundingRect(features[0])
    outlined = .40 <= cv2.contourArea(features[0]) / (w * h) <= .77 and hole_w >= .58 * w and hole_h >= .52 * h
    return 2 if outlined else 0


def _ink_shapes(mask, level, left, top, anchor, width):
    contours, links = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if links is None:
        return
    for i, contour in enumerate(contours):
        if links[0, i, 3] >= 0:
            continue
        x, y, w, h = bounds = cv2.boundingRect(contour)
        if (not max(7, .022 * width) <= min(w, h) <= max(w, h) <= .126 * width or not .76 <= w / h <= 1.58
                or not 0 < x < x + w < mask.shape[1] or not 0 < y < y + h < mask.shape[0]):
            continue
        area = cv2.contourArea(contour)
        holes = []
        child = links[0, i, 2]
        while child >= 0:
            holes.append(contours[child])
            child = links[0, child, 0]
        solid = cv2.drawContours(np.zeros((h, w), np.uint8), [contour], 0, 1, cv2.FILLED, offset=(-x, -y))
        box = _Box(left + x, top + y, left + x + w, top + y + h)
        if evidence := _speech(solid, bounds, area, holes):
            yield _Glyph("comments", box, anchor, level, evidence)
        elif _curved_share((None, bounds, area, contour, [])):
            significant = [cv2.contourArea(hole) / area for hole in holes if cv2.contourArea(hole) >= .025 * area]
            if not significant or (len(significant) == 1 and .21 <= significant[0] <= .83):
                yield _Glyph("share", box, anchor, level)


def _distance(first, second):
    return max(abs(a - b) for a, b in zip(first.xyxy(), second.xyxy(), strict=True))


def _stable_shapes(light, candidates):
    height, width = light.shape
    found = []
    for row in candidates:
        x0, y0, x1, y1 = row["box"]
        extra = max(2, round(.004 * width))
        left, top, right, bottom = max(0, x0 - extra), max(0, y0 - extra), min(width, x1 + extra), min(height, y1 + extra)
        chip = light[top:bottom, left:right]
        for level in _LEVELS:
            found.extend(_ink_shapes(np.uint8(chip >= level), level, left, top, row["id"], width))
    stable = []
    for glyph in found:
        tolerance = max(2, round(.125 * glyph.box.height))
        other = [item for item in found if item.role == glyph.role and _distance(glyph.box, item.box) <= tolerance]
        if len({item.level for item in other}) < 2 or any(item.role == glyph.role and _distance(glyph.box, item.box) <= tolerance
                                                        for item in stable):
            continue
        stable.append(max(other, key=lambda item: (item.evidence, item.level)))
    return stable


def _foot(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (384, round(384 * h / w)), interpolation=cv2.INTER_AREA) if w > 384 else frame
    found = _footer_top(small)
    if found is not None:
        return small, found
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    strip = gray[:, round(.05 * w):round(.95 * w)]
    dark = (strip <= 49).mean(axis=1) >= .67
    kernel = np.ones((max(3, round(.015 * w) | 1), 1), dtype=np.uint8)
    joined = cv2.morphologyEx(np.uint8(dark)[:, None], cv2.MORPH_CLOSE, kernel)[:, 0]
    for top, bottom in reversed(_groups(joined)):
        if not .83 * h <= top <= .954 * h or bottom < .965 * h or bottom - top < .11 * w:
            continue
        ink = gray[top + max(2, round(.015 * w)):top + round(.13 * w)] >= 136
        visible = np.uint8((ink.sum(axis=0) >= max(1, round(.010 * w))))[None, :]
        kernel = np.ones((1, max(3, round(.025 * w) | 1)), dtype=np.uint8)
        joined_columns = cv2.morphologyEx(visible, cv2.MORPH_CLOSE, kernel)[0]
        glyphs = [(a + b) / 2 for a, b in _groups(joined_columns) if .016 * w <= b - a <= .15 * w]
        if len(glyphs) >= 4 and glyphs[0] <= .23 * w and glyphs[-1] >= .77 * w:
            return small, top
    return small, None


def _middle_bookmark(light, x, y, span):
    h, w = light.shape
    a, b = max(0, round(x - .80 * span)), min(w, round(x + .80 * span))
    t, d = max(0, round(y - .78 * span)), min(h, round(y + .80 * span))
    patch = light[t:d, a:b]
    strikes = 0
    for level in (208, 226, 238):
        contours, _ = cv2.findContours(np.uint8(patch >= level), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            l, q, width, height = cv2.boundingRect(contour)
            if (not .61 * span <= height <= 1.42 * span or not .73 <= width / height <= 1.12
                    or abs(a + l + width / 2 - x) > .28 * span or abs(t + q + height / 2 - y) > .36 * span):
                continue
            shape = cv2.drawContours(np.zeros((height, width), np.uint8), [contour], 0, 1, cv2.FILLED, offset=(-l, -q))
            if _bookmark(shape):
                strikes += 1
                break
    return strikes >= 2


def _rail_pairs(marks, candidates, light, footer):
    height, width = light.shape
    context = {row["id"]: row for row in candidates}
    comments = [glyph for glyph in marks if glyph.role == "comments"]
    shares = [glyph for glyph in marks if glyph.role == "share"]
    supported = []
    for bubble, arrow in product(comments, shares):
        bx, by = bubble.box.center
        ax, ay = arrow.box.center
        mean_span = (bubble.box.height + arrow.box.height) / 2
        gap = ay - by
        if (not .58 <= bubble.box.height / arrow.box.height <= 1.72 or
                abs(ax - bx) > min(.032 * width, .44 * mean_span) or not 1.52 * mean_span <= gap <= .21 * height):
            continue
        peers = [row for row in candidates if abs(row["center"][0] - (ax + bx) / 2) <= min(.035 * width, .52 * mean_span)]
        anchors = [context[glyph.anchor] for glyph in (bubble, arrow) if glyph.anchor in context]
        lift = float(np.median([row["center"][1] - glyph.box.center[1] for glyph in (bubble, arrow)
                                if (row := context.get(glyph.anchor)) is not None])) if anchors else 0.
        for steps in (1, 2):
            pitch = gap / steps
            if not 1.42 * mean_span <= pitch <= 4.60 * mean_span:
                continue
            slots = {round((row["center"][1] - lift - by) / pitch) for row in peers
                     if abs((row["center"][1] - lift - by) / pitch - round((row["center"][1] - lift - by) / pitch)) <= .29}
            slots &= set(range(-3, 6))
            if not {0, steps} <= slots or len(slots) < 3:
                continue
            if steps == 2 and 1 not in slots:
                if not _middle_bookmark(light, (ax + bx) / 2, by + pitch, mean_span):
                    continue
                slots.add(1)
            count = sum(_number_band(light, glyph.box, following, pitch) is not None
                        for glyph, following in ((bubble, arrow.box if steps == 1 else None), (arrow, None)))
            bottom_support = footer is not None and .47 * pitch <= footer - ay <= 4.4 * pitch
            if count >= 2 or (len(slots) >= 4 and bottom_support) or (len(slots) >= 4 and count >= 1):
                supported.append(((bubble.evidence, len(slots), count, bottom_support), (bubble, arrow)))
    if not supported:
        return []
    score, best = max(supported, key=lambda item: item[0])
    if any(other_score == score and any(_distance(a.box, b.box) > max(3, .28 * a.box.height)
                                        for a, b in zip(best, other, strict=True)) for other_score, other in supported):
        return []
    return list(best)


def _original_box(box, frame, cropped):
    h, w = frame.shape[:2]
    ch, cw = cropped.shape[:2]
    return _Box(round(box.left * w / cw), round(box.top * h / ch), round(box.right * w / cw), round(box.bottom * h / ch))


def _outlined(frame, small, old_marks):
    peer = find_outlined_short_video_peer(small, {})
    share_box = find_outlined_video_share_box(small, {})
    output = []
    if peer is not None and peer.bubble is not None:
        output.append(_Glyph("comments", _original_box(peer.bubble, frame, small), None, 0))
    if share_box is not None:
        output.append(_Glyph("share", _original_box(_Box(*share_box), frame, small), None, 0))
    elif peer is not None and peer.share_box is not None:
        output.append(_Glyph("share", _original_box(peer.share_box, frame, small), None, 0))
    elif peer is not None and peer.bubble is not None:
        h, w = frame.shape[:2]
        ch, cw = small.shape[:2]
        x, y = peer.comment_x * w / cw, peer.comment_y * h / ch
        span, gap = peer.span * h / ch, peer.pitch * h / ch
        arrows = [glyph for glyph in old_marks if glyph.role == "share" and abs(glyph.box.center[0] - x) <= .37 * span
                  and .72 * gap <= glyph.box.center[1] - y <= 1.28 * gap and .49 * span <= glyph.box.height <= 1.34 * span]
        if len(arrows) == 1:
            output.append(arrows[0])
    return output


def _remaining_outlines(frame, found):
    roles = {glyph.role for glyph in found}
    result = []
    if "comments" not in roles:
        caption = suggest_outlined_comment(frame, {})
        if caption is not None:
            h, w = frame.shape[:2]
            box = _Box(*(round(value * (w - 1 if i % 2 == 0 else h - 1))
                         for i, value in enumerate(caption["box_fraction"])))
            result.append(_Glyph("comments", box, None, 0))
    if "share" not in roles:
        if (box := find_paper_plane_share_box(frame, {})) is not None:
            result.append(_Glyph("share", _Box(*box), None, 0))
    return result


def _toolbar_search(frame, regions, glyphs):
    """Confirm a magnifier in the header of an already verified video rail."""
    height, width = frame.shape[:2]
    header = [row for row in regions["candidates"] if row["visibility"] == "page"
              and .024 * height <= row["center"][1] <= .15 * height]
    if len(header) < 3:
        return None
    edge = [row for row in header if row["center"][0] >= .82 * width]
    if not edge:
        return None
    found = set()
    for mark in glyphs(edge):
        if mark["role"] != "search":
            continue
        box = _Box(*mark["box"])
        x, y = box.center
        peers = []
        for row in header:
            px, py = row["center"]
            l, t, r, b = row["box"]
            if (.047 * width <= x - px <= .31 * width and abs(py - y) <= .57 * box.height
                    and .032 * width <= r - l <= .20 * width and .64 * box.height <= b - t <= 2.8 * box.height
                    and all(abs(px - peer) >= .03 * width for peer in peers)):
                peers.append(px)
        if len(peers) >= 2:
            found.add(box)
    if len(found) != 1:
        return None
    box = next(iter(found))
    return {"role": "search", "function": "搜索", "box": list(box.xyxy()),
            "click_point": list(box.center), "preview_text": "Search"}


def video_control_actions(frame, regions, *, glyphs=None):
    """Return primary and review taps for independently verified rail symbols."""
    layer = regions["layer"]
    height, width = frame.shape[:2]
    if layer not in {"page", "layer_uncertain"} or not 1.60 * width <= height <= 2.75 * width or width < 128:
        return [], []
    if _pale_lower_pane(frame):
        return [], []
    glyphs = glyphs or CandidateGlyphs(frame)
    candidates = list(_side_candidates(regions, width, height))
    old = [mark for mark in glyphs(candidates) if mark["role"] in _ROLES]
    old_marks = [_Glyph(mark["role"], _Box(*mark["box"]), mark["id"], 0) for mark in old]
    proposed = _video_rail(frame, [{**mark, "center": glyph.box.center, "height": glyph.box.height}
                                   for mark, glyph in zip(old, old_marks, strict=True)], candidates)
    found = [glyph for glyph in old_marks if any(row["id"] == glyph.anchor for row in proposed)]
    light = None
    if len({glyph.role for glyph in found}) < 2:
        small, footer = _foot(frame)
        if len(candidates) >= 3:
            light = cv2.min(cv2.min(frame[..., 0], frame[..., 1]), frame[..., 2])
            marks = old_marks + _stable_shapes(light, candidates)
            found.extend(_rail_pairs(marks, candidates, light, footer * height / small.shape[0] if footer is not None else None))
        if footer is not None and layer == "page" and len({glyph.role for glyph in found}) < 2:
            found.extend(_outlined(frame, small, old_marks))
            if len(candidates) >= 3 and len({glyph.role for glyph in found}) < 2:
                found.extend(_remaining_outlines(frame, found))
    output = []
    for role in _ROLES:
        peers = [glyph for glyph in found if glyph.role == role]
        if peers and all(_distance(peers[0].box, glyph.box) <= max(3, .33 * peers[0].box.height) for glyph in peers[1:]):
            output.append(_result(peers[0]))
    if layer == "page" and len(output) == len(_ROLES):
        if (rail := _counted_rail(output, candidates, width)) is not None:
            if light is None:
                light = cv2.min(cv2.min(frame[..., 0], frame[..., 1]), frame[..., 2])
            _attach_count_context(output, rail, light, regions.get("floating_controls", ()))
        if (search := _toolbar_search(frame, regions, glyphs)) is not None:
            output.append(search)
    return (output, []) if layer == "page" else ([], output)
