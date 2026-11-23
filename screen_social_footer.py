"""Own outlined center controls and literal tabs on verified mobile navigation bars."""

import cv2
import numpy as np

from inventory_list_rows import _Box
from inventory_small_reaction_glyphs import _heart_candidates
from screen_box_actions import _video_rail
from screen_candidate_glyphs import CandidateGlyphs
from screen_component_captions import _bands
from screen_mixed_footer import _read_caption, _spans, _text_only
from screen_native_tab_bar import _action, _distance
from screen_video_controls import _side_candidates


def _cross(ink, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    dx, dy = round(.23 * w), round(.23 * h)
    field = ink[top + dy:bottom - dy, left + dx:right - dx]
    x, y, width, height = cv2.boundingRect(field)
    if (not .24 * w <= width <= .52 * w or not .25 * h <= height <= .49 * h
            or not .81 <= width / height <= 1.22 or abs(x + width / 2 - field.shape[1] / 2) > .051 * w
            or abs(y + height / 2 - field.shape[0] / 2) > .053 * h):
        return False
    shape = field[y:y + height, x:x + width]
    middle_x = slice(round(.43 * width), round(.57 * width))
    middle_y = slice(round(.43 * height), round(.57 * height))
    if min(shape[:, middle_x].mean(), shape[middle_y].mean()) < .76:
        return False
    edge_x, edge_y = round(.28 * width), round(.28 * height)
    corners = (shape[:edge_y, :edge_x], shape[:edge_y, -edge_x:],
               shape[-edge_y:, :edge_x], shape[-edge_y:, -edge_x:])
    return max(patch.mean() for patch in corners) <= .11 and .19 <= shape.mean() <= .58


def _shell(mask, contour, hole, left, width):
    x, y, w, h = cv2.boundingRect(contour)
    box = x, y, x + w, y + h
    if (not .059 * width <= w <= .142 * width or not .056 * width <= h <= .128 * width
            or not .88 <= w / h <= 1.40 or abs(x + left + w / 2 - width / 2) > .044 * width
            or mask.shape[0] - box[3] > .132 * width):
        return None
    area = cv2.contourArea(contour)
    if not .70 <= area / (w * h) <= .98 or not .66 <= cv2.contourArea(hole) / area <= .94:
        return None
    hx, hy, hw, hh = cv2.boundingRect(hole)
    if not (.76 * w <= hw <= .95 * w and .75 * h <= hh <= .95 * h
            and abs(hx + hw / 2 - x - w / 2) <= .026 * w
            and abs(hy + hh / 2 - y - h / 2) <= .026 * h and _cross(mask, box)):
        return None
    return x + left, y, x + left + w, y + h


def _outline_plus(distance):
    width = distance.shape[1]
    left, right = round(.33 * width), round(.67 * width)
    center = distance[:, left:right]
    matches = []
    for contrast in (51, 78, 111):
        mask = np.uint8(center >= contrast)
        contours, links = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if links is None:
            continue
        for outline, (_, _, child, parent) in zip(contours, links[0], strict=True):
            if parent >= 0 or child < 0:
                continue
            box = _shell(mask, outline, contours[child], left, width)
            if box is not None:
                matches.append((contrast, box))
    stable = [box for contrast, box in matches if any(
        other != contrast and max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= .007 * width
        for other, peer in matches)]
    if stable and all(max(abs(a - b) for a, b in zip(stable[0], box, strict=True)) <= .009 * width for box in stable):
        return stable[len(stable) // 2]
    return None


def _emphasized(distance, plus, owners):
    width = distance.shape[1]
    _, top, _, bottom = plus
    rise = round(.17 * (bottom - top))
    edges, spans = _spans(width, len(owners) + 2)
    central = len(spans) // 2
    peers = [(box[0] + box[2]) / 2 for box, _, _ in owners]
    tones = []
    for i, (left, right) in enumerate(spans):
        if i == central:
            continue
        area = distance[top + rise:bottom - rise, left:right]
        visible = area[area >= 27]
        if not visible.size:
            return False
        strength = float(np.percentile(visible, 77))
        tones.append((any(edges[i] <= center < edges[i + 1] for center in peers), strength))
    active = [value for peer, value in tones if not peer]
    return len(active) == 1 and active[0] >= max(value for peer, value in tones if peer) + 27


def _social_rail(frame, regions, footer, glyphs):
    h, w = frame.shape[:2]
    candidates = [row for row in _side_candidates({"candidates": regions.get("candidates", ())}, w, h)
                  if row["center"][1] < footer - .04 * w]
    marks = []
    for mark in glyphs(candidates):
        left, top, right, bottom = mark["box"]
        if mark["role"] in {"comments", "share"}:
            marks.append({**mark, "center": ((left + right) / 2, (top + bottom) / 2), "height": bottom - top})
    return {mark["role"] for mark in _video_rail(frame, marks, candidates)} >= {"comments", "share"}


def _adjacent_amount(gray, mark, index):
    x, top, right, bottom = mark.xyxy()
    gap = max(2, round(.10 * (right - x)))
    height = bottom - top
    field = gray[max(0, top - gap):bottom + gap, right + gap:right + round(2.8 * height)]
    if not field.size or np.median(field) < 207:
        return False
    votes = []
    for level in (164, 177, 191):
        ink = np.uint8(field <= level)
        spans = _bands(ink.any(axis=0))
        digits = []
        for a, b in spans[:2]:
            glyph = ink[:, a:b]
            _, _, w, h = cv2.boundingRect(glyph)
            if not .48 * height <= h <= 1.02 * height or not .24 <= w / h <= .86:
                break
            rank = index.rank(glyph, alphabet="code", limit=3)
            if not rank or not rank[0]["text"].isdigit() or rank[0]["distance"] > .22:
                break
            digits.append(rank[0]["text"])
        if len(digits) == 2:
            votes.append("".join(digits))
    return len(votes) >= 2 and len(set(votes)) == 1


def _social_gallery(frame, footer, index):
    h, w = frame.shape[:2]
    top = round(.47 * h)
    if footer <= top:
        return False
    gray = cv2.cvtColor(frame[top:footer], cv2.COLOR_BGR2GRAY)
    scale = min(1., 570 / w)
    search = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    supported = {}
    for box, contrast in _heart_candidates(cv2.bitwise_not(search), round(.034 * search.shape[1])):
        supported.setdefault(box.xyxy(), set()).add(contrast)
    measured = []
    for bounds, evidence in supported.items():
        if len(evidence) < 2:
            continue
        box = _Box(*(round(coord / scale) for coord in bounds))
        if _adjacent_amount(gray, box, index):
            measured.append(box)
    return any(.31 * w <= abs(a.center[0] - b.center[0]) <= .58 * w and abs(a.center[1] - b.center[1]) <= .28 * h
               for a in measured for b in measured)


def _create_context(frame, regions, index, footer, labels, paper, glyphs):
    communication = {"消息", "私信", "訊息", "私訊", "messages", "inbox"}
    account = {"我", "我的", "个人主页", "個人主頁", "me", "profile", "account"}
    names = {name.casefold() for name in labels}
    if not names.intersection(communication) or not names.intersection(account):
        return False
    if float(np.median(paper)) < 78:
        return _social_rail(frame, regions, footer, glyphs or CandidateGlyphs(frame))
    return _social_gallery(frame, footer, index)


def social_footer_actions(frame, regions, index, *, glyphs=None):
    """Read peer labels and name create only when the visible body corroborates posting."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or index is None or width < 240 or not 1.48 * width <= height <= 2.72 * width:
        return []
    offset = height - round(.255 * width)
    tile = frame[offset:]
    bottom = tile[-round(.026 * width):, round(.16 * width):round(.84 * width)]
    paper = np.median(bottom[::2, ::4], axis=(0, 1))
    if np.ptp(paper) > 23:
        return []
    distance = _distance(tile, paper)
    plus = _outline_plus(distance)
    if plus is None:
        return []
    center_left, top, center_right, bottom = plus
    inset = round(.062 * width)
    if (distance[top:bottom, inset:width - inset] <= 13).mean() < .78:
        return []
    owners = _text_only(tile, distance, plus)
    if not owners or not _emphasized(distance, plus, owners):
        return []
    cache, active, printed = {}, [], []
    for (a, y, b, z), (left, first, right, last), (cx, cy) in owners:
        label = _read_caption(frame, (a, y + offset, b, z + offset), index, cache)
        if label:
            printed.append(label)
            active.append(_action(label, [left, first + offset, right, last + offset], [cx, cy + offset]))
    if active and _create_context(frame, regions, index, offset + top, printed, paper, glyphs):
        box = [center_left, offset + top, center_right, offset + bottom]
        active.append({"role": "open_post_composer", "function": "打开创作", "box": box,
                       "click_point": [(center_left + center_right) // 2, offset + (top + bottom) // 2],
                       "preview_text": "Create"})
    return active
