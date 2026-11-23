"""Prove horizontal post actions from their painted carrier and visible ink."""

from dataclasses import dataclass
import re
import unicodedata

import cv2
import numpy as np

from inventory_outline_buttons import _rectangles
from inventory_share_video import _curved_share
from screen_candidate_glyphs import CandidateGlyphs
from screen_caption_transcript import read_caption


Box = tuple[int, int, int, int]
_COMMENT = re.compile(r"(?:[0-9][0-9,.]*(?:[km万萬])?)?(?:comments?|(?:条|條)?(?:评论|評論|留言))")
_SHARE = {"share", "分享"}
_LIKE = {"like", "点赞", "點讚"}


@dataclass(frozen=True)
class _Owner:
    crop: dict
    outline: Box | None
    mark: Box


def _center(box):
    left, top, right, bottom = box
    return ((left + right) / 2, (top + bottom) / 2)


def _size(box):
    return box[2] - box[0], box[3] - box[1]


def _close(first, second, tolerance):
    return max(abs(a - b) for a, b in zip(first, second, strict=True)) <= tolerance


def _caption(frame, box, glyph, index, cache):
    _, top, right, bottom = box
    gl, gt, gr, gb = glyph
    size = gb - gt
    x0 = gr + max(2, round(.12 * size))
    x1 = right - max(2, round(.13 * size))
    y0, y1 = max(top, gt - round(.07 * size)), min(bottom, gb + round(.07 * size))
    if x1 - x0 < max(16, .73 * size):
        return None
    reading = read_caption(frame, (x0, y0, x1, y1), index, cache=cache, numeric=True)
    if reading is None:
        return None
    lx, ly = _center(reading["line_box"])
    if (abs(ly - (gt + gb) / 2) > .21 * size
            or reading["line_box"][0] - gr > .65 * size or lx <= (gl + gr) / 2):
        return None
    return "".join(unicodedata.normalize("NFKC", reading["text"]).casefold().split())


def _bubble(contour, holes, bounds):
    left, top, w, h = bounds
    if not .80 <= w / h <= 1.30 or not .64 <= cv2.contourArea(contour) / (w * h) <= .88:
        return False
    inside = [hole for hole in holes if cv2.contourArea(hole) >= .024 * w * h]
    if len(inside) != 1:
        return False
    x, y, span, depth = cv2.boundingRect(inside[0])
    if not (.53 * w <= span <= .96 * w and .48 * h <= depth <= .92 * h
            and .40 <= cv2.contourArea(inside[0]) / (w * h) <= .74
            and .073 * h <= top + h - y - depth <= .34 * h
            and abs(x + span / 2 - left - w / 2) <= .15 * w):
        return False
    solid = cv2.drawContours(np.zeros((h, w), np.uint8), [contour], 0, 1, cv2.FILLED, offset=(-left, -top))
    x_tip = np.flatnonzero(solid[-max(1, round(.085 * h)):].any(axis=0))
    return bool(x_tip.size and x_tip.size <= .43 * w and abs(float(x_tip.mean()) - w / 2) >= .10 * w)


def _mark_boxes(gray, box, role):
    left, top, right, bottom = box
    w, h = _size(box)
    patch = gray[top:bottom, left:right]
    paper = float(np.median(patch))
    if paper < 219 or (patch >= paper - 19).mean() < .78:
        return []
    found = []
    for level in (48, 57, 66, 84, 111):
        ink = np.uint8(patch <= paper - level)
        contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for i, contour in enumerate(contours):
            if hierarchy[0, i, 3] >= 0:
                continue
            x, y, gw, gh = bounds = cv2.boundingRect(contour)
            if (not .30 * h <= gh <= .68 * h or not .25 * h <= gw <= .76 * h
                    or abs(y + gh / 2 - h / 2) > .21 * h or not 0 < x < x + gw < w):
                continue
            if role != "share" and x > .40 * w:
                continue
            if role == "comments":
                holes = [part for part, parent in zip(contours, hierarchy[0, :, 3], strict=True) if parent == i]
                if not _bubble(contour, holes, bounds):
                    continue
            elif role == "share" and not _curved_share((None, bounds, cv2.contourArea(contour), contour, [])):
                continue
            found.append((level, (left + x, top + y, left + x + gw, top + y + gh)))
    stable = []
    for level, mark in found:
        tolerance = max(2, round(.105 * h))
        if any(peer != level and _close(mark, area, tolerance) for peer, area in found):
            if not any(_close(mark, previous, tolerance) for previous in stable):
                stable.append(mark)
    return sorted(stable, key=lambda item: item[0])


def _outline(gray, owner, inner):
    left, top, right, bottom = owner
    w, h = _size(owner)
    pad = max(4, round(.16 * h))
    l, t = max(0, left - pad), max(0, top - pad)
    patch = gray[t:bottom + pad, l:right + pad]
    found = []
    for cuts in ((8, 25), (13, 34)):
        edge = cv2.Canny(patch, *cuts)
        for measured in _rectangles(edge, (l, t), inner):
            mw, mh = _size(measured)
            if (.80 * w <= mw <= 1.15 * w and .76 * h <= mh <= 1.14 * h
                    and _close(owner, measured, .23 * h)):
                found.append((cuts, tuple(map(int, measured))))
    stable = [box for level, box in found if any(level != other and _close(box, peer, .12 * h) for other, peer in found)]
    if not stable or not all(_close(stable[0], box, .12 * h) for box in stable):
        return None
    return max(stable, key=lambda box: _size(box)[0] * _size(box)[1])


def _wide_line(plain, nominal, height):
    first, last = max(0, round(nominal - .16 * height)), min(plain.shape[0], round(nominal + .17 * height))
    return bool(first < last and np.any((plain[first:last] <= 246).mean(axis=1) >= .86))


def _left_like(frame, gray, rows, comment, share, index, cache):
    width = gray.shape[1]
    centers = [_center(x["box"])[0] for x in (*rows, comment, share)]
    cx, sx = centers[-2:]
    lefts = [item for item, lx in zip(rows, centers[:-2], strict=True)
             if abs((cx - lx) - (sx - cx)) <= .086 * width and .08 * width < lx < .29 * width]
    if not lefts or cx < .38 * width or not .80 * width <= sx <= .93 * width:
        return False
    h = _size(comment["box"])[1]
    plain = gray[:, round(.065 * width):round(.935 * width):3]
    if not (_wide_line(plain, comment["box"][1], h) and _wide_line(plain, comment["box"][3], h)):
        return False
    for item in lefts:
        marks = _mark_boxes(gray, item["box"], "leading")
        if marks and _caption(frame, item["box"], marks[0], index, cache) in _LIKE:
            return True
    return False


def _action(role, owner):
    name, preview = ("打开评论", "Comments") if role == "comments" else ("打开分享菜单", "Share")
    box = owner.outline or tuple(owner.crop["box"])
    target = box if role == "comments" else owner.mark
    return {"role": role, "function": name, "box": list(box), "preview_text": preview,
            "click_point": [round(v) for v in _center(target)]}


def horizontal_post_actions(frame, regions, index, *, glyphs=None):
    """Return page-owned comment and share controls, each with one click point."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or width < 240 or height < 1.20 * width:
        return []
    possible = [row for row in regions["candidates"] if row["visibility"] == "page"
                and .114 * height <= row["center"][1] <= .87 * height
                and .047 * width <= _size(row["box"])[1] <= .138 * width
                and .052 * width <= _size(row["box"])[0] <= .44 * width]
    middle = [row for row in possible if .27 * width <= row["center"][0] <= .66 * width
              and _size(row["box"])[0] >= .15 * width]
    trailing = [row for row in possible if .77 * width <= row["center"][0] <= .958 * width]
    if not middle or not trailing:
        return []
    glyphs = glyphs or CandidateGlyphs(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cache = {}
    found = []
    for comment in middle:
        l, t, r, b = comment["box"]
        h = b - t
        peers = [row for row in possible if row["box"][2] <= l - .008 * width
                 and abs(row["center"][1] - comment["center"][1]) <= .16 * h]
        last = [row for row in trailing if row["box"][0] >= r + .009 * width
                and max(abs(t - row["box"][1]), abs(b - row["box"][3])) <= .21 * h]
        if not peers or not last:
            continue
        speech = [mark for mark in _mark_boxes(gray, comment["box"], "comments")
                  if (label := _caption(frame, comment["box"], mark, index, cache)) and _COMMENT.fullmatch(label)]
        if len(speech) != 1:
            continue
        bubble = speech[0]
        ink = (bubble[0], bubble[1], r - round(.08 * (r - l)), bubble[3])
        comment_outline = _outline(gray, comment["box"], ink)
        pairs = []
        for share in last:
            marks = _mark_boxes(gray, share["box"], "share")
            if not marks:
                marks = [tuple(item["box"]) for item in glyphs([share]) if item["role"] == "share"]
            if len(marks) != 1:
                continue
            mark = marks[0]
            share_outline = _outline(gray, share["box"], mark)
            share_text = _caption(frame, share["box"], mark, index, cache)
            if comment_outline and share_outline and _close((comment_outline[1], comment_outline[3]),
                                                             (share_outline[1], share_outline[3]), .12 * h):
                if share_text is not None and share_text not in _SHARE:
                    continue
            elif not (not comment_outline and not share_outline and share_text in _SHARE
                      and _left_like(frame, gray, peers, comment, share, index, cache)):
                continue
            pairs.append((_Owner(comment, comment_outline, bubble), _Owner(share, share_outline, mark)))
        if len(pairs) == 1 and not any(abs(t - previous[0].crop["box"][1]) <= .25 * h for previous in found):
            found.append(pairs[0])
    return [_action(role, item) for pair in found for role, item in zip(("comments", "share"), pair, strict=True)]
