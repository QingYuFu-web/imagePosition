"""Recognize independent buttons on an evidenced video reaction rail."""

from itertools import combinations, pairwise
from statistics import median

import cv2
import numpy as np

from inventory_back_evidence import _elements
from inventory_comment_evidence import _AUTHOR, _COUNT, _MEDIA, _counted_controls, _reliable_text, _speech_bubble
from inventory_list_rows import _Box


def _count_boxes(observation, width, height):
    points = observation.get("keypoints", ())
    roles = {row["id"]: row.get("role") for row in (observation.get("screen_context") or {}).get("controls", ())}
    boxes = [_Box(*item.box) for item in _elements(observation, width, height)
             if item.kind == "text" and item.confidence >= .85 and _COUNT.fullmatch(item.label.strip())]
    for region in _counted_controls(points, width, height, roles):
        left, top, right, bottom = region.box
        # A detector may own the OCR counter. Its text lies in the lower part
        # of the combined icon-and-count box, on the same vertical axis.
        count = _Box(left, top + round(.67 * region.height), right, bottom)
        if not any(abs(old.center[0] - count.center[0]) < .25 * count.width
                   and abs(old.center[1] - count.center[1]) < .66 * count.height for old in boxes):
            boxes.append(count)
    return sorted((box for box in boxes if .010 * width <= box.height <= .064 * width
                   and min(box.center[0], width - box.center[0]) <= .23 * width
                   and .41 * height <= box.center[1] <= .92 * height), key=lambda box: box.center[1])


def _rail_pitch(counts, width):
    xs, ys = zip(*(box.center for box in counts))
    gaps = [lower - upper for upper, lower in pairwise(ys)]
    text_height = median(box.height for box in counts)
    if (max(xs) - min(xs) > min(.022 * width, .65 * text_height)
            or not 2.35 * text_height <= min(gaps) <= max(gaps) <= 6.6 * text_height
            or max(gaps) > 1.32 * min(gaps)):
        return None
    return median(gaps)


def _marks(gray, count, pitch, *, extra=0):
    x, _ = count.center
    top = max(0, count.top - round(.61 * pitch))
    bottom = count.top - max(1, round(.08 * count.height))
    left, right = max(0, round(x - .36 * pitch)), min(gray.shape[1], round(x + .36 * pitch))
    patch = gray[top:bottom, left:right]
    if min(patch.shape, default=0) < 12:
        return
    baseline = np.median(patch)
    if baseline > 196:
        return
    brightness = max(174, round(baseline + 45)) + extra
    contours, hierarchy = cv2.findContours(np.uint8(patch >= brightness), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return
    owners = hierarchy[0][:, 3]
    for i, outline in enumerate(contours):
        if owners[i] >= 0:
            continue
        u, v, w, h = cv2.boundingRect(outline)
        if (not (.24 * pitch <= w <= .61 * pitch and .22 * pitch <= h <= .55 * pitch)
                or not (0 < u < u + w < patch.shape[1] and 0 < v < v + h < patch.shape[0])
                or abs(left + u + w / 2 - x) > .14 * pitch):
            continue
        area = cv2.contourArea(outline)
        if area < .30 * w * h:
            continue
        box = _Box(left + u, top + v, left + u + w, top + v + h)
        holes = [hole for hole, owner in zip(contours, owners) if owner == i]
        yield box, (u, v, w, h), area, outline, holes


def _curved_share(mark):
    _, (u, v, w, h), area, outline, holes = mark
    if (not .90 <= w / h <= 1.49 or not .36 <= area / (w * h) <= .63
            or any(cv2.contourArea(hole) >= .015 * area for hole in holes)):
        return False
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [outline], 0, 1, cv2.FILLED, offset=(-u, -v))
    narrow = max(1, round(.15 * w))
    tail = np.nonzero(mask[:, :narrow])[0]
    head = np.nonzero(mask[:, -narrow:])[0]
    if not (.65 * h <= np.median(tail) <= .87 * h and .39 * h <= np.median(head) <= .61 * h):
        return False
    heights = mask.sum(axis=0)
    tallest = np.flatnonzero(heights >= .89 * h)
    if not tallest.size or not .43 * w <= np.median(tallest) <= .70 * w:
        return False
    edge = mask.argmax(axis=0)
    if np.median(edge[round(.04 * w):narrow]) - np.median(edge[round(.25 * w):round(.43 * w)]) < .13 * h:
        return False
    # The head tapers toward a mid-height right apex, independently of the tail.
    head_spans = [np.flatnonzero(mask[:, round(fraction * (w - 1))]) for fraction in (.62, .80, .94)]
    return (all(layer.size and .34 * h <= layer.mean() <= .66 * h for layer in head_spans)
            and all(outer.size - inner.size >= .14 * h for outer, inner in pairwise(head_spans)))


def _confirmed_mark(gray, count, pitch, verify):
    first = [mark[0] for mark in _marks(gray, count, pitch) if verify(mark)]
    second = [mark[0] for mark in _marks(gray, count, pitch, extra=24) if verify(mark)]
    if len(first) != 1 or len(second) != 1:
        return None
    box, peer = first[0], second[0]
    if max(abs(a - b) for a, b in zip(box.xyxy(), peer.xyxy(), strict=True)) > .055 * pitch:
        return None
    return box


def _reaction_rails(observation, width, height):
    if observation.get("active_layer"):
        return
    if height <= width or width < 200:
        return
    points = observation.get("keypoints", ())
    if not any(_MEDIA.search(text) for point in points for text in _reliable_text(point, .88)):
        return
    authors = [point["screen_fraction"] for point in points if point.get("screen_fraction")
               and any(_AUTHOR.fullmatch(text) for text in _reliable_text(point, .88))]
    if not authors:
        return
    for leading, comment, share in combinations(_count_boxes(observation, width, height), 3):
        pitch = _rail_pitch((leading, comment, share), width)
        if pitch is None or not any(
                abs(ax * width - comment.center[0]) >= 1.48 * pitch
                and leading.top - .67 * pitch <= ay * height <= share.bottom + .25 * pitch
                for ax, ay in authors):
            continue
        yield leading, comment, share, pitch


def find_video_share_box(gray, observation):
    """Return one half-open box when the same reaction rail owns comment and share."""
    if gray.ndim != 2:
        return None
    matches = set()
    for leading, comment, share, pitch in _reaction_rails(observation, gray.shape[1], gray.shape[0]):
        if (not any(_speech_bubble(*glyph[1:]) for glyph in _marks(gray, comment, pitch))
                or next(_marks(gray, leading, pitch), None) is None):
            continue
        box = _confirmed_mark(gray, share, pitch, _curved_share)
        if box is not None:
            matches.add(box.xyxy())
    return next(iter(matches)) if len(matches) == 1 else None


def _thumb_with_cuff(gray, mark):
    body, (u, v, width, height), area, outline, holes = mark
    if not (.55 <= width / height <= 1.03 and .54 <= area / (width * height) <= .88
            and not any(cv2.contourArea(hole) >= .02 * area for hole in holes)):
        return False
    silhouette = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(silhouette, [outline], 0, 1, cv2.FILLED, offset=(-u, -v))
    tip = silhouette[round(.07 * height):round(.22 * height)]
    hand = silhouette[round(.49 * height):round(.79 * height)]
    if np.median(tip.sum(axis=1)) > .42 * width or np.median(hand.sum(axis=1)) < .74 * width:
        return False
    left = max(0, body.left - round(.72 * width))
    top = body.top + round(.18 * height)
    bottom = min(gray.shape[0], body.bottom + round(.08 * height))
    cuff = gray[top:bottom, left:body.left]
    if not cuff.size:
        return False
    bright = max(181, round(np.median(cuff) + 40))
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(cuff >= bright), connectivity=8)
    return any(.12 * width <= w <= .41 * width and .44 * height <= h <= .84 * height
               and .025 * width <= body.left - (left + x + w) <= .26 * width
               and abs(top + y + h - body.bottom) <= .09 * height
               and area >= .68 * w * h for x, y, w, h, area in stats[1:])


def _comment_mark(mark):
    return _speech_bubble(*mark[1:])


def suggest_counted_comment(frame, observation):
    """Recover the grounded comment icon even if a neighboring glyph is absent."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None
    height, width = frame.shape[:2]
    rails = tuple(_reaction_rails(observation, width, height))
    if not rails:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    matches = set()
    for leading, comment, _, pitch in rails:
        if not any(_thumb_with_cuff(gray, mark) for mark in _marks(gray, leading, pitch)):
            continue
        box = _confirmed_mark(gray, comment, pitch, _comment_mark)
        if box is not None:
            matches.add(box)
    if len(matches) != 1:
        return None
    box = next(iter(matches))
    left, top, right, bottom = box.left, box.top, box.right - 1, box.bottom - 1
    sx, sy = width - 1, height - 1
    return {"role": "comments", "function": "打开视频评论", "anchor_id": None,
            "box_fraction": [left / sx, top / sy, right / sx, bottom / sy],
            "screen_fraction": [(left + right) / (2 * sx), (top + bottom) / (2 * sy)]}
