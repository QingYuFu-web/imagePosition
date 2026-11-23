"""Verify a counted comment control within a compact post footer."""

import re

import cv2
import numpy as np

from inventory_counted_reaction_column import _outline_thumb
from inventory_list_rows import _Box
from inventory_unlabeled_engagement_row import _network
from screen_candidate_glyphs import _overlap
from screen_caption_transcript import read_caption
from screen_inline_post_comments import _separator


_COUNT = re.compile(r"[0-9]{1,3}(?:[.,][0-9]{1,3}){0,2}(?:[kKmM万萬])?\+?")
_LIKE = {"like", "点赞", "點讚", "赞", "讚"}


def _bubble(outline, hole, width, height):
    if not (len(hole) >= 8 and .80 <= width / height <= 1.32
            and .69 <= cv2.contourArea(outline) / (width * height) <= .92
            and .45 <= cv2.contourArea(hole) / (width * height) <= .78):
        return False
    (cx, cy), spans, _ = cv2.fitEllipse(hole)
    radius = .25 * sum(spans)
    if not .80 <= min(spans) / max(spans) or not 0 < radius:
        return False
    points = outline.reshape(-1, 2) - (cx, cy)
    distances = np.linalg.norm(points, axis=1)
    usual = float(np.median(distances))
    if not (1.02 <= usual / radius <= 1.36 and (abs(distances - usual) <= .092 * usual).mean() >= .78):
        return False
    tail = points[distances >= 1.11 * usual]
    if len(tail) < 4 or len(tail) > .21 * len(points) or max(distances) > 1.44 * usual:
        return False
    mean = tail.mean(axis=0)
    if not (mean[1] >= .36 * usual and .30 * usual <= abs(mean[0]) <= 1.10 * usual):
        return False
    # A callout's tail connects to a locally pinched cavity. This distinguishes
    # a short-tail speech bubble from a circular lens with an external handle.
    hull = cv2.convexHull(hole, returnPoints=False)
    defects = cv2.convexityDefects(hole, hull) if len(hull) >= 3 else None
    if defects is None:
        return False
    direction = mean / np.linalg.norm(mean)
    return any(depth >= 256 * max(1.35, .062 * radius)
               and (hole[far, 0] - (cx, cy)) @ direction >= .53 * radius
               for _, _, far, depth in defects[:, 0])


def _solid_icon(gray, role):
    h, w = gray.shape
    paper = float(np.median(gray))
    if paper < 204 or (gray >= paper - 17).mean() < .83:
        return None
    light = 255 - gray if role == "like" else None
    stable = []
    seen = set()
    for offset in (45, 80, 115, 150):
        ink = np.uint8(gray <= paper - offset)
        digest = ink.tobytes()
        if digest in seen:
            continue
        seen.add(digest)
        contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        parents = hierarchy[0, :, 3]
        for i, outline in enumerate(contours):
            x, y, width, height = cv2.boundingRect(outline)
            if (parents[i] >= 0 or not (.34 * h <= height <= .63 * h and .77 <= width / height <= 1.32)
                    or not .34 * h <= y + height / 2 <= .63 * h
                    or not 0 < x < x + width < w or not 0 < y < y + height < h):
                continue
            center = x + width / 2
            bbox = _Box(x, y, x + width, y + height)
            holes = [contour for contour, owner in zip(contours, parents, strict=True) if owner == i]
            if role == "share":
                matched = abs(center - w / 2) <= .14 * w and _network(bbox, holes)
            elif role == "comments":
                matched = (.20 * w <= center <= .44 * w and len(holes) == 1
                           and _bubble(outline, holes[0], width, height))
            else:
                matched = (.15 * w <= center <= .43 * w and len(holes) == 2
                           and _outline_thumb(light, bbox) is not None)
            if matched:
                stable.append((offset, bbox.xyxy()))
    matches = [box for cut, box in stable if any(other != cut and _overlap(box, peer) >= .79
                                               for other, peer in stable)]
    return matches[len(matches) // 2] if matches and all(_overlap(matches[0], box) >= .79 for box in matches) else None


def _rows(boxes, width, height):
    choices = [box for box in boxes if .48 * width <= box.center[0] <= .975 * width
               and .145 * height <= box.center[1] <= .925 * height
               and .051 * width <= box.height <= .123 * width
               and .068 * width <= box.width <= .27 * width]
    for middle in choices:
        scale = middle.height
        if not .92 * scale <= middle.width <= 2.20 * scale:
            continue
        x, y = middle.center
        row = [peer for peer in choices if .09 * width <= abs(peer.center[0] - x) <= .235 * width
               and abs(peer.center[1] - y) <= .12 * scale]
        lefts = [peer for peer in row if peer.right <= middle.left and .70 * scale <= peer.height <= 1.24 * scale
                 and peer.width > middle.width * .89]
        rights = [peer for peer in row if peer.left >= middle.right and .76 * scale <= peer.height <= 1.22 * scale
                  and .61 * scale <= peer.width <= 1.25 * scale]
        for left in lefts:
            for right in rights:
                if .65 <= (x - left.center[0]) / (right.center[0] - x) <= 1.58:
                    yield left, middle, right


def _owns_artwork(gray, boxes, middle, right):
    for media in boxes:
        if (media.left < middle.left - .17 * gray.shape[1] or abs(media.right - right.right) > .68 * middle.height
                or not -.18 * middle.height <= middle.top - media.bottom <= .62 * middle.height
                or not 2.12 * middle.height <= min(media.width, media.height) <= 6.0 * middle.height
                or not .71 <= media.width / media.height <= 1.69):
            continue
        crop = gray[media.top:media.bottom:4, media.left:media.right:4]
        if crop.size and np.std(crop) >= 27:
            return True
    return False


def _reading(frame, box, mark, index, cache):
    _, top, right, bottom = box.xyxy()
    _, gt, gr, gb = mark
    size = gb - gt
    area = (gr + max(2, round(.24 * size)), max(top, gt - 1),
            right - max(2, round(.14 * size)), min(bottom, gb + 2))
    if area[2] - area[0] < .34 * size:
        return None
    result = read_caption(frame, area, index, cache=cache, numeric=True, choice=True)
    if result is None:
        return None
    left, lt, _, lb = result["line_box"]
    text = result["text"].casefold().strip()
    return text if (.18 * size <= left - gr <= .74 * size and .42 * size <= lb - lt <= .76 * size
                    and abs((lt + lb - gt - gb) / 2) <= .15 * size) else None


def compact_post_comments(frame, regions, index):
    """Return the typed comment icon inside a verified page-owned action row."""
    height, width = frame.shape[:2]
    if (index is None or regions.get("layer") != "page" or regions.get("regions")
            or width < 300 or not 1.48 * width <= height <= 2.88 * width):
        return []
    boxes = [_Box(*row["box"]) for row in regions["candidates"]
             if row["visibility"] == "page" and 0 <= row["box"][0] < row["box"][2] <= width
             and 0 <= row["box"][1] < row["box"][3] <= height]
    rows = list(_rows(boxes, width, height))
    if not rows:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    observed, cache, results = {}, {}, []
    for left, middle, right in rows:
        if not (_separator(gray, middle.top, middle.bottom, middle.height) and _owns_artwork(gray, boxes, middle, right)):
            continue
        marks = []
        for box, role in zip((left, middle, right), ("like", "comments", "share"), strict=True):
            key = (box, role)
            if key not in observed:
                l, t, r, b = box.xyxy()
                mark = _solid_icon(gray[t:b, l:r], role)
                observed[key] = ((l + mark[0], t + mark[1], l + mark[2], t + mark[3]) if mark else None)
            marks.append(observed[key])
        if any(mark is None for mark in marks):
            continue
        heights = [box[3] - box[1] for box in marks]
        rows_at = [(box[1] + box[3]) / 2 for box in marks]
        if max(heights) > 1.22 * min(heights) or max(rows_at) - min(rows_at) > .20 * min(heights):
            continue
        hand, bubble, _ = marks
        if _reading(frame, left, hand, index, cache) not in _LIKE:
            continue
        count = _reading(frame, middle, bubble, index, cache)
        if count is None or not _COUNT.fullmatch(count):
            continue
        x, y = round((bubble[0] + bubble[2]) / 2), round((bubble[1] + bubble[3]) / 2)
        if any(l <= x < r and t <= y < d for l, t, r, d in regions.get("floating_controls", ())):
            continue
        owner = middle.xyxy()
        if not any(_overlap(owner, result["box"]) >= .75 for result in results):
            results.append({"role": "comments", "function": "打开评论", "box": list(owner),
                            "click_point": [x, y], "preview_text": "Comments"})
    return results
