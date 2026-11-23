"""Confirm a short-tailed speech bubble in repeated horizontal post controls."""

from itertools import combinations

import cv2
import numpy as np

from inventory_list_rows import _Box
from inventory_small_reaction_glyphs import _heart_shape
from screen_counted_post_actions import _count_is_printed
from screen_horizontal_post_actions import _caption, _mark_boxes
from screen_portrait_peer_actions import _contours


def _short_tail(mask, contour, hole):
    left, top, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    _, y, width, height = cv2.boundingRect(hole)
    if (not .86 <= w / h <= 1.35 or not .67 <= area / (w * h) <= .87
            or not .61 <= cv2.contourArea(hole) / area <= .86
            or not .69 * w <= width <= .94 * w or not .66 * h <= height <= .9 * h
            or not .055 * h <= top + h - y - height <= .16 * h):
        return False
    body = cv2.drawContours(np.zeros((h, w), np.uint8), [contour], 0, 1, cv2.FILLED, offset=(-left, -top))
    upper, toe = np.split(body, [round(.82 * h)])
    symmetric = (upper & upper[:, ::-1]).sum() >= .91 * (upper | upper[:, ::-1]).sum()
    asymmetry = (toe & toe[:, ::-1]).sum() <= .72 * (toe | toe[:, ::-1]).sum()
    bottom = np.nonzero(mask[top + round(.92 * h):top + h, left:left + w])[1]
    return bool(symmetric and asymmetry and bottom.size
                and abs(np.median(bottom) - w / 2) >= .12 * w)


def _shapes(gray, box, purpose):
    for mark in _mark_boxes(gray, box, "share" if purpose == "share" else "leading"):
        left, top, right, bottom = mark
        patch = gray[top - 2:bottom + 2, left - 2:right + 2]
        if patch.size == 0:
            continue
        if purpose != "share":
            votes = 0
            for threshold in (139, 166, 193):
                mask = patch < threshold
                parts = _contours(mask)
                if len(parts) != 1 or len(parts[0][1]) != 1:
                    continue
                outline, (hole,) = parts[0]
                if purpose == "comments":
                    votes += _short_tail(mask, outline, hole)
                else:
                    x, y, w, h = cv2.boundingRect(outline)
                    votes += _heart_shape(255 - patch[y:y + h, x:x + w], outline, hole)
            if votes < 2:
                continue
        yield box, mark, cv2.resize(np.uint8(patch < 158), (24, 24), interpolation=cv2.INTER_NEAREST)


def _counter(gray, owner, mark, index, cache):
    _, top, right, bottom = owner
    _, a, r, b = mark
    height = b - a
    x, y = r + max(2, round(.13 * height)), max(top, a - 2)
    tile = gray[y:min(bottom, b + 4), x:right - 3]
    if not tile.size:
        return False
    dx, dy, w, h = cv2.boundingRect(np.uint8(tile < 150))
    if not (.33 * height <= w <= 2.78 * height and .47 * height <= h <= .84 * height
            and .13 * height <= dx <= .73 * height):
        return False
    return _count_is_printed(gray, _Box(x + dx, y + dy, x + dx + w, y + dy + h), index, cache)


def _align(first, other, width):
    _, mark, raster = first
    _, following, pixels = other
    y = (mark[1] + mark[3]) / 2
    f = (following[1] + following[3]) / 2
    return (abs((mark[0] + mark[2]) - (following[0] + following[2])) <= .086 * width
            and .11 * width <= abs(y - f) <= 1.63 * width
            and (raster & pixels).sum() >= .79 * (raster | pixels).sum())


def repeated_counter_comments(frame, regions, index):
    """Use separate share, comment and heart marks, each repeated on the page."""
    height, width = frame.shape[:2]
    if (index is None or regions["layer"] != "page" or regions["regions"]
            or width < 285 or not 1.47 * width <= height <= 2.82 * width):
        return []
    groups = {"share": [], "comments": [], "heart": []}
    for row in regions["candidates"]:
        if row["visibility"] != "page":
            continue
        left, top, right, bottom = row["box"]
        x, y = row["center"]
        w, h = right - left, bottom - top
        if not (.045 * width <= min(h, w) <= max(h, w) <= .29 * width
                and .072 * height <= y <= .92 * height and .049 * width <= h <= .13 * width):
            continue
        group = "share" if x < .24 * width else "comments" if x <= .48 * width else "heart"
        if x <= .79 * width and (group != "share" or w >= 1.54 * h):
            groups[group].append((left, top, right, bottom))
    if min(map(len, groups.values())) < 2:
        return []
    eligible = {key: set() for key in groups}
    baselines = []
    for middle in groups["comments"]:
        cy = (middle[1] + middle[3]) / 2
        depth = middle[3] - middle[1]
        peers = {role: [box for box in groups[role]
                        if abs((box[1] + box[3]) / 2 - cy) <= .28 * depth]
                 for role in ("share", "heart")}
        if all(peers.values()):
            baselines.append(cy)
            eligible["comments"].add(middle)
            for role, boxes in peers.items():
                eligible[role].update(boxes)
    if len(baselines) < 2 or max(baselines) - min(baselines) < .11 * width:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marks = {role: [hit for box in sorted(proposals) for hit in _shapes(gray, box, role)]
             for role, proposals in eligible.items()}
    rows = []
    for mid in marks["comments"]:
        _, mark, _ = mid
        x, y = (mark[0] + mark[2]) / 2, (mark[1] + mark[3]) / 2
        size = mark[3] - mark[1]
        aligned = lambda peer: (abs((peer[1][1] + peer[1][3]) / 2 - y) <= .24 * size
                                and .84 * size <= peer[1][3] - peer[1][1] <= 1.27 * size)
        first = [peer for peer in marks["share"] if aligned(peer) and .15 * width <= x - peer[1][0] <= .34 * width]
        last = [peer for peer in marks["heart"] if aligned(peer) and .18 * width <= peer[1][0] - x <= .32 * width]
        if len(first) == len(last) == 1:
            rows.append((first[0], mid, last[0]))
    cache, proved = {}, {}
    for first, second in combinations(rows, 2):
        if not all(_align(one, two, width) for one, two in zip(first, second, strict=True)):
            continue
        validated = []
        for share, comment, _ in (first, second):
            owner, mark, _ = comment
            if _caption(frame, share[0], share[1], index, cache) not in {"share", "分享"}:
                break
            if not _counter(gray, owner, mark, index, cache):
                break
            point = [(mark[0] + mark[2]) // 2, (mark[1] + mark[3]) // 2]
            if any(l <= point[0] < r and t <= point[1] < b for l, t, r, b in regions.get("floating_controls", ())):
                break
            validated.append((owner, point))
        if len(validated) == 2:
            proved.update(validated)
    return [{"role": "comments", "function": "打开评论", "box": list(owner), "click_point": point, "preview_text": "Comments"}
            for owner, point in proved.items()]
