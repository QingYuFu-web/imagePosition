"""Verify comment glyphs in counted media and text-only post footers."""

import cv2
import numpy as np

from inventory_counted_reaction_column import _outline_thumb
from inventory_list_rows import _Box
from screen_candidate_glyphs import _overlap
from screen_caption_transcript import _ranked_glyph, _shape_parts
from screen_compact_post_footer import (_COUNT, _LIKE, _owns_artwork, _reading, _rows, _solid_icon)
from screen_context_back import _dots_mark
from screen_horizontal_post_actions import _bubble
from screen_inline_post_comments import _separator


def _cluster(votes):
    stable = [box for index, box in votes if any(
        index != other and _overlap(box, peer) >= .81 for other, peer in votes)]
    return stable[len(stable) // 2] if stable and all(_overlap(stable[0], box) >= .81 for box in stable) else None


def _marks(gray, owner, role):
    left, top, right, bottom = owner.xyxy()
    tile = gray[top:bottom, left:right]
    old = _solid_icon(tile, role)
    if old:
        return tuple(p + offset for p, offset in zip(old, (left, top, left, top), strict=True))
    if role == "share":
        return None
    height, width = tile.shape
    paper = float(np.median(tile))
    if paper < 213 or (tile >= paper - 17).mean() < .83:
        return None
    contrast = paper - tile
    if role == "like":
        # The helper expects bright ink. Normalize strokes from different UI palettes.
        peak = float(np.percentile(contrast[:, :round(.41 * width)], 96))
        if peak < 70:
            return None
        light = np.uint8(np.clip(237 * contrast / peak, 0, 255))
    votes = []
    for cut in (48, 76, 103, 129):
        contours, tree = cv2.findContours(np.uint8(contrast >= cut), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if tree is None:
            continue
        parents = tree[0, :, 3]
        for i, contour in enumerate(contours):
            if parents[i] >= 0:
                continue
            x, y, w, h = bounds = cv2.boundingRect(contour)
            if not (.34 * height <= h <= .66 * height and .81 <= w / h <= 1.32
                    and .18 * width <= x + w / 2 <= .45 * width and abs(y + h / 2 - height / 2) <= .18 * height
                    and 0 < x < x + w < width and 0 < y < y + h < height):
                continue
            holes = [inside for inside, parent in zip(contours, parents, strict=True) if parent == i]
            if role == "like":
                valid = len(holes) == 2 and _outline_thumb(light, _Box(x, y, x + w, y + h)) is not None
            else:
                valid = _bubble(contour, holes, bounds)
            if valid:
                votes.append((cut, (left + x, top + y, left + x + w, top + y + h)))
    return _cluster(votes)


def _overflow(gray, owner):
    left, top, right, bottom = owner.xyxy()
    tile = gray[top:bottom, left:right]
    paper = float(np.median(tile))
    if paper < 219 or (tile >= paper - 17).mean() < .93:
        return None
    height, width = tile.shape
    votes = []
    for cut in (48, 75, 105, 130):
        ink = np.uint8(tile <= paper - cut)
        x, y, w, h = cv2.boundingRect(ink)
        if (.31 * width <= w <= .59 * width and .056 * height <= h <= .13 * height
                and abs(x + w / 2 - width / 2) <= .10 * width and abs(y + h / 2 - height / 2) < .11 * height
                and _dots_mark(ink)):
            votes.append((cut, (left + x, top + y, left + x + w, top + y + h)))
    return _cluster(votes)


def _digit_run(gray, owner, mark, index, cache):
    left, top, right, bottom = owner.xyxy()
    _, gt, gr, gb = mark
    size = gb - gt
    start = gr + max(2, round(.13 * size))
    tile = gray[max(top, gt - 1):min(bottom, gb + 2), start:right - max(2, round(.11 * size))]
    if not tile.size:
        return False
    paper = float(np.median(tile))
    reads = []
    for cut in (44, 55, 70, 84, 102, 110):
        ink = np.uint8(tile <= paper - cut)
        x, y, w, h = cv2.boundingRect(ink)
        if not (.42 * size <= h <= .76 * size and .77 * size <= w <= 2.42 * size
                and .09 * size <= start + x - gr <= .72 * size
                and abs(y + h / 2 - tile.shape[0] / 2) <= .15 * size):
            continue
        parts = _shape_parts(ink[y:y + h, x:x + w], h, choice=True)
        if not 2 <= len(parts) <= 5 or any(not .82 * h <= b - a <= h for _, a, _, b, _ in parts):
            continue
        ranks = [_ranked_glyph(index, cache, "latin", part[-1]) for part in parts]
        transcript = []
        distinct = 0
        for rank in ranks:
            if len(rank) < 2:
                break
            best, next_best = rank[:2]
            text, score = best["text"], best["distance"]
            if text == "0" or text in {"o", "O"}:
                zero = next((opt["distance"] for opt in rank if opt["text"] == "0"), None)
                if zero is None or zero > .12 or zero - score > .046:
                    break
                transcript.append("0")
            elif text.isascii() and text.isdigit() and score <= .27 and next_best["distance"] - score >= .025:
                transcript.append(text)
                distinct += 1
            else:
                break
        if len(transcript) == len(ranks) and distinct:
            reads.append(tuple(transcript))
    return any(reads.count(reading) >= 2 for reading in reads)


def _wide_media(gray, boxes, comment, more):
    width = gray.shape[1]
    for item in boxes:
        if (not .74 * width <= item.width <= width or not 2.2 * comment.height <= item.height <= 8.0 * comment.height
                or not -.10 * comment.height <= comment.top - item.bottom <= .46 * comment.height
                or item.left >= .16 * width or abs(item.right - more.right) > .58 * comment.height):
            continue
        region = gray[item.top:item.bottom:4, item.left:item.right:4]
        if region.size and all(np.std(chip) >= 22 for chip in np.array_split(region, 3, axis=1)):
            return True
    return False


def _tops_match(gray, first, anchor):
    """Check that a text-owned footer repeats one measured media footer."""
    first_boxes, first_icons, first_owner = first
    second_boxes, second_icons, _ = anchor
    _, bubble, _ = first_icons
    y = (bubble[1] + bubble[3]) / 2
    gap = abs((second_icons[1][1] + second_icons[1][3]) / 2 - y)
    size = bubble[3] - bubble[1]
    if (first_owner or not 4 * size <= gap <= .64 * len(gray)
            or max(abs(p.center[0] - q.center[0]) for p, q in zip(first_boxes, second_boxes, strict=True)) > .85 * size):
        return False
    for one, two in zip(first_icons, second_icons, strict=True):
        w, h = one[2] - one[0], one[3] - one[1]
        other_w, other_h = two[2] - two[0], two[3] - two[1]
        if not (.82 <= w / other_w <= 1.22 and .84 <= h / other_h <= 1.20):
            return False
        a, b, c, d = one
        p, q, r, s = two
        first_mask = cv2.resize(np.uint8(gray[b:d, a:c] <= 156), (24, 24), interpolation=cv2.INTER_NEAREST)
        second_mask = cv2.resize(np.uint8(gray[q:s, p:r] <= 156), (24, 24), interpolation=cv2.INTER_NEAREST)
        if (first_mask & second_mask).sum() < .72 * (first_mask | second_mask).sum():
            return False
    # The top cutoff can remove the media, while a few text lines are still visible above the controls.
    top = max(0, round(first_boxes[1].top - 2.85 * size))
    bottom = max(top, round(first_boxes[1].top - .28 * size))
    body = gray[top:bottom, round(.03 * gray.shape[1]):round(.67 * gray.shape[1])]
    if not body.size:
        return False
    parts, _ = cv2.findContours(np.uint8(body <= 153), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glyphs = [box for part in parts if .25 * size <= (box := cv2.boundingRect(part))[3] <= .74 * size
              and .105 * size <= box[2] <= .85 * size]
    for _, t, _, h in glyphs:
        neighbors = [x for x, q, _, z in glyphs if abs(t + h - q - z) <= .10 * size]
        if len(neighbors) >= 6 and 1.48 * size <= max(neighbors) - min(neighbors) <= .92 * body.shape[1]:
            return True
    headline_sizes = [box for part in parts if .56 * size <= (box := cv2.boundingRect(part))[3] <= 1.35 * size
                      and .19 * size <= box[2] <= 1.21 * size]
    for _, t, _, h in headline_sizes:
        neighbors = [x for x, q, _, z in headline_sizes if abs(t + h - q - z) <= .10 * size]
        if len(neighbors) >= 7 and 4.0 * size <= max(neighbors) - min(neighbors) <= .95 * body.shape[1]:
            return True
    return False


def related_footer_comments(frame, regions, index, *, gray=None):
    """Find counted buttons whose source artwork or repeated footer proves ownership."""
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
    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    conventional, supported = [], []
    cache, observed = {}, {}
    for peers in rows:
        left, comment, right = peers
        if not _separator(gray, comment.top, comment.bottom, comment.height):
            continue
        icons = []
        for owner, role in zip(peers, ("like", "comments", "share"), strict=True):
            key = (owner, role)
            if key not in observed:
                observed[key] = _marks(gray, owner, role)
            icons.append(observed[key])
        hand, bubble, share = icons
        if hand is None or bubble is None:
            continue
        sizes = [box[3] - box[1] for box in (hand, bubble)]
        y = (bubble[1] + bubble[3]) / 2
        if max(sizes) > 1.20 * min(sizes) or abs((hand[1] + hand[3]) / 2 - y) > .20 * min(sizes):
            continue
        menu = _overflow(gray, right) if share is None else None
        if menu is None and share is None:
            continue
        count = _reading(frame, comment, bubble, index, cache)
        if (not count or not _COUNT.fullmatch(count)) and not (menu and _digit_run(gray, comment, bubble, index, cache)):
            continue
        if share is not None:
            if _reading(frame, left, hand, index, cache) not in _LIKE:
                continue
            if not .79 * min(sizes) <= share[3] - share[1] <= 1.22 * max(sizes):
                continue
            if abs((share[1] + share[3]) / 2 - y) > .21 * min(sizes):
                continue
            art = _owns_artwork(gray, boxes, comment, right)
            if not art and _wide_media(gray, boxes, comment, right):
                supported.append((comment.xyxy(), bubble))
                art = True
            conventional.append((peers, tuple(icons), art))
        elif (abs((menu[1] + menu[3]) / 2 - y) <= .18 * min(sizes) and _wide_media(gray, boxes, comment, right)
              and _digit_run(gray, left, hand, index, cache)):
            supported.append((comment.xyxy(), bubble))
    owned = [item for item in conventional if item[2]]
    supported.extend((peers[1].xyxy(), icons[1]) for peers, icons, owner in conventional
                     if not owner and any(_tops_match(gray, (peers, icons, owner), proven) for proven in owned))

    actions = []
    for box, mark in supported:
        x, y = round((mark[0] + mark[2]) / 2), round((mark[1] + mark[3]) / 2)
        if any(l <= x < r and t <= y < b for l, t, r, b in regions.get("floating_controls", ())):
            continue
        if not any(_overlap(box, action["box"]) >= .75 for action in actions):
            actions.append({"role": "comments", "function": "打开评论", "box": list(box),
                            "click_point": [x, y], "preview_text": "Comments"})
    return actions
