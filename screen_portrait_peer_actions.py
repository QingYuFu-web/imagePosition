"""Verify visible action icons in repeated mobile cards and video engagement rows."""

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_caption_transcript import _bands, _contrast, _ranked_glyph, read_caption
from screen_landscape_search_field import _available


def _native(box, scale):
    return tuple(round(value / scale) for value in box)


def _action(role, function, box, scale, regions, **metadata):
    left, top, right, bottom = _native(box, scale)
    bounds = (left, top, right, bottom)
    if not _available(bounds, regions):
        return None
    return {"role": role, "function": function, "box": list(bounds),
            "click_point": [(left + right) // 2, (top + bottom) // 2], **metadata}


def _boxes(ink):
    contours, _ = cv2.findContours(np.uint8(ink), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours]


def _engagement_start(gray):
    height, width = gray.shape
    x = round(.08 * width)
    area = gray[round(.12 * height):round(.53 * height):3, x:width - x]
    pale = (area >= 220).mean(axis=1)
    for first, last in _bands(pale >= .88):
        if first >= 13 and last - first >= 5 and np.count_nonzero(pale[first - 13:first] <= .48) >= 11:
            return round(.12 * height) + 3 * first
    return None


def _columns(mask, gap):
    columns = []
    for left, right in _bands(mask.any(axis=0)):
        if columns and left - columns[-1][1] <= gap:
            columns[-1] = (columns[-1][0], right)
        else:
            columns.append((left, right))
    return columns


def _rail_metrics(gray, boxes, pitch):
    height = np.median([box[3] - box[1] for box in boxes])
    bottom = round(np.median([box[3] for box in boxes]))
    top, end = round(bottom + .26 * height), round(bottom + 1.07 * height)
    metrics = []
    for left, _, right, _ in boxes:
        center = (left + right) / 2
        x1, x2 = round(center - .40 * pitch), round(center + .40 * pitch)
        tile = gray[top:end, x1:x2]
        if np.count_nonzero(tile >= 220) < .72 * tile.size:
            return False
        strokes = _boxes(tile < 201)
        meaningful = [(x, y, w, h) for x, y, w, h in strokes if .19 * height <= h <= .76 * height and w >= 2]
        if not 2 <= len(meaningful) <= 12:
            return False
        metrics.append(np.median([y + h for _, y, _, h in meaningful]))
    return max(metrics) - min(metrics) <= .23 * height


def _engagement_rows(gray, start):
    height, width = gray.shape
    end = round(.79 * height)
    for_y = gray[start:end]
    ink = for_y < 176
    seeds = sorted({start + y + h // 2 for _, y, w, h in _boxes(ink)
                    if .022 * width <= h <= .059 * width and .021 * width <= w <= .066 * width})
    tested = set()
    for center in seeds:
        half = round(.027 * width)
        top = center - half
        band = gray[top:center + half + 1] < 176
        columns = _columns(band, round(.011 * width))
        boxes = []
        for left, right in columns:
            x, y, w, h = cv2.boundingRect(np.uint8(band[:, left:right]))
            if (.027 * width <= w <= .073 * width and .026 * width <= h <= .064 * width
                    and 0 < y < 2 * half + 1 - h):
                boxes.append((left + x, top + y, left + x + w, top + y + h))
        if not 4 <= len(boxes) <= 6:
            continue
        key = tuple(boxes)
        if key in tested:
            continue
        tested.add(key)
        centers = [(left + right) / 2 for left, _, right, _ in boxes]
        rows = [(top + bottom) / 2 for _, top, _, bottom in boxes]
        gaps = np.diff(centers)
        pitch = np.median(gaps)
        if (not .14 * width <= pitch <= .27 * width or np.max(np.abs(gaps - pitch)) > .12 * pitch
                or centers[0] > .21 * width or centers[-1] < .79 * width
                or max(rows) - min(rows) > .016 * width):
            continue
        if (gray[top:center + half + 1, round(.03 * width):round(.97 * width)] >= 220).mean() < .85:
            continue
        if _rail_metrics(gray, boxes, pitch):
            yield boxes


def _contours(mask):
    contours, links = cv2.findContours(np.uint8(mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if links is None:
        return []
    return [(contour, [contours[k] for k, link in enumerate(links[0]) if link[3] == j])
            for j, contour in enumerate(contours) if links[0, j, 3] == -1 and cv2.contourArea(contour) >= 9]


def _bubble(mask, components):
    """Require a closed hollow body and a left-side descending tail."""
    if len(components) != 1:
        return False
    contour, holes = components[0]
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    if (len(holes) != 1 or not 1.03 <= w / h <= 1.43 or not .59 <= area / (w * h) <= .82
            or not .55 <= cv2.contourArea(holes[0]) / area <= .84):
        return False
    _, hole_y, hole_w, hole_h = cv2.boundingRect(holes[0])
    tail_columns = np.where(mask[y + round(.92 * h):y + h, x:x + w])[1]
    return bool(.70 * w <= hole_w and .65 * h <= hole_h <= .91 * h and hole_y < y + .22 * h
                and tail_columns.size and np.median(tail_columns) < .25 * w
                and np.max(tail_columns) <= .42 * w)


def _export_arrow(mask, components):
    """Require an outlined arrow leaving a separate open carrier."""
    if len(components) != 2:
        return False
    carrier, arrow = sorted(components, key=lambda pair: len(pair[1]))
    if carrier[1] or len(arrow[1]) != 1:
        return False
    cx, cy, cw, ch = cv2.boundingRect(carrier[0])
    ax, ay, aw, ah = cv2.boundingRect(arrow[0])
    area = cv2.contourArea(arrow[0])
    if (not .77 <= cw / ch <= 1.23 or not .84 <= aw / ah <= 1.55
            or not .35 <= area / (aw * ah) <= .66 or not .26 <= cv2.contourArea(arrow[1][0]) / area <= .60
            or not cx + .15 * cw < ax < cx + .44 * cw or ay >= cy + .13 * ch
            or ax + aw <= cx + 1.04 * cw or ay + ah >= cy + .88 * ch):
        return False
    side = mask[cy:cy + ch, cx:cx + max(2, round(.15 * cw))].any(axis=1)
    base = mask[cy + round(.82 * ch):cy + ch, cx:cx + cw].any(axis=0)
    apex = np.nonzero(mask[ay:ay + ah, ax + round(.88 * aw):ax + aw])[0]
    return bool(side.mean() >= .72 and base.mean() >= .74 and apex.size
                and .27 * ah <= np.median(apex) <= .64 * ah)


def _engagement_actions(gray, start, scale, regions):
    width = gray.shape[1]
    matches = []
    for boxes in _engagement_rows(gray, start):
        found = {}
        for slot, (left, top, right, bottom) in enumerate(boxes):
            pad = max(2, round(.007 * width))
            tile = gray[top - pad:bottom + pad, left - pad:right + pad]
            masks = [tile < contrast for contrast in (151, 183)]
            verified = [(mask, _contours(mask)) for mask in masks]
            if all(_bubble(mask, group) for mask, group in verified):
                found.setdefault("comments", []).append((slot, "打开评论区", (left, top, right, bottom)))
            if all(_export_arrow(mask, group) for mask, group in verified):
                found.setdefault("share", []).append((slot, "打开分享", (left, top, right, bottom)))
        if not found or any(len(hits) != 1 for hits in found.values()):
            continue
        if len(found) == 2 and abs(found["comments"][0][0] - found["share"][0][0]) != 1:
            continue
        results = [_action(role, purpose, box, scale, regions)
                   for role, [(_, purpose, box)] in found.items()]
        if any(results):
            matches.append([result for result in results if result])
    return matches[0] if len(matches) == 1 else []


def _episode_cards(gray, baseline, paper):
    height, width = gray.shape
    top = baseline + round(.018 * width)
    end = min(height, baseline + round(.28 * width))
    area = gray[top:end]
    if not area.size:
        return False
    cards = sorted((x, y, w, h) for x, y, w, h in
                   _boxes((paper - 32 <= area) & (area <= paper - 5))
                   if .21 * width <= w <= .53 * width and .075 * width <= h <= .24 * width
                   and y <= .081 * width)
    return any(
        x < .10 * width and rx + rw >= .65 * width and
        .012 * width <= rx - x - w <= .075 * width and
        abs(w - rw) <= .085 * w and abs(h - rh) <= .09 * h and abs(y - ry) <= .013 * width
        for x, y, w, h in cards for rx, ry, rw, rh in cards
    )


def _episode_total(tile, index):
    distances, _ = _contrast(tile)
    peak = np.percentile(distances, 98)
    cache, transcripts = {}, []
    if peak < 68:
        return None
    for level in (.30, .41):
        ink = np.uint8(distances >= level * peak)
        bands = _bands(ink.any(axis=0))
        if not 3 <= len(bands) <= 8:
            return None
        chars = []
        for n, (left, right) in enumerate(bands):
            _, top, _, h = cv2.boundingRect(ink[:, left:right])
            end = n == 0 or n == len(bands) - 1
            options = _ranked_glyph(index, cache, "all" if end else "choice_latin",
                                    ink[top:top + h, left:right])
            if (len(options) < 2 or options[0]["distance"] > .19
                    or options[1]["distance"] - options[0]["distance"] < .029):
                return None
            char = options[0]["text"]
            if (n == 0 and char != "全" or n == len(bands) - 1 and char not in {"话", "集"}
                    or not end and char not in "0123456789"):
                return None
            chars.append(char)
        transcripts.append("".join(chars))
    return transcripts[0] if transcripts[0] == transcripts[1] else None


def _episode_list(frame, gray, rail, scale, index, regions):
    if index is None:
        return None
    height, width = gray.shape
    edge = round(scale * max(item["box"][3] for item in rail))
    first, last = edge + round(.093 * width), min(height, edge + round(.225 * width))
    right_start = round(.69 * width)
    candidates = []
    for y1, y2 in _bands((gray[first:last, right_start:] < 139).sum(axis=1) >= 3):
        top, bottom = first + y1, first + y2
        if not .019 * width <= bottom - top <= .046 * width:
            continue
        dx, dy, w, h = cv2.boundingRect(np.uint8(gray[top:bottom, right_start:] < 139))
        fx = right_start + dx
        if not .058 * width <= w <= .234 * width or not .89 * width <= fx + w <= .991 * width:
            continue
        paper = int(np.median(gray[top:bottom, round(.54 * width):round(.95 * width)]))
        if paper < 220 or not _episode_cards(gray, bottom, paper):
            continue
        row_start, row_end = top - round(.011 * width), bottom + round(.011 * width)
        hx, hy, hw, hh = cv2.boundingRect(np.uint8(gray[row_start:row_end, :round(.128 * width)] < 103))
        if (not .04 * width <= hw <= .089 * width or hx > .083 * width
                or not .78 * h <= hh <= 1.35 * h):
            continue
        margin = round(.008 * width)
        left_header = _native((max(0, hx - margin), row_start, hx + hw + margin, row_end), scale)
        box = (fx, top + dy, fx + w, top + dy + h)
        left, above, right, below = _native((fx - margin, row_start, fx + w + margin, row_end), scale)
        tile = frame[above:below, left:right]
        paint = np.median(tile, axis=(0, 1))
        dark = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) < paper - 80
        colors = tile[dark]
        if (not colors.size or np.ptp(paint) >= 22 or np.median(np.ptp(colors, axis=1)) < 44):
            continue
        header = read_caption(frame, left_header, index, cache={}, choice=True, join_radicals=True)
        if header is None or header["text"] not in {"合集", "选集", "剧集"}:
            continue
        caption = _episode_total(tile, index)
        if caption:
            action = _action("open_episode_list", "打开选集", box, scale, regions, preview_text=caption)
            if action:
                candidates.append(action)
    return candidates[0] if len(candidates) == 1 else None


def _cards(gray):
    height, width = gray.shape
    top, bottom = round(.076 * height), round(.67 * height)
    area = gray[top:bottom]
    if np.count_nonzero(area >= 221) < .71 * area.size:
        return []
    page = int(np.percentile(area[::3, ::3], 82))
    if page < 219:
        return []
    candidates = []
    for adjustment in (5, 8, 12):
        paint = (area <= page - adjustment) & (area >= page - 37)
        cards = sorted((x, top + y, w, h) for x, y, w, h in _boxes(paint)
                       if .79 * width <= w < .992 * width and .115 * width <= h <= .51 * width
                       and x + w / 2 >= .46 * width and x + w / 2 <= .54 * width)
        cards.sort(key=lambda card: card[1])
        if len(cards) >= 2:
            candidates.append(cards)
    if len(candidates) < 2:
        return []
    cards = max(candidates, key=len)
    height_median = np.median([h for _, _, _, h in cards])
    return cards if all(
        .79 * height_median <= h <= 1.21 * height_median and
        .02 * height_median <= next_y - y - h <= .33 * height_median
        for (_, y, _, h), (_, next_y, _, _) in zip(cards, cards[1:])
    ) else []


def _overflow(card, gray):
    left, top, width, height = card
    x = left + round(.77 * width)
    y = top + round(.15 * height)
    area = gray[y:top + round(.85 * height), x:left + round(.985 * width)]
    paint = int(np.median(area))
    mask = area < paint - 89
    points = []
    for contour, holes in _contours(mask):
        dx, dy, w, h = cv2.boundingRect(contour)
        if (not holes and .006 * width <= h <= .023 * width and .69 <= w / h <= 1.34
                and .41 <= cv2.contourArea(contour) / (w * h) <= .83):
            points.append((dx, dy, dx + w, dy + h))
    points.sort(key=lambda point: point[1])
    matches = []
    for first, middle, last in zip(points, points[1:], points[2:]):
        trio = (first, middle, last)
        xs = [(a + c) / 2 for a, _, c, _ in trio]
        ys = [(b + d) / 2 for _, b, _, d in trio]
        d1, d2 = ys[1] - ys[0], ys[2] - ys[1]
        diameter = np.median([d - b for _, b, _, d in trio])
        if (max(xs) - min(xs) <= .29 * diameter and .985 * diameter <= min(d1, d2) <= 2.75 * diameter
                and abs(d1 - d2) <= .24 * diameter and
                all(.78 * diameter <= d - b <= 1.23 * diameter for _, b, _, d in trio)):
            matches.append((x + min(p[0] for p in trio), y + first[1], x + max(p[2] for p in trio), y + last[3]))
    return matches[0] if len(matches) == 1 else None


def _plus(mask):
    h, w = mask.shape
    horizontal = np.flatnonzero(mask.sum(axis=1) >= .81 * w)
    vertical = np.flatnonzero(mask.sum(axis=0) >= .81 * h)
    if not horizontal.size or not vertical.size or not .79 <= w / h <= 1.26:
        return False
    if (abs(horizontal.mean() + .5 - h / 2) > .10 * h or abs(vertical.mean() + .5 - w / 2) > .10 * w
            or horizontal.size > .25 * h or vertical.size > .25 * w):
        return False
    ys, xs = np.nonzero(mask)
    return bool(((abs(ys + .5 - h / 2) <= .15 * h) | (abs(xs + .5 - w / 2) <= .15 * w)).mean() >= .94)


def _create_action(frame, gray, limit, scale, index, regions):
    width = gray.shape[1]
    top = round(.045 * gray.shape[0])
    area = gray[top:limit]
    if not area.size:
        return None
    ink = area < int(np.median(area)) - 97
    backs, crosses = [], []
    for x, y, w, h in _boxes(ink):
        if not .025 * width <= h <= .057 * width or w < .024 * width:
            continue
        tile = ink[y:y + h, x:x + w]
        box = (x, y + top, x + w, y + top + h)
        if x <= .15 * width and is_shaft_back_arrow(tile):
            backs.append(box)
        if x >= .82 * width and _plus(tile):
            crosses.append(box)
    if len(backs) != 1 or len(crosses) != 1:
        return None
    back, cross = backs[0], crosses[0]
    if abs(sum(back[1::2]) - sum(cross[1::2])) > .024 * width:
        return None
    if index is None:
        return None
    title_left = back[2] + round(.051 * width)
    title_right = round(.56 * width)
    title_top = max(top, min(back[1], cross[1]) - round(.022 * width))
    title_bottom = max(back[3], cross[3]) + round(.024 * width)
    new_x, new_y, new_w, new_h = cv2.boundingRect(np.uint8(gray[title_top:title_bottom, title_left:title_right] < 150))
    if not .10 * width <= new_w <= .39 * width or not .024 * width <= new_h <= .066 * width:
        return None
    padx, pady = round(.015 * width), round(.012 * width)
    title = _native((title_left + new_x - padx, title_top + new_y - pady,
                     title_left + new_x + new_w + padx, title_top + new_y + new_h + pady), scale)
    caption = read_caption(frame, title, index, cache={}, choice=True, join_radicals=True)
    if not caption:
        return None
    text = caption["text"]
    noun = text[:-2]
    if not text.endswith("管理") or not 2 <= len(noun) <= 6 or not all("\u4e00" <= glyph <= "\u9fff" for glyph in noun):
        return None
    return _action("create", f"添加{noun}", cross, scale, regions, preview_text=text)


def portrait_peer_actions(frame, regions, index=None):
    """Find evidenced peer actions on an unobscured portrait phone page."""
    height, width = frame.shape[:2]
    if regions.get("layer") != "page" or regions.get("regions") or width < 325 or not 1.55 <= height / width <= 2.80:
        return []
    scale = min(1.0, 640 / width)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    start = _engagement_start(gray)
    if start is not None:
        actions = _engagement_actions(gray, start, scale, regions)
        if actions and (episode_list := _episode_list(frame, gray, actions, scale, index, regions)):
            actions.append(episode_list)
        return actions
    cards = _cards(gray)
    triplets = [result for card in cards if (result := _overflow(card, gray))]
    if len(triplets) < 2 or max(item[0] for item in triplets) - min(item[0] for item in triplets) > .023 * gray.shape[1]:
        return []
    more = [_action("more", "更多选项", box, scale, regions) for box in triplets]
    create = _create_action(frame, gray, cards[0][1], scale, index, regions)
    return [result for result in [*more, create] if result]
