"""Recognize two owned cards when the artwork fits their canvases differently."""

from dataclasses import dataclass
from itertools import combinations
from math import ceil

import cv2
import numpy as np

from screen_product_rows import _Card, _artwork_click, _price_evidence, _price_index, _small_paired_price, _spans


@dataclass(frozen=True)
class _Bookmark:
    box: tuple[int, int, int, int]
    complete: bool

    @property
    def x(self):
        return (self.box[0] + self.box[2]) / 2


def _components(mask, min_size, max_size):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask), connectivity=8)
    for identity, (x, y, w, h, count) in enumerate(stats[1:].tolist(), 1):
        if min_size <= min(w, h) <= max(w, h) <= max_size and count >= .13 * w * h:
            yield (x, y, x + w, y + h), labels[y:y + h, x:x + w] == identity


def _bookmark_shape(mark):
    h, w = mark.shape
    if not .66 <= w / h <= .96 or not .16 <= mark.mean() <= .36:
        return None
    cap = mark[:round(.17 * h)]
    if (not (cap.mean(axis=1) >= .69).any()
            or mark[round(.21 * h):round(.66 * h), round(.25 * w):round(.76 * w)].mean() > .045):
        return None
    upper = mark[round(.18 * h):round(.45 * h)]
    side = round(.22 * w)
    if any((edge.any(axis=1)).mean() < .86 for edge in (upper[:, :side], upper[:, -side:])):
        return None
    stem = mark[round(.33 * h):round(.70 * h)]
    bottom = mark[round(.69 * h):]
    notch = round(.27 * w)
    notch_tip = bottom[-max(2, round(.09 * h)):, notch:w - notch]
    # A one-pixel edge can enter the notch while resampling a small silhouette.
    if (not bottom[:round(.14 * h), round(.34 * w):round(.67 * w)].any()
            or np.count_nonzero(notch_tip) > int(h <= 32)):
        return None
    complete = (bottom[:, :side].any(axis=1).mean() >= .73 and stem[:, :side].any(axis=1).mean() >= .74
                and stem[:, -side:].any(axis=1).mean() >= .74 and bottom[:, -side:].any(axis=1).mean() >= .73)
    if complete or (stem[:, -side:].any(axis=1).mean() >= .86 and bottom[:, -side:].any(axis=1).mean() >= .86):
        return complete
    return None


def _dark_ink(low, high):
    return (high < 124) & ((high - low) <= 23)


def _bookmarks(frame, candidates):
    width = frame.shape[1]
    output = {}
    for item in candidates:
        left, top, right, bottom = item["box"]
        w, h = right - left, bottom - top
        if (not item.get("visibility") == "page" or not .043 * width <= min(w, h) <= max(w, h) <= .134 * width
                or not (.39 * width <= (left + right) / 2 <= .50 * width or .87 * width <= (left + right) / 2 <= .99 * width)):
            continue
        crop = frame[top:bottom, left:right]
        ink = _dark_ink(crop.min(axis=2), crop.max(axis=2))
        for (a, b, c, d), mark in _components(ink, .026 * width, .070 * width):
            shape = _bookmark_shape(mark)
            if shape is not None:
                box = left + a, top + b, left + c, top + d
                output[box] = _Bookmark(box, shape)
    return tuple(output.values())


def _add_control(dark, low, bookmark, pitch):
    left = round(bookmark.x - .112 * pitch)
    right = min(low.shape[1], round(bookmark.x + .112 * pitch))
    top = bookmark.box[1] + round(.57 * pitch)
    bottom = min(len(low), bookmark.box[1] + round(.95 * pitch))
    if not 0 <= left < right or not top < bottom:
        return None
    found = []
    for (a, b, c, d), mark in _components(dark[top:bottom, left:right], .050 * pitch, .104 * pitch):
        h, w = mark.shape
        if not .88 <= w / h <= 1.14 or not .17 <= mark.mean() <= .32:
            continue
        x, y = left + (a + c) / 2, top + (b + d) / 2
        if abs(x - bookmark.x) > .037 * pitch:
            continue
        hh, hw = max(2, round(.23 * h)), max(2, round(.23 * w))
        center = mark[h // 2 - hh:h // 2 + hh, w // 2 - hw:w // 2 + hw]
        corners = (mark[:hh, :hw], mark[:hh, -hw:], mark[-hh:, :hw], mark[-hh:, -hw:])
        if (center.mean() < .36 or max(block.mean() for block in corners) > .035
                or not mark[h // 2].mean() >= .84 or not mark[:, w // 2].mean() >= .84):
            continue
        radius = round(.96 * max(w, h))
        xx, yy = round(x), round(y)
        if min(xx, yy, low.shape[1] - xx, len(low) - yy) <= radius:
            continue
        coords = np.rint(np.c_[np.cos(np.arange(16) * np.pi / 8), np.sin(np.arange(16) * np.pi / 8)] * (.73 * radius)).astype(int)
        if (low[yy + coords[:, 1], xx + coords[:, 0]] >= 230).mean() < .88:
            continue
        found.append((xx - radius, yy - radius, xx + radius, yy + radius))
    return found[0] if len(found) == 1 else None


def _artwork(low, photos, bookmark, add, pitch):
    found = []
    for box in photos:
        left, top, right, bottom = box
        if (bookmark.x - .94 * pitch <= left <= bookmark.x - .72 * pitch
                and bookmark.x + .035 * pitch <= right <= bookmark.x + .13 * pitch
                and bookmark.box[1] - .055 * pitch <= top <= bookmark.box[1] + .32 * pitch
                and add[1] - .10 * pitch <= bottom <= ceil(add[3] + .04 * pitch)):
            inset = round(.09 * pitch)
            sample = low[top + inset:bottom - inset:3, left + inset:right - inset:3]
            if sample.size and sample.std() >= 22 and (sample < 153).mean() >= .28:
                found.append(box)
    return max(found, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), default=None)


def _caption_price(frame, low, left, right, add, pitch, bank, cache):
    start = add[3] + round(.048 * pitch)
    stop = min(len(low), add[3] + round(.57 * pitch))
    x = left + round(.028 * pitch)
    finish = right - round(.027 * pitch)
    if stop - start < .30 * pitch:
        return None
    lane = low[start:stop, x:finish]
    ink = lane <= 160
    hits = _spans(ink.sum(axis=1) >= max(5, round(.019 * pitch)), max(2, round(.008 * pitch)))
    spans = [(a, b) for a, b in hits if .027 * pitch <= b - a <= .091 * pitch]
    if len(spans) < 2:
        return None
    a, b = spans[0]
    first = lane[a:b]
    text_x, _, text_w, _ = cv2.boundingRect(np.uint8(first < 152))
    parts = _spans((first < 152).any(axis=0), 0)
    start_gap = start + a - add[3]
    above = lane[:a] if a >= round(.018 * pitch) else low[start - round(.047 * pitch):start + a, x:finish]
    if (not .064 * pitch <= start_gap <= .16 * pitch or not 0 <= text_x <= .035 * pitch
            or not .35 * pitch <= text_w <= .94 * pitch or len(parts) < 7 or (above >= 239).mean() < .94):
        return None
    if (frame[start + a:start + b, x:finish].max(axis=2) < 151).mean() < .15:
        return None
    # A price is a separately printed run, with its symbol and amount next to
    # each other across two cuts. A wrapped title or strike-through is not one.
    offers = []
    for y, end in spans[1:]:
        if not .065 * pitch <= y - a <= .46 * pitch:
            continue
        band = start + y, start + end
        price = _price_evidence(frame, x, finish, band, pitch, bank, cache)
        if price is None or price[1] is None:
            price = _small_paired_price(frame, x, finish, band, pitch, bank, cache, saturated=True)
        if price is not None and price[1] is not None:
            offers.append((band, *price))
    return (start + a, *offers[0]) if len(offers) == 1 else None


def _tile_bottom(low, left, right, price_band, pitch):
    start = price_band[1] + round(.018 * pitch)
    end = min(len(low), start + round(.38 * pitch))
    foot = low[start:end, left + round(.026 * pitch):right - round(.16 * pitch)]
    bottom = price_band[1]
    for a, b in _spans(((foot < 244).mean(axis=1) >= .093), max(2, round(.018 * pitch))):
        if start + a - bottom > .057 * pitch or not .023 * pitch <= b - a <= .16 * pitch:
            break
        bottom = start + b
    if end - bottom < .035 * pitch:
        return None
    return bottom


def _gallery(frame, low, dark, candidates, photos, first, second, bank, cache):
    pitch = second.x - first.x
    width = low.shape[1]
    height = len(low)
    left_top, right_top = first.box[1], second.box[1]
    if (not .455 * width <= pitch <= .52 * width or abs(left_top - right_top) > .042 * pitch
            or not .11 * height <= min(left_top, right_top) <= .73 * height
            or max(left_top, right_top) + 1.41 * pitch >= height or not (first.complete or second.complete)):
        return None
    sizes = [(_mark.box[2] - _mark.box[0], _mark.box[3] - _mark.box[1]) for _mark in (first, second)]
    if abs(sizes[0][0] - sizes[1][0]) > .022 * pitch or abs(sizes[0][1] - sizes[1][1]) > .023 * pitch:
        return None
    adds = [_add_control(dark, low, mark, pitch) for mark in (first, second)]
    if not all(adds) or abs((adds[0][1] + adds[0][3]) - (adds[1][1] + adds[1][3])) > .078 * pitch:
        return None
    art = [_artwork(low, photos, mark, add, pitch) for mark, add in zip((first, second), adds, strict=True)]
    if not all(art):
        return None
    margin = round(.029 * pitch)
    outer_left = max(0, round(min(art[0][0], art[1][0] - pitch) - .022 * pitch))
    seam = round(outer_left + pitch - margin / 2)
    if not .47 * width <= seam <= .53 * width:
        return None
    edges = [(outer_left, seam - margin // 2), (seam + margin // 2, min(width, round(outer_left + 2 * pitch - margin)))]
    top = round(min(left_top, right_top) - .052 * pitch)
    seam_ink = low[top:round(top + 1.40 * pitch):3, seam - max(2, margin // 4):seam + max(2, margin // 4)]
    if (seam_ink >= 238).mean() < .97:
        return None
    readings = [_caption_price(frame, low, l, r, add, pitch, bank, cache)
                for (l, r), add in zip(edges, adds, strict=True)]
    if (not all(readings) or readings[0][2] != readings[1][2]
            or abs(readings[0][0] - readings[1][0]) > .074 * pitch):
        return None
    output = []
    for (l, r), picture, (title_top, price_band, _, price), mark, add in zip(edges, art, readings, (first, second), adds, strict=True):
        bottom = _tile_bottom(low, l, r, price_band, pitch)
        if bottom is None:
            return None
        card = _Card((l, top, r, bottom), picture, price)
        blocked = (*candidates, {"box": mark.box}, {"box": add})
        point = _artwork_click(card, blocked)
        if point is None:
            return None
        output.append((card, point, mark.box, add, (l, title_top, r, bottom)))
    return tuple(output)


def vertical_tile_card_actions(frame, regions):
    """Return one product-detail click per complete tile in a proven pair."""
    height, width = frame.shape[:2]
    if regions.get("layer") != "page" or regions.get("regions") or height < 1.59 * width:
        return []
    candidates = regions["candidates"]
    photos = [tuple(item["box"]) for item in candidates if item.get("visibility") == "page"
              and .32 * width <= item["box"][2] - item["box"][0] <= .48 * width
              and .19 * width <= item["box"][3] - item["box"][1] <= .57 * width]
    if not (any(box[2] < .53 * width for box in photos) and any(box[0] > .47 * width for box in photos)):
        return []
    marks = _bookmarks(frame, candidates)
    if len(marks) < 2 or (bank := _price_index()) is None:
        return []
    low = cv2.min(cv2.min(frame[..., 0], frame[..., 1]), frame[..., 2])
    high = cv2.max(cv2.max(frame[..., 0], frame[..., 1]), frame[..., 2])
    dark = _dark_ink(low, high)
    groups, cache = [], {}
    for first, second in combinations(sorted(marks, key=lambda b: b.x), 2):
        if photo_group := _gallery(frame, low, dark, candidates, photos, first, second, bank, cache):
            if any(abs(photo_group[0][0].box[1] - before[0][0].box[1]) <= .024 * width for before in groups):
                continue
            groups.append(photo_group)
    actions = []
    for order, group in enumerate(sorted(groups, key=lambda group: group[0][0].box[1]), 1):
        for side, (card, point, bookmark, add, caption) in zip(("左侧", "右侧"), group, strict=True):
            label = f"第{order}组{side}商品"
            actions.append({"role": "open_product", "function": f"打开{label}（显示价 {card.price}）",
                            "box": list(card.box), "click_point": point, "preview_text": f"{label} {card.price}",
                            "component_artwork": list(card.artwork), "component_caption": list(caption),
                            "component_controls": [list(bookmark), list(add)]})
    return actions
