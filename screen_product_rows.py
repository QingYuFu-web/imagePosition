"""Group visible, priced picture cards into one component per card."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import os
import re

import cv2
import numpy as np

from screen_font_glyphs import _build, _normal, _shortlist


Box = tuple[int, int, int, int]
_CURRENCIES = frozenset("¥$€£")
_AMOUNT = re.compile(r"\d{1,7}(?:\.\d{1,3})?")


@dataclass(frozen=True)
class _Card:
    box: Box
    artwork: Box
    price: str | None


@lru_cache(maxsize=1)
def _price_index():
    root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    names = ("arial.ttf", "arialbd.ttf", "segoeui.ttf", "segoeuib.ttf", "msyh.ttc", "msyhbd.ttc", "simhei.ttf", "calibri.ttf")
    paths = tuple(root / name for name in names if (root / name).is_file())
    return _build(paths, "¥￥$€£0123456789YENS5MC|I") if paths else None


def _size(box):
    return box[2] - box[0], box[3] - box[1]


def _spans(flags, gap):
    merged = []
    edges = np.flatnonzero(np.diff(np.r_[False, flags, False])).reshape(-1, 2)
    for start, end in edges.tolist():
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def _body_pairs(candidates, width):
    bodies = [tuple(map(int, item["box"])) for item in candidates if item["visibility"] == "page"
              and .34 * width <= _size(item["box"])[0] <= .54 * width
              and .70 * _size(item["box"])[0] <= _size(item["box"])[1] <= 1.48 * _size(item["box"])[0]]
    for first, second in combinations(sorted(set(bodies)), 2):
        left, top, right, _ = first
        l2, t2, r2, _ = second
        scale = min(right - left, r2 - l2)
        if (scale >= .88 * max(right - left, r2 - l2) and left <= .07 * width and r2 >= .93 * width
                and -.018 * scale <= l2 - right <= .090 * scale and abs(top - t2) <= .045 * scale + .5):
            yield first, second, scale


def _paired_captions(low, first, second, scale):
    height = low.shape[0]
    left, top, right, _ = first
    l2, t2, r2, _ = second
    y = round((top + t2) / 2)
    if not .10 * height <= y <= .72 * height or y + 1.31 * scale >= height:
        return None
    center = round((right + l2) / 2)
    pad = max(2, round(.004 * scale))
    seam_rows = low[y + round(.10 * scale):y + round(1.29 * scale)]
    centers = (center, center - 1, center + 1) if scale <= 400 else (center,)
    if not any((seam_rows[:, x - pad:x + pad + 1] >= 229).mean() >= .89 for x in centers):
        return None
    start, end = y + round(.86 * scale), min(height, y + round(1.46 * scale))
    lanes = []
    for l, r in ((left, right), (l2, r2)):
        inset = round(.044 * (r - l))
        x0, x1 = l + inset, r - inset
        art = low[y + round(.045 * scale):y + round(.82 * scale), x0:x1]
        if art[::2, ::2].std() < 18 or (art[::2, ::2] < 156).mean() < .095:
            return None
        ink = (low[start:end, x0:x1] < 194).mean(axis=1)
        lanes.append([(start + a, start + b) for a, b in _spans(ink >= .063, max(2, round(.009 * scale)))
                      if .035 * scale <= b - a <= .095 * scale and .105 <= ink[a:b].mean() <= .66])
    aligned = [(min(a, c), max(b, d)) for a, b in lanes[0] for c, d in lanes[1]
               if max(abs(a - c), abs(b - d)) <= .027 * scale]
    for bands in zip(aligned, aligned[1:], aligned[2:]):
        if (not .90 * scale <= bands[0][0] - y <= 1.16 * scale
                or not all(.020 * scale <= b[0] - a[1] <= .105 * scale for a, b in zip(bands, bands[1:]))):
            continue
        end = bands[-1][1] + round(.047 * scale)
        blank = bands[-1][1] + round(.014 * scale)
        if end < height and min((low[blank:end, l:r] >= 232).mean() for l, r in ((left, right), (l2, r2))) >= .83:
            return y, end, bands
    return None


def _glyph(mask, bank, cache):
    key = mask.shape, mask.tobytes()
    if key not in cache:
        scores = {}
        for letter, value, _ in _shortlist(bank, *_normal(np.uint8(mask) * 255)):
            name = "¥" if letter == "￥" else letter
            scores[name] = min(value, scores.get(name, value))
        best, other = sorted(scores, key=scores.__getitem__)[:2]
        cache[key] = best if scores[best] <= .225 and scores[other] - scores[best] >= .019 else None
    return cache[key]


def _marks(mask, em):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask), connectivity=8)
    marks = []
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if .11 * em <= w <= 1.12 * em and .11 * em <= h <= 1.38 * em and area >= max(5, .082 * w * h):
            marks.append((int(x), int(y), int(w), int(h), labels[y:y + h, x:x + w] == i))
    return sorted(marks, key=lambda mark: mark[0])


def _digits_after(mark, tail, em, bank, cache):
    left, top, w, h, _ = mark
    end = left + w
    baseline = top + h
    amount, size = [], None
    for x, y, width, height, raster in tail:
        if x < end - .15 * em or abs(y + height - baseline) > .21 * em:
            continue
        if x - end > .45 * em:
            break
        small = height <= .33 * em and width <= .34 * em and .58 <= width / height <= 1.62 and raster.mean() >= .52
        if small and amount and "." not in amount:
            char = "."
        elif .57 * em <= height <= 1.32 * em and .16 * height <= width <= .94 * height:
            if size is not None and not .65 * size <= height <= 1.28 * size:
                return None, bool(amount)
            char = _glyph(raster, bank, cache)
            if char is None or not char.isascii() or not char.isdigit():
                # The currency and first digit remain a proof of a price, but
                # this cut can only support a literal if all adjoining marks fit.
                return (None, bool(amount))
            if size is None:
                size = height
        else:
            return None, bool(amount)
        amount.append(char)
        end = x + width
    text = "".join(amount)
    return (text if _AMOUNT.fullmatch(text) else None, bool(amount))


def _price_tokens(mask, em, bank, cache):
    parts = _marks(mask, em)
    results = []
    for index, mark in enumerate(parts[:-1]):
        x, _, w, h, raster = mark
        if not (.46 * em <= h <= 1.40 * em and .32 * h <= w <= .98 * h and x < .64 * mask.shape[1]):
            continue
        symbol = _glyph(raster, bank, cache)
        if symbol not in _CURRENCIES:
            continue
        text, witnessed = _digits_after(mark, parts[index + 1:], em, bank, cache)
        if witnessed:
            results.append((symbol, text, x))
    return results


def _verified_price(masks, em, bank, cache, *, require_literal=False):
    witnesses = []
    amounts = Counter()
    for mask in masks:
        tokens = _price_tokens(mask, em, bank, cache)
        if len(tokens) != 1:
            continue
        symbol, amount, x = tokens[0]
        if require_literal and amount is None:
            continue
        witnesses.append((symbol, x))
        if amount is not None:
            amounts[symbol + amount] += 1
    if (len(witnesses) < 2 or len({symbol for symbol, _ in witnesses}) != 1
            or np.ptp([x for _, x in witnesses]) > max(3, .18 * em)):
        return None
    literals = [text for text, count in amounts.items() if count >= 2]
    if require_literal and len(amounts) != 1:
        return None
    return witnesses[0][0], literals[0] if len(literals) == 1 else None


def _price_patch(frame, left, right, band, width):
    pad = max(3, round(.023 * width))
    start, end = max(0, band[0] - pad), min(frame.shape[0], band[1] + pad)
    return frame[start:end, left:right], start


def _price_evidence(frame, left, right, band, width, bank, cache):
    em = band[1] - band[0]
    pixels, _ = _price_patch(frame, left, right, band, width)
    patch = pixels.astype(np.int16)
    blue, green, red = patch.transpose(2, 0, 1)
    # Saturated amounts can sit next to low-contrast statistics. Read the
    # printed ink separately, then also check a neutral-ink rendering.
    red_ink = red - np.maximum(blue, green)
    dark_ink = np.median(patch[::3, ::3], axis=(0, 1)).max() - np.max(patch, axis=2)
    for plane, levels in ((red_ink, (29, 39, 46)), (dark_ink, (57, 73, 96))):
        if result := _verified_price((plane >= level for level in levels), em, bank, cache):
            return result

    # Chroma bleed on recapture can join a decimal point to its neighboring
    # digit. Isolate the stronger pigment only when the entire price agrees.
    pigment = red_ink[red_ink >= 65]
    if pigment.size < max(12, .23 * em * em):
        return None
    strength = float(np.percentile(pigment, 75))
    if strength < 105:
        return None
    distinct = {}
    for fraction in (.35, .43, .51):
        binary = red_ink >= strength * fraction
        distinct[binary.tobytes()] = binary
    return _verified_price(distinct.values(), em, bank, cache, require_literal=True)


def _small_paired_price(frame, left, right, band, width, bank, cache, *, saturated=False):
    """Recover a small printed price only when both scales read the same literal."""
    em = band[1] - band[0]
    if width > 400 or not 12 <= em <= 24:
        return None
    pixels, start = _price_patch(frame, left, right, band, width)
    blue, green, red = pixels.astype(np.int16).transpose(2, 0, 1)
    red_ink = red - np.maximum(blue, green)
    if saturated:
        masks = (red_ink >= 85, red_ink >= 96)
        if np.array_equal(*masks) or (original := _verified_price(masks, em, bank, cache, require_literal=True)) is None:
            return None
    else:
        amounts = set()
        for threshold in (29, 39, 46):
            tokens = _price_tokens(red_ink >= threshold, em, bank, cache)
            if len(tokens) == 1 and tokens[0][1] is not None:
                symbol, amount, _ = tokens[0]
                amounts.add((symbol, symbol + amount))
        if len(amounts) != 1:
            return None
        original, = amounts
    scale = 1.5
    enlarged = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    enlarged_band = tuple(round((value - start) * scale) for value in band)
    checked = _price_evidence(enlarged, 0, enlarged.shape[1], enlarged_band, round(width * scale), bank, cache)
    return checked if checked == original else None


def _artwork_click(card, candidates):
    left, top, right, bottom = card.artwork
    width, height = _size(card.artwork)
    blocked = []
    for item in candidates:
        x0, y0, x1, y1 = item["box"]
        w, h = x1 - x0, y1 - y0
        if (left <= x0 < x1 <= right and top <= y0 < y1 <= bottom
                and .025 * width <= min(w, h) <= max(w, h) <= .38 * width):
            blocked.append((x0, y0, x1, y1))
    pad = max(4, round(.031 * width))
    for xx, yy in ((.50, .48), (.43, .45), (.58, .51), (.39, .58), (.63, .41), (.52, .66)):
        x, y = round(left + xx * width), round(top + yy * height)
        if not any(a - pad <= x <= c + pad and b - pad <= y <= d + pad for a, b, c, d in blocked):
            return [x, y]
    return None


def _horizontal_lanes(candidates, width, *, margin=.017):
    photos = []
    for item in candidates:
        left, top, right, bottom = map(int, item["box"])
        w, h = right - left, bottom - top
        if (item["visibility"] == "page" and margin * width <= left <= .10 * width
                and .24 * width <= w <= .43 * width and .29 * width <= right <= .49 * width
                and .51 * w <= h <= 1.27 * w):
            photos.append((left, top, right, bottom))
    visited = set()
    for box in photos:
        w, _ = _size(box)
        lane = tuple(sorted(peer for peer in photos if abs(peer[0] - box[0]) <= .075 * w
                            and abs(peer[2] - box[2]) <= .085 * w))
        if lane not in visited and len(lane) >= 2 and max(b[1] for b in lane) - min(b[1] for b in lane) >= .75 * w:
            visited.add(lane)
            yield lane


def _canvas_boxes(low, lane):
    height, width = low.shape
    edge_l, edge_r = (round(float(np.median([item[index] for item in lane]))) for index in (0, 2))
    side = edge_r - edge_l
    page = np.median(low[:, :max(6, edge_l - round(.020 * side)):2], axis=1).astype(np.int16)
    clip_l, clip_r = max(0, edge_l - round(.08 * side)), min(width, edge_r + round(.04 * side))
    contrast = page[:, None] - low[:, clip_l:clip_r].astype(np.int16)
    gutter_a, gutter_b = edge_r - round(.18 * side) - clip_l, edge_r - round(.041 * side) - clip_l
    supported = (page >= 225) & ((contrast[:, gutter_a:gutter_b] >= 5).mean(axis=1) >= .67)
    for top, bottom in _spans(supported, max(3, round(.012 * side))):
        h = bottom - top
        if not (.78 * side <= h <= 1.24 * side and top >= .10 * height and bottom <= height - .06 * side):
            continue
        center = (edge_l + edge_r) / 2 - clip_l
        body = contrast[top + round(.10 * h):bottom - round(.10 * h)]
        columns = _spans((body >= 5).mean(axis=0) >= .70, max(2, round(.012 * side)))
        fits = [(a, b) for a, b in columns if a < center < b and .85 * side <= b - a <= 1.09 * side]
        if len(fits) != 1:
            continue
        a, b = fits[0]
        left, right = clip_l + a, clip_l + b
        if not any(min(bottom, y1) - max(top, y0) >= .58 * min(h, y1 - y0) for _, y0, _, y1 in lane):
            continue
        # Use the neutral canvas, with its rounded corners, to refine the two
        # row edges. The object inside can be shorter than the actual tile.
        begin, end = max(0, top - round(.019 * side)), min(height, bottom + round(.019 * side))
        extents = _spans((contrast[begin:end, a:b] >= 3).mean(axis=1) >= .55, 2)
        precise = next(((begin + x, begin + y) for x, y in extents if begin + x < (top + bottom) / 2 < begin + y), None)
        if precise is None:
            continue
        top, bottom = precise
        sample = low[top:bottom:3, left:right:3]
        if not (.76 * side <= bottom - top <= 1.32 * side and sample.std() >= 19 and (sample < 167).mean() >= .075):
            continue
        yield left, top, right, bottom


def _side_caption(low, art):
    width = low.shape[1]
    left, top, art_end, bottom = art
    _, h = _size(art)
    start = art_end + max(4, round(.030 * h))
    crop = low[top:bottom, start:width]
    header = crop[round(.028 * h):round(.56 * h), :round(.62 * crop.shape[1])]
    title_bands = _spans((header < 166).mean(axis=1) >= .073, round(.015 * h))
    if sum(.032 * h <= b - a <= .12 * h for a, b in title_bands) < 2:
        return None
    color_begin = round(.60 * h)
    foot = crop[color_begin:round(.995 * h)]
    footprint = (foot < 247).mean(axis=0) >= .30
    columns = np.flatnonzero(footprint)
    if not len(columns):
        return None
    right = start + int(columns[-1]) + 1
    if right < .86 * width or abs(width - right - left) > .105 * width:
        return None
    probe_end = start + round(.56 * (right - start))
    lower = low[top + round(.56 * h):bottom, start:probe_end]
    y_offset = top + round(.56 * h)
    bands = [(y_offset + a, y_offset + b) for a, b in _spans((lower < 166).mean(axis=1) >= .047, round(.012 * h))
             if .044 * h <= b - a <= .128 * h]
    return (right, start, probe_end, bands) if bands else None


def _subtle_canvas(low, art):
    """Measure the artwork edge against a narrow, brighter text gutter."""
    left, top, right, bottom = art
    width, height = _size(art)
    if not .88 * width <= height <= 1.16 * width:
        return None
    trim = round(.12 * height)
    field = low[top + trim:bottom - trim]
    if field[:, left + round(.10 * width):right - round(.10 * width)].std() < 29:
        return None
    reach = max(3, round(.020 * width))
    options = []
    for edge in range(right - round(.043 * width), right + round(.028 * width) + 1):
        inner = field[:, edge - reach:edge].mean(axis=1)
        outer = field[:, edge:edge + reach].mean(axis=1)
        lift = outer - inner
        if (outer >= 243).mean() >= .92 and (lift >= 1.9).mean() >= .84:
            options.append((float(np.median(lift)), edge))
    if not options or options[-1][1] - options[0][1] > 2 * reach:
        return None
    return left, top, max(options)[1], bottom


def _blank_between(low, first, second):
    upper, lower = sorted((first, second), key=lambda card: card.box[1])
    gap = lower.box[1] - upper.box[3]
    height = min(_size(upper.artwork)[1], _size(lower.artwork)[1])
    if not .023 * height <= gap <= .19 * height:
        return False
    left = max(upper.box[0], lower.box[0]) + round(.08 * height)
    right = min(upper.box[2], lower.box[2]) - round(.08 * height)
    blank = low[upper.box[3]:lower.box[1]:2, left:right:3]
    return bool(blank.size and (blank >= 241).mean() >= .87)


def _horizontal_cards(frame, low, lanes, bank, cache, *, subtle=False):
    rows = []
    for lane in lanes:
        cards = []
        artworks = ((art for box in lane if (art := _subtle_canvas(low, box)) is not None)
                    if subtle else _canvas_boxes(low, lane))
        for art in artworks:
            if any(abs(art[1] - card.box[1]) <= .06 * _size(art)[1] for card in rows):
                continue
            if (caption := _side_caption(low, art)) is None:
                continue
            right, start, price_end, bands = caption
            prices = [(y, found) for y in bands
                      if (found := _price_evidence(frame, start, price_end, y, _size(art)[0], bank, cache)) is not None]
            if len(prices) == 1:
                band, (symbol, price) = prices[0]
                if subtle and price is None:
                    continue
                box = art[0], art[1], right, art[3]
                cards.append((_Card(box, art, price), symbol, band[0] - art[1]))
        for card, symbol, price_y in cards:
            h = _size(card.artwork)[1]
            has_peer = any(other != card and currency == symbol
                           and abs(price_y - baseline) <= (.18 if subtle else .11) * h
                           and .83 * h <= _size(other.artwork)[1] <= 1.19 * h
                           and abs(other.box[0] - card.box[0]) <= .05 * h and abs(other.box[2] - card.box[2]) <= .07 * h
                           and .94 * h <= abs(other.box[1] - card.box[1]) <= 1.32 * h
                           and (not subtle or _blank_between(low, card, other))
                           for other, currency, baseline in cards)
            if has_peer:
                rows.append(card)
    return rows


def _inline_model_name(frame, card, index):
    """Read an independently separated Latin model run in the printed title."""
    from screen_caption_transcript import read_caption

    _, top, art_right, _ = card.artwork
    width, height = _size(card.artwork)
    left = art_right + round(.027 * width)
    first, last = top + round(.026 * height), top + round(.405 * height)
    tile = frame[first:last, left:card.box[2] - round(.022 * width)]
    ink = (tile.max(axis=2) <= 146) & (np.ptp(tile, axis=2) <= 39)
    names = set()
    for a, b in _spans(ink.sum(axis=1) >= .05 * tile.shape[1], 1):
        depth = b - a
        if not .050 * height <= depth <= .106 * height:
            continue
        runs = []
        for x, right in _spans(ink[a:b].any(axis=0), 0):
            if right - x > .84 * depth:
                continue
            if runs and x - runs[-1][-1][1] <= .67 * depth:
                runs[-1].append((x, right))
            else:
                runs.append([(x, right)])
        for run in runs:
            start, stop = run[0][0], run[-1][1]
            if len(run) < 7 or not 4.6 * depth <= stop - start <= 13.3 * depth:
                continue
            px, py = max(2, round(.15 * depth)), max(3, round(.23 * depth))
            box = (left + max(0, start - px), first + max(0, a - py),
                   left + min(tile.shape[1], stop + px), first + min(len(tile), b + py))
            reading = read_caption(frame, box, index, cache={}, choice=True, sentence_case=True, join_letters=True)
            if reading is None or reading["max_distance"] > .13 or reading["min_margin"] < .024:
                continue
            title = re.sub(r"(?<=\d) +(?=\d)", "", reading["text"])
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9 -]{5,32}", title) and any(char.isdigit() for char in title):
                names.add(title)
    return names.pop() if len(names) == 1 else None


def product_card_actions(frame, regions, *, font_index=None):
    """Publish cards with distinct artwork, caption bands, and printed prices."""
    if regions["layer"] != "page":
        return []
    height, width = frame.shape[:2]
    candidates = regions["candidates"]
    possible = list(_body_pairs(candidates, width))
    lanes = list(_horizontal_lanes(candidates, width))
    subtle_lanes = list(_horizontal_lanes(candidates, width, margin=.010)) if not possible and not lanes else []
    if not possible and not lanes and not subtle_lanes:
        return []
    low = cv2.min(cv2.min(frame[..., 0], frame[..., 1]), frame[..., 2])
    groups = []
    cache = {}
    for first, second, scale in possible:
        proof = _paired_captions(low, first, second, scale)
        if proof is None:
            continue
        top, bottom, bands = proof
        if any(abs(top - prior[0].box[1]) <= .05 * scale for prior in groups):
            continue
        bank = _price_index()
        if bank is None:
            return []
        prices = [_price_evidence(frame, box[0], box[2], bands[2], scale, bank, cache) for box in (first, second)]
        if prices.count(None) == 1:
            missing = prices.index(None)
            left, _, right, _ = (first, second)[missing]
            prices[missing] = _small_paired_price(frame, left, right, bands[2], scale, bank, cache)
        if not all(prices) or prices[0][0] != prices[1][0]:
            continue
        art_bottom = min(bands[0][0] - round(.025 * scale), top + round(.96 * scale), height)
        groups.append(tuple(_Card((box[0], top, box[2], bottom), (box[0], top, box[2], art_bottom), price[1])
                            for box, price in zip((first, second), prices, strict=True)))
    if lanes and (bank := _price_index()) is not None:
        groups.extend((card,) for card in _horizontal_cards(frame, low, lanes, bank, cache)
                      if not any(abs(card.box[1] - group[0].box[1]) <= .33 * _size(card.artwork)[1] for group in groups))
    used_subtle = not groups
    if used_subtle and (bank := _price_index()) is not None:
        grouped_lanes = subtle_lanes or _horizontal_lanes(candidates, width, margin=.010)
        groups.extend((card,) for card in _horizontal_cards(frame, low, grouped_lanes, bank, cache, subtle=True))
    result = []
    for row, cards in enumerate(sorted(groups, key=lambda group: group[0].box[1]), 1):
        for index, card in enumerate(cards):
            point = _artwork_click(card, candidates)
            if point is None:
                continue
            side, symbol = (("", "") if len(cards) == 1 else ("左侧", "L") if index == 0 else ("右侧", "R"))
            shown = f"（显示价 {card.price}）" if card.price else ""
            preview = f"Product {row} {symbol}".rstrip()
            if card.price:
                preview += f" Price {card.price if card.price.isascii() else card.price[1:]}"
            action = {"role": "open_product", "function": f"打开第{row}组{side}商品{shown}",
                      "box": list(card.box), "click_point": point, "preview_text": preview,
                      "component_artwork": list(card.artwork)}
            if len(cards) > 1:
                action["component_caption"] = [card.box[0], card.artwork[3], card.box[2], card.box[3]]
            if used_subtle and font_index is not None and (name := _inline_model_name(frame, card, font_index)):
                action.update(function=f"打开{name}商品详情", preview_text=name, product_title=name)
            result.append(action)
    return result
