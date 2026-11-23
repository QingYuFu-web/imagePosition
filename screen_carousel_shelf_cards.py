"""Identify complete owned product cards beneath a verified shop selector."""

from collections import Counter

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_grouped_shop_navigation import _adjacent_groups, _rows, _selected, _strip
from screen_wide_panel_checks import _read_glyph


def _edge(signal, direction, contrast, *, support):
    strength = (direction * (signal[3:] - signal[:-3]) >= contrast).mean(axis=1)
    peaks = [(left, right) for left, right in _bands(strength >= support)
             if right - left <= 5]
    if len(peaks) != 1:
        return None
    left, right = peaks[0]
    indices = np.flatnonzero(strength[left:right] == strength[left:right].max())
    return left + int(np.median(indices)) + 2


def _physical_card(gray, slot, rail_bottom, rail_height, pitch):
    left, right = slot
    inset = round(.17 * pitch)
    start = rail_bottom + round(.14 * rail_height)
    stop = rail_bottom + round(.65 * rail_height)
    roof = _edge(gray[start:stop, left + inset:right - inset], 1, 19, support=.67)
    if roof is None:
        return None
    top = start + roof
    start = top + round(.96 * pitch)
    stop = min(len(gray), top + round(1.66 * pitch))
    if stop - start < .35 * pitch:
        return None
    base = _edge(gray[start:stop, left + inset:right - inset], -1, 22, support=.66)
    if base is None:
        return None
    bottom = start + base
    sample = gray[top + round(.16 * (bottom - top)):top + round(.54 * (bottom - top)):2]
    rise_start = max(0, left - round(.04 * pitch))
    fall_stop = min(gray.shape[1], right)
    rise = _edge(sample[:, rise_start:left + round(.175 * pitch)].T, 1, 16, support=.66)
    fall_start = right - round(.175 * pitch)
    fall = _edge(sample[:, fall_start:fall_stop].T, -1, 16, support=.66)
    if rise is None or fall is None:
        return None
    x, end = rise_start + rise, fall_start + fall
    return (x, top, end, bottom) if .77 * pitch <= end - x <= .96 * pitch else None


def _owned(grid, candidates):
    supported = set()
    for index, (left, top, right, bottom) in enumerate(grid):
        w, h = right - left, bottom - top
        if any(row["visibility"] == "page"
               and max(abs(a - b) for a, b in zip(grid[index], row["box"], strict=True)) <= .085 * h
               and .79 * w <= row["box"][2] - row["box"][0] <= 1.21 * w
               for row in candidates):
            supported.add(index)
    return supported


def _ink_sheared(tile, *, slant=.24):
    h, w = tile.shape
    transform = np.float32([[1, slant, -slant * h / 2], [0, 1, 0]])
    return cv2.warpAffine(tile, transform, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _digit(ink, index, cache):
    key = ink.shape, ink.tobytes()
    if key not in cache:
        options = [item for item in index.rank(ink, alphabet="code", limit=7) if item["text"].isdigit()]
        cache[key] = options[0]["text"] if options and options[0]["distance"] <= .228 and (
            len(options) == 1 or options[1]["distance"] - options[0]["distance"] >= .06
        ) else None
    return cache[key]


def _stock(gray, box, index, cache):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    first, last = top + round(.026 * h), top + round(.118 * h)
    # A printed stock tag is a separate dark rectangle ending at the card wall.
    band = gray[first:last, left + round(.24 * w):right - 4]
    under = gray[last + 6:last + 12, left + round(.30 * w):right - 4]
    if np.percentile(band, 50) > 52 or np.percentile(under, 50) - np.percentile(band, 50) < 34:
        return None
    patch = np.uint8(gray[first:last, right - round(.20 * w):right - 2])
    measured = _stock_digits(_ink_sheared(patch), index, cache)
    if measured is not None:
        return measured
    return _stock_digits(_ink_sheared(patch, slant=.28), index, cache)


def _stock_digits(tile, index, cache):
    reads = []
    for threshold in (133, 154, 176, 193):
        ink = np.uint8(tile >= threshold)
        spans = _bands(ink.any(axis=0))
        if not 2 <= len(spans) <= 5:
            continue
        start, end = spans[-1]
        _, y, width, height = cv2.boundingRect(ink[:, start:end])
        if not (.41 * len(tile) <= height <= .79 * len(tile) and .27 * height <= width <= .80 * height
                and 1 < y < y + height < len(tile) - 1 and end >= .70 * tile.shape[1]):
            continue
        a, b = spans[-2]
        n, _, stats, _ = cv2.connectedComponentsWithStats(ink[:, a:b])
        if n != 3 or not .13 * height <= b - a <= .34 * height or not 1 <= start - b <= .35 * height:
            continue
        _, u, _, hu, area = stats[1]
        _, v, _, hv, area2 = stats[2]
        if not (.26 * height <= v - u <= .65 * height and abs(area - area2) <= .35 * max(area, area2)
                and min(hu, hv) >= 2):
            continue
        digit = _digit(ink[:, start:end], index, cache)
        if digit is not None:
            reads.append(digit)
    if len(reads) >= 2 and len(set(reads)) == 1:
        return int(reads[0])
    return None


def _coin(card, first, last):
    width = card.shape[1]
    a, b = round(.27 * width), round(.63 * width)
    chip = card[first:last, a:b].astype(np.int16)
    ink = np.uint8(np.ptp(chip, axis=2) >= 28) * 255
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (.106 * width <= min(w, h) <= max(w, h) <= .195 * width and .79 <= w / h <= 1.23
                and 0 < y < y + h < len(chip) and cv2.contourArea(contour) >= .55 * w * h):
            shapes.append((a + x + w, first + y, h))
    return shapes[0] if len(shapes) == 1 else None


def _price(frame, gray, box, index, cache):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    first, last = round(.797 * h), round(.937 * h)
    token = _coin(frame[top:bottom, left:right], first, last)
    if token is None:
        return None
    coin_right, coin_top, coin_height = token
    plot = gray[top + first:top + last, left + coin_right + max(3, round(.014 * w)):right - max(3, round(.014 * w))]
    if min(plot.shape) < 11:
        return None
    ink = _ink_sheared(np.uint8(plot))
    ink_level = np.percentile(ink, 10)
    paper = np.percentile(ink, 79)
    if paper - ink_level < 64:
        return None
    reads = []
    for fraction in (.51, .65):
        mask = np.uint8(ink <= paper - fraction * (paper - ink_level))
        spans = _bands(mask.any(axis=0))
        if not 3 <= len(spans) <= 6:
            continue
        digits = []
        for a, b in spans:
            _, y, width, height = cv2.boundingRect(mask[:, a:b])
            if (not .64 * coin_height <= height <= 1.21 * coin_height or not .28 <= width / height <= .76
                    or abs(first + y + height / 2 - coin_top - coin_height / 2) > .24 * coin_height):
                break
            digit = _digit(mask[:, a:b], index, cache)
            if digit is None:
                break
            digits.append(digit)
        if len(digits) == len(spans):
            reads.append("".join(digits))
    return reads[0] if len(reads) == 2 and reads[0] == reads[1] else None


def _title_segments(tile, fraction, width, height):
    paper = np.percentile(tile, 72)
    ink_level = np.percentile(tile, 8)
    if paper - ink_level < 70:
        return None
    mask = np.uint8(tile <= paper - fraction * (paper - ink_level))
    left, top, w, h = cv2.boundingRect(mask)
    if not (0 < top < top + h < len(tile) and max(21, .044 * height) <= h <= .084 * height
            and .32 * width <= w <= .93 * width):
        return None
    result = []
    for start, end in _bands(mask.any(axis=0)):
        _, y, span, depth = cv2.boundingRect(mask[:, start:end])
        center_dot = (.075 * h <= span <= .24 * h and .075 * h <= depth <= .24 * h
                      and .34 * h <= y + depth / 2 - top <= .68 * h)
        if not center_dot and (not .55 * h <= span <= 1.19 * h or depth < .77 * h
                               or abs(y + depth / 2 - top - h / 2) > .14 * h):
            return None
        result.append((start, end, center_dot))
    if not 3 <= len(result) <= 10 or result[0][2] or result[-1][2] or sum(dot for _, _, dot in result) > 1:
        return None
    return result


def _plain_font_votes(signal, peak, index):
    reads = []
    seen = set()
    for cut in (.36, .42, .48, .56):
        mask = np.uint8(signal >= cut * peak)
        identity = mask.tobytes()
        if identity in seen:
            continue
        seen.add(identity)
        ranked = index.rank(mask, alphabet="cjk", limit=2)
        if len(ranked) == 2 and ranked[0]["distance"] <= .232:
            reads.append((ranked[0]["text"], ranked[1]["distance"] - ranked[0]["distance"]))
    if len(reads) >= 3 and len({name for name, _ in reads}) == 1 and np.mean([score for _, score in reads]) >= .021:
        return reads[0][0]
    return None


def _strong_plain_votes(signal, peak, index):
    previous = None
    name = None
    margin = 0.
    for cut in (.36, .42, .48):
        mask = np.uint8(signal >= cut * peak)
        if previous is not None and np.array_equal(previous, mask):
            return None
        previous = mask
        ranked = index.rank(mask, alphabet="cjk", limit=2)
        if len(ranked) < 2 or ranked[0]["distance"] > .19:
            return None
        current = ranked[0]["text"]
        gap = ranked[1]["distance"] - ranked[0]["distance"]
        if gap < .018 or current == "內" or name is not None and current != name:
            return None
        name = current
        margin += gap
    return name if margin >= .087 else None


def _glyph_samples(tile, pieces, index, cache, *, plain):
    output = []
    for a, b, dot in pieces:
        if dot:
            output.append("·")
            continue
        part = tile[:, a:b]
        key = part.shape, part.tobytes()
        key = plain, *key
        if key not in cache:
            paint = float(np.percentile(part, 70))
            signal = np.maximum(paint - part.astype(np.float32), 0)
            peak = float(np.percentile(signal, 98))
            read = None
            if peak >= 73:
                if plain:
                    read = _strong_plain_votes(signal, peak, index)
                if read is None:
                    read = _read_glyph(signal, peak, index)
                if read is None and plain:
                    read = _plain_font_votes(signal, peak, index)
            cache[key] = read
        output.append(cache[key])
    return output


def _title(gray, box, index, cache):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    tile = np.uint8(gray[top + round(.686 * h):top + round(.793 * h),
                         left + round(.104 * w):right - round(.018 * w)])
    first = _title_segments(tile, .48, w, h)
    second = _title_segments(tile, .36, w, h)
    if not first or not second or len(first) != len(second):
        return None
    if any(dot != other or max(abs(a - c), abs(b - d)) > .021 * w
           for (a, b, dot), (c, d, other) in zip(first, second, strict=True)):
        return None
    samples = [_glyph_samples(tile, spans, index, cache, plain=True) for spans in (first, second)]
    if any(not any(pair) for pair in zip(*samples, strict=True)):
        magnified = cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for fraction in (.42, .36):
            spans = _title_segments(magnified, fraction, 2 * w, 2 * h)
            if spans is not None and len(spans) == len(first) and all(
                dot == other and max(abs(a / 2 - c), abs(b / 2 - d)) <= .027 * w
                for (a, b, dot), (c, d, other) in zip(spans, first, strict=True)
            ):
                samples.append(_glyph_samples(magnified, spans, index, cache, plain=False))
    answer = []
    for readings in zip(*samples, strict=True):
        votes = Counter(item for item in readings if item)
        if len(votes) != 1 or max(votes.values()) < 2:
            return None
        answer.append(next(iter(votes)))
    return "".join(answer)


def shelf_card_actions(frame, regions, index):
    """Return one action per legible available card with a complete physical boundary."""
    height, width = frame.shape[:2]
    if index is None or regions.get("layer") != "page" or height < 300 or width < 1.45 * height:
        return []
    if regions.get("regions"):
        return []
    candidates = regions["candidates"]
    row = _rows(candidates, width, height)
    if row is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    strip = _strip(gray, row)
    if strip is None or not _adjacent_groups(gray, candidates, *strip) or _selected(frame, *strip) is None:
        return []
    edges, upper, lower, pitch = strip
    grid = [_physical_card(gray, slot, lower, lower - upper, pitch)
            for slot in zip(edges[:-1], edges[1:], strict=True)]
    if any(box is None for box in grid) or len(grid) < 4:
        return []
    boxes = np.array(grid)
    if (np.ptp(boxes[:, 1]) > .018 * pitch or np.ptp(boxes[:, 3]) > .022 * pitch
            or np.ptp(boxes[:, 2] - boxes[:, 0]) > .062 * pitch):
        return []
    owners = _owned(grid, candidates)
    if len(owners) < 2:
        return []
    actions, digits, glyphs = [], {}, {}
    for left, top, right, bottom in grid:
        w, h = right - left, bottom - top
        card = frame[top:bottom, left:right]
        art = card[round(.19 * h):round(.56 * h):2, round(.08 * w):round(.91 * w):2]
        chroma = np.ptp(art, axis=2)
        if (chroma >= 64).mean() < .45 or np.median(art.max(axis=2)) < 115:
            continue
        box = left, top, right, bottom
        if not _stock(gray, box, index, digits) or _price(frame, gray, box, index, digits) is None:
            continue
        name = _title(gray, box, index, glyphs)
        if name:
            actions.append({"role": "game_offer_card", "function": f"查看{name}", "preview_text": name,
                            "box": list(box), "click_point": [(left + right) // 2, (top + bottom) // 2]})
    counts = Counter(action["function"] for action in actions)
    return [action for action in actions if counts[action["function"]] == 1]
