"""Recognize bounded catalog cards with a shared currency and integral price row."""

from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts
from screen_component_captions import _bands
from screen_game_merchandise import _wall
from screen_product_rows import _glyph, _price_index
from screen_wide_panel_checks import _magnified_label, _read_label


@dataclass(frozen=True)
class _Card:
    left: int
    top: int
    right: int
    shelf: int
    bottom: int | None
    price: str
    coin: np.ndarray
    paint: np.ndarray

    @property
    def width(self):
        return self.right - self.left


def _grid(regions, width, height):
    proposals = set()
    for row in regions["candidates"]:
        if row.get("visibility") != "page":
            continue
        left, top, right, bottom = map(int, row["box"])
        w, h = right - left, bottom - top
        if (0 <= left < right <= width and 0 <= top < bottom <= height
                and .12 * width <= w <= .245 * width and .24 * height <= h <= .47 * height
                and .85 <= w / h <= 1.48):
            proposals.add((left, top, right, bottom))
    rows = set()
    for l, t, r, _ in proposals:
        span = r - l
        peers = tuple(sorted((a, b, c, d) for a, b, c, d in proposals
                             if abs(b - t) <= .043 * span and .91 * span <= c - a <= 1.11 * span))
        if not 3 <= len(peers) <= 6:
            continue
        steps = np.diff([(a + c) / 2 for a, _, c, _ in peers])
        gaps = [b[0] - a[2] for a, b in zip(peers, peers[1:])]
        if .86 * span <= steps.min() <= steps.max() <= 1.23 * span and np.ptp(steps) <= .075 * span:
            if min(gaps) >= -.04 * span and max(gaps) <= .23 * span:
                rows.add(peers)
    rows = sorted(rows, key=lambda boxes: boxes[0][1])
    candidates = []
    for first, second in zip(rows, rows[1:]):
        span = float(np.median([r - l for l, _, r, _ in first]))
        if (len(first) == len(second) and .81 * span <= second[0][1] - first[0][1] <= 1.19 * span
                and all(abs(a + b - c - d) <= .09 * span
                        for (a, _, b, _), (c, _, d, _) in zip(first, second, strict=True))):
            candidates.append((first, second))
    return candidates[0] if len(candidates) == 1 else ()


def _price(gray, left, right, foot, bank, cache):
    span = right - left
    middle = (left + right) // 2
    width = round(.23 * span)
    height = round(.153 * span)
    tile = gray[foot + max(2, round(.01 * span)):foot + height, middle - width:middle + width]
    paint = int(np.median(tile))
    if tile.shape[0] < 12 or paint >= 155:
        return None
    reads, icons, rasters = Counter(), {}, set()
    for lift in (65, 69, 73, 77, 83, 90):
        ink = np.uint8(tile >= paint + lift)
        key = ink.tobytes()
        if key in rasters:
            continue
        rasters.add(key)
        outer, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coin = []
        numbers = []
        for contour in outer:
            x, y, w, h = cv2.boundingRect(contour)
            fill = cv2.contourArea(contour) / (w * h)
            if .082 * span <= min(w, h) <= max(w, h) <= .14 * span and .83 <= w / h <= 1.19 and fill >= .56:
                coin.append((x, y, w, h))
            elif .040 * span <= h <= .082 * span and .22 <= w / h <= .92 and fill >= .115:
                numbers.append((x, y, w, h))
        if len(coin) != 1 or not 1 <= len(numbers) <= 4:
            continue
        x, y, w, h = coin[0]
        numbers.sort()
        if not .016 * span <= numbers[0][0] - (x + w) <= .076 * span:
            continue
        if any(abs(v + p / 2 - y - h / 2) > .019 * span for _, v, _, p in numbers):
            continue
        if any(not 0 <= b[0] - a[0] - a[2] <= .042 * span for a, b in zip(numbers, numbers[1:])):
            continue
        end = numbers[-1][0] + numbers[-1][2]
        if abs((x + end) / 2 - width) > .067 * span:
            continue
        chars = [_glyph(ink[a:a + d, l:l + r] != 0, bank, cache) for l, a, r, d in numbers]
        if all(char is not None and char.isascii() and char.isdigit() for char in chars):
            label = "".join(chars)
            reads[label] += 1
            if label not in icons:
                icons[label] = cv2.resize(np.uint8(tile[y:y + h, x:x + w]), (20, 20), interpolation=cv2.INTER_AREA)
    confirmed = [text for text, count in reads.items() if count >= 2]
    return (confirmed[0], icons[confirmed[0]]) if len(confirmed) == 1 else None


def _card(frame, gray, box, bank, cache):
    a, y, b, _ = box
    width = b - a
    start, end = y + round(.18 * width), y + round(.63 * width)
    spread = max(4, round(.12 * width))
    left = _wall(frame, a, start, end, spread, 1, width)
    right = _wall(frame, b, start, end, spread, -1, width)
    if left is None or right is None or not .86 * width <= right - left <= .995 * width:
        return None
    span = right - left
    middle = (left + right) // 2
    left_wing = np.arange(left + round(.11 * span), middle - round(.21 * span), 3)
    right_wing = np.arange(middle + round(.21 * span), right - round(.11 * span), 3)
    columns = np.r_[left_wing, right_wing]
    gap = max(2, round(.013 * span))
    horizon = np.arange(y + round(.66 * width), min(len(gray) - gap, y + round(.90 * width)))
    jump = gray[horizon[:, None] - gap, columns] - gray[horizon[:, None] + gap, columns]
    rise = np.minimum((jump[:, :len(left_wing)] >= 26).mean(axis=1),
                      (jump[:, len(left_wing):] >= 26).mean(axis=1))
    groups = [(first, last) for first, last in _bands(rise >= .82) if last - first <= .060 * span]
    if len(groups) != 1:
        return None
    first, last = groups[0]
    strength = np.median(jump[first:last], axis=1)
    foot = int(horizon[first + int(strength.argmax())])
    if strength.max() < 38 or foot + .23 * span >= len(gray):
        return None
    reach = round(.30 * span)
    artwork = frame[y + round(.13 * span):foot - round(.20 * span):2, middle - reach:middle + reach:2]
    if np.std(artwork.reshape(-1, 3), axis=0).max() < 23:
        return None
    recognized = _price(gray, left, right, foot, bank, cache)
    if recognized is None:
        return None
    price, coin = recognized
    tail = np.arange(foot + round(.132 * span), foot + round(.211 * span))
    contrast = abs(frame[tail[:, None] - gap, columns].astype(np.int16)
                   - frame[tail[:, None] + gap, columns].astype(np.int16)).max(axis=2)
    rates = (contrast >= 12).mean(axis=1)
    amount = np.median(contrast, axis=1)
    pos = int((amount * rates).argmax())
    bottom = int(tail[pos]) if rates[pos] >= .64 and amount[pos] >= 12 else None
    paint = np.median(frame[foot + round(.055 * span):foot + round(.125 * span), columns], axis=(0, 1))
    return _Card(left, y, right, foot, bottom, price, coin, paint)


def _warning(frame, card):
    height = round(.154 * card.width)
    body = frame[card.shelf - height:card.shelf, card.left:card.right].astype(np.int16)
    red = body[:, :, 2] - np.maximum(body[:, :, 0], body[:, :, 1])
    rows = [(a, b) for a, b in _bands((red >= 64).mean(axis=1) >= .50)
            if .057 * card.width <= b - a <= .135 * card.width]
    if not rows:
        return False
    # A broad high-chroma inscription strip should never be a default tap.
    return any((body[a:b].min(axis=2) >= 174).sum() >= .018 * (b - a) * card.width for a, b in rows)


def _titles(gray, pale, card):
    span = card.width
    center = (card.left + card.right) // 2
    radius = round(.32 * span)
    left, right = center - radius, center + radius
    first = card.shelf - round(.256 * span)
    stop = card.shelf - max(2, round(.012 * span))
    tile = pale[first:stop, left:right]
    bands = []
    for cut in (178, 192, 208, 224):
        ink = np.uint8(tile >= cut)
        for a, b in _bands(ink.sum(axis=1) >= max(4, round(.032 * span))):
            if not .054 * span <= b - a <= .108 * span:
                continue
            x, _, w, h = cv2.boundingRect(ink[a:b])
            if (1.50 * h <= w <= .55 * span and abs(x + w / 2 - radius) <= .085 * span
                    and .19 <= ink[a:b, x:x + w].mean() <= .69):
                bands.append((cut, (left + x - 3, first + a - 2, left + x + w + 3, first + b + 2)))
    balanced = [box for cut, box in bands if any(other != cut and max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= 4
                                            for other, peer in bands)]
    groups = []
    for box in balanced:
        if not any(abs(box[1] - prior[1]) <= 4 for prior in groups):
            groups.append(box)
    return groups


def _outlined_label(frame, gray, pale, box, index, *, short=False):
    left, top, right, bottom = box
    votes = {}
    seen = set()
    for plane in (gray, pale, frame[..., 2]):
        chip = plane[top:bottom, left:right]
        for cut in range(174, 232, 4):
            ink = np.uint8(chip >= cut)
            x, y, w, h = cv2.boundingRect(ink)
            if not .55 * len(chip) <= h <= .79 * len(chip) or not 1 < x < x + w < chip.shape[1] - 1:
                continue
            view = ink[y:y + h, x:x + w]
            witness = view.shape, view.tobytes()
            if witness in seen:
                continue
            seen.add(witness)
            parts = _cjk_parts(view, h, join_radicals=True)
            if not (2 <= len(parts) <= (3 if short else 7)):
                continue
            letters, distances, margins = [], [], []
            for part in parts:
                options = index.rank(part[-1], alphabet="cjk", limit=2)
                if len(options) != 2:
                    break
                best, runner_up = options
                distance = best["distance"]
                margin = runner_up["distance"] - distance
                if distance > .246 or margin < .017:
                    break
                letters.append(best["text"])
                distances.append(distance)
                margins.append(margin)
            else:
                if np.mean(distances) <= .205 and np.median(margins) >= .033:
                    text = "".join(letters)
                    votes.setdefault(text, set()).add(witness)
    verified = [(len(witnesses), text) for text, witnesses in votes.items() if len(witnesses) >= 3]
    return verified[0][1] if len(verified) == 1 and all(verified[0][0] - len(other) >= 2
                                                      for word, other in votes.items() if word != verified[0][1]) else None


def _bounded_title(frame, gray, pale, box, index):
    """Match foreground silhouettes within the title band's aligned baseline."""
    left, top, right, bottom = box
    if not 14 <= bottom - top <= 34 or left < 2 or right + 2 > frame.shape[1]:
        return None
    for trim in (2, 3):
        votes = defaultdict(dict)
        ranks = {}
        for plane_no, plane in enumerate((gray, pale, frame[..., 2])):
            crop = plane[top + trim:bottom - trim, left - 2:right + 2]
            levels = np.quantile(crop[:, 2:-2], (.60, .64, .65, .66, .67, .68, .69, .71, .74), method="higher")
            for cut in sorted(set(levels.tolist())):
                ink = np.uint8(crop >= cut)
                x, y, width, height = cv2.boundingRect(ink)
                if (height < .66 * len(crop) or not 1 < x < x + width < crop.shape[1] - 1
                        or abs(2 * x + width - crop.shape[1]) > .11 * width):
                    continue
                view = ink[y:y + height, x:x + width]
                witness = view.shape, view.tobytes()
                if witness not in ranks:
                    ranks[witness] = _title_silhouette(view, height, index)
                if (match := ranks[witness]) is not None:
                    title, score = match
                    vote = votes[title].setdefault(witness, (score, set()))
                    vote[1].add(plane_no)
        if not votes:
            continue
        title, best = max(votes.items(), key=lambda item: len(item[1]))
        strength = len(best)
        if strength >= 3 and len(set().union(*(planes for _, planes in best.values()))) >= 2:
            best_score = min(score for score, _ in best.values())
            if all(strength > 2 * len(choices) and min(score for score, _ in choices.values()) >= best_score + .009
                   for other, choices in votes.items() if other != title):
                return title
    return None


def _title_silhouette(view, height, index):
    parts = _cjk_parts(view, height, join_radicals=True)
    if not 3 <= len(parts) <= 7 or max(part[1] for part in parts) > .15 * height or min(part[3] for part in parts) < .84 * height:
        return None
    letters, distances, margins = [], [], []
    for *_, mask in parts:
        rows = index.rank(mask, alphabet="cjk", limit=2)
        if len(rows) != 2:
            return None
        best, second = rows
        distance = best["distance"]
        margin = second["distance"] - distance
        if distance > .242 or margin + 1e-8 < .015:
            return None
        letters.append(best["text"])
        distances.append(distance)
        margins.append(margin)
    score = sum(distances) / len(distances)
    if score < .205 and np.median(margins) >= .028:
        return "".join(letters), score
    return None


def _quantity_lane(pale, box, card, bank, cache):
    """Verify a shelf-adjacent qualifier ending with a colon and Arabic numeral."""
    left, top, right, bottom = box
    if not .078 * card.width <= card.shelf - top <= .13 * card.width or right - left < .44 * card.width:
        return False
    tile = pale[top:bottom, right - round(.17 * card.width):right]
    votes = {}
    for level in (153, 160, 168, 176):
        ink = np.uint8(tile >= level)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        pieces = sorted((*map(int, record), label) for label, record in enumerate(stats[1:], 1))
        if len(pieces) < 4:
            continue
        x, y, w, h, _, numeral = pieces[-1]
        if not (x + w >= tile.shape[1] - 6 and .58 * len(tile) <= h <= .87 * len(tile)
                and .19 < w / h <= .80):
            continue
        dots = sorted(pieces[-3:-1], key=lambda dot: dot[1])
        if not all(1 <= min(a, b) <= max(a, b) <= .29 * h and .07 * h <= x - (u + a) <= .37 * h
                   for u, v, a, b, _, _ in dots):
            continue
        first, second = dots
        if not (abs(first[0] - second[0]) <= .15 * h and .34 * h <= second[1] - first[1] <= .57 * h
                and .16 * h <= first[1] - y <= .47 * h and abs(second[1] + second[3] - y - h) <= .20 * h):
            continue
        mark = labels[y:y + h, x:x + w] == numeral
        label = _glyph(mark, bank, cache)
        if label is not None and label.isascii() and label.isdigit():
            votes.setdefault(label, set()).add((ink.shape, ink.tobytes()))
    return len(votes) == 1 and len(next(iter(votes.values()))) >= 2


def _named_item(frame, gray, pale, card, index, bank, cache):
    texts = set()
    lanes = _titles(gray, pale, card)
    for box in lanes:
        if _quantity_lane(pale, box, card, bank, cache):
            continue
        found = _read_label(gray, box, index)
        if not found and card.shelf - box[1] >= .17 * card.width:
            found = _bounded_title(frame, gray, pale, box, index)
        if not found:
            found = _magnified_label(gray, box, index)
        if not found:
            left, top, right, bottom = box
            if card.shelf - top >= .117 * card.width:
                found = _outlined_label(frame, gray, pale, (left - 3, top - 2, right + 3, bottom + 2), index)
        if found:
            texts.add(found)
    if not texts and len(lanes) == 1:
        left, top, right, bottom = lanes[0]
        width = card.width
        if .078 * width <= card.shelf - top <= .13 * width and right - left >= .44 * width:
            center = (card.left + card.right) // 2
            inset = round(.121 * width)
            box = (center - inset, top - round(.143 * width), center + inset, top - round(.027 * width))
            if found := _outlined_label(frame, gray, pale, box, index, short=True):
                texts.add(found)
    return next(iter(texts)) if len(texts) == 1 else None


def priced_catalog_card_actions(frame, regions, index):
    """Return one owned product action per verified item with a legible title."""
    height, width = frame.shape[:2]
    if index is None or regions.get("layer") != "page" or height < 260 or width < 1.48 * height:
        return []
    lattice = _grid(regions, width, height)
    if not lattice or (bank := _price_index()) is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    if frame.dtype == np.uint8 and frame.shape[2] == 3:
        blue, green, red = cv2.split(frame)
        pale = cv2.min(cv2.min(blue, green), red)
    else:
        pale = frame.min(axis=2)
    cache = {}
    inspected = [[_card(frame, gray, box, bank, cache) for box in row] for row in lattice]
    if any(item is None for row in inspected for item in row):
        return []
    cards = [item for row in inspected for item in row]
    ink = cards[0].coin.astype(np.int16)
    paint = np.stack([item.paint for item in cards])
    if np.ptp(paint, axis=0).max() > 26 or any(abs(ink - item.coin).mean() > 19 for item in cards):
        return []
    actions = []
    for row in inspected:
        bottoms = [item.bottom for item in row if item.bottom is not None]
        width_median = float(np.median([item.width for item in row]))
        if len(bottoms) < 2 or np.ptp(bottoms) > .035 * width_median or np.ptp([item.shelf for item in row]) > .034 * width_median:
            return []
        end = round(float(np.median(bottoms)))
        for item in row:
            if _warning(frame, item) or not (label := _named_item(frame, gray, pale, item, index, bank, cache)):
                continue
            actions.append({"role": "game_offer_card", "function": f"查看{label}", "preview_text": label,
                            "box": [item.left, item.top, item.right, end],
                            "click_point": [(item.left + item.right) // 2, (item.top + item.shelf) // 2]})
    counts = Counter(item["function"] for item in actions)
    return [item for item in actions if counts[item["function"]] == 1]
