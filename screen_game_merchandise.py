"""Verify complete stock-aware merchandise cards in a landscape storefront."""

import cv2
import numpy as np

from screen_artwork_title import artwork_title
from screen_component_captions import _bands
from screen_game_offer_cards import _title


def _carrier_row(regions, width, height):
    boxes = [tuple(row["box"]) for row in regions["candidates"] if row.get("visibility") == "page"]
    plausible = []
    for left, top, right, bottom in boxes:
        w, h = right - left, bottom - top
        if (0 <= left < right <= width and 0 <= top < bottom <= height
                and .13 * width <= w <= .31 * width and .23 * height <= h <= .49 * height
                and .87 <= w / h <= 1.43 and top <= .33 * height):
            plausible.append((left, top, right, bottom))
    aligned = []
    for box in plausible:
        w, h = box[2] - box[0], box[3] - box[1]
        row = sorted(peer for peer in plausible if abs(peer[1] - box[1]) <= .043 * h
                     and abs(peer[3] - box[3]) <= .070 * h and .90 * w <= peer[2] - peer[0] <= 1.11 * w)
        if not 3 <= len(row) <= 7 or row in aligned:
            continue
        centers = [(a + r) / 2 for a, _, r, _ in row]
        steps = np.diff(centers)
        if .87 * w <= steps.min() <= steps.max() <= 1.26 * w and np.ptp(steps) <= .09 * w:
            aligned.append(row)
    return aligned[0] if len(aligned) == 1 else []


def _wall(frame, x, top, bottom, spread, direction, width):
    left, right = max(0, x - spread), min(frame.shape[1], x + spread + 1)
    if right - left < 6:
        return None
    swatch = frame[top:bottom:2, left:right].astype(np.int16)
    inside = x + direction * round(.15 * width)
    paint = frame[top:bottom:2, inside].astype(np.int16)
    earlier, later = swatch[:, :-2], swatch[:, 2:]
    inner, outer = (later, earlier) if direction > 0 else (earlier, later)
    strength = abs(swatch[:, 2:] - swatch[:, :-2]).max(axis=2)
    inner_distance = abs(inner - paint[:, None]).max(axis=2)
    outer_distance = abs(outer - paint[:, None]).max(axis=2)
    support = ((strength >= 18) & (inner_distance < 25) & (outer_distance >= inner_distance + 8)).mean(axis=0)
    index = int(support.argmax())
    return int(left + index + 1) if support[index] >= .54 else None


def _stock_line(frame, center, top, bottom, width, contained):
    reach = round(.235 * width)
    patch = frame[top:bottom, center - reach:center + reach]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    paint = np.median(gray)
    mask = np.uint8(gray >= paint + 33)
    bands = [(a, b) for a, b in _bands(mask.sum(axis=1) >= max(4, round(.017 * width)))
             if .30 * len(patch) <= b - a <= .80 * len(patch) and .21 * len(patch) <= (a + b) / 2 <= .85 * len(patch)]
    if len(bands) != 1:
        return False
    y, end = bands[0]
    x, _, w, h = cv2.boundingRect(mask[y:end])
    if not (.10 * width <= w <= .36 * width and .30 * len(patch) <= h <= .80 * len(patch)
            and abs(x + w / 2 - reach) <= .072 * width and 0 < y < y + h < len(patch)):
        return False
    if not contained:
        return True
    spans = _bands(mask[y:y + h].any(axis=0))
    return (3 <= len(spans) <= 7
            and .10 * h <= spans[-1][1] - spans[-1][0] <= .79 * h
            and .17 * h <= spans[-1][0] - spans[-2][1] <= 1.02 * h)


def _card(frame, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    center = (left + right) // 2
    start, end = top + round(.21 * h), top + round(.57 * h)
    first, last = (_wall(frame, x, start, end, round(.095 * w), direction, w)
                   for x, direction in ((left, 1), (right, -1)))
    if first is None or last is None or not .79 * w <= last - first <= .98 * w:
        return None
    pad = max(3, round(.075 * w))
    wings = ((first + pad, first + 2 * pad), (last - 2 * pad, last - pad))
    profile = np.stack([np.median(frame[top:bottom, a:b:2], axis=1) for a, b in wings], axis=1)
    under = np.median(profile[round(.865 * h):round(.937 * h)], axis=0)
    on_footer = (abs(profile - under).max(axis=(1, 2)) <= 20)
    footers = [a for a, b in _bands(on_footer) if .68 * h <= a <= .86 * h and b - a >= .072 * h]
    if len(footers) != 1:
        return None
    foot = footers[0]
    if abs(profile[foot - round(.031 * h)] - under).max(axis=1).min() < 34:
        return None
    mid = np.median(profile[round(.49 * h):round(.62 * h)], axis=0)
    stock = np.median(profile[foot - round(.105 * h):foot - round(.028 * h)], axis=0)
    if abs(under[0] - under[1]).max() > 13 or abs(mid[0] - mid[1]).max() > 17:
        return None
    status = abs(stock - mid).max(axis=1)
    if status.max() <= 17:
        available = True
    elif status.min() >= 36 and abs(stock[0] - stock[1]).max() <= 19:
        available = False
    else:
        return None
    if not _stock_line(frame, center, top + foot - round(.128 * h), top + foot - round(.020 * h), w, available):
        return None
    shelf = top + foot
    label = (center - round(.22 * w), shelf - round(.235 * h),
             center + round(.22 * w), shelf - round(.124 * h))
    painted_box = (first, top, last, bottom - round(.031 * h))
    return painted_box, shelf, available, label, under.mean(axis=0)


def game_merchandise_actions(frame, regions, index):
    """Open one visible product detail per physically verified, legible card."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or height < 240 or width < 1.46 * height:
        return []
    row = _carrier_row(regions, width, height)
    if not row:
        return []
    inspected = [_card(frame, box) for box in row]
    if any(card is None for card in inspected):
        return []
    shelves = [card[1] for card in inspected]
    bases = np.stack([card[4] for card in inspected])
    if (max(shelves) - min(shelves) > .016 * height or np.ptp(bases, axis=0).max() > 19
            or not {False, True} <= {card[2] for card in inspected}):
        return []
    actions = []
    for box, _, available, (left, top, right, bottom), _ in inspected:
        if not available:
            continue
        tile = frame[top:bottom, left:right]
        if name := _title(tile, index) or artwork_title(tile, index):
            x0, y0, x1, y1 = box
            actions.append({"role": "game_offer_card", "function": f"查看{name}", "preview_text": name, "box": list(box),
                            "click_point": [(x0 + x1) // 2, (y0 + y1) // 2]})
    return [action for action in actions if sum(peer["function"] == action["function"] for peer in actions) == 1]
