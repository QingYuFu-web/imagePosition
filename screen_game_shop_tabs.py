"""Recognize inactive captioned game tabs in a physically divided storefront."""

from collections import Counter, defaultdict
from statistics import median

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts
from screen_component_captions import _bands
from screen_priced_catalog_cards import _grid as _priced_card_grid
from screen_wide_panel_checks import _magnified_label


def _card_strip(gray, candidates):
    height, width = gray.shape
    boxes = []
    for proposal in candidates:
        x, y, right, bottom = proposal["box"]
        w, h = right - x, bottom - y
        if (proposal.get("visibility") == "page" and .085 * height <= y <= .48 * height
                and .105 * width <= w <= .235 * width and .225 * height <= h <= .49 * height
                and .72 * w <= h <= 1.23 * w and x >= .145 * width):
            boxes.append((x, y, right, bottom))
    rails = set()
    for _, top, _, bottom in boxes:
        row = tuple(sorted(box for box in boxes if abs(box[1] - top) <= .018 * height
                           and abs(box[3] - bottom) <= .026 * height))
        if not 3 <= len(row) <= 7:
            continue
        widths = [right - x for x, _, right, _ in row]
        typical = median(widths)
        pitch = np.diff([box[0] for box in row])
        if (np.ptp(widths) > .14 * typical or not .84 * typical <= min(pitch) <= max(pitch) <= 1.23 * typical
                or np.ptp(pitch) > .10 * typical):
            continue
        caps = []
        for x, y, right, _ in row:
            pad = round(.20 * (right - x))
            reach = max(2, round(.018 * typical))
            top_band = gray[y - reach:y + 4 * reach, x + pad:right - pad]
            delta = top_band[reach:] - top_band[:-reach]
            support = (delta >= 25).mean(axis=1)
            if not len(support) or support.max() < .71:
                break
            caps.append(y + int(np.argmax(support)))
        if len(caps) == len(row) and np.ptp(caps) <= .025 * typical:
            rails.add((row[0][0], round(median(caps)), round(typical)))
    return next(iter(rails)) if len(rails) == 1 else None


def _pane(gray, strip):
    height, width = gray.shape
    cards_left, top, size = strip
    span = max(2, round(.013 * size))
    sample = gray[max(top, round(.14 * height)):round(.93 * height):3, :cards_left]
    inside = sample[:, span:-span]
    ridge = np.maximum(inside - sample[:, :-2 * span], inside - sample[:, 2 * span:])
    support = (ridge >= 11).mean(axis=0)
    strength = np.median(ridge, axis=0)
    ridges = []
    for start, stop in _bands((support >= .73) & (strength >= 13)):
        column = start + int(np.argmax(strength[start:stop])) + span
        ridges.append(column)
    pairs = [(left, right) for left in ridges for right in ridges
             if .65 * size <= right - left <= 1.23 * size and .071 * size <= cards_left - right <= .32 * size
             and 0 <= left <= .16 * width]
    return pairs[0] if len(pairs) == 1 else None


def _tab_icons(gray, left, right, strip_top):
    height = len(gray)
    width = right - left
    start, stop = max(0, strip_top - round(.10 * width)), min(height, strip_top + round(1.79 * width))
    base = np.median(gray[start:stop, left + round(.48 * width):right - round(.09 * width)], axis=1)
    lower = float(np.percentile(base, 38))
    bands = [(start + a, start + b) for a, b in _bands(base >= lower + 60)
             if .13 * width <= b - a <= .34 * width]
    if len(bands) != 1:
        return None
    selected = bands[0]
    line = gray[(selected[0] + selected[1]) // 2, left + round(.21 * width):right - round(.035 * width)]
    if (line >= lower + 44).mean() < .72:
        return None
    x0, x1 = left + round(.061 * width), left + round(.235 * width)
    ink = abs(gray[start:stop, x0:x1] - base[:, None]) >= 47
    icons = []
    for a, b in _bands(ink.sum(axis=1) >= max(3, round(.021 * width))):
        x, y, w, h = cv2.boundingRect(np.uint8(ink[a:b]))
        if .080 * width <= h <= .174 * width and .64 <= w / h <= 1.48 and x > 0 and x + w < x1 - x0:
            icons.append((x0 + x, start + a + y, x0 + x + w, start + a + y + h))
    if not 4 <= len(icons) <= 8:
        return None
    centers = [(a + b) / 2 for _, a, _, b in icons]
    pitch = median(np.diff(centers))
    active = [i for i, cy in enumerate(centers) if selected[0] <= cy <= selected[1]]
    if (len(active) != 1 or not .19 * width <= pitch <= .40 * width or np.ptp(np.diff(centers)) > .14 * pitch
            or np.ptp([(a + b) / 2 for a, _, b, _ in icons]) > .102 * pitch):
        return None
    ax, ay, ar, ab = icons[active[0]]
    dark_ink = (base[ay - start:ab - start, None] - gray[ay:ab, ax:ar]) >= 47
    return (icons, active[0], pitch) if dark_ink.mean() >= .20 else None


def _without_dots(mask, em):
    """Trim short punctuation-sized fringe components outside the text run."""
    spans = _bands(mask.any(axis=0))
    if len(spans) < 4:
        return mask
    a, b = spans[0]
    c, d = spans[-1]
    if b - a <= .33 * em and cv2.boundingRect(mask[:, a:b])[3] <= .44 * em:
        mask[:, a:b] = 0
    if d - c <= .33 * em and cv2.boundingRect(mask[:, c:d])[3] <= .44 * em:
        mask[:, c:d] = 0
    return mask


def _caption(gray, area, index, cache, *, min_glyph_margin=0.):
    x0, y0, x1, y1 = area
    patch = np.uint8(gray[y0:y1, x0:x1])
    if min(patch.shape) < 8:
        return None
    votes = defaultdict(dict)
    for scale in (1, 2):
        view = patch if scale == 1 else cv2.resize(patch, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        low, high = np.percentile(view, (25, 98))
        if high - low < 54:
            return None
        for cut in (.385, .45, .50, .56, .62):
            mask = np.uint8(view >= low + cut * (high - low))
            a, b, w, h = cv2.boundingRect(mask)
            mask = _without_dots(mask, h)
            a, b, w, h = cv2.boundingRect(mask)
            if not (.26 * view.shape[0] <= h <= .77 * view.shape[0] and .50 * view.shape[1] <= w <= .96 * view.shape[1]):
                continue
            line = mask[b:b + h, a:a + w]
            parts = _cjk_parts(line, h)
            if not 2 <= len(parts) <= 8:
                continue
            ranks = []
            for *_, glyph in parts:
                key = glyph.shape, glyph.tobytes()
                if key not in cache:
                    cache[key] = index.rank(glyph, alphabet="cjk", limit=2)
                ranks.append(cache[key])
            if any(len(r) < 2 or r[0]["distance"] > .200 for r in ranks):
                continue
            margins = [r[1]["distance"] - r[0]["distance"] for r in ranks]
            if median(margins) < .013 or min(margins) < min_glyph_margin:
                continue
            name = "".join(r[0]["text"] for r in ranks)
            votes[name][(scale, line.shape, line.tobytes())] = median(r[0]["distance"] for r in ranks)
    winners = sorted((-len(readings), median(readings.values()), name) for name, readings in votes.items()
                     if len(readings) >= 2 and {sample[0] for sample in readings} == {1, 2})
    if not winners or (len(winners) > 1 and winners[1][0] - winners[0][0] < 2
                       and winners[1][1] - winners[0][1] < .015):
        return None
    return winners[0][2]


def shop_tab_actions(frame, regions, index):
    """Return independently named siblings of a proven selected side tab."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or height < 235 or width < 1.47 * height:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    strip = _card_strip(gray, regions["candidates"])
    if strip is None or (pane := _pane(gray, strip)) is None:
        return []
    left, right = pane
    witness = _tab_icons(gray, left, right, strip[1])
    if witness is None:
        return []
    icons, selected, pitch = witness
    actions, cache = [], {}
    for i, (_, t, r, b) in enumerate(icons):
        if i == selected:
            continue
        cy = (t + b) / 2
        title = (r + max(2, round(.045 * pitch)), round(cy - .249 * pitch),
                 left + round(.651 * (right - left)), round(cy + .249 * pitch))
        caption = _caption(gray, title, index, cache)
        if caption:
            actions.append({"role": "game_shop_tab", "function": f"切换到{caption}", "preview_text": caption,
                            "box": [left + 1, round(cy - .50 * pitch), right, round(cy + .50 * pitch)],
                            "click_point": [(title[0] + title[2]) // 2, round(cy)]})
    counts = Counter(item["function"] for item in actions)
    return [item for item in actions if counts[item["function"]] == 1]


def _selected_catalog_plate(gray, candidates, row, pitch):
    grid_left, grid_top = row[0][:2]
    found = []
    for proposal in candidates:
        if proposal.get("visibility") != "page":
            continue
        left, top, right, bottom = proposal["box"]
        width, depth = right - left, bottom - top
        if not (.84 * pitch <= width <= 1.13 * pitch and .13 * pitch <= depth <= .27 * pitch
                and .05 * pitch <= grid_top - bottom <= .21 * pitch
                and grid_left <= left <= row[-1][2] - .73 * pitch):
            continue
        border = round(.17 * width)
        body = gray[top:bottom, left + border:right - border]
        hline = np.median(body, axis=1)
        ceiling = float(np.percentile(hline, 82))
        floor = np.median(body[:max(2, round(.065 * depth))])
        if ceiling < 157 or ceiling - floor < 68:
            continue
        bands = [(a, b) for a, b in _bands(hline >= floor + .66 * (ceiling - floor))
                 if .60 * depth <= b - a <= .96 * depth and 0 < a < b < depth]
        if len(bands) != 1:
            continue
        a, b = bands[0]
        mid = (left + right) / 2
        high = body[a + round(.24 * (b - a)):b - round(.23 * (b - a))]
        ink = np.uint8(high <= ceiling - 59)
        _, _, w, h = cv2.boundingRect(ink)
        if .24 * width <= w <= .52 * width and .20 * (b - a) <= h <= .54 * (b - a):
            found.append((mid, top + a, top + b, ceiling))
    return found[0] if len(found) == 1 else None


def _catalog_tab_rails(gray, bounds, top, bottom, ceiling):
    left, right = bounds
    inset = round(.13 * (right - left))
    reach = max(2, round(.076 * (bottom - top)))
    if top <= reach or bottom + reach + 1 >= len(gray):
        return None
    rows = np.arange(top - 1, bottom + 2)
    current = gray[rows, left + inset:right - inset]
    above = gray[rows - reach, left + inset:right - inset]
    below = gray[rows + reach, left + inset:right - inset]
    support = ((2 * current - above - below) >= 34).mean(axis=1)
    span = round(.28 * (bottom - top))
    candidates = (support[:span], support[-span:])
    if any(not len(values) or values.max() < .68 for values in candidates):
        return None
    first, last = int(rows[candidates[0].argmax()]), int(rows[-span + candidates[1].argmax()])
    interior = gray[first + reach:last - reach, left + inset:right - inset]
    if last - first < .62 * (bottom - top) or np.median(interior) > ceiling - 72:
        return None
    return first, last


def aligned_catalog_tab_actions(frame, regions, index):
    """Emit one tab action where the rail and a readable inactive caption agree."""
    height, width = frame.shape[:2]
    if (index is None or regions.get("layer") != "page" or regions.get("regions")
            or height < 260 or width < 1.48 * height):
        return []
    lattice = _priced_card_grid(regions, width, height)
    if not lattice:
        return []
    row = lattice[0]
    pitch = np.median(np.diff([(a + b) / 2 for a, _, b, _ in row]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    active = _selected_catalog_plate(gray, regions["candidates"], row, pitch)
    if active is None:
        return []
    center, top, bottom, ceiling = active
    tabs = []
    for step in (-1, 1):
        for offset in range(1, len(row)):
            middle = center + step * offset * pitch
            left, right = round(middle - pitch / 2), round(middle + pitch / 2)
            if left < max(0, row[0][0] - .12 * pitch) or right > min(width, row[-1][2] + .12 * pitch):
                break
            rail = _catalog_tab_rails(gray, (left, right), top, bottom, ceiling)
            if rail is None:
                break
            tabs.append((middle, left, right, *rail))
    if (len(tabs) < 2 or max(np.ptp([tab[i] for tab in tabs]) for i in (3, 4)) > .08 * (bottom - top)):
        return []
    actions, cache = [], {}
    for middle, left, right, upper, lower in sorted(tabs):
        trim = max(2, round(.09 * (lower - upper)))
        half = round(.213 * pitch)
        x = round(middle)
        crop = (x - half, upper + trim, x + half, lower - trim)
        name = _caption(gray, crop, index, cache, min_glyph_margin=.006)
        if not name and 27 <= lower - upper < 32:
            inner = round(.19 * (lower - upper))
            name = _magnified_label(gray, (crop[0], upper + inner, crop[2], lower - inner), index)
        if name:
            actions.append({"role": "game_shop_tab", "function": f"切换到{name}", "preview_text": name,
                            "box": [left, top, right, bottom], "click_point": [x, (upper + lower) // 2]})
    counts = Counter(item["function"] for item in actions)
    return [item for item in actions if counts[item["function"]] == 1]
