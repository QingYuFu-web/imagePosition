"""Own named equipment cards inside a proven wide foreground panel."""

from collections import Counter
from functools import lru_cache
from itertools import product
from pathlib import Path
import os

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts, _shape_parts
from screen_font_glyphs import _normal, _rasterize_bank, _shortlist
from screen_wide_panel_actions import _rail_edge


def _ring(field, x, y, radius):
    angle = np.linspace(0, 2 * np.pi, 64, endpoint=False, dtype=np.float32)
    ux, uy = np.cos(angle)[None, :], np.sin(angle)[None, :]
    step = max(1, round(.027 * radius))
    spread = max(2, round(.14 * radius))
    rings = np.arange(radius - spread, radius + spread + step, step, dtype=np.float32)
    offset = max(1.5, .052 * radius)
    reach = max(2, step)
    centers = list(product(range(x - reach, x + reach + 1), range(y - reach, y + reach + 1)))
    xx, yy = np.asarray(centers, dtype=np.float32).T

    def sample(r):
        px = xx[:, None, None] + r[None, :, None] * ux
        py = yy[:, None, None] + r[None, :, None] * uy
        return cv2.remap(field, px.reshape(-1, len(angle)), py.reshape(-1, len(angle)), cv2.INTER_LINEAR)

    rim = sample(rings)
    sides = np.maximum(sample(rings - offset), sample(rings + offset))
    strength = ((rim - sides >= 7) & (rim >= 44)).mean(axis=1)
    i = int(strength.argmax())
    if strength[i] < .69:
        return None
    center, r = divmod(i, len(rings))
    xx, yy = centers[center]
    return xx, yy, float(rings[r]), float(np.median(rim[i]))


def _highlighted(frame, x, y, radius):
    """A bright concentric focus halo is a selected state."""
    angle = np.linspace(0, 2 * np.pi, 64, endpoint=False, dtype=np.float32)
    scale = np.linspace(.90, 1.30, 13, dtype=np.float32)[:, None] * radius
    rim = cv2.remap(frame, x + scale * np.cos(angle), y + scale * np.sin(angle), cv2.INTER_LINEAR)
    b, g, r = rim.astype(np.int16).transpose(2, 0, 1)
    active = (g >= 118) & (((r >= 148) & (r - b >= 46)) | ((b >= 172) & (b - r >= 63)))
    return bool((active.mean(axis=1) >= .54).any())


def _circle_cells(gray, chroma, rail):
    height, width = gray.shape
    x0, y0 = rail + round(.029 * width), max(0, round(.006 * height))
    x1, y1 = round(.876 * width), round(.84 * height)
    field = gray[y0:y1, x0:x1]
    low, high = max(9, round(.046 * height)), round(.077 * height)
    blurred = cv2.GaussianBlur(field, (0, 0), max(.55, .028 * (low + high) / 2))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, round(1.24 * high),
                               param1=100, param2=max(15, round(.56 * high)),
                               minRadius=low, maxRadius=high)
    if circles is None or len(circles[0]) < 9:
        return []
    found = []
    for x, y, r in np.rint(circles[0]).astype(int):
        if (verified := _ring(chroma, x + x0, y + y0, r)) is not None:
            found.append(verified)
    return found


def _clusters(items, axis, radius):
    result = []
    for center in sorted(items, key=lambda item: item[axis]):
        if result and abs(center[axis] - result[-1][0][axis]) <= .27 * radius:
            result[-1].append(center)
        else:
            result.append([center])
    return [group for group in result if len(group) >= 3]


def _lattice(circles, width, height):
    if len(circles) < 9:
        return None
    radius = float(np.median([item[2] for item in circles]))
    hue = float(np.median([item[3] for item in circles]))
    peers = [item for item in circles if abs(item[2] - radius) <= .115 * radius and abs(item[3] - hue) <= 17]
    columns = _clusters(peers, 0, radius)
    if len(columns) != 3:
        return None
    centers = np.array([np.median([item[0] for item in group]) for group in columns])
    dx = np.diff(centers)
    if not .15 * width <= min(dx) <= max(dx) <= .28 * width or np.ptp(dx) > .10 * min(dx):
        return None
    rows = [row for row in _clusters(peers, 1, radius)
            if len(row) == 3 and all(any(abs(cell[0] - x) <= .27 * radius for cell in row) for x in centers)]
    if len(rows) < 3:
        return None
    yy = [np.median([item[1] for item in group]) for group in rows]
    dy = np.diff(yy)
    if not .12 * height <= min(dy) <= max(dy) <= .27 * height or np.ptp(dy) > .12 * min(dy):
        return None
    return rows, float(np.median(dx))


def _title_patch(gray, x, y, radius, pitch):
    a, b = round(x + 1.12 * radius), min(gray.shape[1], round(x + .80 * pitch))
    top, bottom = round(y - .74 * radius), round(y - .065 * radius)
    if not (0 <= a < b and top >= 0 and bottom > top):
        return None
    patch = gray[top:bottom, a:b]
    background, peak = (float(value) for value in np.percentile(patch, (48, 98)))
    if peak < 148 or peak - background < 80:
        return None
    mask = np.uint8(patch >= max(95, background + .37 * (peak - background)))
    u, v, w, h = cv2.boundingRect(mask)
    if not (0 <= u <= .32 * radius and .32 * radius <= h <= .63 * radius
            and 1.28 * h <= w <= .64 * pitch and .16 <= mask[v:v + h, u:u + w].mean() <= .69):
        return None
    pad = max(2, round(.10 * h))
    begin, end = max(0, u - pad), min(patch.shape[1], u + w + pad)
    return patch[:, begin:end], background, peak, (a + u, top + v, a + u + w, top + v + h)


def _price_rows(frame, x, y, radius, pitch, index):
    start, end = round(x + 1.12 * radius), min(frame.shape[1], round(x + .80 * pitch))
    lines = []
    has_price = False
    for low, high in ((.065, .72), (.71, 1.40)):
        top, bottom = round(y + low * radius), min(frame.shape[0], round(y + high * radius))
        patch = frame[top:bottom, start:end]
        if not patch.size:
            continue
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        b, g, r = patch.astype(np.int16).transpose(2, 0, 1)
        ink = np.uint8((gray >= 86) & (b - r >= 34))
        u, v, w, h = cv2.boundingRect(ink)
        if not (u <= .31 * radius and .32 * radius <= h <= .57 * radius and w >= .56 * radius):
            continue
        lines.append((start + u, top + v, start + u + w, top + v + h))
        if w > 3.75 * h or (g[v:v + h, u:u + w] - r[v:v + h, u:u + w])[ink[v:v + h, u:u + w] != 0].mean() >= 68:
            continue
        readings = []
        for level in (78, 96):
            digits = np.uint8((gray >= level) & (b - r >= 33) & (g - r <= 62))
            dx, dy, dw, dh = cv2.boundingRect(digits)
            if dh < .31 * radius:
                break
            glyphs = _shape_parts(digits[dy:dy + dh, dx:dx + dw], dh)
            if not 2 <= len(glyphs) <= 5:
                break
            guesses = [index.rank(glyph, alphabet="latin", limit=2) for *_, glyph in glyphs]
            if not all(len(options) == 2 and options[0]["text"] in "0123456789"
                       and options[0]["distance"] <= .182 and options[1]["distance"] - options[0]["distance"] >= .005
                       for options in guesses):
                break
            readings.append("".join(options[0]["text"] for options in guesses))
        if len(readings) == 2 and readings[0] == readings[1]:
            has_price = True
    return lines if has_price else []


@lru_cache(maxsize=80)
def _second_font_bank(paths, characters):
    return _rasterize_bank(paths, characters)


def _options(bank, mask):
    best = {}
    normalized = _normal(mask * 255)
    for letter, distance, _ in _shortlist(bank, *normalized):
        best[letter] = min(best.get(letter, 1.), distance)
    return sorted(best.items(), key=lambda item: (item[1], item[0]))


def _agreement(options, *, distance, margin, average):
    votes = {}
    for ranked in options:
        if len(ranked) < 2:
            continue
        letter, error = ranked[0]
        gap = ranked[1][1] - error
        if error <= distance and gap >= margin:
            votes.setdefault(letter, []).append(gap)
    winners = [letter for letter, values in votes.items() if len(values) >= 2 and np.mean(values) >= average]
    return winners[0] if len(winners) == 1 else None


def _literal(patch, background, peak, index):
    lines = []
    for level in (.161, .199, .267):
        cut = np.uint8(patch >= background + level * (peak - background))
        x, y, w, h = cv2.boundingRect(cut)
        parts = _cjk_parts(cut[y:y + h, x:x + w], h)
        if 2 <= len(parts) <= 8:
            lines.append(parts)
    if len(lines) < 2 or len({len(line) for line in lines}) != 1:
        return None
    root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    paths = tuple(path for name in ("Deng.ttf", "Dengb.ttf", "msyhl.ttc", "msjhbd.ttc")
                  if (path := root / name).is_file())
    result = []
    for items in zip(*lines, strict=True):
        masks = {}
        for *_, glyph in items:
            masks[(glyph.shape, glyph.tobytes())] = glyph
        if len(masks) < 2:
            return None
        def rank(mask):
            return [(row["text"], row["distance"]) for row in index.rank(mask, alphabet="cjk", limit=8)]

        raster = list(masks.values())
        initial = [rank(mask) for mask in raster[:2]]
        # A third cut cannot overturn two sufficiently strong agreeing reads.
        early = _agreement(initial, distance=.178, margin=.010, average=.014)
        if early is not None:
            result.append(early)
            continue
        initial.extend(rank(mask) for mask in raster[2:])
        letter = _agreement(initial, distance=.178, margin=.010, average=.012)
        if letter is None and paths:
            characters = "".join(sorted({letter for rows in initial for letter, _ in rows}))
            bank = _second_font_bank(paths, characters) if len(characters) > 1 else None
            if bank is not None:
                alternate = [_options(bank, mask) for mask in masks.values()]
                letter = _agreement(alternate, distance=.19, margin=.004, average=.009)
                if letter is not None and sum(letter in dict(rows[:3]) for rows in initial) < 2:
                    letter = None
        if letter is None:
            return None
        result.append(letter)
    return "".join(result)


def wide_panel_equipment_actions(frame, regions, index):
    """Emit complete named cards whose icon, title, and price share a grid cell."""
    if regions["layer"] != "wide_panel" or len(regions["regions"]) < 2 or index is None:
        return []
    left, top, right, bottom = regions["regions"][0]
    panel = frame[top:bottom, left:right]
    height, width = panel.shape[:2]
    if height < 220 or width < 1.42 * height:
        return []
    chroma = panel[..., 0].astype(np.int16) - panel[..., 2]
    rail = _rail_edge(chroma, width)
    if rail is None:
        return []
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    grid = _lattice(_circle_cells(gray, chroma, rail), width, height)
    if grid is None:
        return []
    rows, pitch = grid
    actions = []
    for row in rows:
        for x, y, radius, _ in sorted(row):
            if _highlighted(panel, x, y, radius):
                continue
            title = _title_patch(gray, x, y, radius, pitch)
            if title is None:
                continue
            lines = _price_rows(panel, x, y, radius, pitch, index)
            if not lines:
                continue
            patch, background, peak, title_box = title
            label = _literal(patch, background, peak, index)
            if label is None:
                continue
            pad = max(2, round(.13 * radius))
            x0, y0 = round(x - radius) - pad, round(y - radius) - pad
            x1 = max([item[2] for item in lines] + [title_box[2]]) + pad
            y1 = max([item[3] for item in lines] + [round(y + radius)]) + pad
            actions.append({"role": "game_equipment_item", "function": f"查看{label}", "preview_text": label,
                            "box": [x0 + left, y0 + top, x1 + left, y1 + top],
                            "click_point": [x + left, y + top]})
    counts = Counter(item["function"] for item in actions)
    return [item for item in actions if counts[item["function"]] == 1]
