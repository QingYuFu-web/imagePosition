"""Measure visible calendar days from printed numbers and a regular date grid."""

from calendar import monthrange
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import os
import string
import unicodedata

import cv2
import numpy as np

from inventory_calendar_lattice import _ENGLISH_TITLE, _MONTH_WORD
from screen_caption_transcript import read_caption
from screen_component_captions import _bands
from screen_font_glyphs import FontGlyphIndex


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class CalendarChoice:
    day: int
    month: int
    year: int | None
    text: str
    box: Box
    label_box: Box
    click_point: tuple[int, int]
    selected: bool
    region_id: int | None = None


@dataclass(frozen=True)
class _Week:
    start: int
    y: float
    columns: tuple[tuple[int, float], ...]
    em: float


@dataclass(frozen=True)
class _Grid:
    first: int
    x: float
    y: float
    dx: float
    dy: float
    em: float
    support: int


@lru_cache(maxsize=1)
def _font_indices():
    root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    names = ("arial.ttf", "arialbd.ttf", "segoeui.ttf", "segoeuib.ttf", "times.ttf",
             "georgia.ttf", "CascadiaMono.ttf", "cour.ttf", "consola.ttf", "simsun.ttc", "msyh.ttc")
    paths = tuple(root / name for name in names if (root / name).is_file())
    glyphs = FontGlyphIndex(fonts=paths or None, charset=string.digits)
    # Use the ordinary Latin outline families for month headings. The reader
    # checks a complete month word, separately from the numeric grid.
    title_paths = tuple(path for path in paths if path.name in {"arial.ttf", "segoeui.ttf", "segoeuib.ttf", "times.ttf", "georgia.ttf"})
    heading = FontGlyphIndex(fonts=title_paths or None, charset=string.digits + string.ascii_letters)
    return glyphs, heading


def _center(box):
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def _height(box):
    return box[3] - box[1]


def _components(mask):
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return tuple(tuple(map(int, row)) for row in stats[1:])


def _groups(boxes, tolerance):
    groups = []
    for box in sorted(boxes, key=lambda item: _center(item)[1]):
        y = _center(box)[1]
        if groups and y - groups[-1][0] <= tolerance:
            groups[-1][1].append(box)
        else:
            groups.append((y, [box]))
    return groups


def _digit(mask, index, cache):
    key = mask.shape, mask.tobytes()
    if key not in cache:
        ranks = index.rank(mask, alphabet="code", limit=2)
        cache[key] = (ranks[0]["text"] if len(ranks) == 2 and ranks[0]["distance"] <= .229
                      and ranks[1]["distance"] - ranks[0]["distance"] >= .016 else None)
    return cache[key]


def _weeks(mask, components, width, index, cache):
    pieces = [(x, y, x + w, y + h) for x, y, w, h, area in components
              if .014 * width <= h <= .049 * width and .20 * h <= w <= .95 * h
              and .16 * w * h <= area <= .91 * w * h and y >= .025 * mask.shape[0]]
    possible = []
    step = width / 7
    for _, boxes in _groups(pieces, .008 * width):
        columns = [[] for _ in range(7)]
        for box in boxes:
            x, _ = _center(box)
            col = int(x / step)
            if 0 <= col < 7 and abs(x / step - col - .5) <= .31:
                columns[col].append(box)
        eligible = {}
        for col, bits in enumerate(columns):
            bits.sort()
            if 1 <= len(bits) <= 2 and (len(bits) == 1 or bits[1][0] - bits[0][2] <= .025 * width):
                eligible[col] = bits
        if len(eligible) >= 4:
            possible.append(eligible)
    # Reading digits can only remove eligible cells. A grid needs at least
    # three week rows, sixteen cells and six distinct columns.
    if (len(possible) < 3 or sum(map(len, possible)) < 16
            or len({col for row in possible for col in row}) < 6):
        return []
    weeks = []
    for columns in possible:
        readings = []
        for col, bits in columns.items():
            digits = [_digit(mask[t:b, l:r], index, cache) for l, t, r, b in bits]
            if any(number is None for number in digits) or digits[0] == "0":
                continue
            day = int("".join(digits))
            if 1 <= day <= 31:
                readings.append((day - col, col, bits))
        starts = Counter(start for start, _, _ in readings).most_common(2)
        if not starts or starts[0][1] < 4 or (len(starts) > 1 and starts[0][1] == starts[1][1]):
            continue
        first = starts[0][0]
        active = [(col, bits) for start, col, bits in readings if start == first]
        ink = [box for _, boxes in active for box in boxes]
        weeks.append(_Week(first, float(np.median([_center(box)[1] for box in ink])),
                           tuple((col, (bits[0][0] + bits[-1][2]) / 2) for col, bits in active),
                           float(np.median([_height(box) for box in ink]))))
    return weeks


def _grids(weeks, width, height):
    found = []
    for a, b in combinations(weeks, 2):
        distance = b.start - a.start
        if not 7 <= distance <= 28 or distance % 7:
            continue
        dy = (b.y - a.y) / (distance // 7)
        if not .087 * width <= dy <= .48 * width:
            continue
        first = 1 - (1 - a.start) % 7
        origin = a.y - (a.start - first) / 7 * dy
        if not .052 * width <= origin < height - 2.2 * dy:
            continue
        peers = [week for week in weeks if first <= week.start <= 31 and (week.start - first) % 7 == 0
                 and abs(week.y - origin - (week.start - first) / 7 * dy) <= max(.012 * width, .33 * week.em)]
        if len({week.start for week in peers}) < 3 or len({week.start for week in peers}) != len(peers):
            continue
        observations = [column for week in peers for column in week.columns]
        if len(observations) < 16 or len({col for col, _ in observations}) < 6:
            continue
        dx, x = np.polyfit(*zip(*observations), 1)
        if not .11 * width <= dx <= .165 * width or any(abs(px - x - col * dx) > .12 * dx for col, px in observations):
            continue
        dy, y = np.polyfit([(week.start - first) / 7 for week in peers], [week.y for week in peers], 1)
        em = float(np.median([week.em for week in peers]))
        grid = _Grid(first, float(x), float(y), float(dx), float(dy), em, len(observations))
        if not any(old.first == first and abs(y - old.y) < .25 * em and abs(dy - old.dy) < .25 * em for old in found):
            found.append(grid)
    return sorted(found, key=lambda grid: grid.support, reverse=True)


def _heading(frame, scope, components, grid, index):
    left, top, right, _ = scope
    width = right - left
    near = [(x, y, x + w, y + h) for x, y, w, h, area in components
            if grid.y - .40 * width <= y <= grid.y - .081 * width
            and .014 * width <= h <= .073 * width and .10 * h <= w <= 1.62 * h
            and .10 * w * h <= area <= .97 * w * h]
    matches = set()
    for _, band in _groups(near, .022 * width):
        letters = []
        span = float(np.median([_height(box) for box in band]))
        for box in sorted(band):
            if letters and box[0] - letters[-1][-1][2] <= .96 * span:
                letters[-1].append(box)
            else:
                letters.append([box])
        for parts in letters:
            x0, x1 = parts[0][0], max(box[2] for box in parts)
            if len(parts) < 3 or not .072 * width <= x1 - x0 <= .54 * width:
                continue
            y0, y1 = min(box[1] for box in parts), max(box[3] for box in parts)
            if not .021 * width <= y1 - y0 <= .084 * width:
                continue
            xp, yp = max(4, round(.006 * width)), max(4, round(.008 * width))
            region = max(left, left + x0 - xp), max(top, top + y0 - yp), min(right, left + x1 + xp), top + y1 + yp
            result = read_caption(frame, region, index, cache={})
            if result is None:
                continue
            text = unicodedata.normalize("NFKC", result["text"]).casefold().strip()
            if (english := _ENGLISH_TITLE.fullmatch(text)) and english[1] in _MONTH_WORD:
                matches.add((_MONTH_WORD[english[1]], int(english[2]) if english[2] else None))
    return matches.pop() if len(matches) == 1 else None


def _filled_surface(patch, screen_paint):
    pixels = patch.astype(np.int16)
    paint = np.median(pixels, axis=(0, 1))
    if np.max(np.abs(paint - screen_paint)) < 43:
        return None
    fill = np.uint8(np.max(np.abs(pixels - paint), axis=2) <= 22)
    contours, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    middle = (float(fill.shape[1] / 2), float(fill.shape[0] / 2))
    for contour in contours:
        if cv2.contourArea(contour) >= .52 * fill.size and cv2.pointPolygonTest(contour, middle, False) >= 0:
            outline = cv2.drawContours(np.zeros_like(fill), [contour], 0, 1, cv2.FILLED)
            return cv2.erode(outline, np.ones((3, 3), np.uint8)) != 0
    return None


def _cell_number(patch, index, cache, screen_paint, em):
    selected = _filled_surface(patch, screen_paint)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.int16)
    delta = np.abs(gray - np.median(gray))
    if selected is not None:
        delta[~selected] = 0
    peak = float(np.percentile(delta, 98))
    if peak < 39:
        return None
    readings = {}
    for ratio in (.28, .35, .42, .55):
        ink = np.uint8(delta >= max(21, ratio * peak))
        spans = _bands(ink.any(axis=0))
        if not 1 <= len(spans) <= 2:
            continue
        values = []
        for start, end in spans:
            _, _, w, h = cv2.boundingRect(ink[:, start:end])
            if not .67 * em <= h <= 1.23 * em or w > .97 * h:
                break
            digit = _digit(ink[:, start:end], index, cache)
            if digit is None:
                break
            values.append(digit)
        if len(values) == len(spans) and values[0] != "0":
            readings.setdefault(int("".join(values)), []).append(cv2.boundingRect(ink))
    confirmed = [(number, boxes) for number, boxes in readings.items() if len(boxes) >= 2]
    if len(confirmed) != 1:
        return None
    number, boxes = confirmed[0]
    x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
    return number, (x, y, x + w, y + h), peak, selected is not None


def _read_grid(frame, scope, grid, month, year, index, cache, paint, layer):
    left, top, right, bottom = scope
    last = monthrange(year or 2000, month)[1]
    measured = []
    for row in range(6):
        cy = top + grid.y + row * grid.dy
        for col in range(7):
            expected = grid.first + row * 7 + col
            if not 1 <= expected <= last:
                continue
            cx = left + grid.x + col * grid.dx
            x0, x1 = max(left, round(cx - .25 * grid.dx)), min(right, round(cx + .25 * grid.dx))
            y0, y1 = max(top, round(cy - .71 * grid.em)), min(bottom, round(cy + .71 * grid.em))
            if x1 - x0 < 10 or y1 - y0 < 10:
                continue
            item = _cell_number(frame[y0:y1, x0:x1], index, cache, paint, grid.em)
            if item is None or item[0] != expected:
                continue
            day, (l, t, r, b), contrast, selected = item
            text = x0 + l, y0 + t, x0 + r, y0 + b
            point = tuple(round(v) for v in _center(text))
            # A page planner can put events lower in the grid; its clickable day
            # heading is the tight area around the number. The chooser is flat.
            reach = .49 * grid.dy if layer != "page" else .78 * grid.em
            box = (max(left, round(cx - .45 * grid.dx)), max(top, round(cy - reach)),
                   min(right, round(cx + .45 * grid.dx)), min(bottom, round(cy + reach)))
            measured.append((CalendarChoice(day, month, year, str(day), box, text, point, selected), contrast))
    ordinary = [contrast for choice, contrast in measured if not choice.selected]
    if len(ordinary) < 16:
        return []
    threshold = .63 * float(np.median(ordinary))
    selected_count = sum(choice.selected for choice, _ in measured)
    return [choice for choice, contrast in measured
            if (choice.selected and selected_count == 1) or (not choice.selected and contrast >= threshold)]


def calendar_choices(frame, regions):
    """Return observed dates of a verified month, including its visible selection."""
    height, width = frame.shape[:2]
    layer = regions["layer"]
    if layer == "page":
        scope = (0, 0, width, height)
    elif layer in {"bottom_sheet", "dialog"} and regions["regions"]:
        x0, y0, x1, y1 = map(int, regions["regions"][0])
        scope = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
    else:
        return []
    left, top, right, bottom = scope
    w, h = right - left, bottom - top
    if w < 240 or h < .66 * w:
        return []
    patch = frame[top:bottom, left:right]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    sampled = gray[4::7, 6::7]
    base = float(np.median(sampled))
    if (np.abs(sampled.astype(np.int16) - base) <= 19).mean() < .43:
        return []
    mask = np.uint8(np.abs(gray.astype(np.int16) - base) > 29)
    components = _components(mask)
    digits, headings = _font_indices()
    cache = {}
    weeks = _weeks(mask, components, w, digits, cache)
    grids = _grids(weeks, w, h)
    if not grids:
        return []
    plans = [(grid, parsed) for grid in grids[:4] if (parsed := _heading(frame, scope, components, grid, headings))]
    if not plans:
        return []
    grid, (month, year) = plans[0]
    if any(other.support >= .89 * grid.support and date != (month, year) for other, date in plans[1:]):
        return []
    paint = np.median(patch[4::7, 6::7], axis=(0, 1))
    return _read_grid(frame, scope, grid, month, year, digits, cache, paint, layer)
