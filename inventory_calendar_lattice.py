"""Recover visible civil dates and measured cells from a monthly calendar."""

from __future__ import annotations

from calendar import monthrange
from collections import Counter
from dataclasses import dataclass
from datetime import date
from itertools import combinations
import re
import unicodedata

import cv2
import numpy as np


_MONTHS = "january february march april may june july august september october november december".split()
_MONTH_WORD = {word: i for i, word in enumerate(_MONTHS, 1)}
_CHINESE_MONTH = {word: i for i, word in enumerate("一 二 三 四 五 六 七 八 九 十 十一 十二".split(), 1)}
_ENGLISH_TITLE = re.compile(r"([a-z]+)(?:\s*((?:19|20)\d{2}))?")
_CHINESE_TITLE = re.compile(r"(?:((?:19|20)\d{2})\s*年\s*)?([0-9一二三四五六七八九十]+)\s*月")
_DAY = re.compile(r"(?:[1-9]|[12][0-9]|3[01])")
_WEEKDAY = {name: index for index, name in enumerate("monday tuesday wednesday thursday friday saturday sunday".split())}
_WEEKDAY.update({name[:3]: index for name, index in list(_WEEKDAY.items())})
_WEEKDAY.update({f"{prefix}{label}": index for prefix in ("星期", "周", "週")
                for index, label in enumerate("一二三四五六日")})


@dataclass(frozen=True)
class DayCell:
    month: int
    day: int
    year: int | None
    cell_box: tuple[int, int, int, int]
    numeral_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class _Numeral:
    day: int
    box: tuple[int, int, int, int]

    @property
    def center(self):
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2, (y0 + y1) / 2

    @property
    def height(self):
        return self.box[3] - self.box[1]


def _text(node):
    return unicodedata.normalize("NFKC", str(node.get("content") or "")).casefold().strip()


def _month(nodes, heading):
    for node in nodes:
        if node.get("kind") != "text" or tuple(map(round, node["box"])) != tuple(heading):
            continue
        label = _text(node)
        if (match := _ENGLISH_TITLE.fullmatch(label)) and match[1] in _MONTH_WORD:
            return _MONTH_WORD[match[1]], int(match[2]) if match[2] else None
        if match := _CHINESE_TITLE.fullmatch(label):
            number = int(match[2]) if match[2].isdecimal() else _CHINESE_MONTH.get(match[2])
            if number is not None and 1 <= number <= 12:
                return number, int(match[1]) if match[1] else None
    return None


def _numerals(nodes, heading, width, height):
    for node in nodes:
        if node.get("kind") != "text" or (node.get("score") or 0) < 89:
            continue
        label = _text(node)
        if not _DAY.fullmatch(label):
            continue
        x0, y0, x1, y1 = map(round, node["box"])
        if (0 <= x0 < x1 <= width and heading[3] < y0 < y1 < .94 * height
                and .008 * width <= y1 - y0 <= .083 * width and x1 - x0 <= .10 * width):
            yield _Numeral(int(label), (x0, y0, x1, y1))


def _tentative_column(x, width):
    position = x / (width / 7)
    column = int(position)
    return column if 0 <= column < 7 and abs(position - column - .5) < .34 else None


def _week_evidence(numerals, width, tolerance):
    rows = []
    for numeral in sorted(numerals, key=lambda it: it.center[1]):
        if rows and numeral.center[1] - rows[-1][0].center[1] <= tolerance:
            rows[-1].append(numeral)
        else:
            rows.append([numeral])
    evidence = []
    for row in rows:
        pairs = {(column, number.day) for number in row
                 if (column := _tentative_column(number.center[0], width)) is not None}
        starts = Counter(day - column for column, day in pairs).most_common(2)
        if not starts or starts[0][1] < 4 or (len(starts) > 1 and starts[1][1] == starts[0][1]):
            continue
        start, support = starts[0]
        evidence.append((start, float(np.median([number.center[1] for number in row])), support))
    return evidence


def _rows(evidence, heading, width, tolerance):
    models = []
    for (left, y0, _), (right, y1, _) in combinations(evidence, 2):
        if not 7 <= right - left <= 28 or (right - left) % 7:
            continue
        spacing = 7 * (y1 - y0) / (right - left)
        if not .13 * width <= spacing <= .46 * width:
            continue
        first = 1 - (1 - left) % 7
        origin = y0 - (left - first) // 7 * spacing
        if not .042 * width <= origin - heading[3] <= .28 * width:
            continue
        matching = [(v, y, count) for v, y, count in evidence if (v - first) % 7 == 0 and first <= v <= 31
                    and abs(y - (origin + (v - first) / 7 * spacing)) <= tolerance]
        if len(matching) >= 3:
            models.append((sum(count for _, _, count in matching), first, matching))
    if not models:
        return None
    models.sort(key=lambda model: model[0], reverse=True)
    count, first, supporting = models[0]
    if any(other_count == count and other_first != first for other_count, other_first, _ in models[1:]):
        return None
    index = [(number - first) / 7 for number, _, _ in supporting]
    spacing, origin = np.polyfit(index, [y for _, y, _ in supporting], 1)
    return first, float(origin), float(spacing)


def _columns(numerals, width, first, origin, spacing, tolerance):
    observations = []
    for numeral in numerals:
        x, y = numeral.center
        column = _tentative_column(x, width)
        row = round((y - origin) / spacing)
        if column is not None and 0 <= row <= 5 and abs(y - origin - row * spacing) <= tolerance:
            if numeral.day == first + 7 * row + column:
                observations.append((column, x))
    if len({column for column, _ in observations}) < 6:
        return None
    stride, left_center = np.polyfit(*zip(*observations), 1)
    if max(abs(x - left_center - column * stride) for column, x in observations) > .12 * stride:
        return None
    return float(left_center), float(stride)


def _day_at(numbers, x, y, width, tolerance, expected):
    matches = [num for num in numbers if abs(num.center[0] - x) <= .22 * width and abs(num.center[1] - y) <= tolerance]
    if len(matches) == 1 and matches[0].day == expected:
        return matches[0]
    return None


def _first_weekday(nodes, heading, first_y, x_origin, stride, label_height):
    labels = []
    for node in nodes:
        if node.get("kind") != "text" or (node.get("score") or 0) < 92:
            continue
        weekday = _WEEKDAY.get(_text(node).rstrip("."))
        x0, y0, x1, y1 = node["box"]
        if weekday is None or not heading[3] < y0 < y1 < first_y - .32 * label_height:
            continue
        x, y = (x0 + x1) / 2, (y0 + y1) / 2
        column = round((x - x_origin) / stride)
        if 0 <= column < 7 and abs(x - x_origin - column * stride) <= .20 * stride:
            labels.append((column, weekday, y))
    for _, _, y in labels:
        row = {(col, weekday) for col, weekday, center in labels if abs(center - y) <= .45 * label_height}
        starts = {(weekday - col) % 7 for col, weekday in row}
        if len(row) >= 4 and len(starts) == 1:
            return starts.pop()
    return None


def _reread(frame, heading, refiner, x, y, stride, numeral_height):
    if refiner is None:
        return []
    h, w = frame.shape[:2]
    region = (max(0, round(x - .25 * stride)), max(0, round(y - .74 * numeral_height)),
              min(w, round(x + .25 * stride)), min(h, round(y + .74 * numeral_height)))
    left, top, right, bottom = region
    crop = frame[top:bottom, left:right]
    if min(crop.shape[:2]) < 10:
        return []
    enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    try:
        nodes = refiner.detect(enlarged)
    except RuntimeError:
        return []
    projected = []
    for node in nodes:
        x0, y0, x1, y1 = node["box"]
        projected.append({**node, "box": (left + x0 / 3, top + y0 / 3, left + x1 / 3, top + y1 / 3)})
    return list(_numerals(projected, heading, w, h))


def find_calendar_cells(frame, nodes, *, month_heading, ocr_refiner=None):
    """Ground day labels only when several OCR-verified weeks imply the same grid."""
    if month_heading is None:
        return []
    parsed = _month(nodes, month_heading)
    if parsed is None:
        return []
    month, year = parsed
    h, w = frame.shape[:2]
    numbers = list(_numerals(nodes, month_heading, w, h))
    if len(numbers) < 15:
        return []
    numeral_height = float(np.median([num.height for num in numbers]))
    tolerance = .62 * numeral_height
    row_model = _rows(_week_evidence(numbers, w, tolerance), month_heading, w, tolerance)
    if row_model is None:
        return []
    first, origin, spacing = row_model
    columns = _columns(numbers, w, first, origin, spacing, tolerance)
    if columns is None:
        return []
    x_origin, stride = columns
    if year is not None:
        first_weekday = _first_weekday(nodes, month_heading, origin, x_origin, stride, numeral_height)
        if first_weekday is not None and (first_weekday + 1 - first) % 7 != date(year, month, 1).weekday():
            return []
    last = monthrange(year or 2000, month)[1]
    days = []
    rereads = 0
    for row in range(6):
        y = origin + row * spacing
        if y + .5 * numeral_height >= .94 * h:
            break
        for column in range(7):
            x = x_origin + column * stride
            number = first + row * 7 + column
            if not 1 <= number <= last:
                continue
            match = _day_at(numbers, x, y, stride, tolerance, number)
            if match is None and rereads < 8 and ocr_refiner is not None:
                rereads += 1
                match = _day_at(_reread(frame, month_heading, ocr_refiner, x, y, stride, numeral_height),
                                x, y, stride, tolerance, number)
            if match:
                cell = (max(0, round(x - .5 * stride)), max(0, round(y - .85 * numeral_height)),
                        min(w, round(x + .5 * stride)), min(h, round(y + spacing - .85 * numeral_height)))
                days.append(DayCell(month, number, year, cell, match.box))
    return days
