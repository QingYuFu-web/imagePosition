"""Read unresolved selector text from independently marked neighbors on the same row."""

from collections import Counter
from itertools import combinations
import unicodedata

import cv2
import numpy as np

from inventory_file_tab_titles import _title_views
from inventory_list_rows import _Box
from inventory_selector_side_titles import _stepper_arrow


def _paired(left, right, first, second):
    height = min(left.height, right.height)
    return (.80 <= left.height / right.height <= 1.25
            and .57 <= left.width / right.width <= 1.75
            and abs(left.top - right.top) <= .16 * height
            and abs(left.bottom - right.bottom) <= .16 * height
            and .90 * height <= right.left - left.right <= 4.25 * height
            and abs(first.center[1] - second.center[1]) <= .14 * height)


def _ink_bounds(gray, lane, height):
    if lane.width < .50 * height or lane.height < .50 * height:
        return None
    pixels = gray[lane.top:lane.bottom, lane.left:lane.right].astype(np.int16)
    foreground = np.abs(pixels - np.median(pixels)) >= 42
    columns = np.flatnonzero(foreground.sum(axis=0) >= 2)
    if not columns.size or columns[-1] - columns[0] + 1 < .40 * height:
        return None
    # A clipped line or vertical border cannot establish a complete title or value.
    if columns[0] == 0 or columns[-1] == lane.width - 1 or (foreground.sum(axis=0) >= .63 * lane.height).any():
        return None
    rows = np.flatnonzero(foreground[:, columns[0]:columns[-1] + 1].sum(axis=1) >= 2)
    if not rows.size or not .23 * height <= rows[-1] - rows[0] + 1 <= .72 * height:
        return None
    return _Box(lane.left + int(columns[0]), lane.top + int(rows[0]),
                lane.left + int(columns[-1]) + 1, lane.top + int(rows[-1]) + 1)


def _pad(ink, lane, horizontal, above, below):
    return _Box(max(lane.left, ink.left - horizontal), max(lane.top, ink.top - above),
                min(lane.right, ink.right + horizontal), min(lane.bottom, ink.bottom + below))


def _hint(box, lines):
    hints = []
    for line in lines:
        original = line.box
        width = min(box.right, original.right) - max(box.left, original.left)
        height = min(box.bottom, original.bottom) - max(box.top, original.top)
        if (line.score >= 90 and width >= .71 * max(box.width, original.width)
                and height >= .55 * max(box.height, original.height)):
            hints.append(line)
    return hints


def _key(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def recapture_selector_pair_text(frame, evidence, occupied, line_reader):
    """Verify bounded title/value crops for unresolved cells of an unambiguous field pair."""
    from inventory_value_select import _overlap

    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    if not callable(recognize):
        return {}
    cells, seen = [], set()
    for field, mark in evidence.strict_closed.items():
        if (evidence.closed.get(field) != mark or field.width < 1.85 * field.height
                or mark in seen or any(_overlap(field, other) for other, _ in cells)):
            continue
        cells.append((field, mark))
        seen.add(mark)
    pairs = [(left, right, first, second) for (left, first), (right, second) in
             combinations(sorted(cells, key=lambda entry: entry[0].left), 2)
             if _paired(left, right, first, second)]
    counts = Counter(field for pair in pairs for field in pair[:2])
    requests, crops = [], []
    gray = evidence.gray
    for left, right, first, second in pairs:
        if counts[left] != 1 or counts[right] != 1:
            continue
        title_lefts = (max(0, left.left - round(2.3 * left.height)), left.right + 1)
        for field, mark, start in zip((left, right), (first, second), title_lefts, strict=True):
            if any(_overlap(field, known) for known in occupied):
                continue
            h = field.height
            inset = max(2, round(.08 * h))
            top, bottom = field.top + inset, field.bottom - inset
            title_right = field.left if field == left else field.left - 1
            title_lane = _Box(start, top, title_right, bottom)
            value_lane = _Box(field.left + inset, top, mark.left - inset, bottom)
            title_ink = _ink_bounds(gray, title_lane, h)
            value_ink = _ink_bounds(gray, value_lane, h)
            if title_ink is None or value_ink is None or _stepper_arrow(gray, field, mark, evidence.rectangles):
                continue
            pad = max(2, round(.13 * h))
            title = _pad(title_ink, title_lane, pad, round(.22 * h), pad)
            value = _pad(value_ink, value_lane, pad, round(.22 * h), pad)
            first_hints, second_hints = _hint(title, evidence.lines), _hint(value, evidence.lines)
            if len(first_hints) > 1 or len(second_hints) > 1 or not (first_hints or second_hints):
                continue
            # Read with OCR padding, but measure enabled text on the observed ink and its near flank.
            value_witness = _pad(value_ink, value_lane, 1, pad, 1)
            requests.append((field, title, value_witness, first_hints, second_hints))
            crops.extend((*_title_views(frame, title, title, title.right), *_title_views(frame, value, value, value.right)))
    if not requests:
        return {}
    try:
        readings = recognize(crops)
    except (RuntimeError, ValueError, cv2.error):
        return {}
    if len(readings) != len(crops):
        return {}
    accepted = {}
    for index, (field, title, value, first_hints, second_hints) in enumerate(requests):
        first, first_alt, second, second_alt = readings[4 * index:4 * index + 4]
        name, shown = first["text"], second["text"]
        title_score = min(first["score"], first_alt["score"])
        value_score = min(second["score"], second_alt["score"])
        if (name != first_alt["text"] or shown != second_alt["text"] or min(title_score, value_score) < 90
                or max(first["score"], first_alt["score"]) < 94 or max(second["score"], second_alt["score"]) < 94
                or not _key(name).endswith(":") or sum(char.isalpha() for char in name) < 2
                or not any(char.isalnum() for char in shown) or _key(name).rstrip(":") == _key(shown)
                or any(_key(line.text) != _key(name) for line in first_hints)
                or any(_key(line.text) != _key(shown) for line in second_hints)):
            continue
        accepted[field] = ((title, name, title_score), (value, shown, value_score))
    return accepted
