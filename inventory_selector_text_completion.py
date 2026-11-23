"""Complete selector text only within already verified control boundaries."""

from collections import Counter
from itertools import pairwise
import re
from statistics import median
import unicodedata

import cv2
import numpy as np

from inventory_list_rows import _Box
from inventory_toolbar_actions import _spans


_COMPACT_CAPTION = re.compile(r"[A-Za-z]{12,48}[:：]?")
_PARTIAL_VALUE = re.compile(r"[A-Za-z]{2,}(?: [A-Za-z]{2,}){1,3}")


def _deviation(patch):
    ground = np.median(patch)
    return patch - ground if ground <= 127 else ground - patch


def _word_bands(gray, box):
    patch = gray[box.top:box.bottom, box.left:box.right].astype(np.int16)
    if patch.shape[0] < 8 or patch.shape[1] < 18:
        return ()
    ink = _deviation(patch)
    strength = float(np.percentile(ink, 97))
    if strength < 60:
        return ()
    profile = ink.max(axis=0)
    minimum_gap = max(2, round(.13 * box.height))
    bands = []
    for first, last in _spans(profile >= max(24, .26 * strength)):
        if bands and first - bands[-1][1] < minimum_gap:
            bands[-1] = (bands[-1][0], last)
        else:
            bands.append((first, last))
    if not 2 <= len(bands) <= 4:
        return ()
    for (_, end), (start, _) in pairwise(bands):
        if start - end > .48 * box.height or np.min(profile[end:start]) > .22 * strength:
            return ()
    if any(last - first < .72 * box.height for first, last in bands):
        return ()
    return tuple((int(first + box.left), int(last + box.left)) for first, last in bands)


def _word_boxes(lane, bands, *, padding):
    left_pad, right_pad = padding
    boxes = []
    for i, (left, right) in enumerate(bands):
        previous = bands[i - 1][1] if i else lane.left
        following = bands[i + 1][0] if i + 1 < len(bands) else lane.right
        boxes.append(_Box(max(previous, left - left_pad), lane.top,
                          min(following, right + right_pad), lane.bottom))
    return tuple(boxes)


def _caption_crops(gray, field, line):
    if (line is None or line.score < 89 or _COMPACT_CAPTION.fullmatch(line.text) is None
            or not 0 <= field.left - line.box.right <= .43 * field.height
            or abs(line.box.center[1] - field.center[1]) > .22 * field.height):
        return ()
    box = line.box
    lane = _Box(max(0, box.left - 2), max(0, box.top - 1),
                min(field.left, box.right + 1), min(gray.shape[0], box.bottom + 1))
    bands = _word_bands(gray, lane)
    if len(bands) < 3:
        return ()
    return _word_boxes(lane, bands, padding=(1, 2))


def _value_crops(gray, field, arrow, line):
    if line is None or line.score < 89 or _PARTIAL_VALUE.fullmatch(line.text) is None:
        return ()
    box = line.box
    if (not field.left - 1 <= box.left <= field.left + .54 * field.height
            or abs(box.center[1] - field.center[1]) > .22 * field.height):
        return ()
    margin = max(2, round(.14 * field.height))
    lane = _Box(max(field.left + 2, box.left - 2), max(field.top + margin, box.top),
                arrow.left - margin, min(field.bottom - margin, box.bottom))
    bands = _word_bands(gray, lane)
    if len(bands) != len(line.text.split()):
        return ()
    # The OCR box must stop inside an observed final word, with additional ink
    # remaining in the same word before the verified end of the control.
    left, right = bands[-1]
    if not left + 2 <= box.right <= right - max(4, round(.30 * lane.height)):
        return ()
    return _word_boxes(lane, bands, padding=(2, 3))


def _views(frame, box):
    original = frame[box.top:box.bottom, box.left:box.right]
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.int16)
    ink = _deviation(gray)
    light = cv2.cvtColor((255 - np.clip(2.3 * ink, 0, 255)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    dark = cv2.cvtColor(np.clip(2.1 * ink, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return tuple(cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                 for crop, scale in ((original, 2), (original, 3), (light, 3), (dark, 3)))


def _confirmed_word(rows):
    votes = Counter()
    for row in rows:
        text = unicodedata.normalize("NFKC", row["text"]).strip()
        if row["score"] >= 90 and text.isascii() and text.isalpha() and len(text) >= 2:
            votes[text] += 1
    confirmed = [text for text, count in votes.items() if count >= 2]
    return confirmed[0] if len(confirmed) == 1 else None


def _complete_caption(raw, words, boxes):
    prefix = "".join(words)
    if not raw.startswith(prefix) or len(raw.rstrip(":：")) - len(prefix) < 3:
        return None
    remainder = raw[len(prefix):]
    expected_width = median(box.width / len(word) for box, word in zip(boxes[:-1], words, strict=True))
    actual_width = boxes[-1].width / len(remainder.rstrip(":："))
    if not .63 * expected_width <= actual_width <= 1.54 * expected_width:
        return None
    return " ".join((*words, remainder))


def _complete_value(raw, words):
    original = raw.split()
    if list(words[:-1]) != original[:-1] or not words[-1].startswith(original[-1]):
        return None
    if not 2 <= len(words[-1]) - len(original[-1]) <= len(original[-1]) + 3:
        return None
    return " ".join(words)


def complete_selector_text(frame, gray, requests, line_reader):
    """Return changed (title, value) for strict field, arrow, and OCR line tuples."""
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    if not callable(recognize):
        return {}
    plans, crops = [], []
    for field, arrow, title, value in requests:
        if not (field.left < arrow.left < arrow.right <= field.right
                and field.top <= arrow.top < arrow.bottom <= field.bottom):
            continue
        for part, line, boxes in (
            (0, title, _caption_crops(gray, field, title)),
            (1, value, _value_crops(gray, field, arrow, value)),
        ):
            if not boxes:
                continue
            reads = boxes[:-1] if part == 0 else boxes
            start = len(crops)
            crops.extend(crop for box in reads for crop in _views(frame, box))
            plans.append((field, part, line.text, boxes, start, len(crops)))
    if not crops:
        return {}
    try:
        readings = recognize(crops)
    except (RuntimeError, ValueError, cv2.error):
        return {}
    if len(readings) != len(crops):
        return {}
    completed = {}
    for field, part, raw, boxes, start, stop in plans:
        words = [_confirmed_word(readings[index:index + 4]) for index in range(start, stop, 4)]
        if None in words:
            continue
        text = _complete_caption(raw, words, boxes) if part == 0 else _complete_value(raw, words)
        if text is not None:
            title, value = completed.get(field, (None, None))
            completed[field] = (text, value) if part == 0 else (title, text)
    return completed
