"""Read an independently visible caption directly below a compact video glyph."""

from dataclasses import dataclass
import re
import unicodedata

import cv2
import numpy as np

from inventory_list_rows import _Box
from inventory_toolbar_actions import _spans


_COUNT = re.compile(r"\d+(?:[.,]\d+)*(?:[kKmMbB万亿])?\+?")
_STROKE_FILTER = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class CaptionReading:
    text: str
    box: tuple[int, int, int, int]
    confidence: float
    is_count: bool


def _box(value):
    return value if isinstance(value, _Box) else _Box(*map(round, value))


def _caption_box(frame, glyph, following):
    height, width = frame.shape[:2]
    h, w = glyph.height, glyph.width
    if (h < 9 or w < 9 or not 0 <= glyph.left < glyph.right <= width
            or not 0 <= glyph.top < glyph.bottom <= height):
        return None
    x = round(glyph.center[0])
    reach = max(10, round(.72 * w))
    left, right = max(0, x - reach), min(width, x + reach)
    top = glyph.bottom + max(1, round(.045 * h))
    bottom = min(height, glyph.bottom + round(.94 * h))
    if following is not None:
        bottom = min(bottom, following.top - max(1, round(.10 * h)))
    if right - left < .95 * w or bottom - top < 8:
        return None
    chip = frame[top:bottom, left:right]
    ground = chip.min(axis=2)
    background = cv2.morphologyEx(ground, cv2.MORPH_OPEN, _STROKE_FILTER)
    ink = cv2.subtract(ground, background) >= 24
    rows = _spans(ink.sum(axis=1) >= 2)
    bands = [(int(first), int(last)) for first, last in rows if last - first >= max(6, .16 * h)]
    if len(bands) != 1:
        return None
    first, last = bands[0]
    if first <= 0 or last >= chip.shape[0] or last - first > .69 * h:
        return None
    cols = np.flatnonzero(ink[first:last].sum(axis=0) >= 2)
    if not cols.size:
        return None
    start, stop = int(cols[0]), int(cols[-1]) + 1
    if (not .34 * w <= stop - start <= 1.35 * w or start <= 0 or stop >= right - left
            or abs(left + (start + stop) / 2 - glyph.center[0]) > .22 * w):
        return None
    horizontal = max(2, round(.11 * w))
    return _Box(max(left, left + start - horizontal), max(top, top + first - round(.16 * h)),
                min(right, left + stop + horizontal), min(bottom, top + last + round(.28 * h)))


def _normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def read_compact_video_caption(frame, glyph_box, line_reader, *, following_box=None):
    """Return corroborated visible text or a count, without assigning a click role."""
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    if not callable(recognize) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        return None
    glyph = _box(glyph_box)
    following = _box(following_box) if following_box is not None else None
    box = _caption_box(frame, glyph, following)
    if box is None:
        return None
    color = frame[box.top:box.bottom, box.left:box.right]
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.int16)
    contrast = np.clip(2.2 * (gray - np.median(gray)), 0, 255).astype(np.uint8)
    try:
        readings = recognize((color, cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)))
    except (RuntimeError, ValueError, cv2.error):
        return None
    if len(readings) != 2:
        return None
    first, second = readings
    name = _normalized(first["text"])
    confidence = min(first["score"], second["score"])
    is_count = _COUNT.fullmatch(name) is not None
    if (confidence < 91 or name != _normalized(second["text"]) or len(name) > 24
            or (not is_count and sum(char.isalpha() for char in name) < 2)):
        return None
    return CaptionReading(name, box.xyxy(), confidence, is_count)
