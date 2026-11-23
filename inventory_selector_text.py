"""Restore visible spaces in text already grounded to a selector field."""

from collections import Counter
from itertools import accumulate, pairwise
from math import ceil, floor
import re
from statistics import median
import unicodedata

import cv2
import numpy as np

from inventory_file_tab_titles import _title_views
from inventory_toolbar_actions import _spans


RESTART_NOTE = re.compile(r"\s*\((?:[^()]{0,30}?)restart\s*(?:required|needed)\)\s*$", re.I)
_NUMBER = r"[0-9]+(?:[.,][0-9]+)*"
_DIMENSIONS = re.compile(rf"{_NUMBER}(?:[xX]{_NUMBER})+")
_NUMERIC_TOKENS = re.compile(rf"{_NUMBER}(?:[xX]{_NUMBER})*%?")


def _space_votes(profile, size, height, ink, known):
    # Work once per unchanged glyph mask rather than once per integer threshold.
    lower, upper = max(12, ceil(.22 * ink)), floor(.84 * ink) + 1
    transitions = sorted({lower, upper, *(int(value) + 1 for value in profile if lower <= value < upper - 1)})
    evidence = max(3, round(.024 * ink))
    minimum_blank = max(2, round(.12 * height))
    quiet = profile < lower
    readings = 0
    votes = Counter()
    for start, end in pairwise(transitions):
        runs = _spans(profile >= start)
        widths = [last - first for first, last in runs]
        if len(runs) != size or max(widths) > max(2.65 * median(widths), .95 * height):
            if readings >= evidence:
                break
            votes.clear()
            readings = 0
            continue
        gaps = [(previous[1], following[0]) for previous, following in pairwise(runs)]
        gap_threshold = max(.16 * height, 1.70 * median(last - first for first, last in gaps))
        blank = {}
        for index, (first, last) in enumerate(gaps, 1):
            if last - first >= gap_threshold:
                width = max((end - start for start, end in _spans(quiet[first:last])), default=0)
                if width >= minimum_blank:
                    blank[index] = width
        # Matching the glyph count alone can include stroke fragments and shift
        # the character positions. The existing OCR word breaks anchor the run.
        if not known.issubset(blank):
            if readings >= evidence:
                break
            votes.clear()
            readings = 0
            continue
        widest = 1.65 * median(blank[index] for index in known) if known else float("inf")
        weight = min(end - start, 2 * evidence - readings)
        for index, width in blank.items():
            if index not in known and width <= widest:
                votes[index] += weight
        readings += weight
        if readings >= 2 * evidence:
            break
    return {index for index, count in votes.items() if count >= evidence and count * 2 > readings}


def restore_select_text(gray: np.ndarray, box: tuple[int, int, int, int], text: str) -> str:
    """Add pixel-supported spaces inside the line's integer xyxy box."""
    words = text.split()
    compact = "".join(words)
    if not compact.isascii() or not 6 <= len(compact) <= 48 or not any(char.isalpha() for char in compact):
        return text
    left, top, right, bottom = box
    roi = gray[top:bottom, left:right].astype(np.int16)
    if not roi.size:
        return text
    contrast = np.abs(roi - np.median(roi))
    ink = float(np.percentile(contrast, 96))
    if ink < 45:
        return text
    known = set(accumulate(map(len, words[:-1])))
    cuts = _space_votes(contrast.max(axis=0), len(compact), bottom - top, ink, known)
    # A neighboring icon in the crop can masquerade as a detached first or last letter.
    if compact[0].isalpha():
        cuts.discard(1)
    if compact[-1].isalpha():
        cuts.discard(len(compact) - 1)
    if not cuts:
        return text

    # Uneven digit spacing inside amounts or dimensions is not a word boundary.
    for match in _NUMERIC_TOKENS.finditer(compact):
        cuts.difference_update(range(match.start() + 1, match.end()))

    # Preserve the recognizer's spaces and use only plausible punctuation sides.
    at = 0
    characters = []
    for letter in text:
        if letter.isspace():
            cuts.discard(at)
        else:
            if at in cuts and letter not in ",.;:!?)]}_-" and compact[at - 1] not in "([{_-":
                characters.append(" ")
            at += 1
        characters.append(letter)
    return "".join(characters)


def recapture_dimension_text(frame, field, arrow, box, text, line_reader):
    """Restore a compressed dimension label only when two larger crops agree."""
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    normalized = unicodedata.normalize("NFKC", text)
    dimension = _DIMENSIONS.search(normalized)
    if (recognize is None or not normalized.isascii() or dimension is None
            or (text == normalized and len(text.split()) > 1)):
        return text
    surrounding = normalized[:dimension.start()] + normalized[dimension.end():]
    if not any(letter.isalpha() for letter in surrounding):
        return text
    try:
        _, contrast = _title_views(frame, box, field, arrow.left)
        crops = [cv2.resize(contrast, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                 for scale in (2, 3)]
        readings = recognize(crops)
    except (RuntimeError, ValueError, cv2.error):
        return text
    if len(readings) == 2:
        first, second = readings
        if (first["text"] == second["text"] and min(first["score"], second["score"]) >= 92
                and "".join(unicodedata.normalize("NFKC", first["text"]).split()) == "".join(normalized.split())):
            return first["text"]
    return text
