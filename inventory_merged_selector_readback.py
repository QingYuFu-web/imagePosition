"""Read merged selector text from both sides of an independently visible rim."""

import unicodedata

import cv2
import numpy as np

from inventory_file_tab_titles import _title_views
from inventory_list_rows import _Box
from inventory_selector_side_titles import _stepper_arrow


def _rim(gray, field, arrow, strict_fields):
    h = field.height
    left_edges = {candidate.left for candidate, mark in strict_fields.items() if mark == arrow
                  and .78 <= candidate.height / h <= 1.27
                  and abs(candidate.right - field.right) <= .22 * h}
    radius = max(2, round(.13 * h))
    probes = {x for left in left_edges for x in range(left - radius, left + radius + 1)
              if 3 <= x < gray.shape[1] - 3 and abs(x - field.left) <= .27 * h}
    center = round(field.center[1])
    half = max(3, round(.34 * h))
    strip = gray[center - half:center + half + 1].astype(np.int16)
    supported = []
    for x in sorted(probes):
        column = strip[:, x]
        left = np.median(strip[:, x - 3:x], axis=1)
        right = np.median(strip[:, x + 1:x + 4], axis=1)
        bright = column - np.maximum(left, right)
        dark = np.minimum(left, right) - column
        if max((bright >= 12).mean(), (dark >= 12).mean()) >= .74:
            supported.append(x)
    return supported[0] if len(supported) == 1 else None


def _ink(gray, box):
    pixels = gray[box.top:box.bottom, box.left:box.right].astype(np.int16)
    return np.abs(pixels - np.median(pixels)) >= 42


def _split_crops(gray, field, arrow, title, value, strict_fields):
    rim = _rim(gray, field, arrow, strict_fields)
    if rim is None:
        return None
    margin = max(2, round(.15 * field.height))
    left = max(0, title.left - 2)
    right = rim - margin
    # Allow the width of an antialiased rim to vary by one pixel at a nearby measured contour.
    if title.right < right and field.left < rim - 2:
        right += 1
    value = _Box(max(rim + 1, value.left), value.top, value.right, value.bottom)
    if right - left < field.height or value.width < .47 * field.height:
        return None
    title_lane = _Box(left, title.top, right, title.bottom)
    title_ink = _ink(gray, title_lane)
    value_ink = _ink(gray, value)
    rows = np.flatnonzero(np.count_nonzero(title_ink, axis=1) >= 2)
    if (not rows.size or rows[-1] - rows[0] + 1 < .28 * field.height
            or np.count_nonzero(value_ink) < 8 or title_ink[:, -1].sum() > 1 or value_ink[:, 0].sum() > 1):
        return None
    pad = max(2, round(.08 * field.height))
    title = _Box(left, max(title.top, title.top + int(rows[0]) - pad), right,
                 min(title.bottom, title.top + int(rows[-1]) + pad + 1))
    return title, value


def _key(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def _matches_combined(title, value, combined, confidence):
    title, value, raw = _key(title), _key(value), _key(combined)
    if not title.endswith(":"):
        return False
    if title + value == raw:
        return True
    before, separator, after = raw.partition(":")
    if (confidence < 94 or not separator or raw.count(":") != 1 or title.count(":") != 1
            or value != after or len(before) < 12 or len(title) != len(before) + 2):
        return False
    corrected = title[:-1]
    return any(corrected[index].isalpha() and corrected[:index] + corrected[index + 1:] == before
               for index in range(2, len(corrected) - 2))


def recapture_merged_selectors(frame, gray, pending, strict_fields, rectangles, line_reader):
    """Confirm both text fragments from one strict field after its usual reads failed."""
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    if not callable(recognize):
        return {}
    requests, crops = [], []
    for field, (combined, title, value) in pending:
        arrow = strict_fields.get(field)
        if arrow is None or ":" not in _key(combined):
            continue
        pair = _split_crops(gray, field, arrow, title, value, strict_fields)
        if pair is None or _stepper_arrow(gray, field, arrow, rectangles):
            continue
        title, value = pair
        color, contrast = _title_views(frame, title, title, title.right)
        crops.extend(cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                     for patch, scale in ((color, 2), (contrast, 3)))
        crops.extend(_title_views(frame, value, value, value.right))
        requests.append((field, combined, title, value))
    if not requests:
        return {}
    try:
        readings = recognize(crops)
    except (RuntimeError, ValueError, cv2.error):
        return {}
    if len(readings) != len(crops):
        return {}
    accepted = {}
    for index, (field, combined, title, value) in enumerate(requests):
        first_title, second_title, first_value, second_value = readings[4 * index:4 * index + 4]
        name, shown = first_title["text"], first_value["text"]
        title_score = min(first_title["score"], second_title["score"])
        value_score = min(first_value["score"], second_value["score"])
        confidence = min(title_score, value_score)
        if (name and shown and name == second_title["text"] and shown == second_value["text"]
                and confidence >= 92 and _matches_combined(name, shown, combined, confidence)):
            accepted[field] = ((title, name, title_score), (value, shown, value_score))
    return accepted
