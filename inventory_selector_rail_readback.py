"""Read the selected value inside an independently reconstructed field."""

import unicodedata

import cv2
import numpy as np

from inventory_file_tab_titles import _title_views
from inventory_list_rows import _Box


def _caption(text):
    return " ".join(unicodedata.normalize("NFKC", text).strip(" :").split())


def _value_bounds(gray, field, arrow):
    h = field.height
    inset = max(2, round(.14 * h))
    lane = _Box(field.left + inset, field.top + inset, arrow.left - 1, field.bottom - inset)
    if lane.width < .70 * h or lane.height < .48 * h:
        return None
    pixels = gray[lane.top:lane.bottom, lane.left:lane.right].astype(np.int16)
    ink = np.abs(pixels - np.median(pixels)) >= 36
    columns = np.flatnonzero(ink.sum(axis=0) >= 2)
    if (not columns.size or columns[0] == 0 or columns[-1] == lane.width - 1
            or columns[-1] - columns[0] + 1 < .50 * h):
        return None
    rows = np.flatnonzero(ink[:, columns[0]:columns[-1] + 1].sum(axis=1) >= 2)
    if not rows.size or not .24 * h <= rows[-1] - rows[0] + 1 <= .84 * h:
        return None
    top, bottom = lane.top + int(rows[0]), lane.top + int(rows[-1]) + 1
    padded = _Box(lane.left, max(lane.top, top - inset), lane.right, min(lane.bottom, bottom + inset))
    witness = _Box(lane.left + int(columns[0]) - 1, max(lane.top, top - 1),
                   lane.left + int(columns[-1]) + 2, min(lane.bottom, bottom + 1))
    return padded, witness


def read_rail_selectors(frame, gray, candidates, line_reader):
    """Check an observed title and read its physically separate value twice."""
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    if not callable(recognize):
        return {}
    requests, crops = [], []
    for field, (arrow, title) in candidates.items():
        bounds = _value_bounds(gray, field, arrow)
        if bounds is None:
            continue
        padded, witness = bounds
        box = title.box
        caption = _Box(max(0, box.left - 2), max(field.top + 1, box.top - 2),
                       min(field.left - 1, box.right + 2), min(field.bottom - 1, box.bottom + 2))
        if caption.width < .75 * field.height or caption.height < .30 * field.height:
            continue
        _, title_contrast = _title_views(frame, caption, caption, caption.right)
        color, contrast = _title_views(frame, padded, padded, padded.right)
        crops.append(cv2.resize(title_contrast, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        crops.extend(cv2.resize(patch, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                     for patch in (color, contrast))
        requests.append((field, title, witness))
    if not requests:
        return {}
    try:
        readings = recognize(crops)
    except (RuntimeError, ValueError, cv2.error):
        return {}
    if len(readings) != len(crops):
        return {}
    accepted = {}
    for i, (field, title, witness) in enumerate(requests):
        label, first, second = readings[3 * i:3 * i + 3]
        shown = first["text"]
        score = min(first["score"], second["score"])
        if (label["score"] >= 90 and _caption(label["text"]) == _caption(title.text)
                and shown == second["text"] and score >= 92 and any(char.isalnum() for char in shown)):
            accepted[field] = witness, shown, score
    return accepted
