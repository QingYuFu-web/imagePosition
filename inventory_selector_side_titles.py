"""Recover a missing inline title inside pixel-bounded selector rows."""

from dataclasses import replace
import unicodedata

import cv2
import numpy as np

from inventory_file_tab_titles import _title_views
from inventory_list_rows import _Box


def _key(text):
    normalized = unicodedata.normalize("NFKC", text).casefold().strip(" :")
    return "".join(normalized.split())


def _overhanging_title(field, lines):
    h = field.height
    candidates = [line for line in lines if line.score >= 90
                  and .35 * h <= line.box.height <= 1.12 * h
                  and abs(line.box.center[1] - field.center[1]) <= .20 * h
                  and line.box.left <= field.left - .80 * h
                  and 0 < line.box.right - field.left <= .40 * h]
    if len(candidates) != 1:
        return None
    line = candidates[0]
    return (_Box(max(0, line.box.left - 2), max(field.top + 1, line.box.top - 2),
                 field.left - 2, min(field.bottom - 1, line.box.bottom + 2)), line.text)


def _between_fields(field, arrow, strict_fields):
    h = field.height
    previous = max((peer for peer, mark in strict_fields.items()
                    if .90 * h <= field.left - peer.right <= 4.25 * h
                    and abs(peer.top - field.top) <= .20 * h
                    and abs(peer.bottom - field.bottom) <= .20 * h
                    and .68 <= peer.height / h <= 1.35
                    and peer.width >= 1.75 * peer.height
                    and abs(mark.center[1] - arrow.center[1]) <= .16 * h),
                   key=lambda peer: peer.right, default=None)
    if previous is None:
        return None
    pad = max(2, round(.10 * h))
    cy = round(field.center[1])
    return (_Box(previous.right + pad, max(field.top + 1, cy - round(.41 * h)),
                 field.left - pad, min(field.bottom - 1, cy + round(.41 * h))), None)


def _title_ink(gray, lane, height):
    if lane.width < .80 * height or lane.height < .36 * height:
        return None
    pixels = gray[lane.top:lane.bottom, lane.left:lane.right].astype(np.int16)
    contrast = np.abs(pixels - np.median(pixels))
    ink = contrast >= 40
    columns = np.flatnonzero(ink.sum(axis=0) >= 2)
    if not columns.size or columns[-1] - columns[0] < .68 * height or columns[-1] >= lane.width - 1:
        return None
    rows = np.flatnonzero(ink[:, columns[0]:columns[-1] + 1].any(axis=1))
    if not .33 * height <= rows[-1] - rows[0] + 1 <= 1.05 * height:
        return None
    return _Box(lane.left + int(columns[0]), lane.top + int(rows[0]),
                lane.left + int(columns[-1]) + 1, lane.top + int(rows[-1]) + 1)


def _stepper_arrow(gray, field, arrow, rectangles):
    from inventory_stepper_actions import _arrow_pair

    x, y = arrow.center
    for outer in rectangles:
        if (not outer.left <= x < outer.right or not outer.top <= y < outer.bottom
                or abs(outer.right - field.right) > .20 * outer.height
                or not .85 <= outer.height / field.height <= 2.35):
            continue
        pair = _arrow_pair(gray, outer)
        if pair is not None:
            down = pair[1].box
            if down.left - 1 <= x <= down.right + 1 and down.top - 1 <= y <= down.bottom + 1:
                return True
    return False


def recover_side_titles(frame, gray, rectangles, strict_fields, pending, lines, line_reader):
    """Read a clipped OCR title or a title in the gap between measured fields."""
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    if recognize is None:
        return {}
    requests, crops = [], []
    seen = set()
    for field, value in pending.items():
        arrow = strict_fields[field]
        if arrow in seen or field.width < 1.85 * field.height or value.score < 90:
            continue
        witness = _overhanging_title(field, lines) or _between_fields(field, arrow, strict_fields)
        if witness is None:
            continue
        lane, hinted = witness
        ink = _title_ink(gray, lane, field.height)
        if ink is None or _stepper_arrow(gray, field, arrow, rectangles):
            continue
        color, contrast = _title_views(frame, lane, lane, lane.right)
        crops.extend((color, cv2.resize(contrast, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)))
        requests.append((field, value, ink, hinted))
        seen.add(arrow)
    if not requests:
        return {}
    try:
        readings = recognize(crops)
    except (RuntimeError, ValueError, cv2.error):
        return {}
    if len(readings) != len(crops):
        return {}
    accepted = {}
    for (field, value, ink, hinted), first, second in zip(requests, readings[::2], readings[1::2], strict=True):
        name = first["text"].strip()
        score = min(first["score"], second["score"])
        if (name == second["text"].strip() and score >= 92
                and 2 <= sum(letter.isalpha() for letter in name) <= 48
                and (hinted is None or _key(hinted) == _key(name)) and _key(name) != _key(value.text)):
            accepted[field] = replace(value, box=ink, text=name, score=score)
    return accepted
