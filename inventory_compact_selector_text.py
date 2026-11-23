"""Read visible selector values independently of a leading graphic."""

import cv2
import numpy as np

from inventory_file_tab_titles import read_tab_title_boxes
from inventory_list_rows import _Box


def _ink_rows(mask, left, right):
    rows = np.flatnonzero(mask[:, left:right].any(axis=1))
    return int(rows[0]), int(rows[-1] + 1)


def _enclosed_icon(mask):
    height, width = mask.shape
    if not .73 <= width / height <= 1.37 or not .20 <= mask.mean() <= .79:
        return False
    rim = max(1, round(.15 * min(height, width)))
    if min(mask[:rim].any(axis=0).mean(), mask[-rim:].any(axis=0).mean(),
           mask[:, :rim].any(axis=1).mean(), mask[:, -rim:].any(axis=1).mean()) < .78:
        return False
    contours, links = cv2.findContours(np.uint8(mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if links is None:
        return False
    for contour, (_, _, _, parent) in zip(contours, links[0], strict=True):
        if parent < 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if (.48 * width <= w <= .92 * width and .48 * height <= h <= .92 * height
                and abs(x + w / 2 - width / 2) <= .15 * width
                and abs(y + h / 2 - height / 2) <= .15 * height
                and cv2.contourArea(contour) >= .24 * width * height):
            return True
    return False


def _icon_and_value(frame, field, arrow, box):
    height = box.height
    if height < 8:
        return None
    pad = max(2, round(.24 * height))
    left = max(field.left + max(2, round(.07 * field.height)), box.left - round(1.20 * height))
    right = min(arrow.left - 1, box.right + pad)
    top = max(field.top + 1, box.top - pad)
    bottom = min(field.bottom - 1, box.bottom + pad)
    if right - left < 1.72 * height or bottom <= top:
        return None
    chip = frame[top:bottom, left:right].astype(np.int16)
    ink = np.abs(chip - np.median(chip, axis=(0, 1))).max(axis=2) >= 42
    columns = np.flatnonzero(ink.sum(axis=0) >= 2)
    gaps = np.flatnonzero(np.diff(columns) >= max(3, round(.21 * height)))
    if not gaps.size:
        return None
    icon_left, icon_right = int(columns[0]), int(columns[gaps[0]] + 1)
    text_left, text_right = int(columns[gaps[0] + 1]), int(columns[-1] + 1)
    # Keep thin letter tips once the longer strokes have located the two groups.
    leading = max(icon_right + max(2, round(.14 * height)), text_left - pad)
    occupied = np.flatnonzero(ink[:, leading:text_left + 1].any(axis=0))
    if occupied.size:
        text_left = leading + int(occupied[0])
    icon_top, icon_bottom = _ink_rows(ink, icon_left, icon_right)
    text_top, text_bottom = _ink_rows(ink, text_left, text_right)
    if (not .42 * height <= icon_right - icon_left <= 1.10 * height
            or not .43 * height <= icon_bottom - icon_top <= 1.10 * height
            or not .14 * height <= text_left - icon_right <= .55 * height
            or text_right - text_left < .90 * height
            or left + icon_left - field.left > .53 * field.height
            or abs((icon_top + icon_bottom) - (text_top + text_bottom)) > .40 * height
            or not _enclosed_icon(ink[icon_top:icon_bottom, icon_left:icon_right])):
        return None
    icon_pixels = chip[:, icon_left:icon_right][ink[:, icon_left:icon_right]]
    text_pixels = chip[:, text_left:text_right][ink[:, text_left:text_right]]
    if np.median(np.ptp(icon_pixels, axis=1)) - np.median(np.ptp(text_pixels, axis=1)) < 11:
        return None
    return left + icon_right, _Box(left + text_left, top + text_top, left + text_right, top + text_bottom)


def _key(text):
    return "".join(text.casefold().split())


def _visible_suffix(original, cropped, touches_icon):
    # Keep the visible spelling and spacing supplied by the whole-field OCR.
    original = original.strip()
    if _key(original) == _key(cropped):
        return original
    stripped = original[1:].lstrip()
    return stripped if touches_icon and _key(stripped) == _key(cropped) else None


def read_compact_selector_value(frame, field, arrow, line, line_reader):
    """Return a corroborated surface value and its icon-free ink bounds."""
    if getattr(line_reader, "recognize_prepared_lines", None) is None:
        return None
    evidence = _icon_and_value(frame, field, arrow, line.box)
    if evidence is None:
        return None
    icon_right, text_box = evidence
    reading = read_tab_title_boxes(frame, (field,), (text_box,), line_reader, right_edges=(arrow.left,))[0]
    if reading is None:
        return None
    name = _visible_suffix(line.text, reading[0], line.box.left < icon_right)
    return (name, text_box) if name else None
