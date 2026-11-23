"""Associate a compact, strictly marked selector with its visible inline label and value."""

from dataclasses import replace

import numpy as np

from inventory_list_rows import _Box


def _line_ink(gray, box):
    pixels = gray[box.top:box.bottom, box.left:box.right].astype(np.int16)
    return np.abs(pixels - np.median(pixels))


def _outside_label(gray, field, line):
    box = line.box
    if box.right <= field.left:
        return line
    right = field.left - 1
    if right - box.left < .65 * box.height:
        return None
    clipped = _Box(box.left, box.top, right, box.bottom)
    pixels = _line_ink(gray, clipped)
    quiet = max(2, round(.09 * field.height))
    columns = np.flatnonzero((pixels >= 42).sum(axis=0) >= 2)
    # Ignore OCR padding crossing the field boundary only when the label has ended.
    if not columns.size or pixels.shape[1] - columns[-1] <= quiet or columns[-1] - columns[0] < .68 * box.height:
        return None
    return replace(line, box=clipped)


def _value_ink(frame, gray, box):
    ink = _line_ink(gray, box)
    if np.quantile(ink, .90) < 74:
        return False
    if box.width >= 1.45 * box.height:
        return True
    # A short value still needs distinct visible characters. A separate square
    # status icon often receives its own OCR line but is not the shown value.
    stroke = ink >= 46
    profile = stroke.sum(axis=0) >= 2
    runs = np.count_nonzero(np.diff(np.r_[False, profile, False])) // 2
    chroma = np.ptp(frame[box.top:box.bottom, box.left:box.right], axis=2)
    return bool(runs >= 2 and np.median(chroma[stroke]) <= 24)


def find_compact_named_selector(frame, gray, field, arrow, lines):
    """Return the one visible label and value of a field with a verified down arrow.

    The caller must establish the field enclosure and strict chevron first.
    """
    height = field.height
    if field.width < 3 * height:
        return None
    labels = []
    values = []
    for line in lines:
        box = line.box
        if (line.score < 90 or not .53 * height <= box.height <= .85 * height
                or abs(box.center[1] - field.center[1]) > .19 * height):
            continue
        if box.left < field.left and -.20 * height <= field.left - box.right <= .65 * height:
            label = _outside_label(gray, field, line)
            if label is not None and np.quantile(_line_ink(gray, label.box), .90) >= 66:
                labels.append(label)
        if (field.left - .13 * height <= box.left <= field.left + 1.12 * height
                and box.right <= arrow.left - .16 * height and _value_ink(frame, gray, box)):
            values.append(line)
    if len(labels) != 1 or len(values) != 1:
        return None
    return labels[0], values[0]
