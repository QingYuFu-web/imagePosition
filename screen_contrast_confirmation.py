"""Find a dim rounded panel with a visibly divided two-button footer."""

import cv2
import numpy as np

from screen_foreground import BlockingOverlay, _working_view


def _bands(mask):
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(changes[::2], changes[1::2])


def _proposals(gray):
    height, width = gray.shape
    seen = []
    for threshold in (32, 44, 56, 70, 87):
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (.53 * width <= w <= .85 * width and .26 * width <= h <= .79 * width
                    and abs(x + w / 2 - width / 2) <= .06 * width
                    and min(x, y, width - x - w, height - y - h) >= 2):
                continue
            box = (x, y, x + w, y + h)
            if any(max(abs(a - b) for a, b in zip(box, old, strict=True)) <= .01 * width
                   for old in seen):
                continue
            if cv2.contourArea(contour) < .89 * w * h:
                continue
            seen.append(box)
            yield box


def _plateau_bounds(gray, box):
    x, y, right, bottom = box
    w, h = right - x, bottom - y
    trim_x, trim_y = round(.13 * w), round(.13 * h)
    inner = gray[y + trim_y:bottom - trim_y, x + trim_x:right - trim_x]
    tone = int(np.median(inner))
    if not 31 <= tone <= 108:
        return None

    x1, y1 = max(0, x - trim_x), max(0, y - trim_y)
    x2, y2 = min(gray.shape[1], right + trim_x), min(gray.shape[0], bottom + trim_y)
    plateau = cv2.inRange(gray[y1:y2, x1:x2], tone - 7, tone + 7) != 0
    inner_y = slice(y + trim_y - y1, bottom - trim_y - y1)
    inner_x = slice(x + trim_x - x1, right - trim_x - x1)
    if plateau[inner_y, inner_x].mean() < .68:
        return None
    columns = np.flatnonzero(plateau[inner_y, :].mean(axis=0) >= .78)
    rows = np.flatnonzero(plateau[:, inner_x].mean(axis=1) >= .78)
    if columns.size == 0 or rows.size == 0:
        return None
    left, right = int(x1 + columns[0]), int(x1 + columns[-1] + 1)
    top, bottom = int(y1 + rows[0]), int(y1 + rows[-1] + 1)
    w, h = right - left, bottom - top
    if (w < .88 * (box[2] - box[0]) or h < .90 * (box[3] - box[1])
            or abs(left + w / 2 - gray.shape[1] / 2) > .047 * gray.shape[1]):
        return None

    rim = max(2, round(.012 * w))
    if min(left, top, gray.shape[1] - right, gray.shape[0] - bottom) < rim:
        return None
    tx, ty = round(.18 * w), round(.15 * h)
    outer = (gray[top - rim:top, left + tx:right - tx],
             gray[bottom:bottom + rim, left + tx:right - tx],
             gray[top + ty:bottom - ty, left - rim:left],
             gray[top + ty:bottom - ty, right:right + rim])
    if any((edge <= tone - 10).mean() < .80 for edge in outer):
        return None

    reach = max(5, round(.047 * w))
    pane = plateau[top - y1:bottom - y1, left - x1:right - x1]
    corners = (pane[:reach, :reach], pane[:reach, -reach:][:, ::-1],
               pane[-reach:, :reach][::-1], pane[-reach:, -reach:][::-1, ::-1])
    corner_tip = max(2, reach // 5)
    if any(corner[:corner_tip, :corner_tip].mean() > .25
           or corner[-corner_tip:, -corner_tip:].mean() < .75 for corner in corners):
        return None
    return (left, top, right, bottom), tone


def _single_divider(support, start, stop, maximum_thickness):
    bands = list(_bands(support[start:stop]))
    if len(bands) != 1:
        return None
    lo, hi = bands[0]
    if hi - lo <= maximum_thickness and 0 < lo and hi < stop - start:
        return start + (lo + hi - 1) // 2
    return None


def _footer_label(ink):
    height, width = ink.shape
    pad_x, pad_y = round(.10 * width), round(.13 * height)
    ys, xs = np.where(ink[pad_y:height - pad_y, pad_x:width - pad_x])
    if not xs.size:
        return None
    x1, x2 = xs.min() + pad_x, xs.max() + pad_x + 1
    y1, y2 = ys.min() + pad_y, ys.max() + pad_y + 1
    if not (.24 * width <= x2 - x1 <= .77 * width and .15 * height <= y2 - y1 <= .44 * height
            and abs((x1 + x2) / 2 - width / 2) <= .12 * width
            and abs((y1 + y2) / 2 - height / 2) <= .16 * height):
        return None
    glyphs = list(_bands(ink[y1:y2, x1:x2].any(axis=0)))
    if len(glyphs) < 3:
        return None
    return (y1, y2)


def _has_split_footer(pane, tone):
    height, width = pane.shape
    rules = (pane >= tone + 9) & (pane <= tone + 53)
    pad = round(.09 * width)
    horizontal = rules[:, pad:width - pad].mean(axis=1) >= .82
    row = _single_divider(horizontal, round(.65 * height), round(.86 * height),
                          max(1, round(.022 * height)))
    if row is None:
        return False
    footer_height = height - row
    vertical = rules[row + round(.08 * footer_height):height - round(.11 * footer_height)].mean(axis=0) >= .76
    col = _single_divider(vertical, round(.45 * width), round(.55 * width),
                          max(1, round(.015 * width)))
    if col is None:
        return False

    ink = pane >= max(142, tone + 66)
    left = _footer_label(ink[row + 1:, :col])
    right = _footer_label(ink[row + 1:, col + 1:])
    if left is None or right is None or max(abs(a - b) for a, b in zip(left, right, strict=True)) > .08 * footer_height:
        return False

    # Require independently centered, separated text within the pane above the T divider.
    body = ink[:row, pad:width - pad]
    centered_lines = 0
    for start, stop in _bands(body.sum(axis=1) >= .045 * width):
        if not .025 * height <= stop - start <= .17 * height:
            continue
        occupied = np.flatnonzero(body[start:stop].any(axis=0))
        if (.24 * width <= occupied[-1] - occupied[0] <= .95 * width
                and abs((occupied[0] + occupied[-1] + 1) / 2 - body.shape[1] / 2) <= .12 * width):
            centered_lines += 1
    return centered_lines >= 2


def detect_contrast_confirmation(frame, *, working_view=None):
    """Return source-image bounds when pane, divider and labels agree."""
    small, gray = _working_view(frame) if working_view is None else working_view
    if small is None:
        return None
    sy, sx = gray.shape
    height, width = frame.shape[:2]
    source_gray = None
    for coarse in _proposals(gray):
        if source_gray is None:
            source_gray = gray if (height, width) == gray.shape else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        box = (round(coarse[0] * width / sx), round(coarse[1] * height / sy),
               round(coarse[2] * width / sx), round(coarse[3] * height / sy))
        refined = _plateau_bounds(source_gray, box)
        if refined is None:
            continue
        (left, top, right, bottom), tone = refined
        if _has_split_footer(source_gray[top:bottom, left:right], tone):
            colors = frame[top:bottom, left:right]
            sample = colors[::4, ::4].astype(np.int16)
            if (sample.max(axis=2) - sample.min(axis=2) <= 13).mean() >= .91:
                return BlockingOverlay("dialog", (left, top, right, bottom))
    return None
