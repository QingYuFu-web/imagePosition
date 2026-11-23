"""Find a bottom share sheet from its surface and aligned icon rows."""

import cv2
import numpy as np

from screen_foreground import _paired_sheet_corners, _sheet_header_control


def _icon_rows(frame):
    """Find repeated, similarly sized touch targets in separate rows."""
    height, width = frame.shape[:2]
    color = frame.astype(np.int16)
    rim = max(2, round(.012 * width))
    rails = np.concatenate((color[:, :rim], color[:, -rim:]), axis=1)
    paper = np.median(rails, axis=1, keepdims=True)
    masks = (np.max(np.abs(color - paper), axis=2) >= 14, np.ptp(color, axis=2) >= 42)
    icons = set()
    for mask in masks:
        ink = cv2.morphologyEx(np.uint8(mask), cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (x >= .013 * width and x + w <= .987 * width and y < .84 * height
                    and .050 * width <= min(w, h) <= max(w, h) <= .175 * width
                    and .74 <= w / h <= 1.30
                    and cv2.contourArea(contour) >= .36 * w * h
                    and not any(abs(x + w / 2 - a - c / 2) <= .026 * width
                                and abs(y + h / 2 - b - d / 2) <= .026 * width for a, b, c, d in icons)):
                icons.add((x, y, w, h))
    rows = {}
    for _, y, _, h in icons:
        center = y + h / 2
        row = sorted((icon for icon in icons if abs(icon[1] + icon[3] / 2 - center) <= .025 * width
                      and .70 * h <= icon[3] <= 1.33 * h))
        if len(row) < 3 or row[-1][0] - row[0][0] < .34 * width:
            continue
        centers = [x + w / 2 for x, _, w, _ in row]
        steps = np.diff(centers)
        stride = float(np.min(steps))
        if not .123 * width <= stride <= .253 * width:
            continue
        multiples = steps / stride
        if np.max(np.abs(multiples - np.rint(multiples))) > .17:
            continue
        rows[tuple(row)] = (float(np.mean([y + h / 2 for _, y, _, h in row])), stride)
    return tuple((center, stride, row) for row, (center, stride) in rows.items())


def _stacked_recipient_buttons(frame):
    """Recognize the repeated short buttons on the right of a recipient list."""
    _, width = frame.shape[:2]
    left = round(.62 * width)
    red = frame[:, left:, 2].astype(np.int16)
    rest = frame[:, left:, :2].max(axis=2)
    mask = np.uint8((red >= 138) & (red - rest >= 66))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    buttons = [(x, y, w, h) for x, y, w, h, area in stats[1:]
               if .115 * width <= w <= .235 * width and .043 * width <= h <= .10 * width
               and area >= .74 * w * h and w >= 1.7 * h]
    for first, second, third in zip(buttons, buttons[1:], buttons[2:]):
        x, y, w, h = first
        intervals = second[1] - y, third[1] - second[1]
        if (min(intervals) >= 1.35 * h and max(intervals) <= 2.5 * h
                and abs(intervals[1] - intervals[0]) <= .22 * h
                and max(abs(x - xx) + abs(w - ww) + abs(h - hh) for xx, _, ww, hh in (second, third)) <= .23 * h):
            return True
    return False


def share_sheet_surface(frame, gray):
    """Measure a viewport-attached plate with repeated horizontal share actions."""
    height, width = frame.shape[:2]
    if not 1.35 * width <= height <= 2.9 * width:
        return None
    pad = max(3, round(.018 * width))
    start, stop = round(.30 * height), round(.81 * height)
    left, right = round(.09 * width), round(.91 * width)
    center = frame[:, left:right].astype(np.int16)
    tone = np.median(center, axis=1).astype(np.int16)
    in_plane = np.max(np.abs(center - tone[:, None]), axis=2) <= 10
    uniform = in_plane.mean(axis=1)
    matches = []
    shifted = np.max(np.abs(tone[1:] - tone[:-1]), axis=1)
    plate_start = (uniform[1:] >= .84) & ((uniform[:-1] < .80) | (shifted >= 34))
    for top in np.flatnonzero(plate_start[start - 1:stop]) + start:
        if np.ptp(tone[top]) > 30 or height - top - pad < .09 * width:
            continue
        surface = np.median(center[top:top + pad], axis=(0, 1))
        above = np.max(np.abs(center[top - pad:top - 1] - surface), axis=2)
        below = np.max(np.abs(center[top:top + pad] - surface), axis=2)
        if (above >= 23).mean() < .59 or (below <= 14).mean() < .82:
            continue
        rim = max(3, round(.021 * width))
        rails = frame[top + round(.082 * width):height - rim, [*range(rim), *range(width - rim, width)]]
        uniform_rails = np.max(rails, axis=2).astype(np.int16) - np.min(rails, axis=2) <= 30
        if uniform_rails.mean() < .82:
            continue
        rows = _icon_rows(frame[top:])
        paired = any(center_a <= .70 * width and .175 * width <= center_b - center_a <= .43 * width
                     and .71 <= gap_a / gap_b <= 1.35
                     for center_a, gap_a, _ in rows for center_b, gap_b, _ in rows)
        light = np.min(surface) >= 212
        close = (_sheet_header_control(gray, int(top), right_only=True, extra_depth=4, corroborate_core=True)
                 if light else None)
        if not paired and not (light and rows and
                (close or (_paired_sheet_corners(gray >= 218, int(top), round(.072 * width))
                           and _stacked_recipient_buttons(frame[top:])))):
            continue
        icons = set()
        for _, stride, row in rows:
            for x, y, w, h in row:
                center_x, center_y = x + w / 2, top + y + h / 2
                radius = max(w / 2, h / 2, .35 * stride)
                icons.add((max(0, round(center_x - radius)), max(int(top), round(center_y - radius)),
                           min(width, round(center_x + radius)), min(height, round(center_y + radius))))
        matches.append(((0, int(top), width, height), close, tuple(sorted(icons, key=lambda b: (b[1], b[0])))))
    if matches and matches[-1][0][1] - matches[0][0][1] <= pad:
        return matches[0]
    return None
