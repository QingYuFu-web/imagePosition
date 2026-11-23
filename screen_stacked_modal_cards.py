"""Locate a framed modal list with aligned available and unavailable buttons."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_game_offer_cards import _title
from screen_lowres_action_text import read_lowres_action_group
from screen_rim_attached_dismiss import _cross_arms


def _rows(boxes, width, height):
    possible = sorted((box for box in boxes if .58 * width <= box[2] - box[0] <= .89 * width
                       and .155 * width <= box[3] - box[1] <= .30 * width and box[1] >= .32 * height),
                      key=lambda box: box[1])
    if not 3 <= len(possible) <= 6:
        return []
    values = np.array(possible)
    pitch = np.diff(values[:, 1])
    typical = np.median(pitch)
    if (not .16 * width <= typical <= .30 * width or np.ptp(pitch) > .12 * typical
            or np.ptp(values[:, 0]) > .034 * width or np.ptp(values[:, 2]) > .034 * width
            or np.ptp(values[:, 3] - values[:, 1]) > .15 * typical):
        return []
    return possible


def _button(frame, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    x0, x1 = left + round(.68 * w), right - round(.017 * w)
    y0, y1 = top + round(.046 * h), top + round(.69 * h)
    tile = frame[y0:y1, x0:x1].astype(np.int16)
    paper = np.median(tile[::3, :max(3, round(.028 * w)):2], axis=(0, 1))
    if paper.mean() < 133:
        return None
    mask = np.uint8(abs(tile - paper).max(axis=2) >= 49)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    matches = []
    for contour in contours:
        x, y, a, b = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (not .17 * w <= a <= .30 * w or not .32 * h <= b <= .56 * h or not 1.3 <= a / b <= 2.65
                or not .76 * a * b <= area <= .986 * a * b or x < 2 or y < 1
                or x + a >= tile.shape[1] - 1 or y + b >= tile.shape[0] - 1):
            continue
        cut = tile[y + round(.26 * b):y + b - round(.20 * b), x + round(.15 * a):x + round(.31 * a)]
        fill = np.median(cut, axis=(0, 1))
        chroma = np.ptp(fill)
        if chroma > 43 or chroma <= 22:
            matches.append(((x0 + x, y0 + y, x0 + x + a, y0 + y + b), chroma > 43, paper))
    return matches[0] if len(matches) == 1 else None


def _round_close(frame, box):
    left, top, right, bottom = box
    tile = frame[top:bottom, left:right]
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(tile.astype(np.int16))
    red = np.uint8((r >= 112) & (r - g >= 83) & (r - b >= 83))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    matches = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (not .54 * tile.shape[1] <= w <= .95 * tile.shape[1]
                or not .54 * tile.shape[0] <= h <= .95 * tile.shape[0]
                or not .79 <= w / h <= 1.23 or not .63 <= cv2.contourArea(contour) / (w * h) <= .88):
            continue
        disk = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(disk, [contour], -1, 1, cv2.FILLED)
        surround = float(np.median(gray[disk != 0]))
        marks = []
        for lift in (42, 60, 79):
            ink = np.uint8((gray >= surround + lift) & (disk != 0))
            a, b, c, d = cv2.boundingRect(ink)
            if (.41 * w <= c <= .73 * w and .41 * h <= d <= .73 * h
                    and abs(a + c / 2 - x - w / 2) <= .085 * w and abs(b + d / 2 - y - h / 2) <= .085 * h
                    and _cross_arms(ink[b:b + d, a:a + c] != 0)):
                marks.append((left + a, top + b, left + a + c, top + b + d))
        if len(marks) >= 2 and np.ptp(marks, axis=0).max() <= .085 * min(w, h):
            matches.append(marks[len(marks) // 2])
    return matches[0] if len(matches) == 1 else None


def _edge(paper, axis, begin, stop, span, sample):
    field = paper if axis == 0 else paper.T
    first, last = span
    scored = []
    for offset in range(max(sample, begin), min(len(field) - sample, stop)):
        change = abs(field[offset - sample, first:last:3].astype(np.int16)
                     - field[offset + sample, first:last:3].astype(np.int16))
        evidence = float((change >= 43).mean())
        if evidence >= .77:
            scored.append((evidence, float(np.median(change)), offset))
    return max(scored)[2] if scored else None


def _outer_panel(frame, rows, close):
    width, height = frame.shape[1], frame.shape[0]
    l, t, r, _ = rows[0]
    b = rows[-1][3]
    w = r - l
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    radius = max(2, round(.006 * w))
    vertical = (t + round(.08 * w), b - round(.08 * w))
    horizontal = (l + round(.084 * w), r - round(.16 * w))
    left = _edge(gray, 1, l - round(.065 * w), l - round(.012 * w), vertical, radius)
    right = _edge(gray, 1, r + round(.010 * w), r + round(.073 * w), vertical, radius)
    bottom = _edge(gray, 0, b + round(.010 * w), b + round(.095 * w), horizontal, radius)
    if left is None or right is None or bottom is None:
        return None
    top = close[1]
    if not (0 < left < right < width and top < t < b < bottom < height):
        return None
    return left, top, right + 1, bottom + 1


def stacked_modal_card_actions(frame, regions, index):
    """Scope a stacked modal list; keep muted or unreadable controls out of actions."""
    height, width = frame.shape[:2]
    if regions["layer"] not in {"page", "dialog", "layer_uncertain"} or width < 260 or not 1.31 * width <= height <= 2.95 * width:
        return None
    boxes = [tuple(row["box"]) for row in regions["candidates"]]
    rows = _rows(boxes, width, height)
    if not rows:
        return None
    buttons = [_button(frame, box) for box in rows]
    if any(button is None for button in buttons) or {item[1] for item in buttons} != {False, True}:
        return None
    colors = np.stack([item[2] for item in buttons])
    if np.ptp(colors, axis=0).max() > 14:
        return None
    l, t, r, _ = rows[0]
    span = r - l
    matched = []
    for box in boxes:
        a, b, c, d = box
        w, h = c - a, d - b
        if (.068 * span <= min(w, h) <= max(w, h) <= .16 * span and .76 <= w / h <= 1.34
                and abs((a + c) / 2 - r) <= .107 * span and .13 * span <= t - (b + d) / 2 <= .34 * span
                and (close := _round_close(frame, box)) and (panel := _outer_panel(frame, rows, box))):
            matched.append((box, close, panel))
    if len(matched) != 1:
        return None
    owner, (a, b, c, d), panel = matched[0]
    actions = [{"role": "close", "function": "关闭", "preview_text": "关闭", "box": list(owner),
                "click_point": [(a + c) // 2, (b + d) // 2]}]
    if index is not None:
        cache = {}
        shared = read_lowres_action_group(frame, [box for box, _, _ in buttons], index)
        for (a, b, c, d), live, _ in buttons:
            if not live:
                continue
            label = shared
            if label is None:
                margin_x, margin_y = round(.18 * (c - a)), round(.22 * (d - b))
                left, top, right, bottom = a + margin_x, b + margin_y, c - margin_x, d - margin_y
                read = read_caption(frame, (left, top, right, bottom), index, cache=cache, choice=True)
                label = read["text"] if read else _title(frame[top:bottom, left:right], index)
            if label:
                actions.append({"role": "modal_primary_action", "function": label, "preview_text": label,
                                "box": [a, b, c, d], "click_point": [(a + c) // 2, (b + d) // 2]})
    return panel, actions
