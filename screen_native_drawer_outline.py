"""Read a portrait drawer that visibly occludes a loading pane."""

from collections import Counter
import re

import cv2
import numpy as np

from screen_caption_transcript import _bands
from screen_grouped_drawer_rows import _headline, _selected, _shapes, _text_groups
from screen_landscape_icon_tabs import _small_latin
from screen_native_tab_bar import _distance
from screen_portrait_drawer import _seam_positions, _split_lane


def _loading_backdrop(gray, seam, top, base):
    height, width = gray.shape
    end = width - round(.030 * width)
    tile = gray[top:height - round(.059 * width), seam:end].astype(np.int16)
    if tile.shape[1] < .16 * width:
        return False
    profiles = tile[:, round(.14 * tile.shape[1]):round(.79 * tile.shape[1])]
    tones = np.median(profiles, axis=1)
    blocks = []
    for a, b in _bands(base - tones >= max(8, .086 * base)):
        if not .016 * width <= b - a <= .066 * width:
            continue
        paint = float(np.median(tones[a:b]))
        core = tile[a:b]
        # The long edge reaches the drawer seam and stops before the exterior margin.
        uniform = abs(core - paint) <= 4
        if (uniform[:, :round(.19 * tile.shape[1])].mean() >= .88
                and np.median(uniform.mean(axis=1)) >= .84):
            blocks.append((a, b, paint))
    if len(blocks) < 9:
        return False
    heads = []
    for i, (a, b, dark) in enumerate(blocks[:-2]):
        following = blocks[i + 1:i + 3]
        if (base - dark < .20 * base
                or any(max(5, .063 * base) > paint - dark or not .80 * (b - a) <= end - start <= 1.22 * (b - a)
                       for start, end, paint in following)
                or any(not .055 * (b - a) <= gap <= .66 * (b - a)
                       for gap in (following[0][0] - b, following[1][0] - following[0][1]))):
            continue
        heads.append(a)
    if len(heads) < 3:
        return False
    gaps = np.diff(heads)
    return .135 * width <= min(gaps) <= max(gaps) <= .40 * width and np.ptp(gaps) <= .14 * np.median(gaps)


def _bottom_corner(gray, seam, base, light):
    height, width = gray.shape
    pad = round(.088 * width)
    band = gray[-pad:, seam - pad:seam + round(.014 * width)]
    paint = (band >= light - min(19, .10 * light))
    if not paint[:, :round(.37 * pad)].all():
        return False
    bounds = np.argmax(~paint, axis=1)
    if (abs(float(np.median(bounds[:max(3, pad // 2)])) - pad) > max(1, .0032 * width)
            or not .016 * width <= pad - bounds[-1] <= .063 * width):
        return False
    curve = np.flatnonzero(bounds < pad - 1)
    if not len(curve) or curve[0] < .28 * pad or (np.diff(bounds[curve[0]:]) <= 1).mean() < .91:
        return False
    return all(abs(float(np.median(band[row, int(bounds[row]) + 2:])) - base) <= 5
               for row in curve[-max(3, len(curve) // 2):])


def _drawer_outline(view, gray, seam, reversed_view=False):
    height, width = gray.shape
    gutter = gray[:, width - round(.022 * width):]
    base = float(np.median(gutter[round(.21 * height):round(.89 * height)]))
    if base < 42 or (abs(gutter[round(.16 * height):].astype(np.int16) - base) <= 5).mean() < .96:
        return None
    outer = (abs(gutter.astype(np.int16) - base) <= 5).mean(axis=1) >= .87
    runs = [(a, b) for a, b in _bands(outer) if a <= .115 * height and b >= .98 * height]
    if len(runs) != 1:
        return None
    top, bottom = runs[0]
    if bottom != height:
        return None
    left, right = round(.42 * seam), round(.89 * seam)
    if reversed_view:
        left, right = seam - right, seam - left
    middle = view[:, left:right:3]
    paint = np.median(middle[round(.26 * height):round(.89 * height)], axis=(0, 1))
    light = float(np.median(gray[round(.42 * height):round(.84 * height), round(.49 * seam):round(.91 * seam):3]))
    if not .22 * light <= base <= .59 * light:
        return None
    quiet = (_distance(middle, paint) <= 11).mean(axis=1)
    flat = np.flatnonzero(quiet[top + round(.115 * height):top + round(.28 * height)] >= .94)
    if not flat.size:
        return None
    body = top + round(.115 * height) + int(flat[0])
    if (quiet[body:round(.91 * height)] >= .83).mean() < .84 or body - top < .105 * height:
        return None
    header = middle[top:body - 1]
    color = np.median(header, axis=(0, 1))
    if (abs(color - paint).max() < 46 or np.ptp(color) < 26 or (_distance(header, color) <= 16).mean() < .73):
        return None
    sharp = gray[body:height - round(.056 * width), seam - 1].astype(np.int16) - gray[body:height - round(.056 * width), seam]
    if (sharp >= 36).mean() < .945 or not _bottom_corner(gray, seam, base, light):
        return None
    if not _loading_backdrop(gray, seam, top, base):
        return None
    return top, body, bottom, paint


def _grouped_layout(frame, box, body, paint):
    left, _, right, bottom = box
    width = right - left
    area = frame[body:bottom, left:right]
    ink = _distance(area, paint)
    mask = ink >= 43
    if not (lane := _split_lane(mask)):
        return None
    text = _shapes(mask[:, lane:round(.94 * width)], (left + lane, body),
                   .016 * width, .088 * width, max(4, round(.006 * width)), max(3, round(.0064 * width) | 1))
    inset = round(.025 * width)
    icons = _shapes(mask[:, inset:lane], (left + inset, body),
                    .028 * width, .100 * width, max(3, round(.0045 * width)), max(3, round(.0078 * width) | 1))
    rows = _text_groups(icons, text, width)
    if len(rows) < 6:
        return None
    gaps = np.diff([item[3] for item in rows])
    pitch = float(np.median(gaps))
    if not .085 * width <= pitch <= .208 * width or (abs(gaps - pitch) <= .105 * pitch).sum() < len(gaps) - 2:
        return None
    selected = [i for i, (_, _, _, center) in enumerate(rows) if _selected(frame, box, center, pitch, paint)]
    if len(selected) != 1:
        return None
    return rows, pitch, selected[0]


def native_drawer_outline_actions(frame, index):
    """Return a verified loading-obscuring drawer and its readable unselected rows."""
    height, width = frame.shape[:2]
    if index is None or width < 240 or not 1.44 * width <= height <= 2.8 * width:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    matches = []
    for right_side in (False, True):
        visual, gray_view = (frame[:, ::-1], gray[:, ::-1]) if right_side else (frame, gray)
        for seam in _seam_positions(gray_view):
            result = _drawer_outline(visual, gray_view, seam, right_side)
            if result:
                top, body, bottom, paint = result
                box = (width - seam, top, width, bottom) if right_side else (0, top, seam, bottom)
                matches.append((box, body, paint))
    if len(matches) != 1:
        return None
    box, body, paint = matches[0]
    layout = _grouped_layout(frame, box, body, paint)
    if not layout:
        return None
    rows, pitch, selected = layout
    actions, cache = [], {}
    for i, (bounds, _, __, center) in enumerate(rows):
        if i == selected:
            continue
        name = _headline(frame, bounds, paint, index, cache)
        if not name and bounds[3] - bounds[1] <= 16:
            name = _small_latin(frame, bounds, paint, index)
        if not name or re.search(r"\d+(?:\.\d+){1,4}", name):
            continue
        left, top, right, bottom = box
        y0, y1 = max(body, round(center - pitch / 2)), min(bottom, round(center + pitch / 2))
        if i:
            y0 = max(y0, round((rows[i - 1][3] + center) / 2))
        if i + 1 < len(rows):
            y1 = min(y1, round((center + rows[i + 1][3]) / 2))
        actions.append({"role": "app_drawer_row", "function": f"打开{name}", "preview_text": name,
                        "box": [left + round(.018 * (right - left)), y0, right - round(.019 * (right - left)), y1],
                        "click_point": [(bounds[0] + bounds[2]) // 2, round(center)]})
    counts = Counter(item["function"] for item in actions)
    return box, [item for item in actions if counts[item["function"]] == 1]
