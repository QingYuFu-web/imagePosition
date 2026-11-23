"""Recognize a painted search key on a visually verified phone keypad."""

from statistics import median

import cv2
import numpy as np

from screen_candidate_glyphs import _glyphs, _overlap
from screen_choice_grid import _is_keypad, _lattices, _size
from screen_component_captions import _rectangular_button


def _search_key_glyph(frame, box):
    left, top, right, bottom = box
    w, h = _size(box)
    left, right = left + round(.20 * w), right - round(.20 * w)
    top, bottom = top + round(.26 * h), bottom - round(.26 * h)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    paint = np.median(crop.reshape(-1, 3), axis=0)
    if np.ptp(paint) < 64 or (np.max(abs(crop.astype(np.int16) - paint), axis=2) <= 18).mean() < .68:
        return None
    scale = max(1., 120 / min(crop.shape[:2]))
    sample = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else crop
    guesses = [(measurement, bounds) for role, measurement, bounds in _glyphs(sample) if role == "search"]
    stable = [bounds for cut, bounds in guesses if sum(other != cut and _overlap(bounds, peer) >= .81
                                                       for other, peer in guesses) >= 2]
    if not stable or any(_overlap(stable[0], peer) < .81 for peer in stable):
        return None
    x0, y0, x1, y1 = stable[len(stable) // 2]
    x, y = left + (x0 + x1) / (2 * scale), top + (y0 + y1) / (2 * scale)
    if abs(x - (left + right) / 2) > .11 * w or abs(y - (top + bottom) / 2) > .13 * h:
        return None
    return x, y


def _keypad_owners(frame, candidates, key):
    height, width = frame.shape[:2]
    expected_w, expected_h = _size(key["box"])
    seeds = [row for row in candidates if row["center"][1] >= .63 * height and
             .75 * expected_w <= _size(row["box"])[0] <= 1.24 * expected_w and
             .37 * expected_h <= _size(row["box"])[1] <= .66 * expected_h]
    if len(seeds) < 10:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = [(row["id"], measured) for row in seeds if (measured := _rectangular_button(gray, row["box"])) is not None]
    groups = [group for group in _lattices(boxes) if _is_keypad(group, candidates, height, width)]
    result = []
    for group in groups:
        surfaces = [box for _, box in group]
        left, top = min(box[0] for box in surfaces), min(box[1] for box in surfaces)
        right, bottom = max(box[2] for box in surfaces), max(box[3] for box in surfaces)
        beat = median(_size(box)[1] for box in surfaces)
        x, y = key["center"]
        if (right - expected_w <= x < right + .11 * expected_w and bottom - 1.36 * beat <= y < bottom and
                key["box"][1] > top + .95 * beat and key["box"][3] <= bottom + .28 * beat and .12 * width >= left):
            result.append((left, top, right, bottom))
    return result


def keyboard_search_actions(frame, regions):
    """Return a search affordance only when the key and surrounding keypad agree."""
    if regions["layer"] != "page":
        return []
    height, width = frame.shape[:2]
    candidates = [row for row in regions["candidates"] if row["visibility"] == "page"]
    # This variant has a tall right-edge mode key next to separate, repeated
    # character keys. Check its own icon before measuring all row surfaces.
    matches = []
    for row in candidates:
        left, top, right, bottom = row["box"]
        w, h = _size(row["box"])
        if not (.70 * height <= top and .80 * height <= row["center"][1] <= .925 * height and
                row["center"][0] >= .79 * width and .10 * width <= w <= .25 * width and .96 * w <= h <= 1.67 * w):
            continue
        point = _search_key_glyph(frame, row["box"])
        if point is not None and _keypad_owners(frame, candidates, row):
            matches.append({"role": "keyboard_search", "function": "键盘搜索", "box": list(row["box"]),
                            "click_point": list(point), "preview_text": "Search key"})
    return matches if len(matches) == 1 else []
