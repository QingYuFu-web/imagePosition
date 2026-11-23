"""Locate the handwriting canvas and verify its printed keyboard actions."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_landscape_search_field import _available
from screen_mail_compose_controls import _down_chevron, _stable_glyph
from screen_native_tab_bar import _distance


def _caps(frame, paint, left, top, right, bottom):
    chip = frame[top:bottom, left:right]
    mask = np.uint8(_distance(chip, paint) >= 9)
    _, _, parts, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    width = frame.shape[1]
    return [(left + int(x), top + int(y), left + int(x + w), top + int(y + h))
            for x, y, w, h, area in parts[1:]
            if .083 * width <= w <= .39 * width and .072 * width <= h <= .175 * width
            and .88 * w * h <= area <= .997 * w * h and 0 < y < y + h < len(mask)]


def _side_keys(frame, paint):
    height, width = frame.shape[:2]
    caps = sorted(_caps(frame, paint, round(.77 * width), round(.48 * height), width,
                        round(.875 * height)), key=lambda box: box[1])
    caps = [box for box in caps if .112 * width <= box[2] - box[0] <= .182 * width
            and .105 * width <= box[3] - box[1] <= .163 * width and box[2] >= .95 * width]
    if not 4 <= len(caps) <= 6:
        return None
    left, top, right, bottom = np.array(caps).T
    depth = bottom - top
    gaps = top[1:] - bottom[:-1]
    if (max(np.ptp(left), np.ptp(right), np.ptp(depth)) > .021 * width
            or min(gaps) < .001 * width or max(gaps) > .044 * width
            or np.ptp(np.diff(top)) > .018 * width):
        return None
    return caps


def _footer_keys(frame, paint, bottom):
    height, width = frame.shape[:2]
    caps = sorted(_caps(frame, paint, 0, bottom - round(.02 * width), width, round(.955 * height)))
    caps = [box for box in caps if .006 * width <= box[1] - bottom <= .044 * width]
    if not 4 <= len(caps) <= 6:
        return None
    left, top, right, after = np.array(caps).T
    gaps = left[1:] - right[:-1]
    if (left[0] > .047 * width or right[-1] < .95 * width or np.ptp(top) > .022 * width
            or np.ptp(after) > .022 * width or min(gaps) < .004 * width or max(gaps) > .042 * width):
        return None
    center = [box for box in caps if box[0] < width / 2 < box[2] and .23 * width <= box[2] - box[0] <= .37 * width]
    return (caps, center[0]) if len(center) == 1 else None


def _keyboard_top(frame, paint, first):
    height, width = frame.shape[:2]
    start = first - round(.22 * width)
    stop = first - round(.061 * width)
    if start < .35 * height:
        return None
    chip = frame[start:stop + round(.012 * width), ::5]
    coverage = (_distance(chip, paint) <= 4).mean(axis=1)
    margin = max(4, round(.006 * width))
    edges = []
    for y in range(margin, stop - start):
        if coverage[y] >= .92 and coverage[y - 1] < .86:
            if coverage[y - margin:y].mean() <= .57 and coverage[y:y + margin].mean() >= .91:
                edges.append(start + y)
    return edges[0] if len(edges) == 1 else None


def _hide_mark(frame, top, first):
    width = frame.shape[1]
    depth = first - top
    section = frame[top:first, :][::4]
    center = section[:, round(.16 * width):round(.79 * width):5]
    if (_distance(center, np.median(center, axis=(0, 1))) <= 6).mean() < .97:
        return None
    left = round(.89 * width)
    gray = cv2.cvtColor(frame[top:first, left:], cv2.COLOR_BGR2GRAY)
    mark = _stable_glyph(gray, (0, round(.14 * depth), gray.shape[1], round(.79 * depth)),
                         _down_chevron,
                         lambda w, h: .16 * depth <= w <= .36 * depth and .096 * depth <= h <= .23 * depth)
    if mark is None:
        return None
    pad = round(.18 * depth)
    a, t, b, d = mark
    return max(0, left + a - pad), top + t - pad, min(width, left + b + pad), top + d + pad


def _strong_label(frame, key, label):
    x, y, r, b = key
    chip = frame[y:b, x:r]
    paint = np.median(chip[::4, ::4], axis=(0, 1))
    l, t, d, e = label["line_box"]
    return (_distance(frame[t:e, l:d], paint) >= 73).mean() >= .14


def handwriting_ime_layout(frame, index):
    """Measure the docked panel using repeated keys, its canvas and a visible space label."""
    height, width = frame.shape[:2]
    if index is None or width < 280 or not 1.55 * width <= height <= 2.85 * width:
        return None
    sample = frame[round(.56 * height):round(.79 * height):10, round(.16 * width):round(.68 * width):10]
    paint = np.median(sample, axis=(0, 1))
    if np.ptp(paint) > 28 or (_distance(sample, paint) <= 6).mean() < .88:
        return None
    side = _side_keys(frame, paint)
    if side is None:
        return None
    footer = _footer_keys(frame, paint, side[-1][3])
    if footer is None:
        return None
    top = _keyboard_top(frame, paint, side[0][1])
    if top is None:
        return None
    keys, space = footer
    cache = {}
    space_label = read_caption(frame, space, index, cache=cache, choice=True)
    if space_label is None or space_label["text"] not in {"空格", "Space", "SPACE", "space"}:
        return None
    key = keys[-1]
    label = read_caption(frame, key, index, cache=cache, choice=True, join_radicals=True)
    enter = None
    if (label and label["text"].casefold() in {"换行", "換行", "回车", "回車", "enter", "return"}
            and _strong_label(frame, key, label)):
        enter = key
    return {"keyboard": (0, top, width, height), "return_key": enter, "hide_key": _hide_mark(frame, top, side[0][1])}


def handwriting_ime_actions(frame, regions, index, *, layout=None):
    """Identify only unambiguous operations on the docked handwriting keyboard."""
    if regions.get("layer") != "page":
        return []
    measured = layout if layout is not None else handwriting_ime_layout(frame, index)
    if measured is None:
        return []
    actions = []
    for key, role, name in (("return_key", "keyboard_enter", "键盘回车"), ("hide_key", "keyboard_hide", "收起键盘")):
        box = measured[key]
        if box is not None and _available(box, regions):
            left, top, right, bottom = box
            actions.append({"role": role, "function": name, "box": list(box),
                            "click_point": [(left + right) // 2, (top + bottom) // 2], "preview_text": name})
    return actions
