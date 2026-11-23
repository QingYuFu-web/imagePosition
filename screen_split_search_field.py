"""Separate a measured search input from its own trailing Search button."""

import cv2
import numpy as np

from screen_back_control import _center
from screen_candidate_glyphs import _overlap
from screen_component_captions import _bands, _rectangular_button
from screen_header_search_plate import _lens
from screen_landscape_search_field import _available
from screen_owned_search_fields import _boxes


def _trailing_peer(box, actions):
    _, _, right, _ = box
    _, y = _center(box)
    depth = box[3] - box[1]
    peers = []
    for action in actions:
        if action["role"] != "search":
            continue
        l, t, r, b = action["box"]
        x, cy = action["click_point"]
        if (0 <= l - right <= .49 * depth and .25 * depth <= r - l <= 1.32 * depth
                and .24 * depth <= b - t <= .92 * depth and abs(cy - y) <= .17 * depth
                and l <= x < r and t <= cy < b):
            peers.append(action)
    return peers[0] if len(peers) == 1 else None


def _leading_back(box, actions, regions):
    left, t, _, b = box
    _, middle = _center(box)
    depth = b - t
    return any(row["role"] == "back" and .34 * depth <= left - row["click_point"][0] <= 1.16 * depth
               and abs(middle - row["click_point"][1]) <= .17 * depth and _available(row["box"], regions)
               for row in actions)


def _plate_glyph(frame, box):
    left, top, right, bottom = box
    depth = bottom - top
    middle = round((top + bottom) / 2)
    a, t = left + round(.14 * depth), middle - round(.39 * depth)
    r, b = left + round(.81 * depth), middle + round(.39 * depth)
    paper = np.median(frame[top + depth // 4:bottom - depth // 4:3,
                            left + (right - left) // 2:right - depth // 2:4], axis=(0, 1))
    if paper.min() < 186 or np.ptp(paper) > 21:
        return None
    patch = frame[t:b, a:r]
    distance = np.max(abs(patch - paper), axis=2)
    if (distance <= 8).mean() < .72:
        return None
    lenses = _lens(distance, .75 * depth)
    if not lenses or any(_overlap(lenses[0], lens) < .84 for lens in lenses[1:]):
        return None
    x, y, r, b = lenses[0]
    return (a + x, t + y, a + r, t + b), paper


def _inline_text(frame, box, glyph, paper):
    """Witness the editable line; search queries often lack a Search prefix."""
    left, top, right, bottom = box
    depth = bottom - top
    _, middle = _center(glyph)
    a, t = glyph[2] + max(2, round(.020 * depth)), round(middle - .30 * depth)
    r, b = min(right - round(.1 * depth), a + round(3.7 * depth)), round(middle + .32 * depth)
    patch = frame[t:b, a:r]
    deviation = np.max(paper - patch, axis=2)
    neutral = np.ptp(patch, axis=2) <= 26
    rows = np.uint8((deviation >= 36) & neutral)
    l, y, span, h = cv2.boundingRect(rows)
    if (not .20 * depth <= h <= .58 * depth or not .49 * depth <= span
            or abs(t + y + h / 2 - middle) > .14 * depth or not 0 <= l <= .52 * depth):
        return None
    care = rows[y:y + h, l:l + span]
    tones = deviation[y:y + h, l:l + span][care != 0]
    if (not .09 <= care.mean() <= .58 or not 38 <= np.percentile(tones, 90) <= 167
            or sum(k - j >= .055 * depth for j, k in _bands(care.any(axis=0))) < 2):
        return None
    interior = frame[top + round(.17 * depth):bottom - round(.17 * depth), left + round(.10 * depth):right]
    if (np.max(abs(interior[::3, ::5] - paper), axis=2) <= 9).mean() < .82:
        return None
    return [a + l, t + y, a + l + span, t + y + h]


def split_search_field_actions(frame, regions, actions, confirmed_submits):
    height, width = frame.shape[:2]
    if (not confirmed_submits or regions.get("layer") != "page" or regions.get("regions")
            or not 1.45 * width <= height <= 2.83 * width or width < 240
            or not any(row["click_point"][0] >= .83 * width
                       and .033 * height <= row["click_point"][1] <= .165 * height for row in confirmed_submits)):
        return []
    found = []
    for proposal in _boxes(regions, width, height):
        a, t, r, b = proposal
        if (not .045 * width <= a <= .25 * width or not .50 * width <= r - a <= .895 * width
                or not .048 * width <= b - t <= .158 * width or not .022 * height <= t <= .125 * height):
            continue
        submit = _trailing_peer(proposal, confirmed_submits)
        if submit is None or not _available(submit["box"], regions):
            continue
        # Keep edge detection local to the proposed control.
        pad = max(3, round(.22 * (b - t)))
        x, y = max(0, a - pad), max(0, t - pad)
        tile = cv2.cvtColor(frame[y:b + pad, x:r + pad], cv2.COLOR_BGR2GRAY)
        measured = _rectangular_button(tile, (a - x, t - y, r - x, b - y))
        if measured is None:
            continue
        l, u, r, b = measured
        field = (x + l, y + u, x + r, y + b)
        if (not _available(field, regions) or not _leading_back(field, actions, regions)
                or any(_overlap(field, row["box"]) >= .65
                for row in actions if row["role"] in {"search", "search_input"})):
            continue
        witness = _plate_glyph(frame, field)
        if witness is None:
            continue
        glyph, paper = witness
        if _inline_text(frame, field, glyph, paper) is None:
            continue
        if not any(_overlap(field, row["box"]) >= .87 for row in found):
            found.append({"role": "search_input", "function": "编辑搜索内容",
                          "box": [int(value) for value in field], "click_point": list(_center(field))})
    return found if len(found) == 1 else []
