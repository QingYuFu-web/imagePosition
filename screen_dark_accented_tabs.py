"""Find short tab pairs whose selected caption overlaps the accent paint."""

import cv2
import numpy as np

from screen_component_captions import _bands, _tight


def _components(mask):
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask), connectivity=8)
    return [tuple(map(int, row)) for row in stats[1:]]


def _search_anchor(frame):
    width = frame.shape[1]
    left, top = round(.75 * width), round(.115 * width)
    hue, saturation, value = cv2.split(cv2.cvtColor(frame[top:round(.257 * width), left:], cv2.COLOR_BGR2HSV))
    mask = (saturation >= 106) & (value >= 95)
    buttons = []
    for x, y, span, depth, area in _components(mask):
        if (not .060 * width <= span <= .145 * width or not .036 * width <= depth <= .083 * width
                or area < .62 * span * depth or x == 0 or y == 0 or y + depth == len(mask)):
            continue
        button_top, button_bottom = y + top, y + top + depth
        a, b = round(.10 * width), round(.79 * width)
        start = button_top - round(.34 * depth)
        stop = button_bottom + round(.31 * depth)
        rows = cv2.cvtColor(frame[start:stop, a:b:3], cv2.COLOR_BGR2GRAY)
        above = rows[:round(.19 * depth)]
        below = rows[-round(.16 * depth):]
        inner = rows[round(.49 * depth):round(1.19 * depth)]
        baseline = np.maximum(np.median(above, axis=0), np.median(below, axis=0))
        if np.median(above) > 105 or np.mean(inner >= baseline + 10) < .83:
            continue
        paint = hue[y:y + depth, x:x + span][mask[y:y + depth, x:x + span]]
        buttons.append((button_top, button_bottom, int(np.median(paint))))
    return buttons[0] if len(buttons) == 1 else None


def _captions(tile, offset, depth):
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    chroma = cv2.subtract(tile.max(axis=2), tile.min(axis=2))
    mask = (gray >= 92) & (chroma <= 35)
    groups = []
    for left, right in _bands(mask.sum(axis=0) >= max(2, round(.039 * depth))):
        if groups and left - groups[-1][1] <= .26 * depth:
            groups[-1] = groups[-1][0], right
        else:
            groups.append((left, right))
    return [box for left, right in groups
            if (box := _tight(mask[:, left:right], offset[0] + left, offset[1])) is not None
            and .28 * depth <= box[3] - box[1] <= .86 * depth and .40 * depth <= right - left <= 3.12 * depth]


def _accent_owner(tile, offset, depth, hue, captions):
    tones, saturation, value = cv2.split(cv2.cvtColor(tile, cv2.COLOR_BGR2HSV))
    difference = cv2.absdiff(tones, hue)
    mask = np.uint8(((difference <= 5) | (difference >= 175)) & (saturation >= 94) & (value >= 89))
    kernel = np.ones((max(1, round(.043 * depth)), max(1, round(.077 * depth))), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    marks = []
    for left, top, span, height, area in _components(mask):
        if (not .53 * depth <= span <= 1.62 * depth or not .115 * depth <= height <= .38 * depth
                or area < .44 * span * height):
            continue
        a, y = offset[0] + left, offset[1] + top
        b, end = a + span, y + height
        owners = [box for box in captions if min(box[2], b) - max(box[0], a) >= .63 * (box[2] - box[0])
                  and abs((a + b - box[0] - box[2]) / 2) <= .24 * depth
                  and .035 * depth <= box[3] - y <= .36 * depth and 0 < end - box[3] <= .29 * depth
                  and y - box[1] >= .29 * depth]
        if len(owners) == 1:
            marks.append((owners[0], (a, y, b, end)))
    return marks[0] if len(marks) == 1 else None


def accented_tab_pair(frame):
    """Return the inactive caption and evidence shared with its selected peer."""
    anchor = _search_anchor(frame)
    if anchor is None:
        return None
    top, bottom, hue = anchor
    depth = bottom - top
    width = frame.shape[1]
    left, right = round(.24 * width), round(.76 * width)
    start, end = top - round(1.47 * depth), top - round(.34 * depth)
    tile = frame[start:end, left:right]
    captions = _captions(tile, (left, start), depth)
    if len(captions) != 2:
        return None
    first, second = captions
    if (not .44 * depth <= second[0] - first[2] <= 1.35 * depth
            or abs(first[3] - second[3]) > .17 * depth
            or abs((first[0] + second[2]) / 2 - width / 2) > .105 * width):
        return None
    selection = _accent_owner(tile, (left, start), depth, hue, captions)
    if selection is None:
        return None
    selected, mark = selection
    left, right = round(.015 * width), round(.925 * width)
    start, end = bottom + round(.47 * depth), bottom + round(1.42 * depth)
    tile = frame[start:end, left:right]
    lower = _captions(tile, (left, start), depth)
    # A second measured selection independently supports the accent's meaning.
    corroboration = _accent_owner(tile, (left, start), depth, hue, lower)
    if corroboration is None:
        return None
    lower_selected, _ = corroboration
    aligned = [label for label in lower if abs(label[3] - lower_selected[3]) <= .17 * depth]
    if len(aligned) < 4 or aligned[-1][2] - aligned[0][0] < .48 * width:
        return None
    peer = second if selected == first else first
    inset = max(2, round(.095 * depth))
    context = [first[0] - inset, min(first[1], second[1]) - inset,
               second[2] + inset, max(mark[3], first[3], second[3]) + inset]
    return peer, context
