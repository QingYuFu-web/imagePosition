"""Find unselected halves of visibly paired controls inside a known filter panel."""

import cv2
import numpy as np

from screen_landscape_paired_switches import _plates


def _ink(gray, box):
    left, top, right, bottom = box
    field = gray[top:bottom, left:right].astype(np.int16)
    paper = float(np.median(field))
    contrast = field - paper
    peak = float(np.percentile(contrast, 98))
    mask = np.uint8(contrast >= max(22, .48 * peak))
    x, y, width, height = cv2.boundingRect(mask)
    return paper, peak, (left + x, top + y, width, height), mask.mean()


def _caption(gray, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    inset_x, inset_y = round(.16 * width), round(.15 * height)
    paper, peak, (x, y, span, depth), density = _ink(
        gray, (left + inset_x, top + inset_y, right - inset_x, bottom - inset_y))
    if (peak >= 45 and .19 * width <= span <= .69 * width and .29 * height <= depth <= .63 * height
            and abs(x + span / 2 - (left + right) / 2) <= .12 * width
            and abs(y + depth / 2 - (top + bottom) / 2) <= .14 * height and .035 <= density <= .45):
        return paper, peak
    return None


def _rim(gray, box, center, *, vertical):
    left, top, right, bottom = box
    height = bottom - top
    delta = max(1, round(.045 * height))
    radius = max(3, round((.18 if vertical else .14) * height))
    start, stop = center - radius - delta, center + radius + delta + 1
    if vertical:
        inset = round(.22 * height)
        field = gray[top + inset:bottom - inset, start:stop]
    else:
        inset = round(.22 * (right - left))
        field = gray[start:stop, left + inset:right - inset].T
    if field.shape[1] <= 2 * delta:
        return center, 0.
    field = field.astype(np.int16)
    profile = np.median(np.abs(field[:, 2 * delta:] - field[:, :-2 * delta]), axis=0)
    position = int(profile.argmax())
    return start + delta + position, float(profile[position])


def _heading(gray, panel, pair):
    left, top, _, bottom = pair
    height = bottom - top
    a, b = max(panel[0], left - round(2.67 * height)), left - round(.46 * height)
    if b - a < height:
        return None
    inset = round(.15 * height)
    _, peak, (x, y, span, depth), density = _ink(gray, (a, top + inset, b, bottom - inset))
    if (peak >= 42 and .52 * height <= span <= 1.45 * height and .25 * height <= depth <= .67 * height
            and abs(y + depth / 2 - (top + bottom) / 2) <= .19 * height and .035 <= density <= .26):
        return x
    return None


def _alternative(gray, panel, selected, selected_caption, direction):
    left, top, right, bottom = selected
    width, height = right - left, bottom - top
    a, b = (left - width, left) if direction < 0 else (right, right + width)
    if not panel[0] + .16 * height < a < b < panel[2] - .16 * height:
        return None
    box = a, top, b, bottom
    caption = _caption(gray, box)
    if caption is None or selected_caption[0] - caption[0] < 23:
        return None
    far, strength = _rim(gray, box, a if direction < 0 else b, vertical=True)
    _, upper = _rim(gray, box, top, vertical=False)
    _, lower = _rim(gray, box, bottom, vertical=False)
    if strength < 9 or min(upper, lower) < 6:
        return None
    edge = far + direction * max(2, round(.07 * height))
    if direction < 0:
        a = edge
    else:
        b = edge
    pair = min(a, left), top, max(right, b), bottom
    heading = _heading(gray, panel, pair)
    pad = max(3, round(.10 * height))
    context = [max(panel[0], (pair[0] if heading is None else heading) - pad), max(panel[1], top - pad),
               min(panel[2], pair[2] + pad), min(panel[3], bottom + pad)]
    result = {"box": [a, top, b, bottom], "click_point": [(a + b) // 2, (top + bottom) // 2],
              "context_box": context}
    if sum(caption) <= min(155, .78 * sum(selected_caption)):
        result["review_only"] = True
    return result


def filter_pair_candidates(frame, panel, reference_row):
    """Measure two paired controls above a verified checked row, without reading labels."""
    if not reference_row:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    panel = tuple(map(round, panel))
    _, reference_y, size = reference_row[0]
    anchors = []
    pairs = []
    for selected in _plates(gray):
        left, top, right, bottom = selected
        height = bottom - top
        if (not 1.65 * size <= height <= 2.45 * size or not 2.8 * size <= reference_y - (top + bottom) / 2 <= 5.7 * size
                or not (panel[0] < left < right < panel[2] and panel[1] < top < bottom < panel[3])
                or any(abs(left - x) <= .21 * height and abs(top - y) <= .17 * height for x, y in anchors)):
            continue
        anchors.append((left, top))
        caption = _caption(gray, selected)
        if caption is None or caption[0] < 35:
            continue
        alternatives = [item for direction in (-1, 1)
                        if (item := _alternative(gray, panel, selected, caption, direction)) is not None]
        if len(alternatives) == 1:
            pairs.extend(alternatives)
    if len(pairs) != 2:
        return []
    first, second = sorted(pairs, key=lambda item: item["box"][0])
    if (abs(first["box"][1] - second["box"][1]) > .25 * size
            or abs(first["box"][3] - second["box"][3]) > .25 * size
            or first["context_box"][2] >= second["context_box"][0]):
        return []
    return [first, second]
