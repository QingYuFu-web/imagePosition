"""Establish an upright settings panel from its asymmetric control layout."""

from itertools import combinations

import cv2
import numpy as np

from screen_component_captions import _bands


def _peaks(signal, minimum):
    return [start + int(np.argmax(signal[start:stop])) for start, stop in _bands(signal >= minimum)]


def _section_bands(gray, visible):
    """Find wide, shallow header plates that run horizontally across the settings pane."""
    height, width = gray.shape
    left, right = round(.217 * width), round(.84 * width)
    lane = gray[:, left:right]
    visible = visible[:, left:right]
    step = max(2, round(.0125 * height))
    ridge = lane[step:] - lane[:-step]
    clear = visible[step:] & visible[:-step]
    required = round(.84 * lane.shape[1])
    starts = _peaks(((ridge >= 6) & clear).sum(axis=1), required)
    ends = _peaks(((ridge <= -6) & clear).sum(axis=1), required)
    plates = [(start + step / 2, end + step / 2) for start in starts for end in ends
              if .027 * height <= end - start <= .056 * height]
    return [(first[1], second[0], second[1]) for first, second in combinations(plates, 2)
            if .095 * height <= first[0] <= .255 * height
            and .225 * height <= second[0] - first[0] <= .415 * height]


def _navigation_rows(gray, visible):
    """Verify a left rail of similarly spaced caption rows beside those plates."""
    height, width = gray.shape
    left, right = round(.122 * width), round(.184 * width)
    lane = gray[:, left:right]
    visible = visible[:, left:right]
    coverage = visible.sum(axis=1)
    order = np.sort(np.where(visible, lane, 256), axis=1)
    profile = order[np.arange(height), (coverage * .36).astype(int)]
    step = max(2, round(.0125 * height))
    rise = profile[step:] - profile[:-step]
    rise[np.minimum(coverage[step:], coverage[:-step]) < .78 * lane.shape[1]] = 0
    peaks = [point + step / 2 for point in _peaks(rise, 21)]
    return any(.17 * height <= first <= .345 * height
               and .087 * height <= second - first <= .142 * height
               and all(any(abs(point - (first + repeat * (second - first))) <= .024 * height
                           for point in peaks) for repeat in (2, 3))
               for first, second in combinations(peaks, 2))


def _selected_controls(frame, visible):
    """Measure slim color accents under matching compact ink, as on checked switches."""
    height, width = frame.shape[:2]
    left, right = round(.24 * width), round(.966 * width)
    top, bottom = round(.19 * height), round(.925 * height)
    hue, saturation, value = cv2.split(cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV))
    accent = (saturation >= 73) & (value >= 68) & visible[top:bottom, left:right]
    ridge = cv2.morphologyEx(np.uint8(accent), cv2.MORPH_OPEN,
                             np.ones((1, max(5, round(.027 * width))), dtype=np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ridge)
    choices = []
    for number, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if (not .039 * width <= w <= .105 * width or not 1 <= h <= .019 * height
                or w < 8 * h or area < .65 * w * h):
            continue
        reach = round(.040 * height)
        gap = max(2, round(.006 * height))
        inset = round(.14 * w)
        a, b = x + inset, x + w - inset
        if y < reach or y + h + reach > len(accent):
            continue
        colors = hue[y:y + h, x:x + w][labels[y:y + h, x:x + w] == number]
        color = np.bincount(colors, minlength=180).argmax()
        offset = np.abs(hue[y - reach:y + h + reach, a:b].astype(np.int16) - color)
        same = accent[y - reach:y + h + reach, a:b] & ((offset <= 9) | (offset >= 171))
        above, below = same[:reach - gap], same[reach + h + gap:]
        if above.sum() < .70 * w or below.sum() > .27 * above.sum():
            continue
        _, _, mark_w, mark_h = cv2.boundingRect(np.uint8(above))
        if .10 * w <= mark_w <= .72 * w and .014 * height <= mark_h <= .039 * height:
            choices.append((left + x + w / 2, top + y, int(color)))
    return choices


def settings_orientation_cue(frame, *, blocked=None):
    """Return whether the photographed display reads upright as a wide settings page."""
    height, width = frame.shape[:2]
    if height < 230 or not 1.68 * height <= width <= 2.60 * height:
        return False
    size = (round(320 * width / height), 320)
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if height >= size[1] else cv2.INTER_LINEAR)
    visible = (cv2.resize(blocked.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) == 0
               if blocked is not None else np.ones(frame.shape[:2], dtype=bool))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    bands = _section_bands(gray, visible)
    if not bands or not _navigation_rows(gray, visible):
        return False
    marks = _selected_controls(frame, visible)
    for left, right in combinations(marks, 2):
        gap = abs(left[2] - right[2])
        if (abs(right[0] - left[0]) < .34 * size[0] or abs(right[1] - left[1]) > .021 * size[1]
                or 9 < gap < 171):
            continue
        # Opposite choices within a switch move its accent by one button width.
        aligned = [y for x, y, hue in marks if min(abs(hue - left[2]), 180 - abs(hue - left[2])) <= 9
                   and min(abs(x - left[0]), abs(x - right[0])) <= .103 * size[0]]
        for first_end, second_top, second_end in bands:
            if (any(first_end < y < second_top for y in aligned)
                    and any(y > second_end + .145 * size[1] for y in aligned)):
                return True
    return False


def _right_button_ridges(gray, height, blocked):
    """Find paired shallow rims where the top edge meets the photographed screen border."""
    left, right = round(.735 * gray.shape[1]), round(.987 * gray.shape[1])
    start, end = round(.046 * height), round(.084 * height)
    step = max(2, round(.008 * height))
    above = gray[start - step:end - step, left:right].astype(np.int16)
    delta = gray[start:end, left:right].astype(np.int16) - above
    if blocked is not None:
        delta[blocked[start:end, left:right] | blocked[start - step:end - step, left:right]] = 0
    bridge = max(3, round(.016 * gray.shape[1]) | 1)
    for contrast in (6, 8, 10):
        spans = cv2.morphologyEx(np.uint8(delta >= contrast), cv2.MORPH_CLOSE, np.ones((1, bridge), np.uint8))
        for y, row in enumerate(spans, start):
            rims = [(left + a, left + b) for a, b in _bands(row != 0)
                    if 0 < a < b < len(row) and .064 * gray.shape[1] <= b - a <= .088 * gray.shape[1]]
            for (first, mid_left), (mid_right, last) in combinations(rims, 2):
                width = min(mid_left - first, last - mid_right)
                if abs(mid_left - first - (last - mid_right)) > .18 * width:
                    continue
                if not .08 * width <= mid_right - mid_left <= .44 * width:
                    continue
                pad = round(.043 * width)
                top, bottom = max(0, y - round(.40 * width)), y + round(.053 * width)
                yield (first - pad, top, mid_left + pad, bottom), (mid_right - pad, top, last + pad, bottom)


def settings_back_box(frame, *, blocked=None):
    """Measure the Back glyph when a verified settings page has two upper-right header buttons."""
    from screen_landscape_header import _opposite_arrow

    if not settings_orientation_cue(frame, blocked=blocked):
        return None
    height, width = frame.shape[:2]
    if blocked is not None:
        blocked = blocked != 0
    gray = cv2.cvtColor(frame[:round(.11 * height)], cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, round(.024 * height) | 1),) * 2)
    contrast = cv2.subtract(gray, cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel))
    matches = []
    for pair in set(_right_button_ridges(gray, height, blocked)):
        if blocked is not None and any(blocked[top:bottom, left:right].mean() > .08
                                       for left, top, right, bottom in pair):
            continue
        mark = _opposite_arrow(gray, contrast, [{"box": box} for box in pair], height, width)
        if mark is None:
            continue
        left, top, right, bottom = mark
        if blocked is None or not blocked[top:bottom, left:right].any():
            matches.append(mark)
    if len(matches) >= 2 and np.ptp(matches, axis=0).max() <= max(2, .004 * height):
        return sorted(matches)[len(matches) // 2]
    return None
