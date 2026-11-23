"""Recognize a screen-wide gesture guide with one painted dismiss button."""

from itertools import combinations

import cv2
import numpy as np


def _hues(hue, saturation, value):
    height, width = hue.shape
    s = np.s_[round(.29 * height):round(.71 * height), round(.25 * width):round(.75 * width)]
    active = (saturation[s] >= 145) & (value[s] >= 190)
    histogram = np.bincount(hue[s][active], minlength=180)
    order = np.argsort(histogram)[::-1]
    peaks = []
    for item in order:
        if histogram[item] < .0020 * active.size or len(peaks) >= 4:
            break
        if all(min(abs(item - old), 180 - abs(item - old)) > 10 for old in peaks):
            peaks.append(int(item))
    return peaks


def _button_shape(mask):
    height, width = mask.shape
    contour = cv2.findNonZero(mask)
    hull = cv2.convexHull(contour)
    extent = cv2.contourArea(hull) / (width * height)
    if not .868 <= extent <= .986 or not .66 <= mask.mean() <= .94:
        return False
    filled = cv2.fillConvexPoly(np.zeros_like(mask), hull, 1)
    middle = filled[round(.30 * height):round(.70 * height)]
    if (middle.sum(axis=1) < .94 * width).any():
        return False
    stroke = max(1, round(.085 * height))
    sides = [filled[:stroke, :stroke], filled[:stroke, -stroke:],
             filled[-stroke:, :stroke], filled[-stroke:, -stroke:]]
    return all(patch.mean() <= .27 for patch in sides)


def _white_label(saturation, value, box):
    x, y, w, h = box
    s = np.s_[y + round(.13 * h):y + round(.85 * h), x + round(.075 * w):x + round(.925 * w)]
    mask = np.uint8((saturation[s] <= 85) & (value[s] >= 193))
    _, _, parts, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    marks = [(a, b, ww, hh) for a, b, ww, hh, area in parts[1:]
             if (.19 * h <= hh <= .59 * h and .045 * h <= ww <= .60 * h and .13 <= area / (ww * hh) <= .90)]
    if len(marks) < 6:
        return False
    start = min(a for a, _, _, _ in marks)
    end = max(a + ww for a, _, ww, _ in marks)
    bases = np.array([b + hh for _, b, _, hh in marks])
    return end - start >= .54 * w and (abs(bases - np.median(bases)) <= .12 * h).mean() >= .74


def _arrow_glyph(mask):
    h, w = mask.shape
    trace = mask if h >= w else mask.T
    length, short = trace.shape
    if not 1.13 <= length / short <= 1.58 or not .39 <= trace.mean() <= .70:
        return None
    for direction, pixels in ((-1, trace), (1, trace[::-1])):
        widths = pixels.sum(axis=1) / short
        head = widths[round(.19 * length):round(.46 * length)]
        stem = widths[round(.44 * length):round(.88 * length)]
        if (head.size and stem.size and widths[:max(1, round(.13 * length))].mean() <= .52
                and head.max() >= .76 and .37 <= np.median(stem) <= .69 and np.ptp(stem) <= .29
                and (pixels & pixels[:, ::-1]).sum() >= .75 * (pixels | pixels[:, ::-1]).sum()):
            return (h >= w), direction
    return None


def _arrow_pairs(arrows, *, vertical):
    marks = [arrow for arrow in arrows if arrow[0] == vertical]
    a, b = (2, 3) if vertical else (3, 2)
    for first, second in combinations(sorted(marks, key=lambda item: item[b]), 2):
        step = (first[4] + second[4]) / 2
        if (first[1] < 0 < second[1] and abs(first[a] - second[a]) <= .22 * step
                and 1.75 * step <= second[b] - first[b] <= 4.25 * step
                and abs(first[4] - second[4]) <= .21 * step):
            cx, cy = (first[2] + second[2]) / 2, (first[3] + second[3]) / 2
            yield cx, cy, step


def _gesture_emblem(accent, saturation, value, pair):
    x, y, size = pair
    span = round(.80 * size)
    cx, cy = round(x), round(y)
    field = np.s_[cy - span:cy + span + 1, cx - span:cx + span + 1]
    ink = accent[field]
    neutral = (saturation[field] <= 79) & (value[field] >= 190)
    if not ink.size or not .17 <= ink.mean() <= .77 or not .055 <= neutral.mean() <= .43:
        return False
    count, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(neutral))
    return count > 1 and any(.24 * size <= h <= .96 * size and .46 * size <= w <= 1.32 * size
                             and area >= .44 * w * h for _, _, w, h, area in stats[1:])


def _instructions(accent, saturation, value, arrows, box):
    height, width = accent.shape
    x, y, w, h = box
    vertical = [item for item in _arrow_pairs(arrows, vertical=True)
                if .17 * height <= item[1] <= min(.61 * height, y + h * .25)]
    horizontal = [item for item in _arrow_pairs(arrows, vertical=False)
                  if abs(item[0] - (x + w / 2)) <= .08 * width and .16 * height <= item[1] < y - .46 * h]
    for left in vertical:
        if not left[0] <= .30 * width or not _gesture_emblem(accent, saturation, value, left):
            continue
        valid_right = [right for right in vertical if right[0] >= .70 * width
                       and abs(right[1] - left[1]) <= .085 * height
                       and abs(right[0] + left[0] - width) <= .13 * width
                       and _gesture_emblem(accent, saturation, value, right)]
        if valid_right and any(_gesture_emblem(accent, saturation, value, item) for item in horizontal):
            return True
    return False


def _caption_rows(accent, stats, box):
    h, w = accent.shape
    x, y, width, height = box
    glyphs = [(cx + ww / 2, cy + hh / 2) for cx, cy, ww, hh, area in stats
              if (.12 * height <= hh <= .51 * height and .047 * height <= ww <= .53 * height
                  and .12 <= area / (ww * hh) <= .83)]
    headline = [x for x, cy in glyphs if .035 * h <= cy <= .20 * h
                and abs(x - w / 2) <= .23 * w]
    if len(headline) < 5 or np.ptp(headline) < .12 * w:
        return False
    footer = [x for x, cy in glyphs if .76 * h <= cy <= .88 * h]
    return len(footer) >= 10 and np.ptp(footer) >= .58 * w


def _backdrop(gray):
    height, width = gray.shape
    tiles = []
    for y in (.08, .65):
        for x in (.04, .37, .70):
            tiles.append(gray[round(y * height):round((y + .19) * height):2,
                              round(x * width):round((x + .24) * width):2])
    return sum(12 <= np.median(tile) <= 124 and tile.std() >= 9 and (tile <= 145).mean() >= .78
               for tile in tiles) >= 5


def landscape_tutorial_actions(frame, regions, index):
    """Return original-pixel foreground scope and its sole confirmed action."""
    height, width = frame.shape[:2]
    if width < 280 or width < 1.40 * height or (regions and regions.get("layer") != "page"):
        return None
    small_w = min(700, width)
    small_h = round(small_w * height / width)
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA) if small_w < width else frame
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv.transpose(2, 0, 1)
    matches = set()
    for color in _hues(hue, saturation, value):
        difference = abs(hue.astype(np.int16) - color)
        mask = np.uint8((np.minimum(difference, 180 - difference) <= 8) & (saturation >= 150) & (value >= 194))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        buttons = [(i, x, y, w, h) for i, (x, y, w, h, _) in enumerate(stats[1:], 1)
                   if (.16 * small_w <= w <= .43 * small_w and .052 * small_h <= h <= .133 * small_h
                       and 2.65 <= w / h <= 7.5 and abs(x + w / 2 - small_w / 2) <= .07 * small_w
                       and .32 * small_h <= y + h / 2 <= .68 * small_h)]
        for i, x, y, w, h in buttons:
            button = x, y, w, h
            if (not _button_shape(np.uint8(labels[y:y + h, x:x + w] == i))
                    or not _white_label(saturation, value, button) or not _caption_rows(mask, stats[1:], button)):
                continue
            arrows = []
            for j, (xx, yy, ww, hh, _) in enumerate(stats[1:], 1):
                size = max(ww, hh)
                if (.31 * h <= size <= .83 * h and .22 * h <= min(ww, hh)
                        and .16 * small_h <= yy + hh / 2 <= .62 * small_h
                        and (shape := _arrow_glyph(labels[yy:yy + hh, xx:xx + ww] == j)) is not None):
                    arrows.append((*shape, xx + ww / 2, yy + hh / 2, size))
            if _instructions(mask, saturation, value, arrows, button):
                matches.add(tuple(map(int, button)))
    if len(matches) != 1 or not _backdrop(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)):
        return None
    x, y, w, h = matches.pop()
    sx, sy = width / small_w, height / small_h
    box = [round(x * sx), round(y * sy), round((x + w) * sx), round((y + h) * sy)]
    click = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    return (0, 0, width, height), [{"role": "close", "function": "关闭提示",
                                    "preview_text": "Close tutorial", "box": box, "click_point": click}]
