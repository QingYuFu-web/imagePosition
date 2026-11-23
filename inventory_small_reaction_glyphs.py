"""Measure small heart and thumbs-up glyphs near a corroborated video rail."""

import cv2
import numpy as np

from inventory_list_rows import _Box


def _local_position(box, anchor_x, anchor_y, span, pitch):
    x, y = box.center
    # Either a single adjacent control or a second intervening control can
    # precede the bubble. This only bounds which strokes can belong to the rail.
    gap = (anchor_y - y) / pitch
    return abs(x - anchor_x) <= .31 * span and (.70 <= gap <= 1.34 or 1.68 <= gap <= 2.30)


def _heart_notch(light):
    height, width = light.shape
    left = light[0, round(.18 * width):round(.42 * width)].max()
    right = light[0, round(.59 * width):round(.83 * width)].max()
    center = slice(round(.41 * width), round(.59 * width))
    cap = np.int16(light[0, center])
    inside = light[round(.36 * height):round(.64 * height), center]
    notch = light[1:max(2, round(.27 * height)), center]
    return bool(min(int(left), int(right)) - cap.min() >= 14
                and int(notch.max()) - np.median(inside) >= 33)


def _heart_shape(light, outline, hole):
    height, width = light.shape
    _, _, hole_width, hole_height = cv2.boundingRect(hole)
    if hole_width < .30 * width or hole_height < .30 * height:
        return False
    edge = cv2.boundingRect(outline)
    shape = cv2.drawContours(np.zeros((height, width), np.uint8), [outline], 0, 1, cv2.FILLED,
                             offset=(-edge[0], -edge[1]))
    symmetric = (shape & shape[:, ::-1]).sum() / (shape | shape[:, ::-1]).sum()
    if symmetric < .66:
        return False
    breadth = shape.sum(axis=1)
    cap = max(2, round(.25 * height))
    tip = np.flatnonzero(shape[-1])
    return bool(breadth[:cap].max() >= .72 * width and breadth[:cap].sum() >= 1.14 * breadth[-cap:].sum()
                and 0 < len(tip) <= .47 * width and np.ptp(tip) + 1 == len(tip)
                and abs(tip.mean() + .5 - width / 2) <= .22 * width and _heart_notch(light))


def _heart_candidates(light, span):
    size = max(3, round(.70 * span) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    relief = cv2.subtract(light, cv2.morphologyEx(light, cv2.MORPH_OPEN, kernel))
    for level, contrast in enumerate((36, 41, 46, 51, 56, 63, 70, 78, 86)):
        ink = np.uint8((relief >= contrast) & (light >= 117))
        contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for index, (outline, (_, _, hole, parent)) in enumerate(zip(contours, hierarchy[0], strict=True)):
            if parent >= 0 or hole < 0:
                continue
            x, y, w, h = cv2.boundingRect(outline)
            if not (max(6, .62 * span) <= h <= 1.26 * span and .88 <= w / h <= 1.34
                    and 0 < x < x + w < light.shape[1] and 0 < y < y + h < light.shape[0]):
                continue
            chip = light[y:y + h, x:x + w]
            interior = max((contour for contour, links in zip(contours, hierarchy[0], strict=True)
                            if links[3] == index), key=cv2.contourArea)
            if _heart_shape(chip, outline, interior):
                yield _Box(x, y, x + w, y + h), level


def _thumb_shape(mask):
    height, width = mask.shape
    first = mask[:max(2, round(.27 * height))]
    base = mask[round(.43 * height):]
    thumb_columns = np.flatnonzero(first.any(axis=0))
    return bool(.52 <= mask.mean() <= .87 and np.median(first.sum(axis=1)) <= .55 * width
                and np.median(base.sum(axis=1)) >= .73 * width and base.mean() >= .70
                and .08 * width <= thumb_columns.mean() + .5 <= .55 * width)


def _wrist(main, wrist):
    gap = main.left - wrist.right
    return (.12 * main.width <= wrist.width <= .43 * main.width and .40 * main.height <= wrist.height <= .76 * main.height
            and 0 <= gap <= .46 * main.width and .21 * main.height <= wrist.top - main.top <= .55 * main.height
            and abs(wrist.bottom - main.bottom) <= .18 * main.height)


def _thumb_candidates(light, span):
    background = np.percentile(light, 45)
    high = np.percentile(light, 98)
    if high < 166 or high - background < 72:
        return
    for level, fraction in enumerate((.54, .63, .72, .81, .89)):
        ink = np.uint8(light >= background + fraction * (high - background))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
        components = [(index, _Box(x, y, x + w, y + h), area)
                      for index, (x, y, w, h, area) in enumerate(stats[1:], 1)]
        for index, main, _ in components:
            width, height = main.width, main.height
            if not (max(6, .50 * span) <= height <= 1.27 * span and .56 <= width / height <= .91):
                continue
            x, y, right, bottom = main.xyxy()
            mask = labels[y:bottom, x:right] == index
            if not _thumb_shape(mask):
                continue
            cuffs = [cuff for _, cuff, area in components if _wrist(main, cuff)
                     and area >= .68 * cuff.width * cuff.height]
            if len(cuffs) == 1:
                yield _Box(cuffs[0].left, y, right, bottom), level


def _near(left, right, span):
    step = max(1, .18 * span)
    return max(abs(a - b) for a, b in zip(left.xyxy(), right.xyxy(), strict=True)) <= step


def find_small_video_like(frame, comment_box, pitch):
    """Return a measured Like glyph. The caller supplies a verified rail anchor in source pixels."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        return None
    height, width = frame.shape[:2]
    span = (comment_box.width + comment_box.height) / 2
    if not (7 <= span <= .095 * width and 1.88 * span <= pitch <= 3.50 * span):
        return None
    x, y = comment_box.center
    left, right = round(x - 1.08 * span), round(x + 1.08 * span)
    top, bottom = round(y - 2.30 * pitch - .67 * span), round(y - .70 * pitch + .67 * span)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None
    light = frame[top:bottom, left:right].min(axis=2)
    matches = []
    for candidates in (_heart_candidates(light, span), _thumb_candidates(light, span)):
        positioned = []
        for box, level in candidates:
            measured = _Box(box.left + left, box.top + top, box.right + left, box.bottom + top)
            if _local_position(measured, x, y, span, pitch):
                positioned.append((measured, level))
        for box, level in positioned:
            if any(other_level != level and _near(box, other, span) for other, other_level in positioned):
                matches.append(box)
    if not matches or not all(_near(matches[0], other, span) for other in matches[1:]):
        return None
    return matches[len(matches) // 2]
