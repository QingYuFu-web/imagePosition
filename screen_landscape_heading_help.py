"""Verify a literal help badge following the title of a landscape header."""

import cv2
import numpy as np

from screen_candidate_glyphs import _overlap


def _diamonds(gray, left, top, height, center_y):
    for inverse, mode in ((True, cv2.CMP_LE), (False, cv2.CMP_GE)):
        for level in (33, 49, 60, 74, 89, 105, 123, 145, 168, 191, 216):
            ink = cv2.compare(gray, level, mode)
            shapes, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in shapes:
                x, y, w, h = cv2.boundingRect(contour)
                if (not .029 * height <= h <= .079 * height or not .84 <= w / h <= 1.16
                        or abs(top + y + h / 2 - center_y) > .29 * h
                        or not 0 < x < x + w < gray.shape[1] or not 0 < y < y + h < len(gray)
                        or not .36 <= cv2.contourArea(contour) / (w * h) <= .68):
                    continue
                poly = cv2.approxPolyDP(contour, .062 * cv2.arcLength(contour, True), True)
                if len(poly) != 4 or not cv2.isContourConvex(poly):
                    continue
                xs, ys = poly[:, 0].T
                if (max(abs(ys[xs.argmin()] - y - h / 2), abs(ys[xs.argmax()] - y - h / 2)) > .17 * h
                        or max(abs(xs[ys.argmin()] - x - w / 2), abs(xs[ys.argmax()] - x - w / 2)) > .19 * w):
                    continue
                yield inverse, level, (left + x, top + y, left + x + w, top + y + h)


def _marks(gray, box, inverse):
    x, y, right, bottom = box
    tile = gray[y:bottom, x:right]
    if tile.shape[0] < 26:
        tile = cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    tile = tile.astype(np.int16)
    h, w = tile.shape
    yy, xx = np.ogrid[:h, :w]
    dx, dy = abs((xx + .5) / w - .5), abs((yy + .5) / h - .5)
    face = (dx + dy <= .36) & (dx >= .19)
    middle = (dx <= .25) & (dy <= .35) & (dx + dy <= .46)
    paint = np.median(tile[face])
    signal = (tile - paint) if inverse else (paint - tile)
    scale = np.percentile(signal[middle], 94)
    if scale < 27:
        return None
    found = []
    for fraction in (.38, .49, .61):
        ink = np.uint8((signal >= max(17, fraction * scale)) & middle)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
        marks = [(i, a, b, p, q, count) for i, (a, b, p, q, count) in enumerate(stats[1:], 1)
                 if count >= max(3, .007 * w * h)]
        if len(marks) != 2:
            continue
        main, second = sorted(marks, key=lambda item: item[2])
        i, a, b, p, q, count = main
        _, c, d, r, s, pixels = second
        if (not .102 * h <= b <= .35 * h or not .035 * h <= d - b - q <= .19 * h
                or not .61 * h <= d + s <= .86 * h
                or max(abs(a + p / 2 - w / 2), abs(c + r / 2 - w / 2)) > .11 * w):
            continue
        if (.075 * w <= p <= .21 * w and .077 * h <= q <= .21 * h and .61 <= p / q <= 1.52
                and .073 * w <= r <= .22 * w and .28 * h <= s <= .52 * h
                and pixels >= .73 * r * s and abs(c + r / 2 - a - p / 2) <= .075 * w):
            found.append("i")
        if (.20 * w <= p <= .47 * w and .26 * h <= q <= .55 * h
                and .063 * w <= r <= .24 * w and .061 * h <= s <= .23 * h and .49 <= r / s <= 2.1):
            hook = labels[b:b + q, a:a + p] == i
            first = hook[:max(1, round(.25 * q))]
            last = hook[max(1, round(.72 * q)):]
            first_x = np.nonzero(first)[1]
            end_x = np.nonzero(last)[1]
            if (first_x.size and end_x.size and np.ptp(first_x) >= .61 * (p - 1)
                    and np.ptp(end_x) <= .63 * (p - 1) and .31 * p <= end_x.mean() <= .73 * p
                    and .18 <= count / (p * q) <= .70):
                found.append("?")
    return next((name for name in ("?", "i") if found.count(name) >= 2), None)


def _title(gray, back, badge):
    left, top, right, bottom = badge
    size = bottom - top
    start = back[2] + max(3, round(.24 * size))
    end = left - max(2, round(.14 * size))
    if not 2.5 * size <= end - start <= 10 * size:
        return False
    y0, y1 = max(0, top - round(.48 * size)), min(len(gray), bottom + round(.43 * size))
    title = gray[y0:y1, start:end]
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, round(.58 * size) | 1),) * 2)
    center = (top + bottom) / 2 - y0
    for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
        delta = cv2.morphologyEx(title, operation, element)
        for level in (32, 51, 73):
            _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.compare(delta, level, cv2.CMP_GE))
            glyphs = sorted((x, x + w) for x, y, w, h, count in stats[1:]
                            if .20 * size <= w <= 2.5 * size and .36 * size <= h <= 1.38 * size
                            and abs(y + h / 2 - center) <= .46 * size and count >= .11 * w * h)
            if (len(glyphs) >= 3 and glyphs[-1][1] - glyphs[0][0] >= 2.32 * size
                    and glyphs[0][0] <= .85 * size and title.shape[1] - glyphs[-1][1] <= .58 * size):
                return True
    return False


def landscape_heading_help_actions(frame, regions, known):
    """Locate a help marker on the same header row as a measured Back control."""
    height, width = frame.shape[:2]
    if (regions.get("layer") != "page" or regions.get("regions")
            or height < 240 or not 1.52 * height <= width <= 2.98 * height):
        return []
    backs = [item["box"] for item in known if item["role"] == "back" and item["box"][2] <= .155 * width
             and item["box"][3] <= .114 * height and item["box"][3] - item["box"][1] >= .024 * height]
    if len(backs) != 1:
        return []
    back = backs[0]
    left = back[2] + round(.043 * height)
    right = round(.393 * width)
    bottom = round(.125 * height)
    gray = cv2.cvtColor(frame[:bottom, :right], cv2.COLOR_BGR2GRAY)
    matches = list(_diamonds(gray[:, left:], left, 0, height, (back[1] + back[3]) / 2))
    stable = [(inverse, box) for inverse, level, box in matches if any(
        sign == inverse and cut != level and _overlap(box, peer) >= .62 for sign, cut, peer in matches)]
    found = []
    for inverse, box in sorted(dict.fromkeys(stable), key=lambda item: item[1][3] - item[1][1], reverse=True):
        if any(_overlap(box, prior["box"]) >= .55 for prior in found):
            continue
        mark = _marks(gray, box, inverse)
        if mark and _title(gray, back, box):
            x, y, right, bottom = box
            point = [(x + right) / 2, (y + bottom) / 2]
            if not any(a <= point[0] < c and b <= point[1] < d for a, b, c, d in (item["box"] for item in known)):
                found.append({"role": "help", "function": "查看规则" if mark == "?" else "查看说明",
                              "preview_text": "规则" if mark == "?" else "说明",
                              "box": list(box), "click_point": point})
    return found if len(found) == 1 else []
