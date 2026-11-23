"""Recognize measured drag handles on an active frequency-response graph."""

from collections import Counter
from functools import lru_cache
import re

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_font_glyphs import FontGlyphIndex
from screen_landscape_search_field import _available


_FREQUENCY = re.compile(r"([1-9][0-9]{0,4})([kK]?)(?:[hH][zZ])?")


def _center(box):
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def _disks(frame):
    height, width = frame.shape[:2]
    tile = frame[round(.115 * height):round(.92 * height)]
    start = round(.115 * height)
    b, g, r = cv2.split(tile)
    maximum = cv2.max(cv2.max(b, g), r)
    minimum = cv2.min(cv2.min(b, g), r)
    mask = cv2.bitwise_and(cv2.compare(maximum, 95, cv2.CMP_GE),
                           cv2.compare(cv2.subtract(maximum, minimum), 66, cv2.CMP_GE))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    circles = []
    for i, (x, y, w, h, count) in enumerate(stats[1:], 1):
        if (not .016 * width <= min(w, h) <= max(w, h) <= .049 * width
                or not .82 <= w / h <= 1.19 or not .66 <= count / (w * h) <= .87):
            continue
        observed = labels[y:y + h, x:x + w] == i
        ys, xs = np.ogrid[:h, :w]
        circular = ((xs - (w - 1) / 2) / (.49 * w)) ** 2 + ((ys - (h - 1) / 2) / (.49 * h)) ** 2 <= 1
        if (observed & circular).sum() < .86 * (observed | circular).sum():
            continue
        interior = ((xs - (w - 1) / 2) / (.33 * w)) ** 2 + ((ys - (h - 1) / 2) / (.33 * h)) ** 2 <= 1
        colors = tile[y:y + h, x:x + w][interior & observed]
        color = np.median(colors, axis=0)
        if (np.abs(colors - color).max(axis=1) <= 24).mean() >= .83:
            box = (int(x), start + int(y), int(x + w), start + int(y + h))
            circles.append((box, color))
    return circles


def _row(circles, width):
    groups = {}
    for source, color in circles:
        size = source[3] - source[1]
        _, cy = _center(source)
        group = sorted((box for box, tone in circles if np.max(np.abs(tone - color)) <= 29
                        and .76 * size <= box[3] - box[1] <= 1.24 * size
                        and abs(_center(box)[1] - cy) <= .19 * width), key=lambda box: box[0])
        if not 6 <= len(group) <= 16:
            continue
        xs, ys = np.asarray(list(map(_center, group))).T
        gaps = np.diff(xs)
        pitch = float(np.median(gaps))
        if (xs[-1] - xs[0] >= .62 * width and .041 * width <= pitch <= .17 * width
                and np.max(np.abs(gaps - pitch)) <= .066 * pitch
                and .20 * size <= np.ptp(ys) <= 2.5 * pitch):
            groups[tuple(group)] = (color, pitch)
    if len(groups) == 1:
        return next(iter(groups.items()))
    return None


def _curve(frame, nodes, tone, pitch):
    for first, second in zip(nodes[:-1], nodes[1:], strict=True):
        ax, ay = _center(first)
        bx, by = _center(second)
        size = first[3] - first[1]
        left = first[2] + max(2, round(.14 * size))
        right = second[0] - max(2, round(.14 * size))
        top = max(0, round(min(ay, by) - .40 * size))
        bottom = min(frame.shape[0], round(max(ay, by) + .40 * size) + 1)
        ink = np.abs(frame[top:bottom, left:right].astype(np.int16) - tone).max(axis=2) <= 58
        if ink.shape[1] < .37 * pitch:
            return False
        along = ink.sum(axis=0)
        best = np.linspace(ay, by, right - left)
        ys, xs = np.nonzero(ink)
        if ((along > 0).mean() < .86 or np.percentile(along, 80) > .48 * size
                or (np.abs(top + ys - best[xs]) <= .56 * size).mean() < .87):
            return False
    return True


def _grid(gray, nodes, pitch):
    xs, ys = np.rint([_center(box) for box in nodes]).astype(int).T
    if xs[0] < .38 * pitch or xs[-1] + .38 * pitch >= gray.shape[1]:
        return None
    left, right = max(0, round(ys.min() - 4.4 * pitch)), min(len(gray), round(ys.max() + 4.4 * pitch))
    tile = gray[left:right].astype(np.int16)
    inset = max(1, round(.025 * pitch))
    flank = round(.12 * pitch)
    lines = np.maximum.reduce([tile[:, xs - inset], tile[:, xs], tile[:, xs + inset]])
    backdrop = (tile[:, xs - flank] + tile[:, xs + flank]) / 2
    shared = ((lines - backdrop) >= 3).mean(axis=1) >= .67
    bridge = np.ones((max(3, round(.21 * pitch)), 1), np.uint8)
    aligned = cv2.morphologyEx(np.uint8(shared[:, None]), cv2.MORPH_CLOSE, bridge).ravel()
    grids = [(left + a, left + b) for a, b in _bands(aligned)
             if left + a < ys.min() - pitch and left + b > ys.max() + pitch
             and b - a >= 3.0 * pitch and shared[a:b].mean() >= .66]
    return grids[0] if len(grids) == 1 else None


def _power_mark(frame, tone, grid_top, regions):
    width = frame.shape[1]
    tile = frame[:grid_top - round(.09 * width)]
    if not tile.size:
        return False
    b, g, r = (cv2.absdiff(plane, int(color)) for plane, color in zip(cv2.split(tile), tone, strict=True))
    mask = np.uint8(cv2.max(cv2.max(b, g), r) <= 35)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    for x, y, w, h, count in stats[1:]:
        if (x < .69 * width or not .062 * width <= h <= .162 * width or not .84 <= w / h <= 1.19
                or not .59 <= count / (w * h) <= .84 or y + h >= grid_top - .18 * width):
            continue
        mark = tile[y:y + h, x:x + w]
        gray = cv2.cvtColor(mark, cv2.COLOR_BGR2GRAY)
        ink = np.uint8((gray >= 190) & (np.ptp(mark, axis=2) <= 82))
        _, ids, parts, _ = cv2.connectedComponentsWithStats(ink)
        for i, (a, t, sw, sh, area) in enumerate(parts[1:], 1):
            if (not .31 * w <= sw <= .53 * w or not .27 * h <= sh <= .55 * h
                    or not .17 <= area / (sw * sh) <= .44 or abs(a + sw / 2 - w / 2) > .08 * w):
                continue
            loop = ids[t:t + sh, a:a + sw] == i
            cap = max(1, round(.20 * sw))
            stem = any(.057 <= ww / sw <= .19 and .41 <= hh / sh <= .73
                       and -.28 <= (yy - t) / sh <= .10 and abs(xx + ww / 2 - a - sw / 2) <= .11 * sw
                       for xx, yy, ww, hh, _ in parts[1:])
            if (stem and loop[:round(.53 * sh), cap:-cap].mean() < .15
                    and loop[round(.76 * sh):, cap:-cap].any()
                    and _available((x, y, x + w, y + h), regions)):
                return True
    return False


def _axis_band(gray, grid, nodes, pitch, *, upper):
    height = len(gray)
    x = np.rint([_center(box)[0] for box in nodes]).astype(int)
    y = grid[0] if upper else grid[1]
    start = max(0, round(y - .99 * pitch)) if upper else y + round(.22 * pitch)
    end = y - round(.24 * pitch) if upper else min(height, round(y + 1.22 * pitch))
    reach = round(.35 * pitch)
    baseline = np.median(gray[start:end, ::5], axis=1)
    if not baseline.size:
        return None
    levels = [np.count_nonzero(np.abs(gray[start:end, cx - reach:cx + reach].astype(np.int16)
                                             - baseline[:, None]) >= 27, axis=1) >= 2 for cx in x]
    bands = [(start + t, start + b) for t, b in _bands(np.mean(levels, axis=0) >= .62)
             if .095 * pitch <= b - t <= .29 * pitch and 0 < t and start + b < end]
    return bands[0] if len(bands) == 1 else None


def _gain_owned(gray, band, nodes, pitch):
    top, bottom = band
    size = bottom - top
    pad = max(2, round(.16 * size))
    reach = round(.34 * pitch)
    confirmed = 0
    for node in nodes:
        x = round(_center(node)[0])
        tile = gray[top - pad:bottom + pad, x - reach:x + reach].astype(np.int16)
        ink = np.abs(tile - np.median(tile)) >= 47
        bands = _bands(ink.any(axis=0))
        if len(bands) != 2:
            continue
        (left, right), (a, b) = bands
        sign = ink[:, left:right]
        _, y, w, h = cv2.boundingRect(np.uint8(sign))
        core = sign[y:y + h]
        digit = cv2.boundingRect(np.uint8(ink[:, a:b]))[3]
        plus = (.71 <= w / max(h, 1) <= 1.34 and core.sum() and
                core.mean(axis=0).max() >= .79 and core.mean(axis=1).max() >= .79 and core.mean() <= .56)
        minus = w >= 2.8 * h and h <= .24 * size
        if 0 < h <= 1.0 * size and .69 * size <= digit <= 1.18 * size and (plus or minus):
            confirmed += 1
    return confirmed >= max(3, len(nodes) // 3)


@lru_cache(maxsize=4)
def _axis_index(fonts):
    return FontGlyphIndex(fonts=fonts, charset="0123456789kKhHzZ")


def _loop_six(ink):
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    holes = [cv2.boundingRect(contour) for contour, (_, _, _, parent) in zip(contours, hierarchy[0], strict=True)
             if parent >= 0]
    return len(holes) == 1 and holes[0][1] + .5 * holes[0][3] >= .58 * ink.shape[0]


def _axis_glyph(raw, bank, height):
    x, y, w, h = cv2.boundingRect(raw)
    if not .66 * height <= h <= height or not .19 <= w / h <= 1.013:
        return None
    glyph = np.ascontiguousarray(raw[y:y + h, x:x + w])
    ranks = bank.rank(glyph, alphabet="choice_latin", limit=4)
    if len(ranks) < 2:
        return None
    first = ranks[0]
    text, error = first["text"], first["distance"]
    if (text in {"5", "6"} and error <= .29 and any(item["text"] == "6" and item["distance"] <= .31
                                                for item in ranks[:2]) and _loop_six(glyph)):
        return "6"
    rival = next((item["distance"] for item in ranks[1:] if item["text"].casefold() != text.casefold()), 1)
    return text if error <= .244 and rival - error >= .014 else None


def _read_axis(gray, area, bank):
    x, y, right, bottom = area
    tile = gray[y:bottom, x:right].astype(np.int16)
    signal = np.abs(tile - np.median(tile))
    peak = np.percentile(signal, 98)
    if peak < 42:
        return None
    reports = Counter()
    seen = set()
    for level in (.25, .33, .41, .50, .59, .68, .75):
        ink = np.uint8(signal >= level * peak)
        a, t, w, h = cv2.boundingRect(ink)
        if not 7 <= h <= .72 * len(tile) or not .37 * h <= w <= 4.2 * h:
            continue
        cut = ink[t:t + h, a:a + w]
        identity = cut.shape, cut.tobytes()
        if identity in seen:
            continue
        seen.add(identity)
        spans = _bands(cut.any(axis=0))
        if not 1 <= len(spans) <= 7:
            continue
        observed = [_axis_glyph(cut[:, left:right], bank, h) for left, right in spans]
        if all(observed) and (match := _FREQUENCY.fullmatch("".join(observed))):
            number, prefix = match.groups()
            hz = int(number) * (1000 if prefix else 1)
            if 15 <= hz <= 30000:
                reports[hz, number + ("kHz" if prefix else "Hz")] += 1
    verified = [label for label, count in reports.items() if count >= 2]
    return verified[0] if len(verified) == 1 else None


def equalizer_curve_actions(frame, regions, index):
    """Return one vertical drag origin per independently read node."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or width < 320 or not 1.43 * width <= height <= 2.70 * width:
        return []
    if width > 900:
        overview = cv2.resize(frame, (700, round(700 * height / width)), interpolation=cv2.INTER_AREA)
        if len(_disks(overview)) < 6:
            return []
    if (lattice := _row(_disks(frame), width)) is None:
        return []
    nodes, (tone, pitch) = lattice
    if not _curve(frame, nodes, tone, pitch):
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (grid := _grid(gray, nodes, pitch)) is None or not _power_mark(frame, tone, grid[0], regions):
        return []
    lower = _axis_band(gray, grid, nodes, pitch, upper=False)
    upper = _axis_band(gray, grid, nodes, pitch, upper=True)
    if lower is None or upper is None or not _gain_owned(gray, upper, nodes, pitch):
        return []
    found = []
    bank = _axis_index(index.fonts)
    xpad, ypad = round(.37 * pitch), round(.10 * pitch)
    for i, box in enumerate(nodes):
        cx, _ = _center(box)
        area = (max(0, round(cx) - xpad), lower[0] - ypad, min(width, round(cx) + xpad), lower[1] + ypad)
        if not _available(box, regions) or not _available(area, regions):
            continue
        if (label := _read_axis(gray, area, bank)) is not None:
            found.append((i, box, *label))
    if len(found) < max(5, len(nodes) - 2):
        return []
    ratios = [(following[2] / previous[2]) ** (1 / (following[0] - previous[0]))
              for previous, following in zip(found[:-1], found[1:], strict=True)]
    if not (1.45 <= np.median(ratios) <= 2.52 and max(ratios) - min(ratios) <= .21):
        return []
    return [{"role": "adjust_slider", "function": f"拖动{label}频段滑块", "preview_text": label,
             "setting_label": label, "setting_display_label": label, "slider_axis": "vertical",
             "slider_thumb_box": list(box), "box": list(box), "click_point": list(_center(box))}
            for _, box, _, label in found]
