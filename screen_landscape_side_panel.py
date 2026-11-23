"""Verify a full-canvas landscape modal with a vertical settings rail."""

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_wide_panel import _diagonal_cross
from screen_wide_panel_checks import _soft_letter


def _rule(gray, start, stop):
    height, width = gray.shape
    gap = max(2, round(.0043 * height))
    strip = gray[:, round(.027 * width):round(.912 * width):3].astype(np.int16)
    ridge = strip[gap:-gap] - np.maximum(strip[:-2 * gap], strip[2 * gap:])
    support = np.pad((ridge >= 12).mean(axis=1), (gap, gap))
    flat = np.pad(strip[gap:-gap].std(axis=1), (gap, gap), constant_values=255)
    mask = (support >= .84) & (flat <= 10)
    rows = np.arange(height)
    spans = _bands(mask & (rows >= start) & (rows < stop))
    if len(spans) != 1 or spans[0][1] - spans[0][0] > max(3, round(.013 * height)):
        return None
    a, b = spans[0]
    return int(a + np.argmax(strip[a:b].mean(axis=1)))


def _divider(gray, header):
    height, width = gray.shape
    a, b = round(.904 * width), round(.979 * width)
    gap = max(2, round(.002 * width))
    reach = max(4, round(.019 * height))
    field = gray[header + reach:height - reach:2, a - gap:b + gap].astype(np.int16)
    ridge = field[:, gap:-gap] - np.maximum(field[:, :-2 * gap], field[:, 2 * gap:])
    supported = (ridge >= 11).mean(axis=0)
    edges = _bands(supported >= .82)
    if len(edges) != 1 or edges[0][1] - edges[0][0] > 2 * gap:
        return None
    l, r = edges[0]
    return a + l, a + r


def _close(gray, header, divider):
    width = gray.shape[1]
    rail_width = width - divider
    first, last = round(.16 * header), round(.83 * header)
    left, right = divider + round(.10 * rail_width), width - max(2, round(.072 * rail_width))
    roi = gray[first:last, left:right]
    if min(roi.shape) < 10 or np.median(roi) > 70:
        return None
    options = []
    for level in (105, 123, 142, 154, 168):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(roi >= level))
        for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if (not .19 * rail_width <= min(w, h) <= max(w, h) <= .47 * rail_width
                    or not .84 <= w / h <= 1.18 or not .28 <= area / (w * h) <= .62
                    or abs(left + x + w / 2 - (divider + width) / 2) > .15 * rail_width
                    or not _diagonal_cross(labels[y:y + h, x:x + w] == i)):
                continue
            options.append((level, (int(left + x), int(first + y), int(left + x + w), int(first + y + h))))
    stable = [box for level, box in options if any(
        other != level and max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= 2 for other, peer in options)]
    if not stable or any(max(abs(a - b) for a, b in zip(stable[0], peer, strict=True)) > 2 for peer in stable):
        return None
    return stable[len(stable) // 2]


def _vertical_letters(gray, header, edge):
    height, width = gray.shape
    left, seam = edge
    rail_width = width - left
    start, stop = seam + round(.08 * rail_width), width - round(.055 * rail_width)
    field = gray[header:, start:stop].astype(np.float32)
    flank = max(2, round(.17 * field.shape[1]))
    paint = np.median(np.concatenate((field[:, :flank], field[:, -flank:]), axis=1), axis=1)
    contrast = np.maximum(0, field - np.float32(paint[:, None]))
    margin = round(.18 * field.shape[1])
    interior = contrast[:, margin:-margin]
    trials = []
    for level in (52, 65, 75, 85):
        occupied = (interior >= level).sum(axis=1) >= max(2, round(.043 * rail_width))
        records = []
        for a, b in _bands(occupied):
            box = cv2.boundingRect(np.uint8(interior[a:b] >= level))
            x, y, w, h = box
            if (.19 * rail_width <= min(w, h) <= max(w, h) <= .47 * rail_width
                    and abs(x + w / 2 - interior.shape[1] / 2) <= .13 * rail_width):
                records.append((a, b))
        groups = []
        for a, b in records:
            if groups and a - groups[-1][-1][1] <= .18 * rail_width:
                groups[-1].append((a, b))
            else:
                groups.append([(a, b)])
        if not 3 <= len(groups) <= 5 or any(not 2 <= len(group) <= 3 for group in groups):
            continue
        centers = [(g[0][0] + g[-1][1]) / 2 for g in groups]
        pitch = np.diff(centers)
        if (pitch.min() < .70 * pitch.max() or np.ptp(pitch) > .048 * (height - header)
                or centers[0] > .73 * pitch[0] or height - header - centers[-1] > .78 * pitch[-1]):
            continue
        trials.append((groups, centers))
    if not trials or any(len(trials[0][0]) != len(groups) or
                         max(abs(a - b) for a, b in zip(trials[0][1], centers, strict=True)) > .12 * rail_width
                         for groups, centers in trials):
        return None
    groups, centers = trials[len(trials) // 2]
    return interior, groups, centers


def _selected(gray, header, edge, centers, contrast, groups):
    left, right = edge
    colors, labels = [], []
    reach = round(.20 * float(np.median(np.diff(centers))))
    for c, group in zip(centers, groups, strict=True):
        y = header + round(c)
        colors.append(float(np.median(gray[y - reach:y + reach, left:right])))
        sample = contrast[group[0][0]:group[-1][1]]
        labels.append(float(np.percentile(sample[sample >= 48], 86)))
    i = int(np.argmax(colors))
    others = [c for j, c in enumerate(colors) if j != i]
    return i if colors[i] - max(others) >= 34 and labels[i] - max(
        c for j, c in enumerate(labels) if j != i) >= 23 else None


def _letter(contrast, start, stop, index):
    first, last = max(0, start - 2), min(len(contrast), stop + 2)
    crop = contrast[first:last]
    peak = float(np.percentile(crop, 99))
    if peak < 52:
        return None
    first_mask, second_mask = (crop >= cut * peak for cut in (.26, .44))
    ranks = [index.rank(mask, alphabet="cjk", limit=7) for mask in (first_mask, second_mask)]
    if (not np.array_equal(first_mask, second_mask) and all(len(row) >= 2 for row in ranks)
            and ranks[0][0]["text"] == ranks[1][0]["text"]
            and max(row[0]["distance"] for row in ranks) <= .128
            and min(row[1]["distance"] - row[0]["distance"] for row in ranks) >= .021):
        return ranks[0][0]["text"]
    ranks.extend(index.rank(crop >= cut * peak, alphabet="cjk", limit=7) for cut in (.35, .53, .62, .66, .70))
    options = tuple(sorted({item["text"] for row in ranks for item in row}))
    paths = tuple(index.fonts if index.fonts is not None else index.cjk_sources)
    answers = {}
    for level in (.30, .38, .46):
        x, y, w, h = cv2.boundingRect(np.uint8(crop >= level * peak))
        if not 8 <= h <= 44 or not .65 <= w / h <= 1.40:
            continue
        target = np.float32(np.pad(np.minimum(1, crop[y:y + h, x:x + w] / peak), 2))
        key = (x, y, w, h)
        if key not in answers:
            answers[key] = _soft_letter(paths, options, target.tobytes(), w, h)
    verified = [value for value in answers.values() if value is not None]
    if not verified or len(set(verified)) != 1:
        return None
    char = verified[0]
    if len(verified) >= 2 or any(len(row) >= 2 and row[0]["text"] == char and row[0]["distance"] <= .26
                                 and row[1]["distance"] - row[0]["distance"] >= .014 for row in ranks):
        return char
    return None


def landscape_side_panel_actions(frame, regions, index):
    """Return a measured full-screen layer and its visibly grounded actions."""
    height, width = frame.shape[:2]
    if regions["layer"] not in {"page", "layer_uncertain"} or height < 280 or width < 1.67 * height:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    header = _rule(gray, .075 * height, .176 * height)
    if header is None:
        return None
    lower = _rule(gray, .775 * height, .931 * height)
    if lower is None or lower - header < .63 * height:
        return None
    edge = _divider(gray, header)
    if edge is None:
        return None
    close = _close(gray, header, edge[0])
    rail = _vertical_letters(gray, header, edge)
    if close is None or rail is None:
        return None
    contrast, groups, centers = rail
    selected = _selected(gray, header, edge, centers, contrast, groups)
    if selected is None:
        return None
    x, t, r, b = close
    actions = [{"role": "close", "function": "关闭", "box": [x, t, r, b],
                "click_point": [(x + r) // 2, (t + b) // 2]}]
    if index is not None:
        boundaries = [header, *(header + round((a + b) / 2) for a, b in zip(centers[:-1], centers[1:], strict=True)),
                      height]
        for i, (group, center) in enumerate(zip(groups, centers, strict=True)):
            if selected == i:
                continue
            letters = [_letter(contrast, a, b, index) for a, b in group]
            if all(letters):
                label = "".join(letters)
                actions.append({"role": "game_settings_tab", "function": label, "preview_text": label,
                                "box": [edge[1], boundaries[i], width, boundaries[i + 1]],
                                "click_point": [(edge[1] + width) // 2, header + round(center)]})
    return (0, 0, width, height), actions
