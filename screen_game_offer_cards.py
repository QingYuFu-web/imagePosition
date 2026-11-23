"""Verify a row of complete game offer cards with independently attached details."""

from collections import Counter

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts


def _rising_edge(values, begin, end):
    indices = np.arange(max(2, begin), min(len(values) - 1, end))
    if not indices.size:
        return 0, 0
    change = np.minimum(values[indices] - values[indices - 2],
                        values[indices + 1] - values[indices - 2])
    strongest = int(np.argmax(change))
    return int(indices[strongest]), float(change[strongest])


def _center_span(strong, center):
    if not strong[center]:
        return None
    left = np.flatnonzero(~strong[:center])
    right = np.flatnonzero(~strong[center + 1:])
    return (int(left[-1]) + 1 if left.size else 0,
            center + 1 + int(right[0]) if right.size else len(strong))


def _frame_edges(gray, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    center = (left + right) // 2
    shoulder = (slice(left + round(.10 * width), left + round(.27 * width)),
                slice(right - round(.27 * width), right - round(.10 * width)))
    sides = [np.median(gray[:, band], axis=1) for band in shoulder]
    mid = np.median(gray[:, center - round(.11 * width):center + round(.11 * width)], axis=1)
    roofs = []
    for icon_top in range(max(2, top - round(.10 * height)), min(len(mid) - 1, top + round(.24 * height))):
        icon_score = min(mid[icon_top], mid[icon_top + 1]) - mid[icon_top - 2]
        if icon_score < 26:
            continue
        span = _center_span(gray[icon_top, left:right] - gray[icon_top - 2, left:right] >= 25, center - left)
        if span is None:
            continue
        icon_left, icon_right = span
        icon_width = icon_right - icon_left
        if (not .26 * width <= icon_width <= .54 * width
                or abs(left + (icon_left + icon_right) / 2 - center) > .052 * width):
            continue
        corners = [_rising_edge(line, icon_top + round(.035 * height), icon_top + round(.15 * height))
                   for line in sides]
        y_left, score_left = corners[0]
        y_right, score_right = corners[1]
        if min(score_left, score_right) >= 29 and abs(y_left - y_right) <= max(2, .025 * height):
            roofs.append((icon_score + min(score_left, score_right), icon_top, icon_width,
                          round((y_left + y_right) / 2)))
    if not roofs:
        return None
    _, icon_top, icon_width, shoulder_top = max(roofs)
    icon_bottom = icon_top + icon_width
    square_slack = max(3, round(.075 * icon_width))
    _, icon_base = _rising_edge(-mid, icon_bottom - square_slack, icon_bottom + square_slack)
    if icon_base < 17:
        return None
    tail = [_rising_edge(-line, bottom - round(.13 * height), bottom + round(.09 * height)) for line in sides]
    if min(score for _, score in tail) < 18 or abs(tail[0][0] - tail[1][0]) > max(2, .02 * height):
        return None
    body_bottom = round(np.mean([row for row, _ in tail]))
    sample_start, sample_end = shoulder_top + round(.20 * height), body_bottom - round(.15 * height)
    if sample_start >= sample_end:
        return None
    row = np.median(gray[sample_start:sample_end], axis=0)
    shift = round(.075 * width)
    body_left, left_score = _rising_edge(row, left - shift, left + shift)
    body_right, right_score = _rising_edge(-row, right - shift, right + shift)
    if min(left_score, right_score) < 28 or body_right - body_left < .76 * width:
        return None
    return (body_left, icon_top, body_right, body_bottom), icon_bottom, icon_width


def _continuous_ridge(strip, start, end, reach, threshold):
    peaks = []
    for y in range(start, end):
        contrast = strip[y] - strip[y - reach:y - 1].min(axis=0)
        if (contrast >= threshold).mean() >= .77:
            peaks.append((float(np.median(contrast)), -y))
    return -max(peaks)[1] if peaks else None


def _detached_details(gray, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    center = (left + right) // 2
    if bottom + round(.34 * height) >= len(gray):
        return None
    half = round(.195 * width)
    strip = gray[:, center - half:center + half]
    reach = max(2, round(.022 * height))
    roof = _continuous_ridge(strip, bottom + round(.042 * height), bottom + round(.15 * height), reach, 18)
    if roof is None:
        return None
    base = _continuous_ridge(strip, roof + round(.125 * height), roof + round(.232 * height), reach, 7)
    if base is None:
        return None
    near, far = round(.215 * width), round(.325 * width)
    body = gray[roof + round(.20 * (base - roof)):base - round(.20 * (base - roof)), center - far:center + far]
    contrast = (abs(body[:, 2:] - body[:, :-2]) >= 10).mean(axis=0)
    reach = far - near
    if min(contrast[:reach].max(), contrast[-reach:].max()) < .65:
        return None
    return roof, base


def _title_reads(patch, index, cuts, minimum_margin):
    low, high = np.percentile(patch, (50, 99))
    if high - low < 59:
        return Counter()
    reads, rasters = Counter(), set()
    for level in cuts:
        mask = np.uint8(patch >= low + level * (high - low))
        x, y, width, height = cv2.boundingRect(mask)
        if not .31 * patch.shape[0] <= height <= .79 * patch.shape[0]:
            continue
        if abs(x + width / 2 - patch.shape[1] / 2) > .09 * patch.shape[1]:
            continue
        line = mask[y:y + height, x:x + width]
        key = line.shape, line.tobytes()
        if key in rasters:
            continue
        rasters.add(key)
        parts = _cjk_parts(line, height)
        if not 2 <= len(parts) <= 10:
            continue
        ranks = [index.rank(glyph, alphabet="cjk", limit=2) for *_, glyph in parts]
        if any(len(options) < 2 or options[0]["distance"] > .23
               or options[1]["distance"] - options[0]["distance"] < minimum_margin for options in ranks):
            continue
        reads["".join(options[0]["text"] for options in ranks)] += 1
    return reads


def _title(patch, index):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    reads = _title_reads(gray, index, (.49, .57), .011)
    supported = [name for name, count in reads.items() if count >= 2]
    if supported:
        return supported[0]

    # Interpolation can preserve subpixel strokes of a small compressed label.
    # Keep the two views separate so the same glyph mask cannot supply both votes.
    supported = set()
    for view in (gray, patch.min(axis=2)):
        enlarged = cv2.resize(view, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        reads = _title_reads(enlarged, index, (.43, .49, .57, .64), .015)
        supported.update(name for name, count in reads.items() if count >= 2)
    return supported.pop() if len(supported) == 1 else None


def game_offer_card_actions(frame, regions, index):
    """Return a point inside each literally named card in one verified row."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or index is None or height < 220 or width < 1.45 * height:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    offers = []
    for candidate in regions["candidates"]:
        left, top, right, bottom = candidate["box"]
        if candidate.get("visibility") != "page" or not (0 <= left < right <= width and 0 <= top < bottom <= height):
            continue
        w, h = right - left, bottom - top
        if not (.10 * width <= w <= .28 * width and .20 * height <= h <= .43 * height and .89 <= w / h <= 1.55):
            continue
        trace = _frame_edges(gray, (left, top, right, bottom))
        if trace is not None and (details := _detached_details(gray, trace[0])) is not None:
            offers.append((*trace, details))
    offers.sort()
    if not 3 <= len(offers) <= 8 or len({offer[0] for offer in offers}) != len(offers):
        return []
    boxes = np.array([offer[0] for offer in offers])
    widths = boxes[:, 2] - boxes[:, 0]
    step = np.diff((boxes[:, 0] + boxes[:, 2]) / 2)
    typical = np.median(widths)
    gaps = boxes[1:, 0] - boxes[:-1, 2]
    if (np.ptp(widths) > .072 * typical or np.ptp(boxes[:, 1]) > .040 * typical
            or np.ptp(boxes[:, 3]) > .045 * typical or np.ptp(step) > .075 * typical
            or min(gaps) < .015 * typical or max(gaps) > .185 * typical
            or np.ptp([details[0] for _, _, _, details in offers]) > .04 * typical):
        return []
    actions = []
    for (left, top, right, bottom), icon_bottom, icon_width, _ in offers:
        pad = round(.054 * (right - left))
        title_top = icon_bottom + round(.049 * icon_width)
        title_bottom = icon_bottom + round(.435 * icon_width)
        if title_bottom >= bottom:
            continue
        label = _title(frame[title_top:title_bottom, left + pad:right - pad], index)
        if label:
            actions.append({"role": "game_equipment_item", "function": f"选择{label}", "preview_text": label,
                            "box": [left, top, right, bottom],
                            "click_point": [(left + right) // 2, (title_top + title_bottom) // 2]})
    counts = Counter(action["function"] for action in actions)
    return [action for action in actions if counts[action["function"]] == 1]
