"""Own a marked strip of text tabs attached to a landscape viewport edge."""

from collections import Counter, defaultdict

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts, read_caption


def _slots(candidates, width, height):
    eligible = []
    for item in candidates:
        left, top, right, bottom = item["box"]
        if (item.get("visibility") != "page" or not 0 <= left < right <= width
                or not 0 <= top < bottom <= height or bottom < .966 * height
                or not .055 * height <= bottom - top <= .19 * height):
            continue
        eligible.append((left, top, right, bottom))
    for count in range(3, 7):
        step = width / count
        rows = []
        for slot in range(count):
            start, end = slot * step, (slot + 1) * step
            matching = [box for box in eligible if abs(box[0] - start) <= .125 * step
                        and abs(box[2] - end) <= .125 * step]
            if not matching:
                break
            rows.append(min(matching, key=lambda box: abs(box[0] - start) + abs(box[2] - end)))
        if len(rows) == count and np.ptp([box[1] for box in rows]) <= .026 * height:
            yield step, rows


def _seams(gray, step, slots):
    height, width = gray.shape
    before = max(2, round(.006 * height))
    typical = round(np.median([box[1] for box in slots]))
    bottom = round(np.median([box[3] for box in slots]))
    pad = max(2, round(.036 * width))
    marks = []
    for y in range(typical - round(.028 * height), typical + round(.026 * height)):
        if not before <= y < height - before:
            continue
        row, upper, lower = gray[y, pad:-pad], gray[y - before, pad:-pad], gray[y + before, pad:-pad]
        rise = np.maximum(row - np.maximum(upper, lower), np.minimum(upper, lower) - row)
        if (rise >= 5).mean() >= .88:
            marks.append((float(np.median(rise)), y))
    if not marks:
        return None
    strength, top = max(marks)
    depth = bottom - top
    if strength < 11 or not .055 * height <= depth <= .19 * height:
        return None
    clip = gray[top + round(.26 * depth):bottom - round(.16 * depth)]
    radius = max(2, round(.041 * step))
    boundaries = [0]
    for i in range(1, len(slots)):
        center = round(i * step)
        changes = abs(np.diff(clip[:, center - radius:center + radius + 1], axis=1))
        support = (changes >= 3).mean(axis=0)
        strongest = int(np.argmax(support))
        if support[strongest] < .69:
            return None
        boundaries.append(center - radius + strongest + 1)
    boundaries.append(width)
    if np.ptp(np.diff(boundaries)) > .10 * step:
        return None
    return top, bottom, boundaries


def _selected(gray, top, bottom, boundaries):
    depth = bottom - top
    band = gray[top + max(2, round(.045 * depth)):top + round(.29 * depth)]
    scores = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        center = (left + right) // 2
        core, near, far = (max(2, round(factor * depth)) for factor in (.255, .47, 1.07))
        if center - far < left or center + far >= right:
            return None
        color = np.concatenate((band[:, center - far:center - near], band[:, center + near:center + far]), axis=1)
        background = np.median(color, axis=1, keepdims=True)
        delta = band[:, center - core:center + core + 1] - background
        scores.append((float(np.percentile(delta, 30)), float((delta >= 19).mean())))
    possible = [i for i, (strength, coverage) in enumerate(scores) if strength >= 19 and coverage >= .66]
    if len(possible) != 1:
        return None
    best = possible[0]
    return best if all(strength < 10 and coverage < .34 for i, (strength, coverage) in enumerate(scores) if i != best) else None


def _caption(tile):
    low, high = np.percentile(tile, (50, 99))
    if high - low < 30:
        return None
    mask = np.uint8(tile >= low + .49 * (high - low))
    x, y, width, height = cv2.boundingRect(mask)
    if (not .27 * tile.shape[0] <= height <= .76 * tile.shape[0]
            or not .67 * height <= width <= .94 * tile.shape[1]
            or abs(x + width / 2 - tile.shape[1] / 2) > .11 * tile.shape[1]):
        return None
    return (x, y, x + width, y + height), low, high


def _cjk_title(tile, index, low, high):
    reports = defaultdict(dict)
    for level in (.60, .64, .68):
        mask = np.uint8(tile >= low + level * (high - low))
        x, y, width, height = cv2.boundingRect(mask)
        cut = mask[y:y + height, x:x + width]
        parts = _cjk_parts(cut, height) if height >= 10 else []
        if not 2 <= len(parts) <= 8:
            continue
        ranked = [index.rank(glyph, alphabet="cjk", limit=2) for *_, glyph in parts]
        if any(len(pair) != 2 or pair[0]["distance"] > .192
               or pair[1]["distance"] - pair[0]["distance"] < .004 for pair in ranked):
            continue
        text = "".join(pair[0]["text"] for pair in ranked)
        reports[text][(cut.shape, cut.tobytes())] = min(pair[1]["distance"] - pair[0]["distance"] for pair in ranked)
    reliable = [text for text, cuts in reports.items() if len(cuts) >= 2 and np.mean(list(cuts.values())) >= .015]
    return reliable[0] if len(reliable) == 1 else None


def _tab_actions(frame, gray, index, top, bottom, boundaries, selected):
    depth = bottom - top
    start, end = top + round(.27 * depth), bottom - round(.095 * depth)
    actions = []
    shape_count = 0
    cache = {}
    for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        if i == selected:
            continue
        center = (left + right) // 2
        reach = min(round(.39 * (right - left)), round(1.52 * depth))
        tile = gray[start:end, center - reach:center + reach]
        shape = _caption(tile)
        if shape is None:
            continue
        shape_count += 1
        (x, y, right_x, bottom_y), low, high = shape
        label = _cjk_title(tile, index, low, high)
        if not label:
            pad = max(2, round(.235 * (bottom_y - y)))
            offset = center - reach
            caption = (max(offset, offset + x - pad), max(start, start + y - pad),
                       min(center + reach, offset + right_x + pad), min(end, start + bottom_y + pad))
            reading = read_caption(frame, caption, index, cache=cache, choice=True)
            label = reading["text"] if reading else None
        if label:
            actions.append({"role": "game_settings_tab", "function": f"切换{label}", "preview_text": label,
                            "box": [left, top, right, bottom], "click_point": [center, (top + bottom) // 2]})
    if shape_count != len(boundaries) - 2:
        return []
    counts = Counter(action["function"] for action in actions)
    return [action for action in actions if counts[action["function"]] == 1]


def landscape_footer_tab_actions(frame, regions, index):
    """Return literally named inactive tabs after the rail owns their pixels."""
    height, width = frame.shape[:2]
    if index is None or height < 220 or width < 1.45 * height or regions["layer"] != "page":
        return []
    groups = list(_slots(regions["candidates"], width, height))
    if not groups:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    rails = [rail for step, slots in groups if (rail := _seams(gray, step, slots)) is not None]
    if len(rails) != 1:
        return []
    top, bottom, boundaries = rails[0]
    selected = _selected(gray, top, bottom, boundaries)
    return _tab_actions(frame, gray, index, top, bottom, boundaries, selected) if selected is not None else []
