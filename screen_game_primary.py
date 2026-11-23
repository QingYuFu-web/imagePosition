"""Verify the printed primary command on a landscape game screen."""

from collections import Counter

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts


_COMMANDS = ("进入比赛", "开始比赛", "开始匹配", "开始游戏", "立即开始", "进入游戏", "开始战斗", "开始挑战")


def _border(tile, start, stop, inset, sign):
    """Locate a continuous bright boundary across the middle of one edge."""
    gradients = sign * (tile[start + 1:stop + 1, inset:-inset] - tile[start - 1:stop - 1, inset:-inset])
    strength = np.median(gradients, axis=1)
    offset, channel = np.unravel_index(strength.argmax(), strength.shape)
    if strength[offset, channel] < 35 or (gradients[offset, :, channel] >= 25).mean() < .82:
        return None
    return start + int(offset)


def _button_box(frame, candidate):
    left, top, right, bottom = candidate
    width, height = right - left, bottom - top
    pad = max(3, round(.18 * height))
    frame_height, frame_width = frame.shape[:2]
    if min(left, top, frame_width - right, frame_height - bottom) < pad:
        return None
    tile = frame[top - pad:bottom + pad, left - pad:right + pad].astype(np.int16)
    across = tile.transpose(1, 0, 2)
    bottom_edge = pad + height
    right_edge = pad + width
    short_inset = pad + round(.18 * height)
    long_inset = pad + round(.12 * width)
    t = _border(tile, 1, 2 * pad, long_inset, 1)
    b = _border(tile, bottom_edge - pad, bottom_edge + pad - 1, long_inset, -1)
    l = _border(across, 1, 2 * pad, short_inset, 1)
    r = _border(across, right_edge - pad, right_edge + pad - 1, short_inset, -1)
    if None in (l, t, r, b) or not (.76 * height <= b - t <= 1.19 * height and
                                     .88 * width <= r - l <= 1.14 * width):
        return None
    return [left - pad + l, top - pad + t, left - pad + r + 1, top - pad + b + 1]


def _matches(command, ranks):
    if len(command) != len(ranks):
        return False
    anchored = 0
    for glyph, choices in zip(command, ranks, strict=True):
        if len(choices) < 2:
            return False
        best = choices[0]
        best_score = best["distance"]
        score = next((choice["distance"] for choice in choices if choice["text"] == glyph), 1.)
        if score > .205 or score - best_score > .045:
            return False
        anchored += best["text"] == glyph and choices[1]["distance"] - best_score >= .025
    return anchored >= (len(command) + 1) // 2


def _printed_command(frame, box, index):
    left, top, right, bottom = box
    height, width = bottom - top, right - left
    x_pad, y_pad = round(.18 * width), round(.15 * height)
    patch = frame[top + y_pad:bottom - y_pad, left + x_pad:right - x_pad]
    blue, green, red = cv2.split(patch)
    light = cv2.min(cv2.min(blue, green), red)
    tone, ink = np.percentile(light, (50, 95))
    if ink - tone < 67:
        return None
    votes, cache, seen = Counter(), {}, set()
    for fraction in (.29, .40, .49, .56):
        mask = np.uint8(light >= round(tone + fraction * (ink - tone)))
        x, y, w, h = cv2.boundingRect(mask)
        if (not .23 * height <= h <= .63 * height or abs(x + w / 2 - patch.shape[1] / 2) > .09 * width
                or abs(y + h / 2 - patch.shape[0] / 2) > .15 * height
                or min(x, y, patch.shape[1] - x - w, patch.shape[0] - y - h) < 1):
            continue
        raster = x, y, w, h, mask[y:y + h, x:x + w].tobytes()
        if raster in seen:
            continue
        seen.add(raster)
        parts = _cjk_parts(mask[y:y + h, x:x + w], h)
        if len(parts) != 4:
            continue
        ranks = []
        for *_, bitmap in parts:
            key = bitmap.shape, bitmap.tobytes()
            if key not in cache:
                cache[key] = index.rank(bitmap, alphabet="cjk", limit=4)
            ranks.append(cache[key])
        names = [command for command in _COMMANDS if _matches(command, ranks)]
        if len(names) == 1:
            votes[names[0]] += 1
    if len(votes) != 1:
        return None
    command, observations = votes.most_common(1)[0]
    return command if observations >= 2 else None


def game_primary_actions(frame, regions, index, existing):
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or width < 1.42 * height:
        return []
    candidates = []
    for candidate in regions["candidates"]:
        left, top, right, bottom = (round(value) for value in candidate["box"])
        w, h = right - left, bottom - top
        if (candidate["visibility"] == "page" and .82 * height <= (top + bottom) / 2 <= .96 * height
                and abs((left + right) / 2 - width / 2) <= .055 * width
                and .052 * height <= h <= .12 * height and .13 * width <= w <= .36 * width
                and 2.65 * h <= w <= 5.1 * h):
            candidates.append([left, top, right, bottom])
    if len(candidates) != 1 or (box := _button_box(frame, candidates[0])) is None:
        return []
    left, top, right, bottom = box
    if any(left <= item["click_point"][0] <= right and top <= item["click_point"][1] <= bottom for item in existing):
        return []
    if (command := _printed_command(frame, box, index)) is None:
        return []
    return [{"role": "game_primary", "function": command, "preview_text": command, "box": box,
             "click_point": [(left + right) // 2, (top + bottom) // 2]}]
