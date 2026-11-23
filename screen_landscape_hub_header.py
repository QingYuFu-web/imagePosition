"""Measure hooked navigation arrows and painted peer choices in landscape headers."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_candidate_glyphs import _overlap
from screen_colored_settings_panel import _printed
from screen_landscape_header import _text_run


_TAB_CAPTIONS = ("好友", "全国", "公会", "榜单", "战区", "世界", "全球", "联盟", "区服", "全服", "本服", "本区",
                 "好友榜", "排行榜", "好友排行", "赛季", "附近", "本地", "系统", "地区", "个人", "团队", "一键",
                 "推荐", "全部", "玩法", "我的", "成就", "朋友", "排行榜单", "日期", "社会", "成员", "好友列表")


def _hook(stroke):
    height, width = stroke.shape
    if not 1.62 <= width / height <= 3.23 or not .20 <= stroke.mean() <= .47:
        return False
    part = cv2.resize(np.uint8(stroke) * 255, (48, 30), interpolation=cv2.INTER_AREA) >= 128
    left, middle, right = part[:, :9], part[:, 19:35], part[:, -10:]
    # The point faces left and is low on the shaft. A long lower stroke
    # curls upwards at the opposite end and returns across the top.
    return (not left[:15].any() and left[-8:].mean() >= .23
            and np.max(part[-8:, :41].mean(axis=1)) >= .85
            and middle[8:19].mean() <= .15 and max(middle[:8].mean(axis=1)) >= .42
            and (right[5:24].mean(axis=1) >= .29).mean() >= .72
            and part[-6:, :24].mean() >= .61)


def _hook_back(frame):
    height, width = frame.shape[:2]
    start, end = round(.013 * width), round(.33 * width)
    crop = frame[:round(.118 * height), start:end]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blue, green, red = cv2.split(crop)
    maximum = cv2.max(cv2.max(blue, green), red)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, round(.024 * height) | 1),) * 2)
    relief = cv2.morphologyEx(maximum, cv2.MORPH_TOPHAT, kernel)
    observed = []
    for level in (23, 27, 31, 42, 54, 66):
        mask = cv2.compare(relief, level, cv2.CMP_GE)
        _, drawn, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i, (x, y, w, h, _) in enumerate(stats[1:], 1):
            if (.028 * height <= h <= .077 * height and .029 * width <= x + start + w / 2 <= .115 * width
                    and .008 * height <= y <= .052 * height and y + h < len(gray)
                    and _hook(drawn[y:y + h, x:x + w] == i)):
                observed.append((level, (int(x), int(y), int(x + w), int(y + h))))
    stable = [box for level, box in observed if any(other != level and _overlap(box, peer) >= .75
              for other, peer in observed)]
    if not stable or any(_overlap(box, stable[0]) < .70 for box in stable):
        return None
    left, top, right, bottom = stable[len(stable) // 2]
    size = bottom - top
    title = (right + round(.28 * size), max(0, top - round(.25 * size)),
             min(crop.shape[1], right + round(7.2 * size)), min(len(crop), bottom + round(.24 * size)))
    contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    if not _text_run(contrast, title, size, (top + bottom) / 2, 3):
        return None
    return [left + start, top, right + start, bottom]


def _face(frame, box):
    left, top, right, bottom = box
    patch = frame[top:bottom, left:right]
    height, width = patch.shape[:2]
    inner = patch[round(.18 * height):round(.82 * height), round(.06 * width):round(.24 * width)]
    outer = patch[round(.18 * height):round(.82 * height), round(.79 * width):round(.94 * width)]
    if min(inner.shape[:2]) < 4 or min(outer.shape[:2]) < 4:
        return None
    colors = np.concatenate((inner, outer), axis=1).astype(np.int16)
    tone = np.median(colors, axis=(0, 1))
    if (abs(colors - tone).max(axis=2) <= 16).mean() < .84:
        return None
    x, r = left + round(.10 * width), right - round(.10 * width)
    y, b = max(0, top - round(.16 * height)), min(len(frame), bottom + round(.15 * height))
    heights = frame[y:b, x:r].astype(np.int16)
    edges = np.max(abs(heights[2:] - heights[:-2]), axis=2)
    margin = max(2, round(.30 * height))
    if min(np.max((edges >= 8).mean(axis=1)[:margin]), np.max((edges >= 8).mean(axis=1)[-margin:])) < .78:
        return None
    return tone


def _caption(frame, window, index, cache):
    left, top, right, bottom = window
    chip = frame[top:bottom, left:right]
    gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
    area = (0, 0, gray.shape[1], gray.shape[0])
    terms = [_printed(channel, area, index, vocabulary=_TAB_CAPTIONS, minimum_score=.85, minimum_margin=.041)
             for channel in (gray, chip.min(axis=2))]
    if terms[0] is not None and terms[0] == terms[1]:
        return terms[0]
    preferred = read_caption(frame, window, index, cache=cache, choice=True, join_radicals=True)
    if not preferred or preferred["support"] < 3 or preferred["min_margin"] < .022 or preferred["max_distance"] > .183:
        return None
    if any(name not in (None, preferred["text"]) for name in terms):
        return None
    x, y, r, d = preferred["line_box"]
    h = bottom - top
    if abs((x + r - left - right) / 2) > .24 * h or abs((y + d - top - bottom) / 2) > .22 * h:
        return None
    return preferred["text"]


def _tab_pair(frame, regions, anchor, index):
    height, width = frame.shape[:2]
    if index is None:
        return []
    candidates = []
    for row in regions["candidates"]:
        left, top, right, bottom = box = row["box"]
        w, h = right - left, bottom - top
        if (row.get("visibility") == "page" and .62 * width <= left < right <= .987 * width
                and .037 * height <= h <= .082 * height and 2.08 * h <= w <= 4.65 * h
                and abs((top + bottom - anchor[1] - anchor[3]) / 2) <= .37 * h
                and 0 < top < bottom <= .104 * height):
            candidates.append(tuple(box))
    pairs = []
    for left, top, right, bottom in candidates:
        w, h = right - left, bottom - top
        for x, y, z, end in candidates:
            if not (-.09 * h <= x - right <= .27 * h and abs(z - x - w) <= .16 * w
                    and max(abs(y - top), abs(end - bottom)) <= .19 * h):
                continue
            peers = ((left, top, right, bottom), (x, y, z, end))
            tones = [_face(frame, box) for box in peers]
            if any(tone is None for tone in tones):
                continue
            selected = int(tones[1].mean() > tones[0].mean())
            delta = tones[selected] - tones[1 - selected]
            if delta.mean() >= 21 and np.ptp(delta) >= 12 and np.min(delta) >= 5:
                pairs.append((peers, selected))
    if len(pairs) != 1:
        return []
    peers, selected = pairs[0]
    captions = []
    cache = {}
    for left, top, right, bottom in peers:
        h = bottom - top
        center = (left + right) / 2
        reach = round(.94 * h)
        window = (round(center - reach), top + round(.13 * h), round(center + reach), bottom - round(.10 * h))
        label = _caption(frame, window, index, cache)
        if not label or not 2 <= len(label) <= 6 or not all("\u3400" <= char <= "\u9fff" for char in label):
            return []
        captions.append(label)
    if captions[0] == captions[1]:
        return []
    left, top, right, bottom = peers[1 - selected]
    name = captions[1 - selected]
    return [{"role": "switch_tab", "function": f"切换到{name}", "preview_text": name,
             "box": [left, top, right, bottom], "click_point": [(left + right) / 2, (top + bottom) / 2]}]


def landscape_hub_header_actions(frame, regions, index):
    """Return the leading hooked Back arrow and verified inactive header tabs."""
    height, width = frame.shape[:2]
    if (regions.get("layer") != "page" or regions.get("regions") or height < 235
            or not 1.55 * height <= width <= 2.95 * height):
        return []
    arrow = _hook_back(frame)
    if arrow is None:
        return []
    a, b, c, d = arrow
    back = {"role": "back", "function": "返回", "preview_text": "返回", "box": arrow,
            "click_point": [(a + c) / 2, (b + d) / 2]}
    return [back, *_tab_pair(frame, regions, arrow, index)]
