"""Measure switch groups, then verify row captions with local fonts."""

from functools import lru_cache

import cv2
import numpy as np

from screen_landscape_paired_switches import (
    SwitchPair, _SWITCH_CAPTIONS, _available, _bands, _caption, _common_index, _fonts,
    _peers, _plates, _profile, _read_line, _red, _remove_help_icons, _word,
)
from screen_landscape_settings_rail import _GLYPH_SIZE, _glyph_scores


_CAPTIONS = tuple(dict.fromkeys((*_SWITCH_CAPTIONS, *(f"自动拾取{level}级包" for level in "一二三四五六"), *((
    "自动加载 自动连接 自动定位 自动加速 自动登录 自动刷新 自动示警 自动推荐 自动使用 自动补充 "
    "允许通知 消息通知 智能推荐 镜头抖动 镜头跟随 显示帧率 智能瞄准 手动瞄准 自动寻路 音效模式 "
    "自动拾取药品 自动拾取物品 自动拾取道具 自动拾取配件 自动拾取弹药 自动拾取防具 自动拾取装备 "
    "侧面瞄准镜 自动装配 侧面瞄准器 瞄准镜 拾取镜 瞄准器 高级设置 基础设置 镜头设置 操作设置 "
    "收起拾取列表时停止自动拾取 展开拾取列表时继续自动拾取 显示拾取列表时开始自动拾取 "
    "收起设置列表时关闭自动播放 展开设置列表时保留自动播放"
).split()))))


def _verified_off(gray, red, box, index):
    if _word(gray, red, box, "关", index):
        return True
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    px, py = round(.22 * w), round(.17 * h)
    if red[top:bottom, left:right].mean() >= .02:
        return False
    tile = gray[top + py:bottom - py, left + px:right - px].astype(np.float32)
    ink = np.maximum(0, tile - np.median(tile))
    peak = float(np.percentile(ink, 99))
    _, _, glyph_w, glyph_h = cv2.boundingRect(np.uint8(ink >= .4 * peak))
    if (peak < 90 or np.max(tile) < 180 or not .36 * h <= glyph_h <= .62 * h
            or not .77 * glyph_h <= glyph_w <= 1.23 * glyph_h):
        return False
    ranks = [index.rank(ink >= q * peak, alphabet="cjk", limit=2) for q in (.28, .34, .43)]
    return (all(len(row) == 2 and row[0]["text"] == "关" for row in ranks)
            and all(row[0]["distance"] <= .125 and row[1]["distance"] - row[0]["distance"] >= .040
                    for row in ranks[:2]))


def _shape_pairs(gray, red, index):
    proposals = []
    for x, y, right, bottom in _plates(gray):
        selected = _profile(gray, (x, y, right, bottom))
        if selected is None or selected[2] - selected[0] < 20:
            continue
        w = right - x
        for left, middle, end, choice in ((x - w, x, right, "开"), (x, right, right + w, "关")):
            if left < 0 or end > gray.shape[1]:
                continue
            first, second = (left, y, middle, bottom), (middle, y, end, bottom)
            other = _profile(gray, first if choice == "开" else second)
            if other and selected[0] - other[0] >= 15 and other[1] - other[0] >= 7:
                proposals.append(SwitchPair(left, y, middle, end, bottom, choice))
    group = [pair for pair in proposals if _peers(proposals, pair)]
    verified = []
    for pair in group:
        if any(abs(other.left - pair.left) <= .13 * pair.width
               and abs(other.top - pair.top) <= .13 * pair.height for other in verified):
            continue
        x, y, mid, end, bottom = pair.left, pair.top, pair.middle, pair.right, pair.bottom
        if _verified_off(gray, red, (x, y, mid, bottom), index) and _word(gray, red, (mid, y, end, bottom), "开", index):
            verified.append(pair)
    return [pair for pair in verified if _peers(verified, pair)]


def _caption_ink(gray, red, pair):
    left = max(0, pair.left - round(2.9 * pair.width))
    right = pair.left - round(.12 * pair.width)
    top, bottom = max(0, pair.top - round(.10 * pair.height)), pair.bottom + round(.10 * pair.height)
    raw = gray[top:bottom, left:right].astype(np.float32)
    if not raw.size:
        return None
    ink = np.maximum(0, raw - np.median(raw))
    ink[red[top:bottom, left:right]] = 0
    peak = float(np.percentile(ink, 99))
    _remove_help_icons(ink, max(37, .28 * peak), pair.height)
    mask = np.uint8(ink >= max(33, .28 * peak))
    _, owners, parts, _ = cv2.connectedComponentsWithStats(mask)
    _, component_y, w, h, n = parts.T
    loose = ((w < .90 * pair.height) & (h < .58 * pair.height) & (n >= 3)
             & (component_y > 0) & (component_y + h < len(ink)))
    loose[0] = True
    ink[~loose[owners]] = 0
    mask[~loose[owners]] = 0
    rows = _bands(mask.sum(axis=1) >= max(4, round(.144 * pair.height)))
    lines = [(a, b) for a, b in rows if .26 * pair.height <= b - a <= .50 * pair.height]
    if not 1 <= len(lines) <= 2 or any(a < 2 or b >= len(ink) - 1 for a, b in lines):
        return None
    return left, top, ink, lines


def _groups(mask, height):
    groups = []
    for a, b in _bands(mask.any(axis=0)):
        if groups and a - groups[-1][1] <= .10 * height and b - groups[-1][0] <= 1.15 * height:
            groups[-1] = groups[-1][0], b
        else:
            groups.append((a, b))
    if not 2 <= len(groups) <= 20:
        return None
    stretches = []
    for a, b in groups:
        _, v, w, h = cv2.boundingRect(mask[:, a:b])
        if w <= .25 * height and h <= .47 * height and v >= .46 * height:
            continue
        if not .56 * height <= min(w, h) <= max(w, h) <= 1.25 * height:
            return None
        stretches.append((a, b))
    if len(stretches) < 2:
        return None
    widths = np.array([(a + b) / 2 for a, b in stretches])
    if np.any(np.diff(widths) < .70 * height) or np.median(np.diff(widths)) > 1.44 * height:
        return None
    return stretches


@lru_cache(maxsize=32)
def _words(count):
    return tuple(word for word in _CAPTIONS if len(word) == count)


def _rank_cut(line, peak, ratio, paths):
    mask = np.uint8(line >= max(30, ratio * peak))
    _, _, _, height = cv2.boundingRect(mask)
    if not 11 <= height <= 44 or not (spans := _groups(mask, height)):
        return None
    options = _words(len(spans))
    if len(options) < 2:
        return None
    letters = []
    for i, (a, b) in enumerate(spans):
        glyph = line[:, a:b]
        cap = float(np.percentile(glyph, 97))
        l, t, w, h = cv2.boundingRect(np.uint8(glyph >= .29 * cap))
        if cap < .47 * peak or not .56 * height <= min(w, h) <= max(w, h) <= 1.28 * height:
            return None
        stamp = np.minimum(1, glyph[t:t + h, l:l + w] / cap).astype(np.float32)
        stamp = cv2.resize(stamp, (_GLYPH_SIZE, _GLYPH_SIZE), interpolation=cv2.INTER_LINEAR)
        chars = tuple(sorted({word[i] for word in options}))
        letters.append(_glyph_scores(paths, chars, np.pad(stamp, 2).tobytes()))
    ranked = []
    for word in options:
        scores = tuple(letters[i][char] for i, char in enumerate(word))
        ranked.append((float(np.mean(scores)), min(scores), word))
    first, second = sorted(ranked, reverse=True)[:2]
    if first[0] >= .91 and first[1] >= .830 and first[0] - second[0] >= .012:
        return first[2], spans[0][0], spans[-1][1]
    return None


def _whole_caption(geometry, index):
    left, top, ink, lines = geometry
    sources = tuple(index.fonts if index.fonts is not None else index.cjk_sources)
    readings = []
    for a, b in lines:
        line = ink[a - 1:b + 1]
        peak = float(np.percentile(line, 99))
        if peak < 55:
            return None
        matches = [found for cut in (.32, .36, .40, .44)
                   if (found := _rank_cut(line, peak, cut, sources)) is not None]
        if len(matches) < 2 or len({found[0] for found in matches}) != 1:
            return None
        readings.append(matches[-1])
    if np.ptp([x for _, x, _ in readings]) > .23 * (lines[0][1] - lines[0][0]):
        return None
    return ("".join(name for name, _, _ in readings),
            (left + min(x for _, x, _ in readings), top + lines[0][0],
             left + max(x for _, _, x in readings), top + lines[-1][1]))


def _aligned_transcript(geometry, pair, index, common, witnesses, anchor):
    if geometry is None:
        return None
    left, top, ink, lines = geometry
    clip = max(0, anchor - left - round(.24 * pair.height))
    readings = [_read_line(ink[a - 1:b + 1, clip:], index, common, witnesses) for a, b in lines]
    if not all(readings):
        return None
    label_left = left + clip + min(x for _, x, _ in readings)
    if abs(label_left - anchor) > .24 * pair.height:
        return None
    return ("".join(name for name, _, _ in readings),
            (label_left, top + lines[0][0], left + clip + max(x for _, _, x in readings), top + lines[-1][1]))


def grouped_switch_actions(frame, regions, index, *, selected_boxes=None):
    """Resolve simultaneous measured row geometry, then decode captions."""
    if index is None or regions.get("layer") != "page" or frame.shape[0] < 260 or frame.shape[1] < 1.55 * frame.shape[0]:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    red = _red(frame)
    common = _common_index(_fonts(index)) if index.charset is None and _fonts(index) else None
    pairs = _shape_pairs(gray, red, common or index)
    observed, accepted, pending = [], {}, []
    for pair in sorted(pairs, key=lambda box: (box.top, box.left)):
        if not _available((pair.left, pair.top, pair.right, pair.bottom), regions):
            continue
        if selected_boxes is not None:
            left = pair.middle if pair.selected == "开" else pair.left
            selected_boxes.append((left, pair.top, left + pair.width, pair.bottom))
        geometry = _caption_ink(gray, red, pair)
        fast = _whole_caption(geometry, index) if geometry is not None else None
        caption = fast
        witnesses = observed.copy()
        if caption is None:
            caption = _caption(gray, red, pair, index, common, witnesses)
        if caption is None:
            pending.append((pair, geometry))
            continue
        if not _available(caption[1], regions):
            continue
        observed = witnesses
        accepted[pair] = caption
    first_pass = tuple(accepted.items())
    for pair, geometry in pending:
        anchors = [heading[0] for peer, (_, heading) in first_pass
                   if abs(peer.left - pair.left) <= .13 * pair.width
                   and .92 * pair.width <= peer.width <= 1.08 * pair.width
                   and abs(peer.top - pair.top) <= 3.2 * pair.height]
        if not anchors or max(anchors) - min(anchors) > .24 * pair.height:
            continue
        reading = _aligned_transcript(geometry, pair, index, common, observed.copy(), round(np.median(anchors)))
        if reading is not None and _available(reading[1], regions):
            accepted[pair] = reading
    result = []
    for pair, (label, heading) in accepted.items():
        target = pair.left if pair.selected == "开" else pair.middle
        verb = "关闭" if pair.selected == "开" else "开启"
        result.append({"role": "select_option", "function": f"{verb}{label.replace('，', '')}", "preview_text": label,
                       "box": list(map(int, (heading[0], min(heading[1], pair.top), pair.right, pair.bottom))),
                       "click_point": [int(target + pair.width // 2), int((pair.top + pair.bottom) // 2)]})
    return sorted(result, key=lambda item: (item["click_point"][1], item["click_point"][0]))
