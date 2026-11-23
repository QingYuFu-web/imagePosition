"""Read binary settings with two adjacent rectangular choices and a printed heading."""

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from screen_component_captions import _bands
from screen_font_glyphs import FontGlyphIndex
from screen_landscape_panel_switches import _glyph
from screen_landscape_search_field import _available
from screen_wide_panel_checks import _clipped_bitmap, _read_glyph
from screen_landscape_settings_rail import _literal_word


_SWITCH_CAPTIONS = tuple((
    "自动拾取 自动拾取手枪 自动拾取侧面瞄准镜 自动装配瞄准镜 停止自动拾取 自动拾取三级包 自动拾取瞄准镜 "
    "自动打开 自动瞄准 自动装填 自动拾取武器 自动装备 自动装备配件 自动录制 自动更新 自动播放 自动翻译 "
    "自动保存 自动开启 自动锁定 自动瞄准目标 恢复默认 自动同步 自动显示 自动切换 画面设置 声音设置"
).split())


@dataclass(frozen=True)
class SwitchPair:
    left: int
    top: int
    middle: int
    right: int
    bottom: int
    selected: str

    @property
    def width(self):
        return self.middle - self.left

    @property
    def height(self):
        return self.bottom - self.top


@lru_cache(maxsize=4)
def _common_index(paths):
    # GB 2312 level-one glyphs are ordered by reading, independent of app labels.
    common = b"".join(bytes((lead, column)) for lead in range(0xB0, 0xD8)
                      for column in range(0xA1, 0xFF)).decode("gb2312", errors="ignore")
    return FontGlyphIndex(fonts=paths, charset=common)


def _red(frame):
    blue, green, red = cv2.split(frame)
    return (red >= 85) & (cv2.subtract(red, cv2.max(blue, green)) >= 28)


def _plates(gray):
    height, width = gray.shape
    contours, _ = cv2.findContours(cv2.Canny(gray, 12, 30), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        left, top, w, h = cv2.boundingRect(contour)
        if (.040 * width <= w <= .145 * width and .045 * height <= h <= .127 * height
                and 1.62 <= w / h <= 3.55 and cv2.contourArea(contour) >= .89 * w * h):
            boxes.append((left, top, left + w, top + h))
    return sorted(boxes, key=lambda p: (p[2] - p[0]) * (p[3] - p[1]), reverse=True)


def _profile(gray, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    margin = max(3, round(.13 * h))
    if top < margin or bottom > gray.shape[0] - margin:
        return None
    inset = round(.20 * w)
    rows = np.median(gray[top - margin:bottom + margin, left + inset:right - inset], axis=1)
    inside = np.median(rows[margin + round(.25 * h):margin + round(.75 * h)])
    return (inside, min(np.median(rows[:2]), np.median(rows[-2:])),
            min(np.max(rows[margin - 1:margin + 3]), np.max(rows[-margin - 3:-margin + 1])))


def _word(gray, red, box, word, index):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    x = left + round(.22 * w)
    y = top + round(.17 * h)
    tile = gray[y:bottom - (y - top), x:right - (x - left)].astype(np.float32)
    covered = red[top:bottom, left:right]
    if covered.mean() > .025 or covered.mean(axis=1).max() > .28 or covered.mean(axis=0).max() > .34:
        return False
    contrast = np.maximum(0, tile - np.median(tile))
    peak = float(np.percentile(contrast, 99))
    if peak < 73 or np.percentile(tile, 99) < 149:
        return False
    _, _, ink_w, ink_h = cv2.boundingRect(np.uint8(contrast >= .40 * peak))
    if not .31 * h <= ink_h <= .68 * h or not .69 <= ink_w / ink_h <= 1.29:
        return False
    ranks = [index.rank(contrast >= level * peak, alphabet="cjk", limit=3) for level in (.34, .43, .52)]
    if word == "关":
        return sum(len(rank) >= 2 and rank[0]["text"] == word and rank[0]["distance"] <= .180
                   and rank[1]["distance"] - rank[0]["distance"] >= .027 for rank in ranks) >= 2
    # At small sizes the two top strokes of 幵 can fuse. Verify the horizontal
    # cap and the two separating lower strokes independently of font ranking.
    if sum(any(item["text"] == word and item["distance"] < .168 for item in rank[:2]) for rank in ranks) < 2:
        return False
    for level in (.36, .44):
        ink = contrast >= level * peak
        a, t, w, h = cv2.boundingRect(np.uint8(ink))
        ink = ink[t:t + h, a:a + w]
        horizontal = ink.mean(axis=1)
        lower = ink[round(.67 * h):].mean(axis=0) >= .32
        if (horizontal[:max(2, round(.15 * h))].max() >= .71
                and horizontal[round(.31 * h):round(.67 * h)].max() >= .76
                and len(_bands(lower)) == 2):
            return True
    return False


def _pairs(gray, red, index):
    result = []
    for x, y, right, bottom in _plates(gray):
        selected = x, y, right, bottom
        measured = _profile(gray, selected)
        if measured is None or measured[2] - measured[0] < 20:
            continue
        w = right - x
        for left, middle, end, choice in ((x - w, x, right, "开"), (x, right, right + w, "关")):
            if (left < 0 or end > gray.shape[1]
                    or any(abs(old.left - left) <= .13 * w and abs(old.top - y) <= .13 * (bottom - y)
                           for old in result)):
                continue
            first, second = (left, y, middle, bottom), (middle, y, end, bottom)
            other = _profile(gray, first if choice == "开" else second)
            if (other is None or measured[0] - other[0] < 15 or other[1] - other[0] < 7
                    or not _word(gray, red, first, "关", index) or not _word(gray, red, second, "开", index)):
                continue
            result.append(SwitchPair(left, y, middle, end, bottom, choice))
    return result


def _peers(pairs, pair):
    for other in pairs:
        if (other.selected != pair.selected and .92 * pair.width <= other.width <= 1.08 * pair.width
                and .84 * pair.height <= other.height <= 1.16 * pair.height
                and abs(other.left - pair.left) <= .13 * pair.width
                and abs(other.right - pair.right) <= .13 * pair.width
                and .88 * pair.height <= abs(other.top - pair.top) <= 4.55 * pair.height):
            return True
    return False


def _remove_help_icons(contrast, threshold, height):
    # The help circle may sit between the baselines of a wrapped heading.
    ink = np.uint8(contrast >= threshold)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        a, t, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if (.33 * height <= h <= .52 * height and .8 <= w / h <= 1.25
                and .60 <= area / (w * h) <= .84 and perimeter and 4 * np.pi * area / perimeter**2 >= .77):
            if .25 < ink[t:t + h, a:a + w].mean() < .62:
                contrast[t:t + h, a:a + w] = 0


@lru_cache(maxsize=1760)
def _rasters(paths, char, width, height):
    result = []
    for path in paths:
        bitmap = _clipped_bitmap(path, 36, char)
        if bitmap is not None and abs(np.log(bitmap.shape[1] * height / (bitmap.shape[0] * width))) <= .32:
            sample = cv2.resize(bitmap, (width, height), interpolation=cv2.INTER_AREA)
            result.append(sample)
    return _normalized(np.float32(result).reshape(-1, width * height))


def _normalized(rows):
    centered = rows - rows.mean(axis=1, keepdims=True)
    return centered / np.maximum(1e-5, np.linalg.norm(centered, axis=1, keepdims=True))


def _compare_fonts(stamp, paths, chars, width, height):
    scores = np.full(len(chars), -1., np.float32)
    for dw, dh in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):
        w, h = width + dw, height + dh
        windows = sliding_window_view(stamp, (h, w)).reshape(-1, w * h)
        windows = _normalized(windows)
        fonts = [_rasters(paths, char, w, h) for char in chars]
        counts = np.array([len(font) for font in fonts])
        available = counts > 0
        if not available.any():
            continue
        matched = np.einsum("ij,kj->ik", np.concatenate(fonts), windows).max(axis=1)
        starts = np.r_[0, counts[available].cumsum()[:-1]]
        scores[available] = np.maximum(scores[available], np.maximum.reduceat(matched, starts))
    return sorted(((float(score), char) for score, char in zip(scores, chars, strict=True) if score > -1),
                  reverse=True)[:2]


def _fonts(index):
    return tuple(index.fonts if index.fonts is not None else index.cjk_sources)


@lru_cache(maxsize=256)
def _letter(index, shape, pixels, *, common=False):
    glyph = np.frombuffer(pixels, np.float32).reshape(shape)
    glyph = np.maximum(glyph - np.percentile(glyph, 20), 0).astype(np.float32)
    peak = float(np.percentile(glyph, 99))
    levels = (.28, .43) if common else (.28, .37, .46, .55, .64)
    ranks = [index.rank(glyph >= level * peak, alphabet="cjk", limit=6) for level in levels]
    if (all(len(row) >= 2 for row in ranks) and len({row[0]["text"] for row in ranks}) == 1
            and max(row[0]["distance"] for row in ranks) <= .128
            and min(row[1]["distance"] - row[0]["distance"] for row in ranks) >= .033):
        return ranks[0][0]["text"]
    chosen = tuple(sorted({item["text"] for row in ranks for item in row}))
    paths = _fonts(index)
    patches = []
    for level in (.23, .31, .40):
        x, y, w, h = cv2.boundingRect(np.uint8(glyph >= level * peak))
        if min(w, h) >= 7:
            stamp = glyph[y:y + h, x:x + w] / peak
            floor = .075 + .30 * (level - .23)
            patches.append((w, h, np.pad(np.clip((stamp - floor) / (.95 - floor), 0, 1), 2)))
    rankings = [_compare_fonts(stamp, paths, chosen, w, h) for w, h, stamp in patches]
    votes = [(scores[0][1], scores[0][0], scores[0][0] - scores[1][0]) for scores in rankings
             if len(scores) == 2 and scores[0][0] >= .700]
    confirmed = {text for text, score, gap in votes if score >= (.785 if common else .747) and gap >= .025}
    if len(confirmed) != 1:
        return None
    text = confirmed.pop()
    gaps = [gap for candidate, _, gap in votes if candidate == text]
    return text if len(gaps) >= 2 and (sum(gap >= .025 for gap in gaps) >= 2 or max(gaps) >= .072) else None


def _imprint(tile):
    ink = np.maximum(tile - np.percentile(tile, 20), 0)
    peak = float(np.percentile(ink, 99))
    if peak < 32:
        return ()
    views = []
    for level in (.23, .32, .41):
        x, y, w, h = cv2.boundingRect(np.uint8(ink >= level * peak))
        glyph = cv2.resize(ink[y:y + h, x:x + w] / peak, (24, 24), interpolation=cv2.INTER_LINEAR)
        views.append((w / h, _normalized(glyph.reshape(1, -1))[0], glyph.ravel() >= .45))
    return views


def _repeat_letter(imprint, observed):
    matches = {}
    for read, source in observed:
        overlap = 0.
        for ar, first, first_ink in imprint:
            for br, second, second_ink in source:
                if abs(np.log(ar / br)) <= .13 and float(first @ second) >= .953:
                    union = np.count_nonzero(first_ink | second_ink)
                    if union:
                        overlap = max(overlap, np.count_nonzero(first_ink & second_ink) / union)
        if overlap >= .78:
            matches.setdefault(read, []).append(overlap)
    if len(matches) == 1:
        read, overlaps = next(iter(matches.items()))
        if len(overlaps) >= 2 and max(overlaps) >= .82:
            return read
    return None


@lru_cache(maxsize=96)
def _isolated_letter(index, common_index, shape, pixels):
    glyph = np.frombuffer(pixels, np.float32).reshape(shape)
    peak = float(np.percentile(glyph, 99))
    narrow = common_index or index
    first = _glyph(glyph, peak, narrow)
    if first is not None and first == _read_glyph(glyph, peak, index):
        return first
    return None


def _read_line(contrast, index, common_index, observed):
    contrast = np.maximum(contrast - np.median(contrast), 0)
    ink = contrast >= max(41, .37 * float(np.percentile(contrast, 99)))
    positions = _bands(ink.any(axis=0))
    if not positions:
        return None
    h = cv2.boundingRect(np.uint8(ink))[3]
    groups = []
    for a, b in positions:
        if groups and a - groups[-1][1] <= .10 * h and b - groups[-1][0] <= 1.13 * h:
            groups[-1] = (groups[-1][0], b)
        else:
            groups.append((a, b))
    if not 2 <= len(groups) <= 21:
        return None
    text = []
    missing = None
    for a, b in groups:
        glyph = contrast[:, a:b]
        peak = float(np.percentile(glyph, 99))
        local = glyph >= .34 * peak
        _, t, w, v = cv2.boundingRect(np.uint8(local))
        if w <= .25 * h and v <= .46 * h and t >= .48 * h:
            text.append("，")
        elif .58 * h <= w <= 1.43 * h and .62 * h <= v <= 1.12 * h:
            tile = contrast[:, max(0, a - 1):b + 1]
            pixels = tile.tobytes()
            imprint = _imprint(tile)
            if not imprint:
                return None
            letter = _repeat_letter(imprint, observed)
            if letter is None:
                if common_index is not None:
                    letter = _letter(common_index, tile.shape, pixels, common=True)
                if letter is None:
                    letter = _letter(index, tile.shape, pixels)
                if letter is not None:
                    observed.append((letter, imprint))
            if letter is None:
                if missing is not None:
                    return None
                missing = len(text), glyph
            text.append(letter)
        else:
            return None
    if missing is not None:
        position, glyph = missing
        letter = _isolated_letter(index, common_index, glyph.shape, glyph.tobytes())
        if letter is None:
            return None
        text[position] = letter
    return "".join(text).strip("，"), groups[0][0], groups[-1][1]


def _caption(gray, red, pair, index, common_index, observed, *, aligned_left=None):
    x = max(0, pair.left - round(2.9 * pair.width))
    if aligned_left is not None:
        x = max(x, aligned_left - round(.24 * pair.height))
    end = pair.left - round(.12 * pair.width)
    y = max(0, pair.top - round(.10 * pair.height))
    bottom = min(gray.shape[0], pair.bottom + round(.10 * pair.height))
    if end <= x:
        return None
    tile = gray[y:bottom, x:end].astype(np.float32)
    contrast = np.maximum(tile - np.median(tile), 0)
    contrast[red[y:bottom, x:end]] = 0
    _remove_help_icons(contrast, max(37, .28 * float(np.percentile(contrast, 99))), pair.height)
    ink = contrast >= max(37, .29 * float(np.percentile(contrast, 99)))
    rows = _bands(ink.sum(axis=1) >= max(4, round(.156 * pair.height)))
    lines = [(a, b) for a, b in rows if .26 * pair.height <= b - a <= .49 * pair.height]
    if not 1 <= len(lines) <= 2:
        return None
    if any(a <= 0 or b >= len(ink) for a, b in lines):
        return None
    readings = [_read_line(contrast[a - 1:b + 1], index, common_index, observed) for a, b in lines]
    if not all(readings):
        if len(lines) != 1 or aligned_left is None:
            return None
        a, b = lines[0]
        chip = np.uint8(contrast[a - 1:b + 2])
        decoded = _literal_word(chip, (0, 0, chip.shape[1], chip.shape[0]), _SWITCH_CAPTIONS, index)
        if decoded is None:
            return None
        text, (left, _, right, _) = decoded
        readings = [(text, left, right)]
    left, right = x + min(a for _, a, _ in readings), x + max(b for _, _, b in readings)
    top, bottom = y + lines[0][0], y + lines[-1][1]
    if (red[top:bottom, left:right].mean() > .025
            or aligned_left is not None and abs(left - aligned_left) > .24 * pair.height):
        return None
    return "".join(label for label, _, _ in readings), (left, top, right, bottom)


def landscape_paired_switch_actions(frame, regions, index, *, selected_boxes=None):
    """Read nonselected choices, and optionally report measured selected halves."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or height < 260 or width < 1.55 * height:
        return []
    from screen_grouped_switch_captions import grouped_switch_actions

    return grouped_switch_actions(frame, regions, index, selected_boxes=selected_boxes)
