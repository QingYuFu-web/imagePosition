"""Read portrait search-result cards with owned artwork, rating, and watch controls."""

from functools import lru_cache

import cv2
import numpy as np
from PIL import ImageFont

from inventory_segmented_actions import _rounded_shape
from screen_colored_settings_panel import _printed
from screen_component_captions import _bands
from screen_font_glyphs import _font_raster, _normal
from screen_game_settings_back import _bold_left_angle


_TAB_WORDS = ("视频", "番剧", "直播间", "用户", "直播", "综合", "电影", "影片", "动漫", "课程", "文章", "影视", "动画")
_WATCH_WORDS = ("观看", "播放", "试看", "追番", "打开", "预约", "查看", "参加", "购买", "下载", "收藏")
_RATINGS = ("评分", "评价", "好评", "分数", "评论", "观看", "排行", "日期", "价格", "年代")


def _center(box):
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def _action(role, function, box, *, text=None):
    target = {"role": role, "function": function, "box": list(box), "click_point": list(_center(box))}
    if text:
        target["preview_text"] = text
    return target


def _project(box, sx, sy):
    return tuple(round(value * scale) for value, scale in zip(box, (sx, sy, sx, sy), strict=True))


def _posters(gray, paper):
    height, width = gray.shape
    lane = gray[:, :round(.355 * width)]
    kernel = np.ones((max(3, round(.011 * width)) | 1,) * 2, np.uint8)
    regions = cv2.morphologyEx(np.uint8(lane <= paper - 47), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for contour in contours:
        left, top, w, h = cv2.boundingRect(contour)
        right, bottom = left + w, top + h
        if (.017 * width <= left <= .087 * width and .208 * width <= w <= .325 * width
                and 1.22 * w <= h <= 1.59 * w and right < lane.shape[1]
                and .125 * height <= top < bottom < height and cv2.contourArea(contour) >= .85 * w * h):
            rows.append((left, top, right, bottom))
    rows.sort(key=lambda box: box[1])
    if len(rows) < 2:
        return []
    props = np.array(rows)
    w = float(np.median(props[:, 2] - props[:, 0]))
    h = float(np.median(props[:, 3] - props[:, 1]))
    pitch = np.diff(props[:, 1])
    if (np.ptp(props[:, 0]) > .042 * w or np.ptp(props[:, 2]) > .055 * w
            or np.ptp(props[:, 3] - props[:, 1]) > .12 * h or pitch.min() < 1.01 * h
            or pitch.max() > 1.46 * h):
        return []
    return rows


def _plate(gray, paper, poster):
    left, top, right, bottom = poster
    span, depth = right - left, bottom - top
    x0, y0 = right + round(.018 * span), top + round(.63 * depth)
    x1, y1 = min(gray.shape[1], right + round(.99 * span)), min(len(gray), bottom + round(.025 * depth))
    patch = gray[y0:y1, x0:x1]
    matches = []
    for drop in (6, 9):
        mask = np.uint8(patch <= paper - drop)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, w, h = bounds = cv2.boundingRect(contour)
            if (.55 * span <= w <= .85 * span and .22 * span <= h <= .39 * span
                    and 1.85 <= w / h <= 3.2 and .02 * span <= x <= .21 * span
                    and x + w < patch.shape[1] and top + .73 * depth <= y + y0 <= top + .90 * depth
                    and _rounded_shape(contour, bounds)):
                candidates.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
        if len(candidates) != 1:
            return None
        matches.append(candidates[0])
    if np.max(abs(np.asarray(matches[0]) - matches[1])) <= max(2, .022 * span):
        return matches[-1]
    return None


def _rating(gray, area, index):
    left, top, right, bottom = area
    tile = gray[top:bottom, left:right]
    paper = float(np.median(tile))
    mask = np.uint8(tile <= paper - 75)
    for a, b in _bands(mask.sum(axis=1) >= 5):
        if b - a < 12:
            continue
        line = mask[a:b]
        spans = _bands(line.any(axis=0))
        if len(spans) < 4:
            continue
        first, second, punct, digits = spans[:4]
        h = b - a
        if (not .68 * h <= first[1] - first[0] <= 1.20 * h
                or not .68 * h <= second[1] - second[0] <= 1.20 * h
                or not .06 * h <= punct[1] - punct[0] <= .34 * h or not .24 * h <= digits[1] - digits[0] <= .79 * h):
            continue
        pad = max(3, round(.22 * h))
        label_area = (left + max(0, first[0] - pad), max(top, top + a - pad),
                      left + min(second[1] + pad, punct[0]), min(bottom, top + b + pad))
        if _printed(gray, label_area, index, vocabulary=_RATINGS, light=False, minimum_score=.75) == "评分":
            return True
    return False


@lru_cache(maxsize=200)
def _letter_shapes(paths, letter):
    shapes = []
    for path in paths:
        try:
            font = ImageFont.truetype(str(path), 38)
        except OSError:
            continue
        raster = _font_raster(font, letter)
        if raster is None or (match := _normal(raster)) is None:
            continue
        shapes.append(cv2.GaussianBlur(match[0].astype(np.float32) / 255, (3, 3), .60))
    return shapes


def _side_score(query, sprites, *, right):
    result = []
    for n in (11, 13, 14):
        columns = slice(-n, None) if right else slice(0, n)
        chip = query[:, columns]
        qnorm = np.linalg.norm(chip)
        for sprite in sprites:
            part = sprite[:, columns]
            result.append(float(np.sum(chip * part) / max(.1, qnorm * np.linalg.norm(part))))
    return max(result, default=0.)


def _glyph(signal, index, ranks):
    peak = float(np.percentile(signal, 96))
    if peak < 57:
        return None
    readings = []
    for cutoff in (.29, .35, .44, .53):
        mask = np.uint8(signal >= cutoff * peak)
        key = (mask.shape, mask.tobytes())
        if key not in ranks:
            ranks[key] = index.rank(mask, limit=3)
        options = ranks[key]
        if len(options) >= 2 and options[0]["distance"] <= .181:
            readings.append((mask, options))
    if len(readings) < 3:
        return None
    strong = {row[0]["text"] for _, row in readings if row[1]["distance"] - row[0]["distance"] >= .021}
    if len(strong) == 1:
        best = next(iter(strong))
        if sum(row[0]["text"] == best for _, row in readings) >= 3:
            return best
    if strong:
        return None
    paths = tuple(index.fonts or index.cjk_sources)
    winner = max({row[0]["text"] for _, row in readings}, key=lambda name: (sum(row[0]["text"] == name for _, row in readings), name))
    peers = {row[1]["text"] for _, row in readings if row[0]["text"] == winner}
    if len(peers) != 1 or sum(row[0]["text"] == winner for _, row in readings) < 3:
        return None
    opposite = next(iter(peers))
    if not all("\u3400" <= name <= "\u9fff" for name in (winner, opposite)):
        return None
    referents = [_letter_shapes(paths, name) for name in (winner, opposite)]
    shape_evidence = [[], []]
    for mask, row in readings:
        if row[0]["text"] != winner or row[0]["distance"] > .132 or row[1]["distance"] - row[0]["distance"] > .025:
            continue
        raster, _ = _normal(mask * 255)
        query = cv2.GaussianBlur(raster.astype(np.float32) / 255, (3, 3), .60)
        for side in (0, 1):
            strongest, rival = (_side_score(query, chips, right=side) for chips in referents)
            if strongest >= .895:
                shape_evidence[side].append((strongest - rival, mask.tobytes()))
    for votes in shape_evidence:
        strong_votes = {raster: margin for margin, raster in votes if margin >= .012}
        if len(strong_votes) >= 2 and max(strong_votes.values()) >= .021:
            return winner
    return None


def _title(gray, box, index, cache):
    left, top, right, bottom = box
    crop = gray[top:bottom, left:right]
    paper = round(float(np.median(crop)))
    ink = cv2.subtract(paper, crop)
    mask = np.uint8(ink >= 73)
    u, v, w, h = cv2.boundingRect(mask)
    if not .37 * crop.shape[0] <= h <= .82 * crop.shape[0] or v < 2 or v + h + 1 >= len(crop):
        return None
    stroke = ink[max(0, v - 1):min(len(ink), v + h + 1), :]
    groups = []
    for a, b in _bands(mask.any(axis=0)):
        if groups and a - groups[-1][1] <= .073 * h and b - groups[-1][0] <= 1.14 * h:
            groups[-1] = groups[-1][0], b
        else:
            groups.append((a, b))
    if not 2 <= len(groups) <= 18:
        return None
    title = []
    for i, (a, b) in enumerate(groups):
        pad = min(2, (a - groups[i - 1][1] - 1) // 2 if i > 0 else 2,
                  (groups[i + 1][0] - b - 1) // 2 if i < len(groups) - 1 else 2)
        if not (letter := _glyph(stroke[:, max(0, a - pad):b + pad], index, cache)):
            return None
        title.append({"(": "（", ")": "）", ":": "："}.get(letter, letter))
    label = "".join(title)
    if (sum("\u3400" <= char <= "\u9fff" for char in label) < 2 or label.count("（") != label.count("）")
            or any(not (char.isalnum() or char in "（）·：") for char in label)):
        return None
    return label


def _tabs(gray, small, paper, first, index, sx, sy):
    left, top, right, bottom = first
    depth = bottom - top
    y0, y1 = top - round(.36 * depth), top - round(.062 * depth)
    if y0 < 0 or y1 <= y0:
        return []
    band = small[y0:y1]
    contours, _ = cv2.findContours(np.uint8(band <= paper - 21), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fills = []
    for shape in contours:
        x, y, w, h = bounds = cv2.boundingRect(shape)
        if (.12 * depth <= h <= .24 * depth and 1.30 <= w / h <= 3.10 and x <= .67 * small.shape[1]
                and _rounded_shape(shape, bounds)):
            fills.append((x, y0 + y, x + w, y0 + y + h))
    if len(fills) != 1:
        return []
    fill = fills[0]
    fleft, ftop, fright, fbottom = _project(fill, sx, sy)
    h = fbottom - ftop
    l, r, t, b = round(.034 * gray.shape[1]), round(.91 * gray.shape[1]), ftop + round(.22 * h), fbottom - round(.22 * h)
    line = gray[t:b, l:r] <= paper - 73
    groups = []
    for start, stop in _bands(line.any(axis=0)):
        if groups and start - groups[-1][1] <= .38 * h:
            groups[-1] = groups[-1][0], stop
        else:
            groups.append((start, stop))
    if not 3 <= len(groups) <= 6:
        return []
    recognized = []
    for a, z in groups:
        pad = max(3, round(.14 * h))
        box = (max(0, l + a - pad), t - pad, min(gray.shape[1], l + z + pad), b + pad)
        word = _printed(gray, box, index, vocabulary=_TAB_WORDS, light=False)
        if word is None:
            return []
        selected = fleft < _center(box)[0] < fright
        recognized.append((word, box, selected))
    if len({word for word, _, _ in recognized}) != len(recognized) or sum(state for _, _, state in recognized) != 1:
        return []
    return [_action("switch_tab", f"切换到{word}", box, text=word) for word, box, selected in recognized if not selected]


def _header(gray, small, paper, poster, index, cache, sx, sy):
    _, top, _, bottom = poster
    width = small.shape[1]
    depth = bottom - top
    first = max(0, top - round(.58 * depth))
    last = top - round(.30 * depth)
    if last <= first:
        return None
    lines = small[first:last, round(.062 * width):round(.93 * width)]
    spans = _bands((lines <= paper - 5).mean(axis=1) >= .78)
    rules = [first + a for a, b in spans if 1 <= b - a <= .042 * depth]
    if len(rules) != 1:
        return None
    y0, y1 = rules[0] - round(.310 * depth), rules[0] - round(.043 * depth)
    x0, x1 = round(.019 * width), round(.105 * width)
    x0, y0, x1, y1 = _project((x0, y0, x1, y1), sx, sy)
    if not 0 <= y0 < y1 <= len(gray):
        return None
    patch = gray[y0:y1, x0:x1]
    candidates = []
    for level in (102, 137, 170):
        foreground = np.uint8(patch <= level)
        x, y, w, h = cv2.boundingRect(foreground)
        if (.105 * depth * sy <= h <= .168 * depth * sy and 0 < x < x + w < x1 - x0
                and 0 < y < y + h < y1 - y0 and _bold_left_angle(foreground[y:y + h, x:x + w].astype(bool))):
            candidates.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
    if len(candidates) < 2 or np.ptp(np.asarray(candidates), axis=0).max() > .06 * depth * sy:
        return None
    border = candidates[-1]
    x, y, right, bottom = border
    size = bottom - y
    q0, q1 = right + round(.91 * size), min(gray.shape[1], right + round(8.0 * size))
    query = gray[y - 6:bottom + 6, q0:q1]
    a, b, w, h = cv2.boundingRect(np.uint8(query < paper - 85))
    if not .64 * size <= h <= 1.02 * size or not 1.18 * h <= w <= 9 * h:
        return None
    pad = max(6, round(.24 * h))
    box = (max(q0, q0 + a - pad), max(0, y - 6 + b - pad), min(gray.shape[1], q0 + a + w + pad), y - 6 + b + h + pad)
    return _action("back", "返回", border) if _title(gray, box, index, cache) else None


def search_result_row_actions(frame, overlay, index):
    """Return a visible watch control only with artwork, rating, and literal row title."""
    height, width = frame.shape[:2]
    if index is None or overlay is not None or width < 330 or not 1.61 * width <= height <= 2.72 * width:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preview_w = min(560, width)
    preview = cv2.resize(gray, (preview_w, round(height * preview_w / width)), interpolation=cv2.INTER_AREA)
    paper = round(float(np.median(preview[:, round(.90 * preview_w):])))
    if paper <= 208:
        return []
    posters = _posters(preview, paper)
    if len(posters) < 2:
        return []
    sx, sy = width / preview_w, height / preview.shape[0]
    cache = {}
    observed = []
    for small_box in posters:
        plate = _plate(preview, paper, small_box)
        if plate is None:
            continue
        x, y, r, b = button = _project(plate, sx, sy)
        h, w = b - y, r - x
        if _printed(gray, (x + round(.20 * w), y + round(.11 * h), r - round(.20 * w), b - round(.11 * h)),
                    index, vocabulary=_WATCH_WORDS, light=False) != "观看":
            continue
        l, t, right, bottom = _project(small_box, sx, sy)
        span, depth = right - l, bottom - t
        x0 = right + round(.064 * span)
        area = (x0, t + round(.222 * depth), min(width, x0 + round(.79 * span)), t + round(.369 * depth))
        if not _rating(gray, area, index):
            continue
        observed.append((button, (x0, t + round(.026 * depth), round(.96 * width), t + round(.171 * depth))))
    if len(observed) < 2:
        return []
    results = []
    for button, region in observed:
        if text := _title(gray, region, index, cache):
            results.append(_action("open_result", f"观看{text}", button, text=text))
    if len({item["function"] for item in results}) != len(results):
        results = [item for item in results if sum(peer["function"] == item["function"] for peer in results) == 1]
    tabs = _tabs(gray, preview, paper, posters[0], index, sx, sy)
    if tabs and (header := _header(gray, preview, paper, posters[0], index, cache, sx, sy)):
        return [header, *tabs, *results]
    return tabs + results
