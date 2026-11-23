"""Recognize literal game-mode names on individually outlined peer plates."""

from collections import defaultdict
from math import ceil

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts, read_caption
from screen_colored_settings_panel import _printed
from screen_component_captions import _bands


_MODE_TEXT = {
    "Battle": {"对战", "對戰", "战斗", "戰鬥"},
    "Ranking": {"排位", "排位赛", "排位賽"},
    "Ranked": {"排位", "排位赛", "排位賽", "竞技", "競技"},
    "Match": {"匹配", "匹配赛", "匹配賽"},
    "Casual": {"休闲", "休閒", "休闲模式", "休閒模式"},
    "Classic": {"经典", "經典", "经典模式", "經典模式"},
    "Competitive": {"竞技", "競技", "竞技场", "競技場"},
    "Challenge": {"挑战", "挑戰", "挑战模式", "挑戰模式"},
    "Training": {"训练", "訓練", "训练营", "訓練營"},
    "Practice": {"练习", "練習"},
    "Quick Play": {"快速对战", "快速對戰"},
    "Free Play": {"自由对战", "自由對戰"},
    "Tournament": {"赛事", "賽事"},
    "Solo": {"单人", "單人"},
    "Multiplayer": {"多人"},
}
_MODES = frozenset().union(*_MODE_TEXT.values())
_LATIN = (*_MODE_TEXT, "Profile", "Records", "Battle Pass", "Results", "Settings",
          "Schedule", "History", "Season", "Current", "Rules")


def _peers(regions, width, height):
    # Carrier proposals may include the drop shadow beneath one plate. The
    # observed upper edges and horizontal run are more stable than total height.
    candidates = set()
    for candidate in regions["candidates"]:
        left, top, right, bottom = map(int, candidate["box"])
        span, depth = right - left, bottom - top
        if (candidate.get("visibility") == "page" and 0 <= left < right <= width
                and .48 * height <= top <= .83 * height and bottom <= height
                and .10 * width <= span <= .25 * width and .084 * height <= depth <= .235 * height
                and 1.18 <= span / depth <= 3.75):
            candidates.add((left, top, right, bottom))
    groups = set()
    for seed in candidates:
        typical = seed[2] - seed[0]
        row = sorted(box for box in candidates if abs(box[1] - seed[1]) <= .023 * height
                     and .69 * typical <= box[2] - box[0] <= 1.43 * typical)
        if len(row) not in (3, 4, 5) or row[-1][2] - row[0][0] < .48 * width:
            continue
        centers = np.array([(box[0] + box[2]) / 2 for box in row])
        gaps = np.diff(centers)
        if gaps.max() > 1.46 * gaps.min() or not .69 * typical <= gaps.min() <= gaps.max() <= 1.49 * typical:
            continue
        groups.add(tuple(row))
    return next(iter(groups)) if len(groups) == 1 else ()


def _plate_edges(frame, peers):
    height = frame.shape[0]
    top_hint = int(round(np.median([box[1] for box in peers])))
    start = max(0, top_hint - round(.029 * height))
    stop = min(height, top_hint + round(.187 * height))
    colors = frame[start:stop]
    stride = max(1, round(.0024 * height))
    blue, green, red = cv2.split(cv2.absdiff(colors[2 * stride:], colors[:-2 * stride]))
    changes = cv2.max(cv2.max(blue, green), red)
    row_y = np.arange(start + stride, stop - stride)
    roofs, floors = [], []
    for left, _, right, _ in peers:
        margin = round(.24 * (right - left))
        section = changes[:, left + margin:right - margin]
        support = (section >= 22).mean(axis=1)
        peak = np.median(section, axis=1) * support
        upper = (row_y <= top_hint + .029 * height) & (support >= .70)
        if not upper.any():
            return []
        roof = int(row_y[np.argmax(np.where(upper, peak, 0))])
        lower = ((row_y >= roof + .083 * height) & (row_y <= roof + .154 * height)
                 & (support >= .74))
        if not lower.any():
            return []
        floor = int(row_y[np.argmax(np.where(lower, peak, 0))])
        roofs.append(roof)
        floors.append(floor)
    if np.ptp(roofs) > .025 * height or np.ptp(floors) > .022 * height:
        return []
    return list(zip(roofs, floors, strict=True))


def _strokes(gray, panel, roof, floor):
    left, _, right, _ = panel
    width, depth = right - left, floor - roof
    side = round(.12 * width)
    x, r = left + side, right - side
    y, b = roof + round(.12 * depth), floor - round(.31 * depth)
    patch = gray[y:b, x:r]
    tone = float(np.median(patch))
    observations = []
    for gain in (64, 79, 92):
        ink = np.uint8(patch >= max(150, tone + gain))
        cols = _bands(ink.sum(axis=0) >= max(2, round(.032 * depth)))
        grouped = []
        for a, end in cols:
            if grouped and a - grouped[-1][1] <= .13 * depth:
                grouped[-1] = grouped[-1][0], end
            else:
                grouped.append((a, end))
        runs = []
        for a, end in grouped:
            sx, sy, sw, sh = cv2.boundingRect(ink[:, a:end])
            first, last = a + sx + x, a + sx + sw + x
            top, stop = y + sy, y + sy + sh
            if (.26 * depth <= sh <= .50 * depth and 1.52 <= sw / sh <= 7.1
                    and abs(first + last - left - right) <= .27 * width
                    and first > x + 1 and last < r - 1 and top > y + 1 and stop < b - 1):
                runs.append((first, top, last, stop))
        if len(runs) == 1:
            observations.append(runs[0])
    if len(observations) < 2 or np.ptp(observations, axis=0).max() > max(2, .045 * depth):
        return None
    return tuple(map(int, np.rint(np.median(observations, axis=0))))


def _second_line(gray, bounds, index):
    left, top, right, bottom = bounds
    height = bottom - top
    pad = round(.29 * height)
    x, r = max(0, left - pad), min(gray.shape[1], right + pad)
    y, b = bottom + max(1, round(.04 * height)), min(len(gray), bottom + round(.54 * height))
    crop = gray[y:b, x:r]
    if min(crop.shape) < 8:
        return None
    k = max(5, round(.26 * crop.shape[0]) | 1)
    relief = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    peak = float(np.percentile(relief, 98))
    if peak < 15:
        return None
    detail = np.uint8(np.minimum(255, relief.astype(np.float32) * (180 / peak)))
    ink = (detail >= 72).sum(axis=1)
    runs = [(a, b) for a, b in _bands(ink >= round(.095 * detail.shape[1]))
            if .20 * height <= b - a <= .46 * height]
    if len(runs) != 1:
        return None
    a, b = runs[0]
    margin = max(2, round(.24 * (b - a)))
    line = cv2.copyMakeBorder(detail[a:b], margin, margin, 0, 0, cv2.BORDER_CONSTANT)
    return _printed(line, (0, 0, line.shape[1], len(line)), index, vocabulary=_LATIN,
                    minimum_score=.69, minimum_margin=.034)


def _verify_title(gray, bounds, choices, index, cache):
    if not choices:
        return None
    left, top, right, bottom = bounds
    pad = max(2, round(.07 * (bottom - top)))
    patch = gray[top - pad:bottom + pad, left - pad:right + pad]
    checked = defaultdict(set)
    for level in (142, 150, 158, 173, 191):
        ink = np.uint8(patch >= level)
        x, y, w, h = cv2.boundingRect(ink)
        if h < 10 or abs(h - (bottom - top)) > .19 * h:
            continue
        masks = _cjk_parts(ink[y:y + h, x:x + w], h, join_radicals=True)
        options = [name for name in choices if len(name) == len(masks)]
        if not options:
            continue
        ranked = []
        for *_, glyph in masks:
            key = glyph.shape, glyph.tobytes()
            if key not in cache:
                cache[key] = index.rank(glyph, alphabet="cjk", limit=5)
            ranked.append(cache[key])
        for name in options:
            for wanted, ranks in zip(name, ranked, strict=True):
                match = next((row for row in ranks if row["text"] == wanted), None)
                if match is None or match["distance"] > .196 or match["distance"] - ranks[0]["distance"] > .033:
                    break
            else:
                checked[name].add(ink.tobytes())
    verified = [name for name, masks in checked.items() if len(masks) >= 2]
    return verified[0] if len(verified) == 1 else None


def _compact_marks(ink, spine, height, left, top):
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(ink), connectivity=8)
    return [(left + x + (w - 1) / 2, top + y + (h - 1) / 2)
            for x, y, w, h, count in stats[1:]
            if max(2, round(.0038 * height)) <= min(w, h) <= max(w, h) <= .018 * height
            and .62 <= w / h <= 1.56 and count >= .52 * w * h
            and abs(left + x + (w - 1) / 2 - spine) <= max(1.5, .0035 * height)]


def _outlined_marker(light, x, y, height):
    angles = np.arange(24) * np.pi / 12
    radii = height * np.array([.009, .010, .011, .012])
    # A selected marker has a bright rim separated from its filled center.
    # Use both circles so that an isolated glow cannot stand in for a ring.
    rings = np.concatenate((radii, .69 * radii))[:, None]
    xs = np.float32(x + rings * np.cos(angles))
    ys = np.float32(y + rings * np.sin(angles))
    outer, inner = np.split(cv2.remap(light, xs, ys, cv2.INTER_LINEAR), 2)
    return bool(((np.percentile(outer, 25, axis=1) >= 99)
                 & (np.median(outer, axis=1) - np.median(inner, axis=1) >= 11)).any())


def _spine_markers(frame):
    height, width = frame.shape[:2]
    top, bottom = round(.13 * height), round(.63 * height)
    first, last = round(.028 * width), round(.16 * width)
    reach = round(.018 * height)
    pixels = frame[top:bottom, :last + reach]
    tone = pixels[..., 0].astype(np.int16) - pixels[..., 2]
    relief = cv2.morphologyEx(tone, cv2.MORPH_TOPHAT, np.ones((1, max(7, reach | 1)), np.uint8))
    columns = _bands((relief[:, first:last] >= 20).mean(axis=0) >= .40)
    if len(columns) != 1 or columns[0][1] - columns[0][0] > max(2, ceil(.006 * height)):
        return None
    a, b = columns[0]
    spine = first + (a + b - 1) / 2
    left, right = round(spine - reach), round(spine + reach) + 1
    patch = pixels[:, left:right]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    light = patch.min(axis=2)
    selected = [point for point in _compact_marks(light >= 190, spine, height, left, top)
                if _outlined_marker(light, point[0] - left, point[1] - top, height)]
    if len(selected) != 1:
        return None
    diameter = max(5, round(.017 * height) | 1)
    detail = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
    ink = (detail >= 19) & (tone[:, left:right] >= 15) & (light < 164)
    dots = _compact_marks(ink, spine, height, left, top)
    _, current = selected[0]
    choices = [(x, y) for x, y in dots if abs(y - current) >= .105 * height]
    if not 1 <= len(choices) <= 8:
        return None
    axis = round(spine)
    connected = []
    for x, y in choices:
        a, b = sorted((y, current))
        middle = relief[round(a + .036 * height) - top:round(b - .036 * height) - top, axis - 1:axis + 2]
        if middle.size and (middle.max(axis=1) >= 16).mean() >= .73:
            connected.append((x, y))
    return (spine, current, connected) if connected else None


def _spine_caption(frame, spine, y, index, *, selected=False):
    height = frame.shape[0]
    left, right = round(spine + .019 * height), round(spine + .16 * height)
    top, bottom = round(y - .022 * height), round(y + .022 * height)
    tile = frame[top:bottom, left:right]
    if selected:
        masks = (tile.min(axis=2) >= cut for cut in (163, 181, 197))
    else:
        diameter = max(5, round(.009 * height) | 1)
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        detail = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
        masks = (detail >= cut for cut in (25, 29, 33, 37))
    bounds, words, cache = [], defaultdict(set), {}
    for ink in masks:
        x, t, w, h = cv2.boundingRect(np.uint8(ink))
        if not (.018 * height <= h <= .039 * height and 1.45 * h <= w <= 6.5 * h
                and abs(top + t + h / 2 - y) <= .012 * height and x <= .024 * height
                and t >= 1 and t + h < len(tile) and x + w < tile.shape[1]):
            continue
        raster = np.uint8(ink[t:t + h, x:x + w])
        parts = _cjk_parts(raster, h, join_radicals=True)
        if not 2 <= len(parts) <= 6:
            continue
        bounds.append((left + x, top + t, left + x + w, top + t + h))
        if selected or index is None:
            continue
        ranks = []
        for *_, bitmap in parts:
            key = bitmap.shape, bitmap.tobytes()
            if key not in cache:
                cache[key] = index.rank(bitmap, alphabet="cjk", limit=5)
            ranks.append(cache[key])
        for word in _MODES:
            if len(word) != len(ranks):
                continue
            for char, choices in zip(word, ranks, strict=True):
                score = next((option["distance"] for option in choices if option["text"] == char), 1.)
                if not choices or score > .212 or score - choices[0]["distance"] > .032:
                    break
            else:
                words[word].add(raster.tobytes())
    if len(bounds) < 2 or np.ptp(bounds, axis=0).max() > max(3, .009 * height):
        return None
    verified = [word for word, proofs in words.items() if len(proofs) >= 2]
    return (tuple(map(int, np.rint(np.median(bounds, axis=0)))), verified[0] if len(verified) == 1 else None)


def marked_mode_rail_actions(frame, index=None):
    """Read a marked navigation rail, with optional caption recognition."""
    if (rail := _spine_markers(frame)) is None:
        return []
    spine, current, choices = rail
    if _spine_caption(frame, spine, current, index, selected=True) is None:
        return []
    actions = []
    rows = [current]
    for x, y in choices:
        if (caption := _spine_caption(frame, spine, y, index)) is None:
            continue
        rows.append(y)
        (left, top, right, bottom), label = caption
        pad = max(2, round(.15 * (bottom - top)))
        actions.append({"role": "game_mode" if label else "side_navigation",
                        "function": f"切换到{label}" if label else "切换侧栏选项",
                        **({"preview_text": label} if label else {}),
                        "box": [round(x) - pad, top - pad, right + pad, bottom + pad],
                        "click_point": [(left + right) / 2, (top + bottom) / 2]})
    if not 1 <= len(actions) <= 4:
        return []
    gaps = np.diff(sorted(rows))
    pitch = float(np.median(gaps))
    return actions if .105 * frame.shape[0] <= pitch <= .24 * frame.shape[0] and np.ptp(gaps) <= .21 * pitch else []


def landscape_mode_rail_actions(frame, regions, index):
    """Recognize distinct game-mode tiles and vertically marked navigation rows."""
    height, width = frame.shape[:2]
    if (index is None or regions.get("layer") != "page" or regions.get("regions")
            or height < 260 or not 1.45 * height <= width <= 2.75 * height):
        return []
    peers = _peers(regions, width, height)
    if not peers or not (edges := _plate_edges(frame, peers)):
        return marked_mode_rail_actions(frame, index)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    actions, cache, ranked = [], {}, {}
    for panel, (roof, floor) in zip(peers, edges, strict=True):
        if (bounds := _strokes(gray, panel, roof, floor)) is None:
            continue
        left, top, right, bottom = bounds
        pad = max(3, round(.15 * (bottom - top)))
        area = (left - pad, top - pad, right + pad, bottom + pad)
        reading = read_caption(frame, area, index, cache=cache, choice=True, join_radicals=True)
        name = reading["text"] if reading and reading["text"] in _MODES else None
        if name is None:
            gloss = _second_line(gray, bounds, index)
            name = _verify_title(gray, bounds, _MODE_TEXT.get(gloss, ()), index, ranked)
        if name is not None:
            inset = round(.31 * (bottom - top))
            actions.append({"role": "game_mode", "function": f"进入{name}", "preview_text": name,
                            "box": [max(panel[0], left - inset), roof, min(panel[2], right + inset), floor],
                            "click_point": [(left + right) / 2, (top + bottom) / 2]})
    names = {item["preview_text"] for item in actions}
    return actions if len(names) >= 2 and len(names) == len(actions) else []
