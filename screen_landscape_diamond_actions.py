"""Verify framed and captioned shortcut peers in a landscape edge rail."""

from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np

from screen_activity_icon_groups import activity_icon_group_actions, _dimensions, _tile
from screen_caption_transcript import _cjk_parts


def _outline_maps(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blue = frame[:, :, 0]
    result = []
    for plane in (gray, blue):
        dx = cv2.Sobel(plane, cv2.CV_16S, 1, 0)
        dy = cv2.Sobel(plane, cv2.CV_16S, 0, 1)
        parallel, crossed = abs(dx + dy), abs(dx - dy)
        result.append((np.maximum(0, parallel - .55 * crossed), np.maximum(0, crossed - .55 * parallel)))
    return tuple(np.maximum(a, b) for a, b in zip(*result, strict=True))


def _frames(maps, xs, ys, radii):
    """Measure a diamond against all four actual diagonal stroke runs."""
    height, width = maps[0].shape
    steps = np.linspace(.18, .82, 11)
    lower, upper = 3, 4
    choices = []
    cx, cy = np.meshgrid(xs, ys)
    for radius in radii:
        valid = (cx - radius > 1) & (cx + radius < width - 2) & (cy - radius > 1) & (cy + radius < height - 2)
        if not valid.any():
            continue
        x, y = cx[valid], cy[valid]
        along = np.rint(steps * radius).astype(int)
        across = radius - along
        strengths = []
        offset = max(1, round(.046 * radius))
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            plane = maps[sx != sy]
            columns = x[:, None] + sx * along[None, :]
            rows = y[:, None] + sy * across[None, :]
            samples = np.maximum.reduce([plane[rows + sy * shift, columns + sx * shift]
                                         for shift in (-offset, 0, offset)])
            # Linear 31st percentile on these eleven ordered samples.
            ranked = np.partition(samples, (lower, upper), axis=1)
            strengths.append(.90 * ranked[:, lower] + .10 * ranked[:, upper])
        sides = np.stack(strengths)
        best = np.sort(sides, axis=0)
        quality = best[0] + .46 * best[1] + .33 * best[2] + .11 * best[3]
        valid_rows = np.flatnonzero((best[0] >= 16) & (best[1] >= 22))
        for pos in valid_rows:
            choices.append((float(quality[pos]), int(x[pos]), int(y[pos]), int(radius)))
    ranked = sorted(choices, reverse=True)
    kept = []
    for quality, x, y, radius in ranked:
        if not any(max(abs(x - a), abs(y - b), abs(radius - r)) <= .19 * max(radius, r)
                   for _, a, b, r in kept):
            kept.append((quality, x, y, radius))
            if len(kept) == 4:
                break
    return kept


def _rail_pairs(frame, proposals):
    height, width = frame.shape[:2]
    seeds = []
    for row in proposals:
        box = tuple(row["box"])
        x, y, w, h = _dimensions(box)
        if (row.get("visibility") == "page" and .025 * height <= min(x, width - x) <= .145 * height
                and .055 * height <= w <= .150 * height and .074 * height <= h <= .166 * height
                and .75 <= w / h <= 1.12 and .074 * height <= y <= .71 * height and _tile(frame, box) is not None):
            seeds.append(box)
    found = []
    for box in seeds:
        x, y, w, h = _dimensions(box)
        l = max(0, round(x - .64 * w))
        r = min(width, round(x + .64 * w))
        t = max(0, round(box[1] - .94 * h))
        b = min(height, round(box[3] + 1.10 * h))
        maps = _outline_maps(frame[t:b, l:r])
        step = max(1, round(.029 * w))
        radii = range(round(.32 * w), round(.47 * w) + 1, step)
        xs = np.arange(round(x - l - .11 * w), round(x - l + .11 * w) + 1, step)
        ys = np.arange(max(1, round(box[1] - t - .58 * h)),
                       min(b - t - 1, round(box[3] - t + .56 * h)), step)
        for first, second in combinations(_frames(maps, xs, ys, radii), 2):
            _, x1, y1, r1 = first
            _, x2, y2, r2 = second
            if (abs(x1 - x2) > .11 * w or abs(r1 - r2) > .24 * max(r1, r2)
                    or not .83 * h <= abs(y2 - y1) <= 1.30 * h
                    or not any(box[1] <= t + seed_y <= y + .06 * h for seed_y in (y1, y2))):
                continue
            rails = sorted([(x1 + l, y1 + t, r1), (x2 + l, y2 + t, r2)], key=lambda pair: pair[1])
            if not any(rows and all(max(abs(a - b) for a, b in zip(left, right, strict=True)) <= .16 * w
                                   for left, right in zip(rows, rails, strict=True)) for rows in found):
                found.append(rails)
    return found


_WORDS = {
    "奖励": "game_rewards", "獎勵": "game_rewards", "排行": "game_rank", "排行榜": "game_rank",
    "排名": "game_rank", "前十": "game_rank", "战绩": "game_history", "戰績": "game_history",
    "记录": "game_history", "記錄": "game_history", "历史": "game_history", "歷史": "game_history",
}


def _caption(frame, diamond, index):
    x, y, radius = diamond
    badge = frame[max(0, y):min(len(frame), y + round(1.82 * radius)),
                  max(0, x - round(1.29 * radius)):min(frame.shape[1], x + round(1.29 * radius))]
    gray = cv2.cvtColor(badge, cv2.COLOR_BGR2GRAY)
    alternative = badge.min(axis=2)
    choices = defaultdict(dict)
    cache = {}
    cuts = (.51, .59, .67, .75, .84, .92)
    for plane, start in ((plane, start) for plane in (gray, alternative) for start in cuts):
        tile = plane[round(start * radius):]
        if min(tile.shape) < 8:
            continue
        for scale in (1, 2, 3):
            if radius >= 30 and scale == 3:
                continue
            view = tile if scale == 1 else cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            low, high = np.percentile(view, (30, 98))
            if high < 137 or high - low < 40:
                continue
            for cut in (.33, .38, .43, .48, .53, .58, .63, .68):
                ink = cv2.compare(view, int(np.ceil(low + cut * (high - low))), cv2.CMP_GE)
                a, b, w, h = cv2.boundingRect(ink)
                if (not .35 * radius <= h / scale <= .77 * radius or not 1.65 * h <= w <= 5.7 * h
                        or abs(a + w / 2 - view.shape[1] / 2) > .14 * view.shape[1]):
                    continue
                line = ink[b:b + h, a:a + w]
                key = scale, line.shape, line.tobytes()
                if key in cache:
                    continue
                cache[key] = True
                parts = _cjk_parts(line, h)
                if not 2 <= len(parts) <= 5:
                    continue
                ranks = [index.rank(part[-1], alphabet="cjk", limit=2) for part in parts]
                if not all(len(options) == 2 and options[0]["distance"] <= .247 for options in ranks):
                    continue
                name = "".join(options[0]["text"] for options in ranks)
                suffix = next((word for word in sorted(_WORDS, key=len, reverse=True)
                               if name.endswith(word)), None)
                if suffix is not None:
                    distance = float(np.mean([options[0]["distance"] for options in ranks]))
                    choices[(name, _WORDS[suffix])][key] = distance
        if any(len(readings) >= 4 and len({key[0] for key in readings}) >= 2
               for readings in choices.values()):
            break
    found = [(name, role, len(readings), min(readings.values()))
             for (name, role), readings in choices.items() if len(readings) >= 3]
    return sorted(found, key=lambda row: (-row[2], row[3]))


def landscape_diamond_actions(frame, regions, index):
    """Read both named controls of a proven same-size diamond shortcut pair."""
    height, width = frame.shape[:2]
    if index is None or regions.get("layer") != "page" or regions.get("regions") or height < 320 or width < 1.45 * height:
        return []
    pairs = _rail_pairs(frame, regions["candidates"])
    if len(pairs) != 1:
        return []
    labels = [_caption(frame, box, index) for box in pairs[0]]
    if not all(labels):
        return []
    first, second = labels[0][0], labels[1][0]
    if "game_rank" not in (first[1], second[1]) or first[1] == second[1]:
        return []
    if any(len(rows) > 1 and rows[0][2] - rows[1][2] < 2 and rows[1][3] - rows[0][3] < .014
           for rows in labels):
        return []
    results = []
    for position, ((x, y, radius), (word, role, _, _)) in enumerate(zip(pairs[0], (first, second), strict=True)):
        half = max(radius, round(.29 * radius * len(word)))
        bottom = min(height, y + round(1.49 * radius))
        if position == 0:
            _, next_y, next_radius = pairs[0][1]
            bottom = min(bottom, next_y - next_radius)
        results.append({"role": role, "function": f"查看{word}", "preview_text": word,
                        "box": [max(0, x - half), y - radius, min(width, x + half + 1), bottom],
                        "click_point": [x, y]})
    return results


def landscape_side_actions(frame, regions, index):
    """Use repeated shortcut structure and suppress muted or covered rails."""
    if regions.get("layer") != "page" or regions.get("regions"):
        return []
    rows = []
    for action in activity_icon_group_actions(frame, regions, index):
        left, top, right, bottom = action["box"]
        tile = frame[top:bottom:2, left:right:2]
        if ((tile.max(axis=2) >= 185).mean() >= .12):
            rows.append(action)
    if len(rows) >= 2:
        return rows
    return landscape_diamond_actions(frame, regions, index)
