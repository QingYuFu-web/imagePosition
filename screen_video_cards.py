"""Own complete video thumbnails and their printed captions as single controls."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np

from screen_product_rows import _artwork_click, _body_pairs, _Card, _size, _spans


@dataclass(frozen=True)
class _Tile:
    box: tuple[int, int, int, int]
    artwork: tuple[int, int, int, int]


def _candidate_bodies(candidates, width):
    art = []
    captions = []
    for item in candidates:
        if item["visibility"] != "page":
            continue
        box = tuple(map(int, item["box"]))
        w, h = _size(box)
        if not .34 * width <= w <= .54 * width:
            continue
        if .45 * w <= h <= .80 * w:
            art.append(box)
        elif .23 * w <= h <= .44 * w:
            captions.append(box)
    for left, top, right, split in art:
        w = right - left
        for l, t, r, b in captions:
            if max(abs(l - left), abs(r - right), abs(t - split)) <= .029 * w:
                yield {"box": (left, top, right, b), "visibility": "page"}


def _tile(low, candidate):
    l, t, r, b = candidate
    w = r - l
    height = low.shape[0]
    if not .09 * height <= t or b >= height - .04 * w:
        return None
    inset = round(.065 * w)
    end = min(height, b + round(.071 * w))
    field = low[t:end, l + inset:r - inset:2]
    blank = (field >= 247).mean(axis=1)
    reach = round(.032 * w)
    possible = []
    for s, e in _spans(blank >= .83, 0):
        if (not .49 * w <= s <= .76 * w or e - s < reach
                or blank[s - reach:s].mean() > .63):
            continue
        art = field[::2][:s // 2]
        if art.std() < 19 or (art < 152).mean() < .10:
            continue
        possible.append(s)
    if len(possible) != 1:
        return None
    split = possible[0]
    # A caption must have its own dark line on a light canvas, and end
    # before the next thumbnail. Short titles can fit on a single line.
    glyph_fraction = (field[split:] < 170).mean(axis=1)
    ranges = _spans(glyph_fraction[:round(.32 * w)] >= .056, max(1, round(.005 * w)))
    lines = [(a, z) for a, z in ranges if .032 * w <= z - a <= .095 * w]
    if not lines or not .025 * w <= lines[0][0] <= .10 * w:
        return None
    # Ignore darker print across the center when finding the lower card edge.
    canvas = (low[t + split:end, l + inset:r - inset:2] >= 245).mean(axis=1)
    edge = next((a for a, z in _spans(canvas < .58, 1)
                 if .31 * w <= a <= .49 * w and z - a >= .022 * w), None)
    if edge is None or (canvas[:edge] >= .58).mean() < .76:
        return None
    return _Tile((l, t, r, t + split + edge), (l, t, r, t + split))


@lru_cache(maxsize=96)
def _badge_shape(width, height, play):
    yy, xx = np.mgrid[:height, :width]
    curve = .23 * height

    def rounded(inset):
        x = np.maximum(abs(xx - (width - 1) / 2) - (.5 * width - inset - curve), 0)
        y = np.maximum(abs(yy - (height - 1) / 2) - (.5 * height - inset - curve), 0)
        return x * x + y * y <= curve * curve

    rim = rounded(.025 * height) & ~rounded(.10 * height)
    marks = np.zeros((height, width), np.uint8)
    if play:
        points = np.int32([[.34 * width, .34 * height], [.34 * width, .63 * height],
                           [.64 * width, .485 * height]])
        cv2.fillConvexPoly(marks, points, 1)
    else:
        for baseline in (.39, .65):
            row = round(baseline * height)
            stroke = max(1, round(.065 * height))
            marks[row:row + stroke, round(.42 * width):round(.78 * width)] = 1
    pad = max(3, round(.17 * height) | 1)
    quiet = rounded(.175 * height) & ~cv2.dilate(marks, np.ones((pad, pad), np.uint8)).astype(bool)
    marks = marks.astype(bool)
    if min(rim.sum(), marks.sum(), quiet.sum()) == 0:
        return None
    weights = .52 * rim / rim.sum() + .48 * marks / marks.sum() - quiet / quiet.sum()
    return weights.astype(np.float32), rim, marks, quiet


def _badge(low, box, *, play, anchor=None):
    l, _, r, bottom = box
    w = r - l
    if play:
        x0, x1 = l + round(.022 * w), l + round(.20 * w)
        y0, y1 = bottom - round(.135 * w), bottom - round(.019 * w)
    else:
        x, y = anchor
        x0, x1 = x + round(.22 * w), min(r, x + round(.49 * w))
        y0, y1 = y - round(.020 * w), min(bottom, y + round(.094 * w))
    search = low[y0:y1, x0:x1].astype(np.float32)
    matches = []
    for scale in (.91, 1., 1.08):
        width, height = round(.082 * w * scale), round(.069 * w * scale)
        if height < 12 or search.shape[0] < height or search.shape[1] < width:
            continue
        shape = _badge_shape(width, height, play)
        if shape is None:
            continue
        weights, rim, marks, quiet = shape
        response = cv2.matchTemplate(search, weights, cv2.TM_CCORR)
        _, score, _, (x, y) = cv2.minMaxLoc(response)
        patch = search[y:y + height, x:x + width]
        core, surround, surface = patch[marks], patch[rim], patch[quiet]
        if (score >= (100 if play else 80) and core.mean() >= (195 if play else 178)
                and surround.mean() >= 142 and (core >= 180).mean() >= .66
                and (surround >= 180).mean() >= .32 and (surface >= 180).mean() <= .23):
            matches.append((score, x0 + x, y0 + y))
    return max(matches)[1:] if matches else None


def _duration(low, box, anchor, index):
    if index is None:
        return False
    l, _, r, bottom = box
    w = r - l
    _, y = anchor
    field = low[y:y + round(.076 * w), r - round(.22 * w):r - round(.027 * w)]
    times = Counter()
    for threshold in (150, 160, 170, 178, 192):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(field >= threshold), connectivity=8)
        parts = [(i, *map(int, stat)) for i, stat in enumerate(stats[1:], 1)]
        digits = [(i, a, b, span, h) for i, a, b, span, h, area in parts
                  if .027 * w <= h <= .055 * w and .27 * h <= span <= .90 * h and area >= .22 * span * h]
        dots = [(a, b, span, h) for _, a, b, span, h, area in parts
                if .004 * w <= h <= .013 * w and .51 <= span / h <= 1.95 and area >= .56 * span * h]
        digits.sort(key=lambda entry: entry[1])
        for first, second, third in combinations(digits, 3):
            _, a, b, aw, ah = first
            _, c, d, cw, ch = second
            _, e, f, _, fh = third
            if (not .43 * ah <= c - a - aw <= 1.05 * ah or not 1 <= e - c - cw <= .40 * ah
                    or max(abs(b - d), abs(b - f), abs(ah - ch), abs(ah - fh)) > .23 * ah):
                continue
            punctuation = [(x, t) for x, t, span, hh in dots if a + aw <= x < c
                           and b + .10 * ah <= t + hh / 2 <= b + .96 * ah]
            if not any(abs(x - xx) <= .14 * ah and .27 * ah <= abs(t - tt) <= .68 * ah
                       for (x, t), (xx, tt) in combinations(punctuation, 2)):
                continue
            best = [index.rank(labels[dy:dy + h, dx:dx + span] == i, alphabet="latin", limit=4)
                    for i, dx, dy, span, h in (first, second, third)]
            if all(ranks and ranks[0]["text"] in "0123456789" and ranks[0]["distance"] <= .245
                   for ranks in best) and best[1][0]["text"] <= "5":
                times["".join(ranks[0]["text"] for ranks in best)] += 1
                break
        if any(count >= 2 for count in times.values()):
            return True
    return False


def video_card_actions(frame, regions, index):
    """Return one action for each fully visible, locally evidenced video tile."""
    if regions["layer"] != "page" or frame.shape[0] < frame.shape[1]:
        return []
    candidates = regions["candidates"]
    width = frame.shape[1]
    bodies = [*candidates, *_candidate_bodies(candidates, width)]
    pairs = list(_body_pairs(bodies, width))
    if len(pairs) < 2:
        return []
    low = cv2.min(cv2.min(frame[..., 0], frame[..., 1]), frame[..., 2])
    result = []
    tiles = {}
    for first, second, scale in sorted(pairs):
        pair = []
        for box in (first, second):
            if box not in tiles:
                tiles[box] = _tile(low, box)
            pair.append(tiles[box])
        left, right = pair
        if (left is None or right is None or abs(left.artwork[3] - right.artwork[3]) > .030 * scale
                or abs(left.box[3] - right.box[3]) > .030 * scale):
            continue
        row_gap = low[left.box[1] + round(.036 * scale):min(left.box[3], right.box[3]),
                      round((first[2] + second[0]) / 2)]
        if (row_gap >= 225).mean() < .86:
            continue
        if any(abs(first[1] - row[0].box[1]) <= .045 * scale for row in result):
            continue
        result.append((left, right))
    if len(result) < 2:
        return []
    actions = []
    for row_number, row in enumerate(result, 1):
        for position, tile in zip(("左", "右"), row, strict=True):
            play = _badge(low, tile.artwork, play=True)
            if play is None or (not _badge(low, tile.artwork, play=False, anchor=play)
                                and not _duration(low, tile.artwork, play, index)):
                continue
            click = _artwork_click(_Card(tile.box, tile.artwork, None), candidates)
            if click is not None:
                actions.append({"role": "video_card", "function": "打开视频",
                                "preview_text": f"第{row_number}排{position}侧视频",
                                "box": list(tile.box), "click_point": click})
    return actions
