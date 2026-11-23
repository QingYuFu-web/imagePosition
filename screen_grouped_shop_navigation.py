"""Read inactive captions in a shop's horizontal group selector."""

from collections import Counter
from statistics import median

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts, read_caption
from screen_component_captions import _bands, _tight
from screen_game_shop_tabs import _caption


def _rows(candidates, width, height):
    labels = []
    for item in candidates:
        left, top, right, bottom = item["box"]
        w, h = right - left, bottom - top
        if (item.get("visibility") == "page" and .089 * height <= top <= .33 * height
                and .078 * width <= w <= .21 * width and .047 * height <= h <= .115 * height
                and 2.3 <= w / h <= 4.1):
            labels.append((left, top, right, bottom))
    groups = set()
    for anchor in labels:
        row = tuple(sorted(box for box in labels if abs(box[1] - anchor[1]) <= .014 * height
                           and abs(box[3] - anchor[3]) <= .014 * height))
        if not 4 <= len(row) <= 8:
            continue
        centers = [(a + b) / 2 for a, _, b, _ in row]
        pitch = median(np.diff(centers))
        if (np.ptp(np.diff(centers)) > .082 * pitch or np.ptp([b - a for a, _, b, _ in row]) > .12 * pitch
                or not 0 <= centers[0] - pitch / 2 < centers[-1] + pitch / 2 <= width):
            continue
        groups.add(row)
    return next(iter(groups)) if len(groups) == 1 else None


def _strip(gray, row):
    centers = [(a + b) / 2 for a, _, b, _ in row]
    pitch = median(np.diff(centers))
    dx = [round(centers[0] + (i - .5) * pitch) for i in range(len(row) + 1)]
    first, last = dx[0] + round(.028 * pitch), dx[-1] - round(.028 * pitch)
    top = round(median(box[1] for box in row))
    h = round(median(box[3] - box[1] for box in row))
    lo, hi = top - round(.20 * h), top + round(.48 * h)
    surface = gray[lo:hi, first:last]
    rail = [(lo + a, lo + b) for a, b in _bands((surface >= 184).mean(axis=1) >= .90)
            if max(2, .023 * h) <= b - a <= .185 * h]
    if len(rail) != 1:
        return None
    top = rail[0][1]
    bottom = round(median(box[3] for box in row))
    yy = np.arange(bottom - round(.18 * h), bottom + round(.14 * h))
    a, b = dx[0] + round(.22 * pitch), dx[0] + round(.72 * pitch)
    ledge = np.median(gray[yy - 2, a:b] - gray[yy + 1, a:b], axis=1)
    if ledge.max() < 6:
        return None
    bottom = int(yy[ledge.argmax()])
    return (dx, top, bottom, pitch) if .68 * h <= bottom - top <= 1.2 * h else None


def _adjacent_groups(gray, candidates, edges, top, bottom, pitch):
    height, width = gray.shape
    h = bottom - top
    floors = []
    for rec in candidates:
        a, t, b, end = rec["box"]
        if (rec.get("visibility") == "page" and -.015 * pitch <= a - edges[-1] <= .13 * pitch
                and .69 * pitch <= b - a <= 1.20 * pitch and 1.19 * h <= end - t <= 2.2 * h):
            floors.append((a, t, b, end))
    floors.sort(key=lambda box: box[1])
    stop = next((i for i in range(1, len(floors)) if floors[i][1] - floors[i - 1][3] > .26 * h), len(floors))
    floors = floors[:stop]
    if len(floors) < 4 or abs(floors[0][1] - top) > .24 * h:
        return False
    jumps = np.diff([(t + end) / 2 for _, t, _, end in floors])
    if not 1.31 * h <= min(jumps) <= max(jumps) <= 2.08 * h or np.ptp(jumps) > .23 * h:
        return False
    roofs = []
    y0, y1 = bottom + round(.15 * h), bottom + round(.63 * h)
    for x, right in zip(edges[:-1], edges[1:], strict=True):
        a, b = round(x + .15 * pitch), round(right - .15 * pitch)
        patch = gray[y0:y1, a:b]
        diff = patch[3:] - patch[:-3]
        support = (diff >= 19).mean(axis=1)
        if support.max() < .69:
            return False
        roofs.append(y0 + int(support.argmax()) + 3)
    return np.ptp(roofs) <= .075 * h


def _selected(frame, edges, top, bottom, pitch):
    h = bottom - top
    surfaces = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        patch = frame[top + round(.27 * h):bottom - round(.21 * h), a + round(.19 * pitch):b - round(.16 * pitch)]
        surfaces.append(np.median(patch[::2, ::3], axis=(0, 1)))
    samples = np.asarray(surfaces)
    median_sample = np.median(samples, axis=0)
    difference = np.max(abs(samples - median_sample), axis=1)
    active = np.flatnonzero(difference >= 51)
    if len(active) != 1 or (difference < 15).sum() != len(samples) - 1:
        return None
    chosen = int(active[0])
    paint = samples[chosen]
    if np.ptp(paint) < 47 or np.ptp(median_sample) > 27:
        return None
    left, right = edges[chosen:chosen + 2]
    patch = frame[top + round(.12 * h):bottom - round(.12 * h), left + round(.09 * pitch):right - round(.09 * pitch)]
    chroma = np.ptp(patch, axis=2) >= 46
    off_surface = np.max(abs(patch.astype(np.int16) - median_sample), axis=2) >= 51
    return chosen if (chroma & off_surface).mean() >= .71 else None


def _parts(gray, left, top, right, bottom):
    w, h = right - left, bottom - top
    y0, y1 = top + round(.135 * h), bottom - round(.115 * h)
    a, mid, b = left + round(.098 * w), left + round(.343 * w), right - round(.095 * w)
    paper = np.median(gray[y0:y1, a:b])
    icon = _tight(gray[y0:y1, a:mid] - paper >= 30, a, y0)
    words = _tight(gray[y0:y1, mid:b] - paper >= 31, mid, y0)
    if not icon or not words:
        return None
    x, y, r, d = icon
    xa, ya, xb, yb = words
    if (not .24 * h <= d - y <= .64 * h or not .54 <= (r - x) / (d - y) <= 1.54
            or not .27 * h <= yb - ya <= .63 * h or not .28 * w <= xb - xa <= .55 * w
            or abs(y + d - ya - yb) > .20 * h or not .018 * w <= xa - r <= .085 * w):
        return None
    return words


def _mixed_title(frame, gray, crop, index, cache):
    """Read an all-caps prefix immediately followed by wider ideographs."""
    x, y, right, bottom = crop
    tile = gray[y:bottom, x:right]
    low, high = np.percentile(tile, (25, 98))
    if high - low < 52:
        return None
    observations = []
    for ratio in (.29, .39, .49, .59):
        mask = np.uint8(tile >= low + ratio * (high - low))
        a, b, w, h = cv2.boundingRect(mask)
        if h < 12:
            continue
        line = mask[b:b + h, a:a + w]
        spans = _bands(line.any(axis=0))
        split = next((i for i, (l, r) in enumerate(spans) if r - l >= .76 * h), len(spans))
        if not 3 <= split <= 6 or len(spans) - split < 2:
            continue
        boundary = spans[split][0]
        if not 2 <= len(_cjk_parts(line[:, boundary:], h)) <= 5:
            continue
        prefix = []
        for l, r in spans[:split]:
            glyph = line[:, l:r]
            _, gy, _, gh = cv2.boundingRect(glyph)
            if not (.72 * h <= gh <= .96 * h and .06 * h <= r - l <= .69 * h and abs(gy + gh - h) <= .17 * h):
                break
            options = index.rank(glyph, alphabet="code", limit=2)
            if (len(options) < 2 or not options[0]["text"].isalpha() or options[0]["distance"] > .116
                    or options[1]["distance"] - options[0]["distance"] < .046):
                break
            prefix.append(options[0]["text"])
        if len(prefix) == split:
            observations.append(("".join(prefix), x + a + boundary, x + a + spans[split - 1][1], h))
    if len(observations) < 2 or len({part[0] for part in observations}) != 1:
        return None
    name, boundary, preceding, h = observations[0]
    suffix_box = max(preceding + 1, boundary - round(.20 * h)), y, right, bottom
    reading = read_caption(frame, suffix_box, index, cache=cache)
    return name + reading["text"] if reading and all("\u3400" <= ch <= "\u9fff" for ch in reading["text"]) else None


def grouped_shop_navigation_actions(frame, regions, index):
    """Return one literal target for each named inactive horizontal shop tab."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or width < 1.45 * height or height < 260:
        return []
    row = _rows(regions["candidates"], width, height)
    if row is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    strip = _strip(gray, row)
    if strip is None:
        return []
    edges, top, bottom, pitch = strip
    if not _adjacent_groups(gray, regions["candidates"], edges, top, bottom, pitch):
        return []
    selected = _selected(frame, edges, top, bottom, pitch)
    if selected is None:
        return []
    cells = list(zip(edges[:-1], edges[1:], strict=True))
    parts = [(i, _parts(gray, a, top, b, bottom)) for i, (a, b) in enumerate(cells) if i != selected]
    if sum(words is not None for _, words in parts) < len(cells) - 2:
        return []
    actions, cache = [], {}
    for i, words in parts:
        if words is None:
            continue
        x, y, right, lower = words
        pad = max(3, round(.23 * (lower - y)))
        a, b = cells[i]
        crop = max(a, x - pad), max(top, y - pad), min(b, right + pad), min(bottom, lower + pad)
        reading = read_caption(frame, crop, index, cache=cache)
        name = reading["text"] if reading else _caption(gray, crop, index, cache) or _mixed_title(frame, gray, crop, index, cache)
        if name:
            actions.append({"role": "game_shop_tab", "function": f"切换到{name}", "preview_text": name,
                            "box": [a, top, b, bottom], "click_point": [(x + right) // 2, (y + lower) // 2]})
    counts = Counter(action["function"] for action in actions)
    return [action for action in actions if counts[action["function"]] == 1]
