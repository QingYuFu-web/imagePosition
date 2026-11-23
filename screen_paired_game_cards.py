"""Verify two equally framed landscape option cards and their printed titles."""

from collections import Counter, defaultdict
from itertools import combinations

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts
from screen_component_captions import _bands


def _roof(gray, box):
    left, top, right, bottom = box
    height, width = bottom - top, right - left
    stride = max(2, round(.008 * width))
    start, end = max(stride, top - round(.05 * height)), min(len(gray), top + round(.07 * height))
    inset = round(.14 * width)
    band = gray[start:end, left + inset:right - inset]
    delta = band - gray[start - stride:end - stride, left + inset:right - inset]
    measure = ((delta >= 42) & (band >= 91)).mean(axis=1)
    row = int(measure.argmax())
    return start + row if measure[row] >= .82 else None


def _wall(gray, x, center, width, y0, y1):
    width_pad = round(.072 * width)
    step = max(2, round(.015 * width))
    inside = 1 if x < center else -1
    start, end = max(step, x - width_pad), min(gray.shape[1] - step, x + width_pad)
    band = gray[y0:y1:2, start:end]
    background = gray[y0:y1:2, start - inside * step:end - inside * step]
    filled = ((band - background >= 24) & (band >= 58)).mean(axis=0)
    col = int(filled.argmax())
    return start + col if filled[col] >= .65 else None


def _surface(gray, box):
    left, top, right, bottom = box
    h, w = bottom - top, right - left
    head = _roof(gray, box)
    if head is None:
        return None
    y0, y1 = head + round(.09 * h), head + round(.63 * h)
    center = (left + right) / 2
    edges = [_wall(gray, x, center, w, y0, y1) for x in (left, right)]
    if None in edges or not .85 * w <= edges[1] - edges[0] <= 1.10 * w:
        return None
    x0, x1 = edges
    width = x1 - x0
    art = gray[head + round(.26 * h):head + round(.77 * h):3, x0 + round(.08 * width):x1 - round(.08 * width):3]
    if art.std() < 27 or np.abs(np.diff(art, axis=1)).mean() < 7:
        return None
    return (x0, head, x1 + 1, bottom), art


def _roof_peer(gray, box):
    """Search the existing roof row for a second detached, equal-length edge."""
    left, top, right, bottom = box
    width = right - left
    head = _roof(gray, box)
    if head is None:
        return None
    reach = max(2, round(.012 * width))
    first, last = head - reach, head + reach + 1
    rows = gray[first:last]
    delta = rows - gray[first - reach:last - reach]
    ink = np.uint8(((delta >= 42) & (rows >= 91)).any(axis=0))[None, :]
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, max(3, round(.025 * width))), np.uint8))[0]
    segments = [(a, b) for a, b in _bands(closed) if .67 * width <= b - a <= 1.07 * width]
    primary = [segment for segment in segments if left <= segment[0] < segment[1] <= right]
    if len(primary) != 1:
        return None
    start, stop = primary[0]
    peers = []
    for a, b in segments:
        advance = a - start
        if (1.013 * width <= abs(advance) <= 1.29 * width and .89 * (stop - start) <= b - a <= 1.11 * (stop - start)
                and 0 <= left + advance < right + advance <= gray.shape[1]):
            peers.append((left + advance, top, right + advance, bottom))
    return peers[0] if len(peers) == 1 else None


def _caption(frame, box, index, cache):
    left, top, right, bottom = box
    width = right - left
    x = (left + right) / 2
    patch = frame[top + round(.095 * width):top + round(.23 * width),
                  round(x - .268 * width):round(x + .268 * width)]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    candidates = (gray, patch.min(axis=2), cv2.GaussianBlur(gray, (3, 3), .46))
    votes = defaultdict(set)
    for family, values in enumerate(candidates):
        low, high = np.percentile(values, (28, 98))
        if high - low < 55:
            continue
        for fraction in (.255, .31, .365, .41):
            mask = np.uint8(values >= low + fraction * (high - low))
            a, b, w, h = cv2.boundingRect(mask)
            if (not .062 * width <= h <= .132 * width or abs(a + w / 2 - patch.shape[1] / 2) > .055 * width
                    or a == 0 or a + w >= patch.shape[1] or b == 0 or b + h >= patch.shape[0]):
                continue
            word = []
            parts = _cjk_parts(mask[b:b + h, a:a + w], h)
            if not 3 <= len(parts) <= 8:
                continue
            for *_, glyph in parts:
                key = glyph.shape, glyph.tobytes()
                if key not in cache:
                    cache[key] = index.rank(glyph, alphabet="cjk", limit=2)
                options = cache[key]
                if (len(options) != 2 or options[0]["distance"] > .227
                        or options[1]["distance"] - options[0]["distance"] < .010):
                    break
                word.append(options[0]["text"])
            if len(word) == len(parts):
                votes["".join(word)].add((family, mask.shape, mask.tobytes()))
    agreed = [word for word, cuts in votes.items() if len(cuts) >= 2 and len({cut[0] for cut in cuts}) >= 2]
    return agreed[0] if len(agreed) == 1 else None


def paired_game_card_actions(frame, regions, index):
    """Return one action per literally named card in an independently bounded pair."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or height < 220 or width < 1.45 * height:
        return []
    carriers = []
    for candidate in regions["candidates"]:
        left, top, right, bottom = candidate["box"]
        w, h = right - left, bottom - top
        if (candidate.get("visibility") == "page" and 0 <= left < right <= width and 0 <= top < bottom <= height
                and .36 * height <= h <= .78 * height and .12 * width <= w <= .29 * width
                and 1.33 * w <= h <= 2.3 * w and .10 * height <= top <= .47 * height):
            carriers.append((left, top, right, bottom))
    pairs = []
    for a, b in combinations(sorted(carriers), 2):
        typical = a[2] - a[0]
        if (.025 * typical <= b[0] - a[2] <= .23 * typical and .90 * typical <= b[2] - b[0] <= 1.11 * typical
                and abs(a[1] - b[1]) <= .061 * typical
                and .82 * (a[3] - a[1]) <= b[3] - b[1] <= 1.22 * (a[3] - a[1])):
            pairs.append((a, b))
    if len(pairs) != 1 and len(carriers) != 1:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    if pairs:
        pair = pairs[0]
    else:
        peer = _roof_peer(gray, carriers[0])
        if peer is None:
            return []
        pair = sorted((carriers[0], peer))
    faces = [_surface(gray, box) for box in pair]
    if any(face is None for face in faces):
        return []
    (first, first_art), (second, second_art) = faces
    if abs(first[1] - second[1]) > max(2, .009 * height):
        return []
    resized_art = [cv2.resize(art.astype(np.float32), (64, 48), interpolation=cv2.INTER_AREA) for art in (first_art, second_art)]
    if np.mean(abs(resized_art[0] - resized_art[1])) < 18:
        return []
    cache = {}
    actions = []
    for box in (first, second):
        if caption := _caption(frame, box, index, cache):
            left, top, right, bottom = box
            actions.append({"role": "game_mode", "function": f"打开{caption}", "preview_text": caption,
                            "box": list(box), "click_point": [(left + right) // 2, top + round(.38 * (bottom - top))]})
    counts = Counter(item["function"] for item in actions)
    return [item for item in actions if counts[item["function"]] == 1]
