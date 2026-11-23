"""Scope a framed game panel with a separate selected navigation column."""

from collections import Counter

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts, read_caption
from screen_component_captions import _bands
from screen_game_offer_cards import _title
from screen_wide_panel import _diagonal_cross


def _close(frame, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    dx, dy = round(.20 * w), round(.20 * h)
    tile = frame[top + dy:bottom - dy, left + dx:right - dx]
    lo = tile.min(axis=2)
    stable = []
    for cut in (211, 224, 237):
        mask = np.uint8(lo >= cut)
        x, y, a, b = cv2.boundingRect(mask)
        if (.225 * w <= a <= .57 * w and .225 * h <= b <= .57 * h and .76 <= a / b <= 1.31
                and abs(x + a / 2 - tile.shape[1] / 2) <= .10 * w
                and abs(y + b / 2 - tile.shape[0] / 2) <= .10 * h
                and _diagonal_cross(mask[y:y + b, x:x + a] > 0)):
            stable.append((left + dx + x, top + dy + y, left + dx + x + a, top + dy + y + b))
    if len(stable) >= 2 and np.ptp(stable, axis=0).max() <= max(3, .075 * h):
        return stable[-1]
    return None


def _side_rows(frame, boxes, header):
    left, top, _, _ = header
    image_height, image_width = frame.shape[:2]
    rows = sorted((box for box in boxes if .078 * image_width <= box[2] - box[0] <= .19 * image_width
                   and .084 * image_height <= box[3] - box[1] <= .188 * image_height
                   and abs(box[2] - left) <= .024 * image_width and box[1] >= top),
                  key=lambda box: box[1])
    if not 4 <= len(rows) <= 9:
        return None
    values = np.array(rows)
    pitch = np.diff(values[:, 1])
    normal = float(np.median(pitch))
    if not (.071 * image_height <= normal <= .175 * image_height and np.ptp(pitch) <= .19 * normal
            and np.ptp(values[:, 0]) <= .016 * image_width and np.ptp(values[:, 2]) <= .016 * image_width
            and np.ptp(values[:, 3] - values[:, 1]) <= .22 * normal):
        return None
    colors = []
    for left, top, right, bottom in rows:
        w, h = right - left, bottom - top
        chips = [frame[top + round(.16 * h):top + round(.29 * h),
                       right - round(.20 * w):right - round(.10 * w)],
                 frame[top + round(.48 * h):top + round(.60 * h),
                       right - round(.158 * w):right - round(.075 * w)]]
        colors.append([np.median(chip[::2, ::2], axis=(0, 1)) for chip in chips])
    paint = np.array(colors)
    ordinary = np.median(paint, axis=0)
    contrast = abs(paint - ordinary).max(axis=2)
    selected = np.flatnonzero(contrast.min(axis=1) >= 31)
    if len(selected) != 1 or np.count_nonzero(contrast.max(axis=1) <= 19) < len(rows) - 1:
        return None
    return rows, int(selected[0]), ordinary.mean(axis=0)


def _disabled_peers(frame, boxes, header, bottom):
    left, _, right, start = header
    span, height = right - left, frame.shape[0]
    peers = []
    for l, t, r, b in boxes:
        w, h = r - l, b - t
        if not (.124 * span <= w <= .30 * span and .046 * height <= h <= .105 * height
                and .022 * span <= right - l <= .30 * span and abs(right - r) <= .119 * span
                and start < t < b <= bottom):
            continue
        x, y = round(.16 * w), round(.22 * h)
        tile = frame[t + y:b - y:2, l + x:r - x:2]
        tone = np.median(tile, axis=(0, 1))
        if (np.ptp(tone) <= 19 and 47 <= np.mean(tone) <= 176
                and (np.ptp(tile, axis=2) <= 20).mean() >= .73):
            peers.append((l, t, r, b))
    peers.sort(key=lambda p: p[1])
    for a, b, c in zip(peers, peers[1:], peers[2:]):
        first, second = b[1] - a[1], c[1] - b[1]
        if (.087 * height <= min(first, second) <= max(first, second) <= .25 * height
                and abs(first - second) <= .13 * first
                and np.ptp([p[0] for p in (a, b, c)]) <= .021 * span):
            return True
    return False


def _rims(frame, header, bottom):
    left, top, right, _ = header
    h, w = frame.shape[:2]
    a, b = left + round(.045 * (right - left)), right - round(.081 * (right - left))
    dy = max(2, round(.0024 * h))
    reach = round(.026 * h)
    def row(nominal):
        values = []
        for y in range(max(dy, nominal - reach), min(h - dy, nominal + reach + 1)):
            diff = abs(frame[y - dy, a:b:4].astype(np.int16) - frame[y + dy, a:b:4].astype(np.int16)).max(axis=1)
            if (diff >= 29).mean() >= .84:
                values.append((float(np.median(diff)), y))
        return max(values)[1] if values else None

    head, tail = row(top), row(bottom)
    if head is None or tail is None or tail - head < .52 * h:
        return None
    start, stop = head + round(.14 * h), tail - round(.079 * h)
    for col in range(max(2, right - round(.018 * w)), min(w - 2, right + round(.009 * w))):
        diff = abs(frame[start:stop:3, col - 2].astype(np.int16) - frame[start:stop:3, col + 2].astype(np.int16)).max(axis=1)
        if (diff >= 26).mean() >= .78:
            return head, tail
    return None


def _read_thin_strokes(tile, index):
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, (2, 65))
    if high - low < 68:
        return None
    reads = Counter()
    seen = set()
    for fraction in (.20, .225, .25, .275, .30, .325, .35):
        mask = np.uint8(gray <= high - fraction * (high - low))
        x, y, w, h = cv2.boundingRect(mask)
        if not (.57 * len(gray) <= h <= .83 * len(gray) and .82 * gray.shape[1] <= w <= .98 * gray.shape[1]
                and abs(2 * x + w - gray.shape[1]) <= .065 * gray.shape[1]
                and .14 <= mask[y:y + h, x:x + w].mean() <= .52):
            continue
        line = mask[y:y + h, x:x + w]
        key = line.shape, line.tobytes()
        if key in seen:
            continue
        seen.add(key)
        parts = _cjk_parts(line, h, join_radicals=True)
        if not 2 <= len(parts) <= 8:
            continue
        letters, distances, margins = [], [], []
        for *_, glyph in parts:
            options = index.rank(glyph, alphabet="cjk", limit=2)
            if len(options) != 2:
                break
            distance = options[0]["distance"]
            margin = options[1]["distance"] - distance
            if distance > .197 or margin < .011:
                break
            letters.append(options[0]["text"])
            distances.append(distance)
            margins.append(margin)
        else:
            if np.mean(distances) <= .157 and np.median(margins) >= .032:
                reads["".join(letters)] += 1
    if len(reads) == 1:
        text, count = next(iter(reads.items()))
        return text if count >= 3 else None
    return None


def _title_text(frame, box, paint, index, cache):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    a, b = left + round(.212 * w), right - round(.112 * w)
    first, last = top + round(.15 * h), bottom - round(.12 * h)
    image = frame[first:last, a:b]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tone = cv2.cvtColor(np.uint8([[paint]]), cv2.COLOR_BGR2GRAY).item()
    mask = np.uint8(gray <= tone - 60)
    bands = [(t, z) for t, z in _bands(mask.sum(axis=1) >= max(5, round(.023 * w)))
             if .129 * h <= z - t <= .34 * h]
    if not 1 <= len(bands) <= 2:
        return None
    if len(bands) == 1 and abs(first + sum(bands[0]) / 2 - (top + bottom) / 2) > .10 * h:
        return None
    lines = []
    for t, z in bands:
        spans = _bands(mask[t:z].any(axis=0))
        depth = z - t
        if len(spans) >= 4 and max(spans[0][1] - spans[0][0], spans[-1][1] - spans[-1][0]) <= .47 * depth:
            spans = spans[1:-1]
        if not spans:
            return None
        pad = max(3, round(.145 * depth))
        area = (a + max(0, spans[0][0] - pad), first + t - pad,
                a + min(b - a, spans[-1][1] + pad), first + z + pad)
        record = read_caption(frame, area, index, cache=cache, choice=True)
        if record:
            lines.append(record["text"])
        else:
            lx, ty, rx, by = area
            tile = frame[ty:by, lx:rx]
            label = _title(255 - tile, index) or _read_thin_strokes(tile, index)
            if label is None:
                return None
            lines.append(label)
    return "".join(lines)


def game_section_panel_actions(frame, regions, index):
    """Return the foreground bounds, safe close point, and verified inactive sections."""
    height, width = frame.shape[:2]
    if regions["layer"] not in {"page", "layer_uncertain"} or height < 260 or width < 1.48 * height:
        return None
    boxes = [tuple(row["box"]) for row in regions["candidates"]]
    proof = []
    for header in boxes:
        left, top, right, bottom = header
        if not (.32 * width <= right - left <= .64 * width and .103 * height <= bottom - top <= .32 * height
                and .050 * height <= top <= .28 * height and .56 * width <= right <= .94 * width):
            continue
        side = _side_rows(frame, boxes, header)
        if side is None:
            continue
        rows, selected, paint = side
        rims = _rims(frame, header, rows[-1][3])
        if rims is None or not _disabled_peers(frame, boxes, header, rims[1]):
            continue
        for box in boxes:
            a, b, c, d = box
            w, h = c - a, d - b
            if (.035 * height <= min(w, h) <= max(w, h) <= .105 * height and .68 <= w / h <= 1.47
                    and abs((a + c) / 2 - right) <= .065 * (right - left)
                    and top <= (b + d) / 2 <= top + .107 * height and (cross := _close(frame, box))):
                proof.append((header, rows, selected, paint, rims, box, cross))
    if len(proof) != 1:
        return None
    header, rows, selected, paint, (head, tail), owner, glyph = proof[0]
    l, t, r, b = glyph
    actions = [{"role": "close", "function": "关闭", "preview_text": "关闭", "box": list(owner),
                "click_point": [(l + r) // 2, (t + b) // 2]}]
    if index is not None:
        cache = {}
        for i, box in enumerate(rows):
            if i == selected or not (label := _title_text(frame, box, paint, index, cache)):
                continue
            l, t, r, b = box
            first = max(t, (rows[i - 1][3] + t) // 2) if i else t
            last = min(b, (b + rows[i + 1][1]) // 2) if i + 1 < len(rows) else b
            actions.append({"role": "game_panel_section", "function": f"打开{label}", "preview_text": label,
                            "box": [l, first, min(r, header[0]), last], "click_point": [(l + r) // 2, (t + b) // 2]})
    return (min(box[0] for box in rows), min(head, owner[1]), max(header[2], owner[2]),
            max(tail, rows[-1][3])), actions
