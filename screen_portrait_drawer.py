"""Own repeated icon-caption rows in a visually isolated portrait drawer."""

from collections import Counter
import re

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_native_tab_bar import _distance


def _seam_positions(gray):
    height, width = gray.shape
    start, stop = round(.47 * width), round(.91 * width)
    stripe = gray[round(.225 * height):round(.864 * height):max(3, height // 220), start - 1:stop].astype(np.int16)
    delta = stripe[:, :-1] - stripe[:, 1:]
    fraction = (delta >= 32).mean(axis=0)
    for a, b in _bands(fraction >= .89):
        if b - a <= max(3, .008 * width):
            best = max(range(a, b), key=lambda i: (fraction[i], np.median(delta[:, i])))
            yield start + best


def _masked_overflow_rail(gray, left, top, bottom, base, width):
    strip = gray[top:bottom, left:]
    cut = np.uint8(strip <= base - max(23, .23 * base))
    _, _, components, centers = cv2.connectedComponentsWithStats(cut, connectivity=8)
    dots = [(float(x), float(y), (w + h) / 2) for (_, _, w, h, area), (x, y)
            in zip(components[1:], centers[1:], strict=True)
            if .006 * width <= min(w, h) <= max(w, h) <= .017 * width
            and .79 <= w / h <= 1.27 and .59 <= area / (w * h) <= .91]
    if not 9 <= len(dots) <= 180:
        return False
    groups = []
    for x, y, size in dots:
        peers = sorted(b for a, b, d in dots if abs(a - x) <= .25 * size and .74 * size <= d <= 1.26 * size
                       and 1.11 * size <= abs(y - b) <= 1.94 * size)
        if len(peers) == 2 and peers[0] < y < peers[1] and abs(peers[0] + peers[1] - 2 * y) <= .24 * size:
            groups.append((x, y, size))
    for x, _, size in groups:
        lane = sorted(y for a, y, d in groups if abs(a - x) <= .41 * size and .79 * size <= d <= 1.21 * size)
        gaps = np.diff(lane)
        for a, b in zip(gaps, gaps[1:]):
            if 5.0 * size <= min(a, b) <= max(a, b) <= .25 * (bottom - top) and abs(a - b) <= .18 * max(a, b):
                return True
    return False


def _quiet_scrim(gray, seam, top, bottom, backdrop):
    """Prove a drawer shadow and a dimmed real UI even on a flat canvas."""
    width = gray.shape[1]
    depth = max(12, round(.042 * width))
    if seam + 2 * depth >= width or backdrop < 38:
        return False
    for portion in np.array_split(gray[top:bottom:3, seam:seam + depth], 3):
        profile = np.median(portion, axis=0)
        start = float(np.median(profile[:max(2, depth // 9)]))
        end = float(np.median(profile[-max(3, depth // 5):]))
        if not (7 <= end - start <= .29 * backdrop and abs(end - backdrop) < 8
                and (np.diff(profile) >= -1.5).mean() >= .92):
            return False
    return _masked_overflow_rail(gray, seam + depth, top, bottom, backdrop, width)


def _painted_scope(view, gray, seam):
    height, width = gray.shape
    start, stop = round(.44 * seam), round(.885 * seam)
    panel = view[:, start:stop:4]
    paint = np.median(panel[round(.22 * height):round(.895 * height)], axis=(0, 1))
    light = float(np.median(gray[round(.23 * height):round(.89 * height):4, start:stop:4]))
    if light < 118:
        return None
    sheet = (_distance(panel, paint) <= 13).mean(axis=1) >= .83
    force = gray[:, seam - 1].astype(np.int16) - gray[:, seam]
    joint = np.uint8((sheet & (force >= 29))[:, None])
    seal = max(3, round(.037 * width) | 1)
    solid = cv2.morphologyEx(joint, cv2.MORPH_CLOSE, np.ones((seal, 1), np.uint8)).ravel()
    runs = [(a, b) for a, b in _bands(solid != 0) if a <= .30 * height and b >= .875 * height
            and b - a >= .52 * height]
    if len(runs) != 1:
        return None
    body, bottom = runs[0]
    if (force[body:bottom] >= 31).mean() < .89:
        return None
    if body >= .05 * height:
        head = abs(view[:body, seam - 1].astype(np.int16) - view[:body, seam]).max(axis=1)
        if (head >= 17).mean() < .64:
            return None
    pad, tail = round(.021 * width), round(.019 * width)
    exterior = gray[body:bottom:3, seam + pad:width - tail:2].astype(np.int16)
    if exterior.shape[1] < .064 * width:
        return None
    floor, bright, peak = np.percentile(exterior, (12, 90, 98))
    if bright > .64 * light or peak > .76 * light:
        return None
    textured = bright - floor >= 23 and (abs(np.diff(exterior, axis=0)) >= 5).mean() >= .042
    if not textured and (bright - floor > 16 or not _quiet_scrim(gray, seam, body, bottom, bright)):
        return None
    if bottom <= .965 * height:
        footer = gray[bottom + round(.008 * height):bottom + round(.027 * height), seam - pad:seam + pad]
        if footer.shape[0] < 3 or (abs(footer.astype(np.int16) - np.median(footer)) <= 14).mean() < .91:
            return None
    return body, bottom, paint, textured


def _drawer_bounds(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    width = frame.shape[1]
    hypotheses = []
    for reverse in (False, True):
        candidate = gray[:, ::-1] if reverse else gray
        for seam in _seam_positions(candidate):
            proof = _painted_scope(frame[:, ::-1] if reverse else frame, candidate, seam)
            if proof:
                top, bottom, paint, textured = proof
                bounds = (width - seam, 0, width, bottom) if reverse else (0, 0, seam, bottom)
                hypotheses.append((bounds, top, paint, textured))
    return hypotheses[0] if len(hypotheses) == 1 else None


def _split_lane(mask):
    _, width = mask.shape
    presence = mask.mean(axis=0)
    gaps = [(a, b) for a, b in _bands(presence <= .017)
            if .077 * width <= a < b <= .35 * width and .028 * width <= b - a <= .131 * width
            and presence[max(0, a - round(.077 * width)):a].max(initial=0) >= .047
            and presence[b:b + round(.078 * width)].max(initial=0) >= .053]
    if not gaps:
        return None
    first, last = max(gaps, key=lambda ends: ends[1] - ends[0])
    return (first + last) // 2


def _row_shapes(frame, box, body, paint):
    left, _, right, bottom = box
    width = right - left
    mask = _distance(frame[body:bottom, left:right], paint) >= 42
    divider = _split_lane(mask)
    if divider is None:
        return None
    captions = mask[:, divider:round(.96 * width)]
    ink = np.uint8((captions.sum(axis=1) >= max(4, round(.006 * width)))[:, None])
    kernel = np.ones((max(3, round(.0068 * width) | 1), 1), np.uint8)
    strokes = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel).ravel()
    items = []
    for a, b in _bands(strokes != 0):
        if not .027 * width <= b - a <= .105 * width:
            continue
        x, y, w, h = cv2.boundingRect(np.uint8(captions[max(0, a - 3):b + 3]))
        start = max(0, a - 3)
        cell = (divider + x, body + start + y, divider + x + w, body + start + y + h)
        if not .127 * width <= cell[0] <= .36 * width or not .95 * h <= w <= .653 * width:
            continue
        center = (cell[1] + cell[3]) / 2 - body
        y0, y1 = max(0, round(center - .71 * h)), min(len(mask), round(center + .71 * h))
        side = mask[y0:y1, :divider]
        x, y, w, ih = cv2.boundingRect(np.uint8(side))
        if (not .51 * h <= min(w, ih) <= max(w, ih) <= 1.48 * h
                or not .59 <= w / ih <= 1.61 or abs(y + ih / 2 + y0 - center) > .29 * h
                or not .030 * width <= x <= .155 * width):
            continue
        caption = (left + cell[0], cell[1], left + cell[2], cell[3])
        items.append((caption, left + x, center + body, h))
    if not 5 <= len(items) <= 19:
        return None
    typical = float(np.median([item[3] for item in items]))
    origin = float(np.median([item[0][0] for item in items]))
    icon_left = float(np.median([item[1] for item in items]))
    stable = [item for item in items if abs(item[0][0] - origin) <= .022 * width
              and abs(item[1] - icon_left) <= .036 * width and .82 * typical <= item[3] <= 1.20 * typical]
    if len(stable) < max(5, len(items) - 2):
        return None
    gaps = np.diff([item[2] for item in stable])
    repeats = gaps[(gaps >= 1.78 * typical) & (gaps <= 4.56 * typical)]
    if len(repeats) < 4:
        return None
    pitch = float(np.median(repeats))
    if ((abs(gaps - pitch) <= .105 * pitch).sum() < max(4, len(gaps) - 3)
            or not .082 * width <= pitch <= .238 * width):
        return None
    return stable, pitch


def _glyph_box(frame, caption, index, cache):
    x, y, right, bottom = caption
    pad = max(3, round(.13 * (bottom - y)))
    area = (max(0, x - pad), max(0, y - pad), min(frame.shape[1], right + pad), min(frame.shape[0], bottom + pad))
    read = read_caption(frame, area, index, cache=cache, choice=True)
    return read["text"] if read else None


def _fallback_caption(frame, caption, paint, index, cache):
    x, y, right, bottom = caption
    height = bottom - y
    tile = frame[y:bottom, x:right].astype(np.int16)
    contrast = abs(tile - paint).max(axis=2)
    spans = _bands((contrast >= 58).any(axis=0))
    gaps = [(b - a, a, b) for (_, a), (b, _) in zip(spans[:-1], spans[1:], strict=True)]
    gaps.sort(reverse=True)
    if (len(gaps) >= 2 and .295 * height <= gaps[0][0] <= .99 * height
            and gaps[0][0] >= 1.57 * gaps[1][0]):
        split = x + (gaps[0][1] + gaps[0][2]) // 2
        first, second = _glyph_box(frame, (x, y, split, bottom), index, cache), _glyph_box(frame, (split, y, right, bottom), index, cache)
        if first and second:
            return first + " " + second
    votes = {}
    for level in (54, 84, 118, 151):
        mask = np.uint8(contrast >= level)
        kernel = np.ones((1, max(3, round(.064 * height) | 1)), dtype=np.uint8)
        groups = _bands(cv2.morphologyEx(np.uint8(mask.any(axis=0)[None, :]), cv2.MORPH_CLOSE, kernel).ravel() != 0)
        if not 2 <= len(groups) <= 9:
            continue
        glyphs = []
        for a, b in groups:
            _, _, w, h = cv2.boundingRect(mask[:, a:b])
            if not .60 * height <= min(w, h) <= max(w, h) <= 1.13 * height:
                break
            rank = index.rank(mask[:, a:b], alphabet="cjk", limit=2)
            if len(rank) != 2 or rank[0]["distance"] > .162 or rank[1]["distance"] - rank[0]["distance"] < .027:
                break
            glyphs.append(rank[0]["text"])
        if len(glyphs) == len(groups):
            votes.setdefault("".join(glyphs), set()).add(mask.tobytes())
    verdicts = [text for text, variants in votes.items() if len(variants) >= 2]
    return verdicts[0] if len(verdicts) == 1 else None


def portrait_drawer_actions(frame, index):
    """Return the isolated drawer box and its literally named menu entries."""
    height, width = frame.shape[:2]
    if index is None or width < 255 or not 1.46 * width <= height <= 2.76 * width:
        return None
    scope = _drawer_bounds(frame)
    if scope is None:
        return None
    box, body, paint, textured = scope
    if not textured:
        from screen_grouped_drawer_rows import grouped_drawer_rows

        return (box, actions) if (actions := grouped_drawer_rows(frame, box, body, paint, index)) else None
    proof = _row_shapes(frame, box, body, paint)
    if proof is None:
        return None
    rows, pitch = proof
    left, _, right, bottom = box
    inset = right - left
    actions, cache = [], {}
    for i, (caption, _, center, _) in enumerate(rows):
        text = _glyph_box(frame, caption, index, cache) or _fallback_caption(frame, caption, paint, index, cache)
        if not text or re.search(r"\d+(?:\.\d+){1,4}(?:[-+][a-z0-9.]+)?", text, flags=re.I):
            continue
        first, last = max(body, round(center - pitch / 2)), min(bottom, round(center + pitch / 2))
        if i:
            first = max(first, round((rows[i - 1][2] + center) / 2))
        if i + 1 < len(rows):
            last = min(last, round((center + rows[i + 1][2]) / 2))
        x = (caption[0] + caption[2]) // 2
        actions.append({"role": "app_drawer_row", "function": f"打开{text}", "preview_text": text,
                        "box": [left + round(.047 * inset), first, right - round(.012 * inset), last],
                        "click_point": [x, round(center)]})
    counts = Counter(row["function"] for row in actions)
    stable = [row for row in actions if counts[row["function"]] == 1]
    return (box, stable) if len(stable) >= max(3, len(rows) // 2) else None
