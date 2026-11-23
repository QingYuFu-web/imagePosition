"""Find native landscape icon-and-caption tabs on a shared bottom surface."""

from collections import Counter, defaultdict
from functools import lru_cache
import string

import cv2
import numpy as np
from PIL import ImageFont

from screen_component_captions import _bands, _outlier, _tight
from screen_native_tab_bar import _action, _distance, _ink_color, _transcript


def _grids(candidates, width, height):
    rows = []
    for candidate in candidates:
        left, top, right, bottom = candidate["box"]
        if (candidate.get("visibility") == "page" and 0 <= left < right <= width
                and 0 <= top < bottom <= height and bottom >= .970 * height
                and .098 * height <= bottom - top <= .225 * height):
            rows.append((left, top, right, bottom))
    for n in range(3, 7):
        edges = [round(i * width / n) for i in range(n + 1)]
        tolerance = .112 * width / n
        fitting = [[row for row in rows if abs(left - row[0]) <= tolerance and abs(right - row[2]) <= tolerance]
                   for left, right in zip(edges[:-1], edges[1:], strict=True)]
        if all(len(matches) == 1 for matches in fitting):
            tops = [matches[0][1] for matches in fitting]
            if np.ptp(tops) <= .030 * height:
                yield edges, round(np.median(tops))


def _anchor(frame, approximate):
    height, width = frame.shape[:2]
    inset = round(.023 * width)
    painted = frame[height - round(.05 * height):, inset:width - inset]
    paint = np.median(painted[::2, ::4], axis=(0, 1))
    gap = max(2, round(.0065 * height))
    start, stop = approximate - round(.022 * height), approximate + round(.026 * height)
    surround = _distance(frame[start:stop + gap, inset:width - inset], paint)
    rails = [(float(np.median(a - b.astype(np.int16))), start + i + 1)
             for i, (a, b) in enumerate(zip(surround[:-gap], surround[gap:], strict=True))
             if ((a.astype(np.int16) - b) >= 9).mean() >= .86 and (b <= 11).mean() >= .90]
    if not rails:
        return None
    _, top = max(rails)
    safe = height - round(.052 * height)
    mask = np.uint8(_distance(frame[safe:], paint) >= 36)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count != 2:
        return None
    x, y, w, h, area = stats[1]
    if (not .049 * width <= w <= .151 * width or not .0035 * height <= h <= .020 * height
            or abs(x + w / 2 - width / 2) > .018 * width or w / h < 8
            or area < .65 * w * h or safe + y + h >= height):
        return None
    return top, int(safe + y), paint


def _parts(frame, edges, top, gesture, paint):
    depth, width = gesture - top, frame.shape[1]
    if not .088 * frame.shape[0] <= depth <= .195 * frame.shape[0]:
        return None
    bottom = gesture - round(.115 * depth)
    area = frame[top + round(.073 * depth):bottom]
    shift = top + round(.073 * depth)
    distance = _distance(area, paint)
    mask = distance >= 24
    break_at = top + round(.493 * depth) - shift
    if (distance <= 13).mean() < .89:
        return None
    parts = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        margin = round(.12 * (right - left))
        a, b = left + margin, right - margin
        icon = _tight(mask[:break_at, a:b], a, shift)
        label = _tight(mask[break_at:, a:b], a, shift + break_at)
        if not icon or not label:
            return None
        ix, iy, iX, iY = icon
        lx, ly, lX, lY = label
        iw, ih, lw, lh = iX - ix, iY - iy, lX - lx, lY - ly
        center = (left + right) / 2
        if (not .068 * depth <= ih <= .43 * depth or not .13 * depth <= iw <= .48 * depth
                or not .58 <= iw / ih <= 4.22 or not .11 * depth <= lh <= .34 * depth
                or not .70 * lh <= lw <= .74 * (right - left)
                or not .083 * depth <= ly - iY <= .315 * depth
                or abs((ix + iX) / 2 - center) > .023 * width
                or abs((lx + lX) / 2 - center) > .024 * width):
            return None
        colors = [_ink_color(frame[y:Y, x:X], _distance(frame[y:Y, x:X], paint))
                  for x, y, X, Y in (icon, label)]
        parts.append((icon, label, *colors))
    if np.ptp([label[1] for _, label, _, _ in parts]) > .07 * depth:
        return None
    icon_colors, label_colors = ([item[i] for item in parts] for i in (2, 3))
    selected = _outlier(icon_colors, same_max=42, gap=43)
    if selected is None or selected != _outlier(label_colors, same_max=43, gap=38):
        return None
    others = [i for i in range(len(parts)) if i != selected]
    icon_tone = icon_colors[selected] - np.median([icon_colors[i] for i in others], axis=0)
    label_tone = label_colors[selected] - np.median([label_colors[i] for i in others], axis=0)
    if (np.ptp(icon_tone) < 40 or np.ptp(label_tone) < 33
            or (icon_tone @ label_tone) < .77 * np.linalg.norm(icon_tone) * np.linalg.norm(label_tone)):
        return None
    return parts, selected, bottom


def _features(values):
    values = np.pad(values.astype(np.float32, copy=False), ((0, 0),) * (values.ndim - 2) + ((2, 2), (2, 2)))
    mean = values.mean(axis=(-2, -1), keepdims=True)

    def unit(mask):
        flat = mask.reshape((*mask.shape[:-2], -1))
        return flat / np.maximum(1e-6, np.linalg.norm(flat, axis=-1, keepdims=True))

    return np.concatenate((.55 ** .5 * unit(values), .30 ** .5 * unit(values - mean),
                           .075 ** .5 * unit(np.diff(values, axis=-2)),
                           .075 ** .5 * unit(np.diff(values, axis=-1))), axis=-1)


@lru_cache(maxsize=12)
def _font_inks(paths):
    inks = []
    for path in paths:
        font = ImageFont.truetype(str(path), 50)
        _, upper, _, baseline = font.getbbox("H")
        cap = baseline - upper
        for character in string.ascii_letters + string.digits:
            render = font.getmask(character, mode="L")
            width, height = render.size
            if not width or not height:
                continue
            ink = np.asarray(render, np.uint8).reshape(height, width)
            x, y, w, h = cv2.boundingRect(ink)
            y_shift = font.getbbox(character)[1] + y - baseline
            inks.append((character, ink[y:y + h, x:x + w], (y_shift / cap, (y_shift + h) / cap)))
    return inks


@lru_cache(maxsize=40)
def _glyph_bank(paths, width, height):
    chars, mats, levels = [], [], []
    for char, ink, alignment in _font_inks(paths):
        if not .51 <= (width / height) / (ink.shape[1] / ink.shape[0]) <= 1.76:
            continue
        for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            w, h = width + dx, height + dy
            if min(w, h) < 1:
                continue
            x, y = (width - w) // 2 + 1, (height - h) // 2 + 1
            raster = np.zeros((height + 2, width + 2), np.float32)
            raster[y:y + h, x:x + w] = cv2.resize(ink, (w, h), interpolation=cv2.INTER_AREA)
            chars.append(char)
            mats.append(raster)
            levels.append(alignment)
    if not mats:
        return (), np.empty(0), np.empty(0), np.empty(0, dtype=int)
    order = np.argsort(chars, kind="stable")
    unique, starts = np.unique(np.asarray(chars)[order], return_index=True)
    return tuple(unique), _features(np.stack(mats)[order]), np.asarray(levels)[order], starts


def _short_shoulder(glyph, height, cap):
    if not .61 * cap <= height <= .88 * cap or not 3 <= glyph.shape[1] <= .59 * height:
        return False
    half = glyph.shape[1] // 2
    split = max(2, round(.34 * height))
    upper_right, lower_right = glyph[:split, half:], glyph[split:, half:]
    stem = glyph[split:, :half]
    return upper_right.max() >= .34 and lower_right.max() <= .22 and np.percentile(stem.max(axis=1), 30) >= .38


def _soft_letter(ink, bounds, paths, cap, baseline):
    x, y, right, bottom = bounds
    glyph = ink[y:bottom, x:right]
    h, w = glyph.shape
    chars, library, levels, starts = _glyph_bank(paths, w, h)
    if not chars:
        return None
    scores = np.einsum("ij,j->i", library, _features(np.pad(glyph, 1)), optimize=False)
    scores -= .20 * abs(levels - ((y - baseline) / cap, (bottom - baseline) / cap)).sum(axis=1)
    best = np.maximum.reduceat(scores, starts)
    if len(best) < 2:
        return None
    first, second = np.argsort(-best, kind="stable")[:2]
    if _short_shoulder(glyph, h, cap):
        return "r" if "r" in chars and best[chars.index("r")] >= .59 else None
    if best[first] >= .771 and best[first] - best[second] >= .021:
        return chars[first]
    return None


def _small_latin(frame, area, paint, index):
    x, y, right, bottom = area
    patch = frame[y - 1:bottom + 1, x - 2:right + 2]
    shade = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    paper = float(cv2.cvtColor(np.uint8([[paint]]), cv2.COLOR_BGR2GRAY)[0, 0])
    ink = abs(shade - paper)
    peak = float(np.percentile(ink, 99))
    if peak < 47:
        return None
    ink /= peak
    base = index.fonts or index.latin_sources
    if not base:
        return None
    standard = [base[0].parent / name for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf", "tahoma.ttf",
                                                   "NotoSansSC-VF.ttf")]
    paths = index.fonts or tuple(font for font in standard if font.is_file()) or base
    votes = defaultdict(set)
    readings = {}
    for level in (.175, .214, .254, .302):
        mask = np.uint8(ink >= level)
        spans = _bands(mask.any(axis=0))
        if not 2 <= len(spans) <= 12:
            continue
        bounds = [_tight(mask[:, a:b], a, 0) for a, b in spans]
        baseline = float(np.median([b for _, _, _, b in bounds]))
        cap = max(baseline - t for _, t, _, _ in bounds)
        if not 6 <= cap <= 16:
            continue
        letters = []
        for box in bounds:
            key = (box, cap, baseline)
            if key not in readings:
                readings[key] = _soft_letter(ink, box, paths, cap, baseline)
            letters.append(readings[key])
        if all(letters):
            votes["".join(letters)].add(mask.tobytes())
    agreed = [name for name, signals in votes.items() if len(signals) >= 2]
    return agreed[0] if len(agreed) == 1 else None


def landscape_icon_tab_actions(frame, regions, index):
    """Publish only named unselected tabs on a verified mobile edge."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or height < 225 or width < 1.44 * height:
        return []
    grids = list(_grids(regions["candidates"], width, height))
    if len(grids) != 1:
        return []
    edges, approximate = grids[0]
    anchor = _anchor(frame, approximate)
    if not anchor:
        return []
    top, gesture, paint = anchor
    match = _parts(frame, edges, top, gesture, paint)
    if match is None:
        return []
    parts, selected, bottom = match
    actions, cache = [], {}
    for i, (icon, label, _, _) in enumerate(parts):
        if i == selected:
            continue
        text = _transcript(frame, label, index, cache)
        if not text and label[3] - label[1] <= 15:
            text = _small_latin(frame, label, paint, index)
        if text:
            left, y, right, end = icon
            actions.append(_action(text, [edges[i], top, edges[i + 1], bottom],
                                   [(left + right) // 2, (y + end) // 2]))
    counts = Counter(item["function"] for item in actions)
    return [item for item in actions if counts[item["function"]] == 1]
