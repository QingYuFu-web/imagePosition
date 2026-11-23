"""Verify radio-option rows on paired landscape settings cards."""

import cv2
import numpy as np

from inventory_choice_glyphs import find_choice_glyphs
from screen_component_captions import _bands, _tight
from screen_wide_panel_checks import _soft_letter


def _joined(mask, gap):
    merged = []
    for a, b in _bands(mask):
        if merged and a - merged[-1][1] <= gap:
            merged[-1] = merged[-1][0], b
        else:
            merged.append((a, b))
    return merged


def _card_pairs(gray):
    height, width = gray.shape
    top, bottom = round(.195 * height), round(.685 * height)
    diff = gray[top + 2:bottom + 2].astype(np.int16) - gray[top - 2:bottom - 2]
    found = []
    for row, scores in enumerate(diff, top):
        spans = [(a, b) for a, b in _joined(scores >= 5, round(.004 * width))
                 if .30 * width <= b - a <= .422 * width]
        if len(spans) != 2:
            continue
        (a, b), (x, r) = spans
        if (not .07 * width <= a <= .21 * width or r < .82 * width
                or not .003 * width <= x - b <= .030 * width or abs((b - a) - (r - x)) > .032 * width):
            continue
        confidence = float(np.median(scores[a:b]) + np.median(scores[x:r]))
        found.append((row, spans, confidence))
    groups = []
    for row, spans, confidence in found:
        if groups and row - groups[-1][-1][0] <= 3:
            groups[-1].append((row, spans, confidence))
        else:
            groups.append([(row, spans, confidence)])
    pairs = [(row, spans) for group in groups
             for row, spans, _ in [max(group, key=lambda item: (min(b - a for a, b in item[1]), item[2]))]]
    if len(pairs) < 2:
        return []
    first = pairs[0][1]
    return [(y, spans) for y, spans in pairs if max(abs(a - b) for one, two in zip(spans, first, strict=True)
                                                   for a, b in zip(one, two, strict=True)) <= .019 * width]


def _has_title(gray, top, lane):
    height = gray.shape[0]
    left, right = lane
    reach = right - left
    field = gray[top + round(.015 * height):top + round(.068 * height),
                 left + round(.045 * reach):right - round(.18 * reach)].astype(np.int16)
    background = float(np.median(field))
    bands = _joined((field >= background + 26).sum(axis=0) >= 2, 3)
    return any(.10 * reach <= b - a <= .63 * reach for a, b in bands)


def _carrier_tone(frame, glyph):
    left, top, right, bottom = glyph.box
    height = bottom - top
    width = right - left
    source = frame[top:bottom, left:right]
    y, x = np.ogrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    radius = x * x + y * y
    interior = np.median(source[radius <= .37], axis=0)
    if glyph.state == "off" and np.ptp(interior) <= 24:
        return "off"
    if glyph.state == "on" and np.ptp(interior) >= 33:
        return "on"
    return None


def _label_box(gray, glyph, limit):
    _, top, right, bottom = glyph.box
    size = bottom - top
    left = right + max(2, round(.15 * size))
    last = min(limit, right + round(5.0 * size))
    field = gray[top - 2:bottom + 2, left:last].astype(np.float32)
    if field.size < 10:
        return None
    contrast = np.maximum(field - np.median(field), 0)
    peak = float(np.percentile(contrast, 99))
    if peak < 39:
        return None
    mask = contrast >= max(22, .32 * peak)
    bands = _joined(mask.sum(axis=0) >= 2, max(2, round(.153 * size)))
    if not bands or bands[0][0] > .34 * size:
        return None
    x, r = bands[0]
    box = _tight(mask[:, x:r], left + x, top - 2)
    if box is None:
        return None
    l, t, rr, b = box
    if (.47 * size <= b - t <= .88 * size and 1.53 * (b - t) <= rr - l <= 5.30 * (b - t)
            and abs(t + b - top - bottom) <= .41 * size):
        return box
    return None


def _rows(frame, gray, header, lane):
    left, right = lane
    width, height = right - left, gray.shape[0]
    if not _has_title(gray, header, lane):
        return ()
    top, bottom = header + round(.077 * height), header + round(.159 * height)
    bounds = left, top, right, bottom
    scale = (max(8, round(.027 * height)), max(11, round(.056 * height)))
    observed = [glyph for glyph in find_choice_glyphs(gray, bounds, scale)
                if glyph.kind == "radio" and glyph.confidence >= .68]
    if not 2 <= len(observed) <= 3:
        return ()
    glyphs = sorted(observed, key=lambda glyph: glyph.center[0])
    xs, ys = zip(*(glyph.center for glyph in glyphs), strict=True)
    diameters = [glyph.box[3] - glyph.box[1] for glyph in glyphs]
    size = float(np.median(diameters))
    gaps = np.diff(xs)
    if (not .055 * width <= xs[0] - left <= .165 * width or min(gaps) < .205 * width
            or max(gaps) > .354 * width or np.ptp(gaps) > .030 * width
            or np.ptp(ys) > .25 * size or np.ptp(diameters) > .23 * size):
        return ()
    cut_a, cut_b = round(max(ys) + .70 * size), round(max(ys) + 1.80 * size)
    cells = gray[cut_a:cut_b, left + round(.06 * width):right - round(.075 * width)].astype(np.int16)
    if len(cells) < 5 or not np.any(((cells[:-4] - cells[4:]) >= 4).mean(axis=1) >= .71):
        return ()
    states = [_carrier_tone(frame, glyph) for glyph in glyphs]
    if states.count("on") != 1 or states.count("off") != len(glyphs) - 1:
        return ()
    limits = [glyph.box[0] - round(.24 * size) for glyph in glyphs[1:]] + [right]
    labels = [_label_box(gray, glyph, limit) for glyph, limit in zip(glyphs, limits, strict=True)]
    if any(box is None for box in labels):
        return ()
    # A readable orphan at the next measured pitch indicates a missing ring.
    if len(glyphs) == 2:
        pitch = float(gaps[0])
        expected = round(xs[-1] + pitch + .70 * size)
        field = gray[round(min(ys) - .53 * size):round(max(ys) + .52 * size),
                     expected:min(right, expected + round(2.8 * size))]
        if field.size and ((field.astype(np.int16) - np.median(field)) >= 26).mean() >= .12:
            return ()
    return tuple(zip(glyphs, states, labels, strict=True))


def _read_letter(glyph, peak, index):
    ranks = [index.rank(glyph >= v * peak, alphabet="cjk", limit=7) for v in (.22, .29, .40)]
    if (all(len(r) >= 2 for r in ranks) and len({r[0]["text"] for r in ranks}) == 1
            and min(r[1]["distance"] - r[0]["distance"] for r in ranks) >= .023
            and max(r[0]["distance"] for r in ranks) <= .151):
        return ranks[0][0]["text"]
    ranks.extend(index.rank(glyph >= v * peak, alphabet="cjk", limit=7) for v in (.52, .64, .72))
    options = tuple(sorted({item["text"] for reading in ranks for item in reading}))
    paths = tuple(index.fonts if index.fonts is not None else index.cjk_sources)
    used = {}
    for cut in (.23, .31, .40, .47):
        x, y, w, h = cv2.boundingRect(np.uint8(glyph >= cut * peak))
        if not .61 <= w / max(1, h) <= 1.28 or h < 9:
            continue
        key = x, y, w, h
        if key not in used:
            raster = np.float32(np.pad(np.minimum(1, glyph[y:y + h, x:x + w] / peak), 2))
            used[key] = _soft_letter(paths, options, raster.tobytes(), w, h)
    readings = [v for v in used.values() if v is not None]
    if not readings or len(set(readings)) != 1:
        return None
    best = readings[0]
    if (len(readings) >= 2 or any(r and r[0]["text"] == best and r[0]["distance"] <= .253 for r in ranks)):
        return best
    return None


def _read_label(gray, box, index):
    l, t, r, b = box
    patch = gray[t - 3:b + 4, l - 3:r + 5].astype(np.float32)
    contrast = np.maximum(patch - np.median(patch), 0)
    peak = float(np.percentile(contrast, 99))
    if peak < 42:
        return None
    x, y, w, h = cv2.boundingRect(np.uint8(contrast >= .32 * peak))
    count = round(w / max(1, 1.075 * h))
    if not 2 <= count <= 5 or not 12 <= h <= 29:
        return None
    contrast = contrast[y:y + h, x:x + w]
    columns = contrast.sum(axis=0)
    boundaries = [0]
    for i in range(1, count):
        middle = i * w / count
        reach = .16 * h
        edge = min(range(round(middle - reach), round(middle + reach) + 1),
                   key=lambda v: (columns[v], abs(v - middle)))
        if columns[edge] > .22 * h * peak:
            return None
        boundaries.append(edge)
    boundaries.append(w)
    names = [_read_letter(contrast[:, a:b], peak, index) for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)]
    return "".join(names) if all(names) else None


def landscape_radio_actions(frame, regions, index):
    """Return one click for each inactive and literally read radio option."""
    h, w = frame.shape[:2]
    if regions["layer"] != "page" or index is None or h < 320 or w < 1.65 * h:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    groups = _card_pairs(gray)
    if len(groups) < 2:
        return []
    actions = []
    for header, lanes in groups:
        for lane in lanes:
            for glyph, state, caption in _rows(frame, gray, header, lane):
                if state == "on" or (label := _read_label(gray, caption, index)) is None:
                    continue
                left, top, right, bottom = map(int, glyph.box)
                _, t, r, b = map(int, caption)
                actions.append({"role": "game_settings_option", "function": label, "preview_text": label,
                                "box": [left, min(top, t), max(right, r), max(bottom, b)],
                                "click_point": [(left + right) // 2, (top + bottom) // 2]})
    return actions
