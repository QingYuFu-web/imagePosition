"""Read a verified wide footer button and its two balanced sibling alternatives."""

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts, read_caption
from screen_component_captions import _bands
from screen_foreground import detect_blocking_overlay


def _panel(frame, regions):
    if regions["layer"] == "dialog" and len(regions["regions"]) == 1:
        panel = tuple(regions["regions"][0])
    elif regions["layer"] in {"page", "layer_uncertain"}:
        height, width = frame.shape[:2]
        if not any(.61 * width <= r - l <= .89 * width and .028 * height <= b - t <= .074 * height
                   and abs((l + r) / 2 - width / 2) <= .075 * width and .48 * height <= (t + b) / 2 <= .72 * height
                   for l, t, r, b in (item["box"] for item in regions.get("candidates", []))):
            return None
        overlay = detect_blocking_overlay(frame)
        if overlay is None or overlay.kind != "dialog":
            return None
        panel = overlay.panel
    else:
        return None
    height, width = frame.shape[:2]
    a, t, b, d = panel
    w, h = b - a, d - t
    if not (.48 * width <= w <= .94 * width and .62 * w <= h <= 1.45 * w and 0 < t < d < height
            and abs((a + b) / 2 - width / 2) <= .08 * width and min(a, width - b) > .015 * width):
        return None
    inset = max(5, round(.045 * w))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    interior = gray[t + inset:d - inset:3, a + inset:b - inset:4]
    paper = float(np.median(interior))
    outside = [gray[t + inset:d - inset:3, l:r] for l, r in ((max(0, a - inset), a), (b, min(width, b + inset)))]
    if (paper < 162 or (abs(interior - paper) <= 24).mean() < .72
            or any((margin <= paper - 55).mean() < .76 for margin in outside)):
        return None
    return panel, gray


def _primary_outline(frame, panel):
    x, top, right, bottom = panel
    width, height = right - x, bottom - top
    start, stop = top + round(.52 * height), top + round(.86 * height)
    blue, green, red = cv2.split(frame[start:stop, x:right].astype(np.int16))
    accent = (red - green >= 32) & (red - blue >= 35) & (red >= 121)
    pad = round(.092 * width)
    rails = _bands(accent[:, pad:-pad].mean(axis=1) >= .605)
    if len(rails) != 2:
        return None
    (y0, _), (_, y1) = rails
    h = y1 - y0
    if not .084 * height <= h <= .179 * height or not .126 * height <= bottom - start - y1 <= .263 * height:
        return None
    side = accent[y0 + round(.17 * h):y1 - round(.15 * h)].mean(axis=0)
    left = np.flatnonzero((np.arange(width) <= .149 * width) & (side >= .66))
    right_side = np.flatnonzero((np.arange(width) >= .85 * width) & (side >= .66))
    if not len(left) or not len(right_side) or right_side[-1] - left[0] < .82 * width:
        return None
    return x + int(left[0]), start + y0, x + int(right_side[-1]) + 1, start + y1


def _ink(gray, window):
    x, y, right, bottom = window
    patch = gray[y:bottom, x:right]
    base = np.percentile(patch, 65)
    mask = np.uint8(patch <= base - 62)
    a, b, w, h = cv2.boundingRect(mask)
    parts = _cjk_parts(mask[b:b + h, a:a + w], h) if h >= 12 else []
    return (x + a, y + b, x + a + w, y + b + h), parts


def _owned_labels(gray, panel, button):
    _, top, _, bottom = panel
    x, y, right, end = button
    w, h = right - x, end - y
    mid = (x + right) // 2
    inset = round(.17 * w)
    primary = (x + inset, y + round(.14 * h), right - inset, end - round(.16 * h))
    t, d = end + round(.023 * (bottom - top)), bottom - round(.038 * (bottom - top))
    gap = round(.037 * w)
    pairs = [(a, t, b, d) for a, b in ((x + gap, mid - gap), (mid + gap, right - gap))]
    measured = [_ink(gray, crop) for crop in (primary, *pairs)]
    boxes = [box for box, _ in measured]
    sizes = [b - a for (_, a, _, b) in boxes]
    if (any(not .31 * h <= z <= .67 * h for z in sizes) or max(sizes) > 1.24 * min(sizes)
            or abs(boxes[1][1] + boxes[1][3] - boxes[2][1] - boxes[2][3]) > .15 * h):
        return None
    for box, crop in zip(boxes, (primary, *pairs), strict=True):
        a, c, b, d = box
        if (not 1.42 * (d - c) <= b - a <= .36 * w or abs(a + b - crop[0] - crop[2]) > .074 * w
                or not crop[0] < a < b < crop[2] or not crop[1] < c < d < crop[3]):
            return None
    return measured


def _literal(frame, box, index, cache, *, limit=None):
    x, y, right, bottom = box
    pad = max(3, round(.16 * (bottom - y)))
    crop = (x - pad, y - pad, min(right + pad, limit) if limit is not None else right + pad, bottom + pad)
    result = read_caption(frame, crop, index, cache=cache, choice=True)
    if result is None:
        result = read_caption(frame, crop, index, cache=cache)
    if result and all("\u3400" <= ch <= "\u9fff" for ch in result["text"]):
        return result["text"]
    return None


def _same_suffix(all_parts, other_parts):
    def rasters(parts):
        return [cv2.resize(part[-1].astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA) for part in parts]

    def soft_shape(raster):
        smoothed = cv2.GaussianBlur(raster, (5, 5), .8).flatten()
        return smoothed / np.linalg.norm(smoothed)

    if min(len(all_parts), len(other_parts)) < 3:
        return 0
    source, sibling = rasters(all_parts), rasters(other_parts)
    original, other = np.array(source) >= .50, np.array(sibling) >= .50
    projections = None
    count = 0
    for i in range(1, min(len(original), len(other)) - 1):
        choice = other[-i]
        scores = [float((candidate & choice).sum() / (candidate | choice).sum()) for candidate in original]
        index = len(original) - i
        if scores[index] - max(score for j, score in enumerate(scores) if j != index) < .14:
            break
        if scores[index] < .87:
            # Resampling a small caption changes which edge pixels cross the ink cutoff.
            # Only accept the softer match when the original masks already agree on its owner.
            if (scores[index] < .75 or max(all_parts[index][-1].shape[0], other_parts[-i][-1].shape[0]) > 32):
                break
            if projections is None:
                projections = np.array([soft_shape(raster) for raster in source])
            smooth_scores = projections @ soft_shape(sibling[-i])
            if smooth_scores[index] < .944 or smooth_scores[index] - np.max(np.delete(smooth_scores, index)) < .10:
                break
        count += 1
    return count


def modal_footer_choices(frame, regions, index):
    """Return only choices owned by a bright, bounded modal action group."""
    if index is None or frame.shape[0] < 1.46 * frame.shape[1] or (owned := _panel(frame, regions)) is None:
        return None
    panel, gray = owned
    primary_box = _primary_outline(frame, panel)
    if primary_box is None or (labels := _owned_labels(gray, panel, primary_box)) is None:
        return None
    cache = {}
    (top_label, prototypes), *siblings = labels
    suffix_lengths = [_same_suffix(prototypes, glyphs) for _, glyphs in siblings]
    if len(set(suffix_lengths)) != 1 or suffix_lengths[0] < 2:
        return None
    primary = _literal(frame, top_label, index, cache)
    if not primary or len(primary) != len(prototypes):
        return None
    names = [primary]
    suffix = suffix_lengths[0]
    for (a, b, _, d), glyphs in siblings:
        end = a + glyphs[-suffix - 1][2]
        limit = a + glyphs[-suffix][0] - 1
        prefix = _literal(frame, (a, b, end, d), index, cache, limit=limit)
        if not prefix or len(prefix) != len(glyphs) - suffix:
            return None
        names.append(prefix + primary[-suffix:])
    if len(set(names)) != 3:
        return None
    a, b, right, d = primary_box
    mid = (a + right) // 2
    controls = [primary_box, (a, d, mid, panel[3]), (mid, d, right, panel[3])]
    actions = [{"role": "modal_footer_action", "function": name, "preview_text": name, "box": list(cell),
                "click_point": [(x + X) // 2, (y + Y) // 2]}
               for name, cell, ((x, y, X, Y), _) in zip(names, controls, labels, strict=True)]
    return panel, actions
