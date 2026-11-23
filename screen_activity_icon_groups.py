"""Recover captions owned by a repeated icon-and-label rail."""

from collections import defaultdict
from statistics import median

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts
from screen_component_captions import _bands


def _tile(frame, box):
    left, top, right, bottom = box
    h, w = bottom - top, right - left
    height, width = frame.shape[:2]
    cx, cy = (left + right) / 2, (top + bottom) / 2
    a, b = max(0, round(cx - .42 * w)), min(width, round(cx + .42 * w))
    t, z = max(0, round(cy - .56 * h)), min(height, round(cy + .48 * h))
    span = max(3, round(.10 * w))
    side = np.r_[np.arange(max(0, a - span), a + 3), np.arange(b - 3, min(width, b + span))]
    base = np.median(frame[t:z, side], axis=1)
    delta = (frame[t:z, a:b].astype(np.int16) - base[:, None]).max(axis=2)
    for threshold in (46, 57, 70):
        ink = delta >= threshold
        bands = _bands(ink.sum(axis=1) >= max(3, round(.055 * w)))
        for (i, j), (y, end) in reversed(list(zip(bands, bands[1:]))):
            if not (.27 * h <= j - i <= .79 * h and .12 * h <= end - y <= .27 * h
                    and 1 <= y - j <= .105 * h and cy + .04 * h <= t + (y + end) / 2 <= cy + .43 * h):
                continue
            x, _, iw, _ = cv2.boundingRect(np.uint8(ink[i:j]))
            u, _, tw, _ = cv2.boundingRect(np.uint8(ink[y:end]))
            if (not .30 * w <= iw <= .85 * w or not .25 * w <= tw <= .89 * w
                    or abs(a + x + iw / 2 - cx) > .16 * w or abs(a + u + tw / 2 - cx) > .135 * w):
                continue
            pad = max(2, round(.12 * (end - y)))
            area = (max(a, a + u - pad), max(t + j, t + y - pad), min(b, a + u + tw + pad), t + end + pad)
            return [a + min(x, u), t + i, a + max(x + iw, u + tw), t + end], area, [a + x + iw // 2, t + (i + j) // 2]
    return None


def _literal(frame, box, index, cache):
    x, y, right, bottom = box
    patch = frame[y:bottom, x:right]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    reports = defaultdict(dict)
    for family, view in enumerate((gray, patch.min(axis=2), patch.max(axis=2))):
        low, high = np.percentile(view, (28, 98))
        if high - low < 39:
            continue
        # A lighter cut retains radicals that are lost from small outlined type.
        for fraction in (.27, .31, .34, .39):
            mask = np.uint8(view >= low + fraction * (high - low))
            a, b, w, h = cv2.boundingRect(mask)
            if h < 11 or abs(a + w / 2 - view.shape[1] / 2) > .13 * view.shape[1]:
                continue
            raster = mask[b:b + h, a:a + w]
            parts = _cjk_parts(raster, h)
            if not 2 <= len(parts) <= 6:
                continue
            ranks = []
            for *_, glyph in parts:
                key = glyph.shape, glyph.tobytes()
                if key not in cache:
                    cache[key] = index.rank(glyph, alphabet="cjk", limit=2)
                ranks.append(cache[key])
            if any(len(choices) != 2 or choices[0]["distance"] > .231 for choices in ranks):
                continue
            margin = median(choices[1]["distance"] - choices[0]["distance"] for choices in ranks)
            if margin < .010:
                continue
            word = "".join(choices[0]["text"] for choices in ranks)
            error = median(choices[0]["distance"] for choices in ranks)
            reports[word][(raster.shape, raster.tobytes())] = (family, error)
    qualified = sorted((min(error for _, error in matches.values()), word) for word, matches in reports.items()
                       if len(matches) >= 2 and len({view for view, _ in matches.values()}) >= 2)
    if not qualified or len(qualified) > 1 and qualified[1][0] - qualified[0][0] < .014:
        return None
    return qualified[0][1]


def _dimensions(box):
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2, right - left, bottom - top


def activity_icon_group_actions(frame, regions, index):
    """Return only corroborated labels from a repeated, visible icon rail."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or width < 1.43 * height or height < 240:
        return []
    flags = [tuple(candidate["box"]) for candidate in regions["candidates"] if candidate.get("visibility") == "page"]
    seeds = []
    for box in flags:
        x, y, w, h = _dimensions(box)
        if (min(x, width - x) <= .15 * height and .093 * height <= y <= .79 * height
                and .078 * height <= w <= .159 * height and .081 * height <= h <= .165 * height
                and .79 <= w / h <= 1.28):
            seeds.append(box)
    actions, cache, seen = [], {}, set()
    for seed in seeds:
        x, y, w, h = _dimensions(seed)
        peers = [box for box in seeds if abs(_dimensions(box)[0] - x) <= .13 * w
                 and .75 * h <= box[3] - box[1] <= 1.36 * h]
        if len(peers) < 3:
            continue
        peers.sort(key=lambda box: box[1])
        if tuple(peers) in seen:
            continue
        seen.add(tuple(peers))
        steps = np.diff([_dimensions(box)[1] for box in peers])
        pitch = min(steps)
        if not .81 * h <= pitch <= 1.62 * h or any(min(abs(step - pitch), abs(step - 2 * pitch)) > .23 * pitch for step in steps):
            continue
        tiles = [part for box in peers if (part := _tile(frame, box)) is not None]
        if len(tiles) < 3:
            continue
        for box, title, point in tiles:
            if name := _literal(frame, title, index, cache):
                actions.append({"role": "activity_shortcut", "function": f"打开{name}", "preview_text": name,
                                "box": box, "click_point": point})
    counts = {item["function"]: sum(other["function"] == item["function"] for other in actions) for item in actions}
    return [item for item in actions if counts[item["function"]] == 1]
