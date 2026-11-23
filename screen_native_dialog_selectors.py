"""Own repeated dropdowns by their visible labels and aligned triangles."""

from collections import defaultdict

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_dialog_search_mode import _down_triangle
from screen_landscape_search_field import _available


_FUNCTIONS = {
    "archive format": "选择压缩格式",
    "compression level": "选择压缩级别",
    "encryption": "选择加密方式",
    "split into volumes": "选择分卷",
}


def _triangles(gray, paint, top, span):
    matches = defaultdict(set)
    for cut in (42, 70, 100):
        ink = np.uint8(gray >= paint + cut)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
        for i, (x, y, w, h, _) in enumerate(stats[1:], 1):
            if (.019 * span <= w <= .043 * span and .009 * span <= h <= .023 * span
                    and .35 * span <= x <= .966 * span and 1.56 * h <= w <= 2.39 * h
                    and _down_triangle(labels[y:y + h, x:x + w] == i)):
                key = next((key for key in matches if max(abs(x - key[0]), abs(y - key[1])) <= 2), None)
                matches[tuple(map(int, (x, y, x + w, y + h))) if key is None else key].add(cut)
    return [(l, t + top, r, b + top) for (l, t, r, b), cuts in matches.items() if len(cuts) >= 2]


def _caption(frame, gray, x, r, y, span, paint, index, cache, frame_top):
    t, b = y - round(.094 * span), y - round(.028 * span)
    chip = gray[t - frame_top:b - frame_top, x:r]
    ink = np.uint8(chip >= paint + 58)
    u, v, w, h = cv2.boundingRect(ink)
    if (not .019 * span <= h <= .061 * span or .019 * span >= w
            or u > .042 * span or w > .86 * ink.shape[1]):
        return None
    margin = max(2, round(.12 * h))

    def read(stop):
        area = (x + max(0, u - margin), t + max(0, v - margin), x + stop,
                t + min(ink.shape[0], v + h + margin))
        result = read_caption(frame, area, index, cache=cache, choice=True, sentence_case=True, join_letters=True)
        return result["text"].strip(" :：") if result else None

    if name := read(min(chip.shape[1], u + w + margin)):
        return name
    columns = _bands(ink.any(axis=0))
    breaks = [(a + n) // 2 for (_, a), (n, _) in zip(columns, columns[1:]) if n - a >= .245 * h]
    if breaks and (prefix := read(breaks[0])) and prefix.casefold() == "compression":
        return "Compression level"
    return None


def native_dialog_selector_actions(frame, regions, index, panel):
    """Keep sibling dropdowns separate after a verified dialog restricts the raster."""
    if regions["layer"] != "dialog" or index is None:
        return []
    left, top, right, bottom = panel
    w, h = right - left, bottom - top
    scan = top + round(.23 * h)
    gray = cv2.cvtColor(frame[scan - round(.115 * w):top + round(.76 * h), left:right], cv2.COLOR_BGR2GRAY)
    frame_top = scan - round(.115 * w)
    paper = frame[:, left:right]
    paint = float(np.median(gray[::7, ::7]))
    if paint > 105:
        return []
    marks = _triangles(gray, paint, frame_top, w)
    if len(marks) < 3:
        return []
    cache, actions = {}, []
    for l, t, r, b in marks:
        y = (t + b) // 2
        if y < scan:
            continue
        peers = [other for other in marks if other != (l, t, r, b)
                 and abs(other[1] + other[3] - 2 * y) <= .027 * w]
        side = .79 * w <= l <= .955 * w
        if side:
            if sum(abs(x - l) <= .022 * w for x, _, _, _ in marks) < 2:
                continue
        elif not any(x - l >= .32 * w and x >= .79 * w for x, _, _, _ in peers):
            continue
        paired = side and any(l - x >= .32 * w for x, _, _, _ in peers)
        start = round(.50 * w) if paired else round(.006 * w)
        stop = l - round(.021 * w)
        if stop - start <= .16 * w:
            continue
        chip = gray[y - frame_top - round(.034 * w):y - frame_top + round(.035 * w), start:stop]
        a, _, dw, dh = cv2.boundingRect(np.uint8(chip >= paint + 67))
        if a > .045 * w or not .018 * w <= dh <= .061 * w or not .021 * w <= dw <= .70 * w:
            continue
        name = _caption(paper, gray, start, stop, y, w, paint, index, cache, frame_top)
        if name is None:
            continue
        box = [left + start, y - round(.033 * w), min(right, left + r + round(.018 * w)), y + round(.031 * w)]
        if _available(box, regions):
            actions.append({"role": "dropdown", "function": _FUNCTIONS.get(name.casefold(), f"选择{name}"),
                            "preview_text": name,
                            "box": box, "click_point": [(box[0] + box[2]) // 2, y]})
    return actions
