"""Identify an empty, labelled comment input on a detailed light thread."""

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_component_captions import _bands
from screen_dark_barrage_input import _printed
from screen_landscape_search_field import _available
from screen_wide_panel_checks import _clipped_bitmap


_INPUT = ("评论", "評論")
_SEND = ("发送", "發送")
_DETAILS = ("详情", "詳情")


def _neutral(tile):
    return np.ptp(tile, axis=2) <= 37


def _consensus(hits, tolerance):
    agreeing = [box for level, box in hits if any(other != level and max(abs(a - b) for a, b in
                zip(box, peer, strict=True)) <= tolerance for other, peer in hits)]
    if not agreeing or np.ptp(agreeing, axis=0).max() > 2 * tolerance:
        return None
    return tuple(map(int, np.median(agreeing, axis=0)))


def _literal(signal, index, names, rivals):
    peak = float(np.percentile(signal, 98))
    if peak < 23:
        return None
    fonts = index.fonts or index.cjk_sources[:4]
    seen, reports, strong = set(), {}, set()
    for amount in (.32, .43, .56):
        ink = np.uint8(signal >= max(12, round(amount * peak)))
        x, y, w, h = cv2.boundingRect(ink)
        if h < 7 or not 1.37 <= w / h <= 2.52:
            continue
        identity = ink.tobytes()
        if identity in seen:
            continue
        seen.add(identity)
        crop = signal[y:y + h, x:x + w].astype(np.float32) / peak
        ranks = []
        for word in (*names, *rivals):
            score = 0.
            for font in fonts:
                raw = _clipped_bitmap(font, 64, word)
                if raw is None or not .79 <= raw.shape[1] * h / (raw.shape[0] * w) <= 1.24:
                    continue
                paint = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
                score = max(score, float(cv2.matchTemplate(crop, paint, cv2.TM_CCOEFF_NORMED)[0, 0]))
            ranks.append((score, word))
        ranks.sort(reverse=True)
        score, word = ranks[0]
        if word in names and score >= .807 and score - ranks[1][0] >= .133:
            return word
        if word in names and score >= .768 and score - ranks[1][0] >= .105:
            strong.add(word)
        if word in names and score >= .718 and score - ranks[1][0] >= .101:
            reports[word] = reports.get(word, 0) + 1
    normal = np.uint8(np.clip(signal.astype(np.float32) * (104 / peak), 0, 255))
    if signal.shape[0] < 22:
        scales = (1., 1.48, 1.65, 1.85)
    else:
        scales = (1., 1.48) if signal.shape[0] < 28 else (1.,)
    for size in scales:
        pixels = normal if size == 1 else cv2.resize(normal, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC)
        if text := _printed(pixels, index, set(names)):
            return text
    exact = strong | {word for word, support in reports.items() if support >= 2}
    return next(iter(exact)) if len(exact) == 1 else None


def _footer(gray):
    height, width = gray.shape
    x, r = round(.42 * width), round(.84 * width)
    top, bottom = height - round(.176 * width), height - round(.094 * width)
    strip = gray[:, x:r]
    baseline = np.median(strip, axis=1)
    choices = [y for y in range(top, bottom) if (strip[y] <= 246).mean() >= .88
               and baseline[y] <= min(baseline[y - 3], baseline[y + 3]) - 6
               and min(baseline[y - 3], baseline[y + 3]) >= 246]
    if not choices or choices[-1] - choices[0] > max(2, round(.007 * width)):
        return None
    return choices[-1] + 1


def _outlined_send(frame, gray, start):
    height, width = gray.shape
    x0, end = round(.829 * width), width
    tile = gray[start:height - max(1, round(.004 * width)), x0:end]
    neutral = _neutral(frame[start:height - max(1, round(.004 * width)), x0:end])
    lines = []
    for level in (212, 224, 235, 242):
        ink = np.uint8((tile <= level) & neutral)
        shapes, nesting = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if nesting is None:
            continue
        for contour, (_, _, child, parent) in zip(shapes, nesting[0], strict=True):
            x, y, w, h = cv2.boundingRect(contour)
            if (child < 0 or parent != -1 or not .061 * width <= h <= .112 * width or not 1.18 <= w / h <= 1.69
                    or not .846 * width <= x + x0 <= .92 * width or y < .016 * width
                    or not .85 <= cv2.contourArea(contour) / (w * h) <= .98
                    or not .78 <= cv2.contourArea(shapes[child]) / cv2.contourArea(contour) <= .990):
                continue
            owned = neutral[y + round(.22 * h):y + round(.78 * h), x + round(.14 * w):x + round(.86 * w)]
            surface = tile[y + round(.22 * h):y + round(.78 * h), x + round(.14 * w):x + round(.86 * w)]
            if owned.mean() >= .94 and np.median(surface) >= 240:
                lines.append((level, (x0 + x, start + y, x0 + x + w, start + y + h)))
    return _consensus(lines, max(2, round(.006 * width)))


def _face_strokes(gray):
    h, w = gray.shape
    x, y = round(.16 * w), round(.16 * h)
    chip = gray[y:h - y, x:w - x]
    for level in (217, 231, 241):
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(chip <= level))
        eyes = [(a + u / 2 + x, t + v / 2 + y) for a, t, u, v, n in stats[1:]
                if .056 * h <= min(u, v) <= max(u, v) <= .21 * h and .64 * u * v <= n
                and .23 * h <= t + v / 2 + y <= .44 * h]
        smiles = [n for a, t, u, v, n in stats[1:]
                  if .37 * w <= u <= .71 * w and .17 * h <= v <= .39 * h
                  and .56 * h <= t + v / 2 + y <= .80 * h and abs(a + u / 2 + x - w / 2) <= .14 * w]
        if len(eyes) == 2 and smiles:
            first, second = sorted(eyes)
            if .16 * w <= second[0] - first[0] <= .43 * w and abs(first[1] - second[1]) <= .095 * h:
                return True
    return False


def _emoji(frame, gray, send):
    width = gray.shape[1]
    x, y, right, bottom = send
    h = bottom - y
    start = x - round(1.53 * h)
    top = y - round(.16 * h)
    rows, columns = slice(top, bottom + round(.12 * h)), slice(start, x - round(.18 * h))
    neutral = _neutral(frame[rows, columns])
    tile = np.where(neutral, gray[rows, columns], 255).astype(np.uint8)
    matches = []
    for level in (224, 232, 239, 245):
        contours, tree = cv2.findContours(np.uint8(tile <= level), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if tree is None:
            continue
        for contour, (_, _, child, parent) in zip(contours, tree[0], strict=True):
            a, t, w, d = cv2.boundingRect(contour)
            if (child < 0 or parent >= 0 or not .051 * width <= d <= .092 * width or not .84 <= w / d <= 1.19
                    or not .64 <= cv2.contourArea(contour) / (w * d) <= .85
                    or not .77 <= cv2.contourArea(contours[child]) / cv2.contourArea(contour) <= .99
                    or abs(top + t + d / 2 - y - h / 2) > .20 * h):
                continue
            if _face_strokes(tile[t:t + d, a:a + w]):
                matches.append((level, (start + a, top + t, start + a + w, top + t + d)))
    return _consensus(matches, 2)


def _header(frame, gray, index):
    height, width = gray.shape
    rows = np.median(gray[:, round(.11 * width):round(.91 * width):max(2, round(.024 * width))], axis=1)
    bars = [(a, b) for a, b in _bands((rows >= 23) & (rows <= 86)) if .021 * height <= a <= .080 * height
            and .096 * width <= b - a <= .169 * width and b + 3 < height and np.median(rows[b + 1:b + 4]) >= 226]
    if len(bars) != 1:
        return None
    top, bottom = bars[0]
    tile = gray[top:bottom, :round(.12 * width)]
    neutral = _neutral(frame[top:bottom, :tile.shape[1]])
    matches = []
    for value in (51, 74, 102):
        ink = np.uint8((tile.astype(np.int16) - int(np.median(tile)) >= value) & neutral)
        _, ids, stats, _ = cv2.connectedComponentsWithStats(ink)
        for i, (x, y, w, h, _) in enumerate(stats[1:], 1):
            if (.026 * width <= h <= .067 * width and .018 * width <= x <= .077 * width
                    and abs(y + h / 2 - len(tile) / 2) <= .17 * (bottom - top)
                    and is_shaft_back_arrow(ids[y:y + h, x:x + w] == i)):
                matches.append((value, (x, top + y, x + w, top + y + h)))
    back = _consensus(matches, 2)
    if back is None:
        return None
    _, t, r, b = back
    size = b - t
    label = r + round(1.34 * size), max(top, t - round(.48 * size)), r + round(4.4 * size), min(bottom, b + round(.45 * size))
    x, y, r, b = label
    caption = frame[y:b, x:r]
    pale = gray[y:b, x:r]
    signal = cv2.subtract(pale, int(np.median(pale)))
    signal[~_neutral(caption)] = 0
    return (back, label) if _literal(signal, index, _DETAILS, ("帮助", "评语", "詳解", "资料", "搜索")) else None


def _empty_label(frame, gray, footer, send, emoji, index):
    width = gray.shape[1]
    _, top, _, bottom = send
    h = bottom - top
    left, right = round(.030 * width), emoji[0] - round(.032 * width)
    y, stop = top + round(.12 * h), bottom - round(.24 * h)
    tile = frame[y:stop, left:right]
    pane = gray[y:stop, left:right]
    neutral = _neutral(tile)
    if (neutral.mean() < .83 or np.median(pane[neutral]) < 245
            or (pane[neutral] < 142).mean() > .01):
        return None
    signal = cv2.subtract(int(np.median(pane[neutral])), pane)
    signal[~neutral] = 0
    x, t, w, d = cv2.boundingRect(np.uint8(signal >= 27))
    if (not .024 * width <= d <= .062 * width or not 1.31 <= w / d <= 2.54
            or left + x > .095 * width or signal.max() > 113):
        return None
    pad = max(2, round(.19 * d))
    caption = signal[max(t - pad, 0):min(len(signal), t + d + pad), max(0, x - pad):x + w + pad]
    if _literal(caption, index, _INPUT, ("回复", "评价", "评语", "評語", "评说", "测试")) is None:
        return None
    point = [left + x + w / 2, y + t + d / 2]
    return [left + x - max(2, round(.006 * width)), max(footer, y), min(right, left + x + w + round(.17 * width)), stop], point


def empty_comment_footer_actions(frame, regions, index):
    """Return only the actual empty input and its toolbar's Back control."""
    height, width = frame.shape[:2]
    if (index is None or regions["layer"] != "page" or regions.get("regions")
            or width < 225 or not 1.52 * width <= height <= 2.62 * width):
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (footer := _footer(gray)) is None or (send := _outlined_send(frame, gray, footer)) is None:
        return []
    if (emoji := _emoji(frame, gray, send)) is None or not all(_available(box, regions) for box in (send, emoji)):
        return []
    left, top, right, bottom = send
    h = bottom - top
    pad, inset = round(.16 * h), round(.18 * h)
    rows, columns = slice(top + pad, bottom - pad), slice(left + inset, right - inset)
    patch = gray[rows, columns]
    signal = cv2.subtract(int(np.median(patch)), patch)
    signal[~_neutral(frame[rows, columns])] = 0
    if _literal(signal, index, _SEND, ("发布", "发达", "发话", "配送", "发言")) is None:
        return []
    if (header := _header(frame, gray, index)) is None or not all(_available(box, regions) for box in header):
        return []
    if (empty := _empty_label(frame, gray, footer, send, emoji, index)) is None:
        return []
    back, _ = header
    x, y, r, b = back
    actions = [{"role": "back", "function": "返回", "box": list(back),
                "click_point": [(x + r) / 2, (y + b) / 2]}]
    box, point = empty
    if _available(box, regions):
        actions.append({"role": "comment_input", "function": "输入评论", "box": box, "click_point": point})
    return actions
