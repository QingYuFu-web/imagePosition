"""Find an on-video chat composer from the visible footer and placeholder."""

from itertools import combinations

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts
from screen_component_captions import _bands


_PROMPTS = {
    "说点什么", "说点什麼", "說點什麼", "说些什么", "說些什麼",
    "聊点什么", "聊點什麼", "输入评论", "輸入評論", "输入留言", "輸入留言",
}
_ANGLES = np.linspace(0, 2 * np.pi, 40, endpoint=False)
_X, _Y = np.cos(_ANGLES), np.sin(_ANGLES)


def _light_strokes(gray, size):
    span = 2 * (round(size) // 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def _menus(gray):
    height, width = gray.shape
    left = round(.722 * width)
    mark = gray[:, left:]
    ink = _light_strokes(mark, max(5, .018 * width))
    _, _, stats, centers = cv2.connectedComponentsWithStats(np.uint8(ink >= 32))
    dots = []
    for (x, y, w, h, n), (cx, cy) in zip(stats[1:], centers[1:], strict=True):
        if (.006 * width <= min(w, h) <= max(w, h) <= .0165 * width and .66 <= w / h <= 1.50
                and n >= .55 * w * h and .041 * width <= height - cy <= .155 * width
                and mark[round(cy), round(cx)] >= 145):
            dots.append((cx + left, cy, w, h))
    if len(dots) > 14:
        return []
    matches = []
    for triple in combinations(sorted(dots), 3):
        xs, ys, ws, hs = np.asarray(triple).T
        steps = np.diff(xs)
        if (.014 * width <= min(steps) <= max(steps) <= .030 * width
                and np.ptp(steps) <= .0032 * width and np.ptp(ys) <= .0035 * width
                and np.ptp(ws) <= 1.6 and np.ptp(hs) <= 1.6 and .775 * width <= xs[1] <= .977 * width):
            matches.append((int(round(xs[1])), int(round(np.mean(ys)))))
    return matches if len(matches) == 1 else []


def _rays(gray, x, y, radii, *, half=False):
    angles = _X >= 0 if half else np.ones(len(_X), bool)
    xs = np.rint(x + radii[:, None] * _X[angles]).astype(np.int32)
    ys = np.rint(y + radii[:, None] * _Y[angles]).astype(np.int32)
    if min(xs.min(), ys.min()) < 0 or xs.max() >= gray.shape[1] or ys.max() >= gray.shape[0]:
        return None
    return gray[ys, xs].astype(np.int16)


def _circle(gray, x, y, radii, *, half=False):
    delta = max(2, round(.0036 * gray.shape[1]))
    height, width = gray.shape
    radius_limit = min(y, height - y - 1, width - x - 1, width if half else x)
    radii = radii[(radii > delta) & (radii + delta <= radius_limit)]
    if not len(radii):
        return 0., 0
    inside = _rays(gray, x, y, radii - delta, half=half)
    outside = _rays(gray, x, y, radii + delta, half=half)
    if inside is None or outside is None:
        return 0., 0
    signed = outside - inside
    scores = np.mean(np.clip(signed - 11, 0, 24), axis=1)
    coverage = np.mean(signed >= 16, axis=1)
    scores[coverage < (.61 if half else .73)] = 0
    i = np.argmax(scores)
    return scores[i], int(radii[i])


def _chain(gray, menu, radius):
    x, y = menu
    options = []
    search = range(round(1.82 * radius), round(2.72 * radius) + 1)
    radii = np.arange(max(3, round(.89 * radius)), round(1.11 * radius) + 1)
    seen = {}
    for pitch in search:
        strengths = []
        for prior in range(1, 5):
            center = x - prior * pitch
            if center not in seen:
                seen[center] = _circle(gray, center, y, radii)[0] if center > radius else 0
            if seen[center] < 10:
                break
            strengths.append(seen[center])
        if len(strengths) >= 2:
            options.append((len(strengths), sum(strengths), pitch))
    if not options:
        return None
    count, _, pitch = max(options)
    return x - (count + 1) * pitch, y


def _face_marks(gray, x, y, radius):
    extent = round(.73 * radius)
    if extent < 5:
        return False
    chip = gray[y - extent:y + extent + 1, x - extent:x + extent + 1]
    paper = int(np.median(chip))
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(chip >= paper + 31))
    eyes, mouths = [], []
    for xx, yy, w, h, n in stats[1:]:
        cx, cy = xx + w / 2 - extent, yy + h / 2 - extent
        if (min(xx, yy) < 1 or max(xx + w, yy + h) >= len(chip) or .11 * radius > min(w, h)
                or n < .52 * w * h):
            continue
        if -.53 * radius <= cy <= -.08 * radius and .50 <= w / h <= 1.26:
            eyes.append((cx, cy, h))
        if .12 * radius <= cy <= .51 * radius and abs(cx) <= .16 * radius and 1.36 <= w / h <= 3.3:
            mouths.append((cx, cy))
    if len(eyes) != 2 or len(mouths) != 1:
        return False
    (a, b, h1), (c, d, h2) = sorted(eyes)
    return (a < 0 < c and .43 * radius <= c - a <= .90 * radius and abs(a + c) <= .23 * radius
            and abs(b - d) <= .16 * radius and abs(h1 - h2) <= .20 * radius)


def _face(gray, expected, wrapper):
    x, y = expected
    dx, dy = max(2, round(.22 * wrapper)), max(1, round(.10 * wrapper))
    radii = np.arange(max(6, round(.45 * wrapper)), round(.72 * wrapper) + 1)
    choices = []
    offset = max(2, round(.115 * wrapper))
    for cx in range(x - dx, x + dx + 1):
        for cy in range(y - dy, y + dy + 1):
            rays = _rays(gray, cx, cy, np.concatenate((radii - offset, radii, radii + offset)))
            if rays is None:
                continue
            before, center, after = np.split(rays, 3)
            delta = center - np.maximum(before, after)
            scores = np.mean(np.clip(delta - 19, 0, 50), axis=1)
            scores[np.mean(delta >= 22, axis=1) < .76] = 0
            i = np.argmax(scores)
            if scores[i] >= 25:
                choices.append((scores[i], cx, cy, int(radii[i])))
    for _, cx, cy, radius in sorted(choices, reverse=True):
        if _face_marks(gray, cx, cy, radius):
            return cx, cy
    return None


def _capsule(gray, face, radius):
    x, y = face
    radii = np.arange(round(.91 * radius), round(1.11 * radius) + 1)
    score, r = _circle(gray, x, y, radii, half=True)
    if score < 10 or not .34 * gray.shape[1] <= x + r <= .65 * gray.shape[1]:
        return None
    offset = max(2, round(.12 * r))
    left, right = round(.025 * gray.shape[1]), x - round(.19 * r)
    if y - r - offset < 0 or y + r + offset >= len(gray):
        return None
    top = gray[y - r - offset, left:right].astype(np.int16) - gray[y - r + offset, left:right]
    bottom = gray[y + r + offset, left:right].astype(np.int16) - gray[y + r - offset, left:right]
    aligned = np.uint8((top >= 13) & (bottom >= 13))
    kernel = np.ones((1, max(3, r // 3)), np.uint8)
    aligned = cv2.morphologyEx(aligned[None], cv2.MORPH_CLOSE, kernel).ravel()
    runs = [(a, b) for a, b in _bands(aligned) if b - a >= .24 * gray.shape[1]
            and a + left <= .14 * gray.shape[1] and right - b - left <= .44 * r]
    if len(runs) != 1:
        return None
    start, _ = runs[0]
    return max(round(.018 * gray.shape[1]), left + start - round(.75 * r)), y - r, x + r, y + r


def _prompt_shape(ink, radius, index):
    spans = _bands(ink.any(axis=0))
    tall = [(a, b) for a, b in spans if .42 * radius <= cv2.boundingRect(ink[:, a:b])[3] <= 1.02 * radius
            and b - a >= .10 * radius]
    if not 3 <= len(tall) <= 10:
        return None
    left, right = tall[0][0], tall[-1][1]
    bx, by, bw, bh = cv2.boundingRect(ink[:, left:right])
    if (left > 1.23 * radius or not .60 * radius <= bh <= .94 * radius or
            not 3.23 * bh <= bw <= 5.8 * bh or abs(by + bh / 2 - len(ink) / 2) > .20 * radius):
        return None
    parts = _cjk_parts(ink[by:by + bh, left + bx:left + bx + bw], bh, join_radicals=True)
    if not 4 <= len(parts) <= 5:
        return None
    expected = {word for word in _PROMPTS if len(word) == len(parts)}
    for position, (_, _, _, _, mask) in enumerate(parts):
        result = index.rank(mask, alphabet="cjk", limit=2)
        if len(result) != 2 or result[0]["distance"] > .195 or result[1]["distance"] - result[0]["distance"] < .014:
            return None
        expected = {word for word in expected if word[position] == result[0]["text"]}
        if not expected:
            return None
    return next(iter(expected))


def _verify_placeholder(crop, r, index):
    if not crop.size:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    strokes = _light_strokes(gray, max(5, .177 * r))
    for low, high in ((42, 46), (23, 32)):
        first = _prompt_shape(np.uint8(strokes >= low), r, index)
        if first in _PROMPTS and first == _prompt_shape(np.uint8(strokes >= high), r, index):
            return True
    if _white_placeholder(crop, gray, strokes, r, index):
        return True
    # Recover thin strokes in small text without amplifying a dimmed footer.
    if r >= 32 or np.percentile(gray, 98) < 224:
        return False
    reference_radius = 48
    scale = reference_radius / r
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    strokes = _light_strokes(gray, .177 * reference_radius)
    return _white_placeholder(enlarged, gray, strokes, reference_radius, index)


def _placeholder(frame, box, index):
    left, top, right, bottom = box
    r = (bottom - top) / 2
    y = (top + bottom) / 2
    x0, x1 = left + round(.54 * r), min(right - round(2 * r), left + round(5.35 * r))
    lane = frame[round(y - .62 * r):round(y + .62 * r)]
    if _verify_placeholder(lane[:, x0:x1], r, index):
        return True
    # Video can hide the leading part of the capsule rim. Keep the measured
    # click geometry and recheck the missing glyphs on the same baseline.
    width = frame.shape[1]
    if left <= .073 * width:
        return False
    earlier = max(round(.025 * width), x0 - round(.84 * r))
    extension = lane[:, earlier:x0]
    if not extension.size or np.percentile(extension.min(axis=2), 98) < 211:
        return False
    return _verify_placeholder(lane[:, earlier:x1], r, index)


def _white_placeholder(crop, gray, strokes, radius, index):
    # The translucent field shares texture with the video. A bright color
    # floor rejects scene edges that can join a printed radical after JPEG.
    white = crop.min(axis=2)
    fine = _light_strokes(gray, max(5, .14 * radius))
    samples = (
        (strokes, gray, 20, 155),
        (strokes, gray, 30, 155),
        (fine, gray, 18, 183),
        (strokes, white, 26, 175),
        (strokes, white, 34, 145),
        (strokes, white, 20, 175),
        (strokes, white, 32, 145),
    )
    readings = {}
    for contrast, tone, cutoff, floor in samples:
        mask = np.uint8((contrast >= cutoff) & (tone >= floor))
        word = _prompt_shape(mask, radius, index)
        if word is None:
            continue
        if word in readings and not np.array_equal(mask, readings[word]):
            return True
        readings[word] = mask
    return False


def live_chat_input_actions(frame, regions, font_index):
    """Return the input action only when the full-page video footer proves it."""
    height, width = frame.shape[:2]
    if (font_index is None or regions.get("layer") != "page" or regions.get("regions")
            or not 1.66 * width <= height <= 3.08 * width or width < 350):
        return []
    top = height - round(.208 * width)
    crop = cv2.cvtColor(frame[top:], cv2.COLOR_BGR2GRAY)
    span = min(560, width)
    scale = span / width
    gray = cv2.resize(crop, (span, round(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    menus = _menus(gray)
    if not menus:
        return []
    menu = menus[0]
    strength, radius = _circle(gray, *menu, np.arange(round(.032 * span), round(.069 * span)))
    anchor = _chain(gray, menu, radius) if strength >= 12 else None
    face = _face(gray, anchor, radius) if anchor else None
    owner = _capsule(gray, face, radius) if face else None
    if owner is None:
        return []
    a, b, c, d = owner
    box = (round(a / scale), top + round(b / scale), round(c / scale), top + round(d / scale))
    if not _placeholder(frame, box, font_index):
        return []
    left, upper, right, lower = box
    return [{"role": "comment_input", "function": "输入直播评论", "box": list(box),
             "click_point": [round(left + .39 * (right - left)), (upper + lower) // 2]}]
