"""Locate printed mention and smile glyphs in a verified comment composer."""

import cv2
import numpy as np


_SIZE = 36
_ANGLES = np.deg2rad(np.arange(0, 360, 15))
_RADII = np.array((.745, .80, .855, .905, .96, 1.015))[:, None]
_OUTER_X = np.float32((np.cos(_ANGLES) * _RADII + 1) * _SIZE / 2 - .5)
_OUTER_Y = np.float32((np.sin(_ANGLES) * _RADII + 1) * _SIZE / 2 - .5)
_AXIS = (2 * np.arange(_SIZE) + 1) / _SIZE - 1
_FACE_INTERIOR = _AXIS[:, None] ** 2 + _AXIS ** 2 <= .73 ** 2
_CLOSE_GAPS = np.ones((3, 3), np.uint8)


def _contrast(gray, width, *, dark):
    paint = 255 - gray if dark else gray
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (round(.115 * width), round(.085 * width)))
    return cv2.subtract(cv2.morphologyEx(paint, cv2.MORPH_CLOSE, kernel), paint)


def _mention(sample, peak, rim):
    # A typographic @ is a nearly circular arc with a separately enclosed,
    # much smaller counter and an opening on its lower-right shoulder.
    if (rim >= .57).sum() < 20 or rim[6:23].min() < .39:
        return False
    confirmed = 0
    for fraction in (.41, .54, .65, .74, .82):
        ink = np.uint8(sample >= peak * fraction)
        contours, tree = cv2.findContours(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, _CLOSE_GAPS),
                                          cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if tree is None:
            continue
        possible = []
        for i, hole in enumerate(contours):
            parent = tree[0, i, 3]
            if parent < 0:
                continue
            x, y, w, h = (value / _SIZE for value in cv2.boundingRect(hole))
            area = cv2.contourArea(hole) / (_SIZE ** 2)
            main = cv2.contourArea(contours[parent]) / (_SIZE ** 2)
            if (.025 <= area <= .105 and .17 <= w <= .38 and .20 <= h <= .45
                    and .42 <= x + w / 2 <= .58 and .41 <= y + h / 2 <= .59 and .20 <= main <= .61):
                possible.append(hole)
        if len(possible) == 1:
            confirmed += 1
    # The arc may have few gray pixels over its opening after resampling.
    return confirmed >= 2 and (rim[2:7].min() < .73 or confirmed >= 3)


def _face(sample, peak, rim):
    if (rim >= .57).sum() < 21:
        return False
    confirmed = 0
    for fraction in (.33, .46, .59):
        interior = np.uint8((sample >= peak * fraction) & _FACE_INTERIOR)
        _, _, components, _ = cv2.connectedComponentsWithStats(interior, connectivity=8)
        pieces = ((x / _SIZE, y / _SIZE, w / _SIZE, h / _SIZE, area / (_SIZE ** 2))
                  for x, y, w, h, area in components[1:])
        eyes, mouths = [], []
        for x, y, w, h, area in pieces:
            cx, cy = x + w / 2, y + h / 2
            if .079 <= w <= .205 and .089 <= h <= .237 and .009 <= area <= .046 and .29 <= cy <= .45:
                eyes.append((cx, cy))
            if .216 <= w <= .475 and .090 <= h <= .273 and .017 <= area <= .085:
                if .43 <= cx <= .57 and .59 <= cy <= .72:
                    mouths.append((cx, cy))
        if len(eyes) == 2 and len(mouths) == 1:
            (a, y), (b, y2) = sorted(eyes)
            if .22 <= a <= .44 and .56 <= b <= .79 and .21 <= b - a <= .44 and abs(y - y2) <= .06:
                confirmed += 1
    return confirmed >= 2


def _icons(ink, width):
    """Require the same bounded role at more than one contrast level."""
    samples = []
    measured = {}
    for threshold in (25, 41, 62, 87):
        contours, _ = cv2.findContours(np.uint8(ink >= threshold), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for outline in contours:
            x, y, w, h = cv2.boundingRect(outline)
            if (not .035 * width <= min(w, h) <= max(w, h) <= .083 * width or not .79 <= w / h <= 1.21
                    or min(x, y, ink.shape[1] - x - w, ink.shape[0] - y - h) <= 0
                    or not .013 * width <= ink.shape[0] - y - h / 2 <= .228 * width):
                continue
            box = (x, y, x + w, y + h)
            if box not in measured:
                sample = cv2.resize(ink[y:y + h, x:x + w], (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR)
                peak = float(np.percentile(sample, 95))
                measured[box] = None
                if peak >= 43:
                    rim = cv2.remap(sample, _OUTER_X, _OUTER_Y, cv2.INTER_LINEAR).max(axis=0) / peak
                    if _face(sample, peak, rim):
                        measured[box] = "emoji"
                    elif _mention(sample, peak, rim):
                        measured[box] = "mention"
            role = measured[box]
            if role is not None:
                samples.append((role, threshold, box))
    accepted = []
    for role, level, box in samples:
        cluster = [other for kind, other_level, other in samples if kind == role and other_level != level
                   and max(abs(a - b) for a, b in zip(box, other, strict=True)) <= .014 * width]
        if not cluster:
            continue
        average = tuple(np.rint(np.median([box, *cluster], axis=0)).astype(int))
        if not any(kind == role and max(abs(a - b) for a, b in zip(average, other, strict=True)) <= .019 * width
                   for kind, other in accepted):
            accepted.append((role, average))
    return accepted


def _left_caption(gray, y, end, *, dark):
    """Observe a printed input-length line independently of the toolbar glyphs."""
    height, width = gray.shape
    radius = round(.055 * width)
    left, right = round(.027 * width), min(round(.637 * width), end - round(.095 * width))
    top, bottom = max(0, y - radius), min(height, y + radius)
    plane = gray[top:bottom, left:right]
    if not plane.size:
        return False
    ink = _contrast(plane, width, dark=dark)
    for threshold in (27, 42):
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(ink >= threshold))
        glyphs = [(x + w / 2, y0 + h / 2) for x, y0, w, h, area in stats[1:]
                  if .005 * width <= w <= .068 * width and .016 * width <= h <= .055 * width
                  and abs(top + y0 + h / 2 - y) <= .034 * width and area >= .15 * w * h]
        for _, yc in glyphs:
            xs = [x + left for x, yy in glyphs if abs(yy - yc) <= .014 * width]
            if len(xs) >= 5 and min(xs) <= .33 * width and max(xs) - min(xs) >= .13 * width:
                return True
    return False


def _action(role, box):
    left, top, right, bottom = box
    purpose, preview = {"mention": ("在评论中提及用户", "@"), "emoji": ("打开评论表情", "表情")}[role]
    return {"role": role, "function": purpose, "preview_text": preview, "box": list(box),
            "click_point": [(left + right) / 2, (top + bottom) / 2]}


def comment_composer_icons(frame, panel, index=None):
    """Return source-pixel glyph actions within a separately proven comment sheet."""
    left, top, right, bottom = panel
    native_width = right - left
    if native_width < 245 or bottom - top < .35 * native_width:
        return []
    y0 = max(top, bottom - round(.34 * native_width))
    width = 560
    height = round((bottom - y0) * width / native_width)
    interpolation = cv2.INTER_AREA if native_width >= width else cv2.INTER_LINEAR
    footer = cv2.resize(frame[y0:bottom, left:right], (width, height), interpolation=interpolation)
    gray = cv2.cvtColor(footer, cv2.COLOR_BGR2GRAY)
    dark = np.median(gray[:, round(.61 * width):]) < 107
    x0, x1 = round(.615 * width), round(.987 * width)
    blue, green, red = cv2.split(footer[:, x0:x1])
    chroma = cv2.subtract(cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red))
    ink = _contrast(gray[:, x0:x1], width, dark=dark)
    ink[chroma > 46] = 0
    candidates = _icons(ink, width)
    if not candidates:
        return []
    rows = []
    for role, (l, t, r, b) in candidates:
        x, y = x0 + (l + r) / 2, (t + b) / 2
        rows.append((role, (x, y), (x0 + l, t, x0 + r, b)))
    matches = []
    for role, (x, y), box in rows:
        peers = [other for kind, (xx, yy), other in rows if kind != role and abs(yy - y) <= .020 * width
                 and .064 * width <= abs(xx - x) <= .251 * width]
        if (peers or x >= .762 * width) and _left_caption(gray, round(y), box[0], dark=dark):
            matches.append((role, box))
    if len(matches) != len({role for role, _ in matches}):
        return []

    scale_x, scale_y = native_width / width, (bottom - y0) / height
    return [_action(role, (round(left + l * scale_x), round(y0 + t * scale_y),
                           round(left + r * scale_x), round(y0 + b * scale_y)))
            for role, (l, t, r, b) in sorted(matches, key=lambda row: row[1][0])]
