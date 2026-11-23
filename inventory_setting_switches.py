"""Find switch glyphs from a rounded track and an independently visible thumb."""

from __future__ import annotations

from itertools import product

import cv2
import numpy as np


_ANGLES = np.arange(32)[:, None] * (2 * np.pi / 32)
_RADII = np.array((.43, .72, .89, 1.07, 1.17))
_PROBE_X = np.cos(_ANGLES) * _RADII
_PROBE_Y = np.sin(_ANGLES) * _RADII
_INSIDE_RADII = np.array((.44, .72, .89, 1.17))
_INSIDE_PROBE_X = np.cos(_ANGLES) * _INSIDE_RADII
_INSIDE_PROBE_Y = np.sin(_ANGLES) * _INSIDE_RADII


def _track(crop, box, side, background):
    """Recognize one exposed semicircular end of a track on the opposite side from its thumb."""
    x, y, width, height = box
    sample_x = round(x + (.20 if side == 'left' else .80) * width)
    sample_y = round(y + .44 * height)
    tone = np.median(crop[sample_y - 1:sample_y + 2, sample_x - 1:sample_x + 2], axis=(0, 1))
    distance = np.max(np.abs(tone - background))
    if distance < 22:
        return None
    tolerance = max(7, min(16, .20 * distance))
    mask = cv2.inRange(crop, tuple(tone - tolerance), tuple(tone + tolerance))
    _, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    index = components[sample_y, sample_x]
    if index == 0:
        return None
    x0, y0, w, h, area = map(int, stats[index])
    outer_x = x0 if side == 'left' else x0 + w
    outer_limit = x if side == 'left' else x + width
    if (not .27 * width <= h <= .55 * width or not .23 * width <= w <= .64 * width
            or not .54 <= w / h <= 1.66 or not .53 <= area / (w * h) <= .96
            or abs(outer_x - outer_limit) > .09 * width):
        return None
    ink = components[y0:y0 + h, x0:x0 + w] == index
    if side == 'right':
        ink = ink[:, ::-1]
    slices = (ink[round(.09 * h):round(.15 * h) + 1], ink[round(.41 * h):round(.59 * h) + 1],
              ink[round(.85 * h):round(.91 * h) + 1])
    lefts = []
    for pixels in slices:
        edge = [np.flatnonzero(row)[0] for row in pixels if row.any()]
        if not edge:
            return None
        lefts.append(float(np.median(edge)))
    if min(lefts[0], lefts[2]) - lefts[1] < .12 * h or abs(lefts[0] - lefts[2]) > .26 * h:
        return None
    return (x0, y0, x0 + w, y0 + h), tone, float(distance)


def _thumb(crop, box, track, side):
    """Fit a nearly uniform circular thumb using radial transitions, including its soft shadow."""
    x, _, width, _ = box
    x0, y0, x1, y1 = track
    h = y1 - y0
    cy = (y0 + y1 - 1) / 2
    max_x = crop.shape[1] - 1
    max_y = crop.shape[0] - 1
    search = []
    for radius, inset, offset in product(np.linspace(.57 * h, .83 * h, 9),
                                         (-.035 * width, 0, .025 * width, .055 * width),
                                         (-.075 * h, 0, .075 * h)):
        cx = x + width - 1 - radius - inset if side == 'left' else x + radius + inset
        if abs(cx - (x0 + x1) / 2) < .28 * width:
            continue
        center_y = cy + offset
        if cx - 1.18 * radius < 0 or cx + 1.18 * radius > max_x or center_y - 1.18 * radius < 0 or center_y + 1.18 * radius > max_y:
            continue
        search.append((cx, center_y, radius))
    if not search:
        return None
    values = np.asarray(search)
    xc, yc, radius = values.T[:, :, None, None]
    px = np.rint(xc + radius * _PROBE_X).astype(int)
    py = np.rint(yc + radius * _PROBE_Y).astype(int)
    colors = crop[py, px].astype(np.float32)
    tone = np.median(colors[:, :, :2, :], axis=(1, 2))
    contrast = np.max(np.abs(colors - tone[:, None, None, :]), axis=-1)
    edge = contrast[:, :, 3:].max(axis=-1)
    inside = (contrast[:, :, :3] <= 8).mean(axis=(1, 2))
    weak = (edge >= 3).mean(axis=1)
    strong = (edge >= 8).mean(axis=1)
    valid = (inside >= .88) & (weak >= .70) & (strong >= .39)
    if not valid.any():
        return None
    scores = inside + weak + .30 * strong - .12 * np.abs(values[:, 2] / h - .71)
    best = int(np.argmax(np.where(valid, scores, -1)))
    return *search[best], tone[best], scores[best]


def _color_state(track, track_delta, thumb, side):
    track_chroma, thumb_chroma = np.ptp(track), np.ptp(thumb)
    if side == 'left' and track_chroma >= 12 and thumb_chroma >= 27:
        track_hue = track - track.mean()
        thumb_hue = thumb - thumb.mean()
        agreement = np.dot(track_hue, thumb_hue) / (np.linalg.norm(track_hue) * np.linalg.norm(thumb_hue))
        if agreement >= .87:
            return 'on'
    if side == 'right' and max(track_chroma, thumb_chroma) <= 19 and track_delta >= 32:
        return 'off'
    return 'unknown'


def _switch(crop, local_box, background):
    matches = []
    for side in ('left', 'right'):
        track = _track(crop, local_box, side, background)
        if track is None:
            continue
        step, tone, contrast = track
        thumb = _thumb(crop, local_box, step, side)
        if thumb is None:
            continue
        cx, cy, radius, thumb_tone, score = thumb
        x0, y0, x1, y1 = step
        box = (min(x0, round(cx - radius)), min(y0, round(cy - radius)),
               max(x1, round(cx + radius) + 1), max(y1, round(cy + radius) + 1))
        matches.append((score, box, _color_state(tone, contrast, thumb_tone, side)))
    return matches[0] if len(matches) == 1 else None


def _stadium_mask(height, width, inset=0):
    y, x = np.ogrid[:height, :width]
    radius = height / 2
    dx = np.maximum(0, np.maximum(radius - .5 - x, x - (width - radius - .5)))
    return dx ** 2 + (y - (height - 1) / 2) ** 2 <= (radius - inset) ** 2


def _stadium(component):
    height, width = component.shape
    if not 1.46 <= width / height <= 2.37:
        return False
    contours, _ = cv2.findContours(np.uint8(component), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    exterior = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(exterior, contours, -1, 1, cv2.FILLED)
    ideal = _stadium_mask(height, width)
    return (exterior & ideal).sum() >= .92 * (exterior | ideal).sum()


def _oblong_thumb(chip, track, side):
    height, width = chip.shape[:2]
    interior = _stadium_mask(height, width, inset=max(1, round(.037 * height)))
    contrast = np.max(np.abs(chip.astype(float) - track), axis=2)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8((contrast >= 28) & interior), connectivity=8)
    for index, (x, y, w, h, _) in enumerate(stats[1:], start=1):
        cx, cy = x + w / 2, y + h / 2
        offset = cx if side == "left" else width - cx
        if (.234 * width <= offset <= .395 * width and abs(cy - .5 * height) <= .095 * height
                and .63 * height <= h <= .91 * height and .38 * width <= w <= .66 * width
                and _stadium(labels[y:y + h, x:x + w] == index)):
            return 1.97, track, (cx, cy, h / 2, w / h)
    return None


def _inner_thumb(chip, side):
    height, width = chip.shape[:2]
    opposite = .73 if side == "left" else .27
    center = round(opposite * (width - 1))
    mid = (height - 1) / 2
    patch = max(1, round(.055 * height))
    track = np.median(chip[round(mid) - patch:round(mid) + patch + 1,
                           center - patch:center + patch + 1], axis=(0, 1))
    centers = np.linspace(.244, .366, 8) * (width - 1)
    if side == "right":
        centers = width - 1 - centers
    aspects = (1, 1.23, 1.46, 1.63) if width / height >= 1.84 else (1,)
    hypotheses = np.asarray(list(product(centers, mid + height * np.array((-.04, 0, .04)),
                                        np.linspace(.204, .456, 12) * height, aspects)))
    xc, yc, radii, stretch = hypotheses.T[:, :, None, None]
    x = np.rint(xc + radii * stretch * _INSIDE_PROBE_X).astype(int)
    y = np.rint(yc + radii * _INSIDE_PROBE_Y).astype(int)
    valid_xy = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    colors = chip[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)].astype(np.float32)
    thumb = np.median(colors[:, :, :2], axis=(1, 2))
    delta = np.max(np.abs(thumb - track), axis=1)
    deviation = np.max(np.abs(colors - thumb[:, None, None]), axis=-1)
    from_track = np.max(np.abs(colors[:, :, -1] - track), axis=-1)
    inside = ((deviation[:, :, :3] <= 13) & valid_xy[:, :, :3]).mean(axis=(1, 2))
    outside = ((from_track <= 16) & valid_xy[:, :, -1]).mean(axis=1)
    valid = (delta >= 23) & (inside >= .86) & (outside >= .76)
    if not valid.any():
        return _oblong_thumb(chip, track, side) if width / height >= 1.84 else None
    score = inside + outside
    best = int(np.argmax(np.where(valid, score, -1)))
    return score[best], track, hypotheses[best]


def _embedded_switch(crop, box, component, background):
    x, y, width, height = box
    if not _stadium(component):
        return None
    chip = crop[y:y + height, x:x + width]
    sides = [(side, result) for side in ("left", "right") if (result := _inner_thumb(chip, side)) is not None]
    if not sides:
        return None
    sides.sort(key=lambda item: item[1][0], reverse=True)
    side, (score, track, (_, _, radius, _)) = sides[0]
    if len(sides) == 2 and score - sides[1][1][0] < .12:
        return None
    # The closed surface must be clearly visible even for an outlined off track.
    exposed = round((.73 if side == "left" else .27) * (width - 1))
    rim = float(np.max(np.abs(chip[:max(2, round(.16 * height)), exposed].astype(float) - background)))
    if rim < 42:
        return None
    chroma = float(np.ptp(track))
    track_contrast = float(np.max(np.abs(track - background)))
    if side == "right" and (chroma >= 25 or track_contrast >= 65) and radius >= .30 * height:
        state = "on"
    elif side == "left" and chroma <= 38:
        state = "off"
    else:
        state = "unknown"
    return score, (x, y, x + width, y + height), state


def _near(first, second):
    x0, y0, x1, y1 = first
    a, b, c, d = second
    overlap = max(0, min(x1, c) - max(x0, a)) * max(0, min(y1, d) - max(y0, b))
    return overlap >= .58 * min((x1 - x0) * (y1 - y0), (c - a) * (d - b))


def find_setting_switches(frame_bgr, *, top=0, bottom=None):
    """Return measured right-trailing switch glyphs. Unknown state is kept explicit."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        return []
    height, width = frame_bgr.shape[:2]
    top = max(0, round(top))
    bottom = height if bottom is None else min(height, round(bottom))
    if min(height, width) < 140 or bottom - top < 20:
        return []
    left = round(.68 * width)
    surface = frame_bgr[top:bottom, left:]
    row_tones = np.median(surface[:, ::max(1, round(.009 * width))], axis=1)
    contrast = np.max(np.abs(surface.astype(np.int16) - row_tones[:, None, :]), axis=2)
    proposals = []
    measured = []
    for threshold in (12, 20, 25, 27, 38):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(contrast >= threshold), connectivity=8)
        for index, (x, y, w, h, _) in enumerate(stats[1:].tolist(), start=1):
            if (not .034 * width <= w <= .15 * width or not .018 * width <= h <= .11 * width
                    or not 1.20 <= w / h <= 2.30 or not 1 <= x < x + w < surface.shape[1] - 1
                    or not 1 <= y < y + h < len(surface) - 1):
                continue
            if any(_near((x, y, x + w, y + h), other) for other in measured):
                continue
            padding = round(.27 * w)
            a, b = max(0, x - padding), max(0, y - padding)
            c, d = min(surface.shape[1], x + w + padding), min(len(surface), y + h + padding)
            chip = surface[b:d, a:c]
            local_box = (x - a, y - b, w, h)
            background = row_tones[y + h // 2]
            component = labels[y:y + h, x:x + w] == index
            result = _embedded_switch(chip, local_box, component, background)
            style = "embedded"
            if result is None and threshold <= 20:
                result = _switch(chip, local_box, background)
                style = "protruding"
            if result is None:
                continue
            score, (u, v, r, end), state = result
            box = (u + a + left, v + b + top, r + a + left, end + b + top)
            if 0 <= box[0] < box[2] <= width and top <= box[1] < box[3] <= bottom:
                proposals.append((score, box, state, style))
                measured.append((x, y, x + w, y + h))
    accepted = []
    for _, box, state, style in sorted(proposals, reverse=True):
        if any(_near(box, row['box']) for row in accepted):
            continue
        x0, y0, x1, y1 = box
        accepted.append({'box': box, 'center': ((x0 + x1) / 2, (y0 + y1) / 2), 'state': state, 'style': style})
    return sorted(accepted, key=lambda row: row['center'][1])
