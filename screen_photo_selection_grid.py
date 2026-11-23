"""Measure the separate image and corner-marker geometry of a photo picker."""

from math import ceil, floor

import cv2
import numpy as np

from screen_component_captions import _bands


_ANGLES = np.arange(48) * (2 * np.pi / 48)
_UNIT_CIRCLE = np.column_stack((np.cos(_ANGLES), np.sin(_ANGLES)))


def _footer_bounds(gray, neutral, width):
    height = len(gray)
    inset = round(.055 * width)
    support = (neutral[:, inset:width - inset:2]).mean(axis=1)
    # The row buttons interrupt the otherwise continuous dark footer.
    bands = _bands(support >= .64)
    reach = 2 * max(1, round(.026 * width)) + 1
    joined = cv2.morphologyEx(np.uint8((support >= .64)[None, :]), cv2.MORPH_CLOSE,
                              np.ones((1, reach), np.uint8))[0]
    for start, bottom in _bands(joined != 0):
        top = next((a for a, b in bands if start <= a < bottom and b - a >= .10 * width), None)
        if top is None:
            continue
        if not max(.49 * height, .53 * width) <= top or not .13 * width <= bottom - top <= .42 * width:
            continue
        clear = gray[top + max(2, round(.009 * width)):top + round(.034 * width)]
        paint = float(np.median(clear[:, round(.25 * width):round(.75 * width)]))
        rails = [(a, b) for a, b in _bands((abs(clear.astype(np.int16) - paint) <= 11).mean(axis=0) >= .80)
                 if a <= .08 * width and b >= .92 * width]
        if len(rails) == 1:
            left, right = rails[0]
            yield left, top, right, bottom


def _grid_columns(neutral, footer):
    left, top, right, _ = footer
    width = right - left
    start = max(0, top - round(.77 * width))
    band = neutral[start:top - round(.032 * width):2, left:right]
    gutters = [(left + a, left + b) for a, b in _bands(band.mean(axis=0) >= .72)
               if .105 * width <= a < b <= .90 * width and b - a <= .016 * width]
    if not 2 <= len(gutters) <= 6:
        return None
    centers = np.array([(a + b) / 2 for a, b in gutters])
    gaps = np.diff(centers)
    pitch = float(np.median(gaps))
    if not .14 * width <= pitch <= .35 * width or max(abs(gaps - pitch)) > .075 * pitch:
        return None
    if not (.76 * pitch <= centers[0] - left <= 1.19 * pitch
            and .76 * pitch <= right - centers[-1] <= 1.19 * pitch):
        return None
    columns = list(zip([left, *(b for _, b in gutters)], [*(a for a, _ in gutters), right], strict=True))
    return pitch, columns


def _grid_rows(neutral, footer, pitch):
    left, footer_top, right, _ = footer
    inset = round(.035 * (right - left))
    strengths = neutral[:footer_top, left + inset:right - inset:2].mean(axis=1)
    bands = _bands(strengths >= .69)
    tails = [(a, b) for a, b in bands if 1 <= b - a <= .047 * pitch and .005 * pitch <= footer_top - b <= .17 * pitch]
    if len(tails) > 1:
        return ()
    bottom = tails[0][0] if tails else footer_top
    rows = []
    for a, b in reversed(bands):
        depth = bottom - b
        if depth < .79 * pitch:
            continue
        if depth > 1.16 * pitch:
            break
        rows.append((b, bottom))
        if b - a > .047 * pitch:
            break
        bottom = a
    return tuple(reversed(rows)) if 2 <= len(rows) <= 6 else ()


def _marker_chip(frame, gray, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    x0, y0 = left + round(.60 * w), top + round(.024 * h)
    x1, y1 = right, top + round(.39 * h)
    color = frame[y0:y1, x0:x1].astype(np.int16)
    patch = gray[y0:y1, x0:x1]
    size = min(w, h)
    blue, green, red = cv2.split(color)
    mask = np.uint8((green >= 99) & (green - red >= 60) & (green - blue >= 41))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marked = []
    for contour in contours:
        x, y, span, depth = cv2.boundingRect(contour)
        fill = cv2.contourArea(contour) / (span * depth)
        if (.16 * size <= min(span, depth) <= max(span, depth) <= .34 * size
                and .83 <= span / depth <= 1.20 and .63 <= fill <= .89
                and mask[y:y + depth, x:x + span].mean() >= .57):
            marked.append((x0 + x, y0 + y, x0 + x + span, y0 + y + depth))
    if marked:
        return (marked[0], "selected") if len(marked) == 1 else (None, None)

    radius_min = max(5, round(.078 * size))
    radius_max = round(.168 * size)
    circles = cv2.HoughCircles(cv2.GaussianBlur(patch, (3, 3), 0), cv2.HOUGH_GRADIENT, 1,
                               radius_min, param1=110, param2=max(7, round(.055 * size)),
                               minRadius=radius_min, maxRadius=radius_max)
    if circles is None:
        return None, None
    rings = []
    for x, y, radius in circles[0]:
        if min(x, y, patch.shape[0] - y) < .97 * radius or patch.shape[1] - x < .70 * radius:
            continue
        # Recheck the Hough suggestion at three radii using actual ring ink.
        points = np.rint(np.array([x, y]) + _UNIT_CIRCLE[None, :, :]
                         * np.array([.92, 1., 1.08])[:, None, None] * radius).astype(int)
        xs, ys = points[..., 0], points[..., 1]
        visible = (0 <= xs) & (xs < patch.shape[1]) & (0 <= ys) & (ys < patch.shape[0])
        angles = visible.any(axis=0)
        if angles.mean() < .80:
            continue
        ink = np.zeros(visible.shape, np.uint8)
        ink[visible] = patch[ys[visible], xs[visible]]
        painted = (ink.max(axis=0)[angles] >= 169).mean()
        mid = patch[round(y - .43 * radius):round(y + .43 * radius),
                    round(x - .43 * radius):round(x + .43 * radius)]
        if painted >= .84 and mid.size and (mid >= 169).mean() <= .15:
            rings.append((float(painted),
                          (x0 + floor(x - radius), y0 + floor(y - radius),
                           min(x1, x0 + ceil(x + radius)), y0 + ceil(y + radius))))
    if not rings:
        return None, None
    return max(rings)[1], "unselected"


def _visible_tiles(frame, gray, columns, rows):
    tiles = []
    profiles = []
    normal_width = float(np.median([right - left for left, right in columns]))
    for top, bottom in rows:
        for left, right in columns:
            width, height = right - left, bottom - top
            x = round(.075 * width)
            chip = gray[top + round(.36 * height):bottom - round(.075 * height):2, left + x:right - x:2]
            if not chip.size or chip.std() < 15:
                continue
            box = (left, top, right, bottom)
            marker, state = _marker_chip(frame, gray, box)
            profile = None
            if marker is not None:
                a, b, c, d = marker
                # Reference the lattice width at the two clipped viewport edges.
                origin = min(left, right - normal_width) if left == columns[0][0] else left
                profile = ((a + c - 2 * origin) / (2 * normal_width), (b + d - 2 * top) / (2 * height),
                           (c - a) / normal_width, (d - b) / height)
                if .67 <= profile[0] <= .93 and .09 <= profile[1] <= .31:
                    profiles.append(profile)
                else:
                    profile = None
            tiles.append((box, marker, state, profile))
    if len(profiles) < 3:
        return None
    typical = np.median(profiles, axis=0)
    confirmed = []
    output = []
    for box, marker, state, profile in tiles:
        if profile is not None and abs(np.asarray(profile) - typical).max() <= .072:
            confirmed.append(box)
            output.append((box, marker, state))
        else:
            output.append((box, None, None))
    if (len(confirmed) < max(3, ceil(.43 * len(tiles)))
            or len({box[0] for box in confirmed}) < 2 or len({box[1] for box in confirmed}) < 2):
        return None
    return tuple(output)


def photo_selection_grid(frame):
    """Return row-major (photo, marker, visual state) and independently measured footer."""
    height, width = frame.shape[:2]
    if width < 260 or not .74 * width <= height <= 2.72 * width:
        return None
    small_width = min(960, width)
    small_height = round(height * small_width / width)
    small = (cv2.resize(frame, (small_width, small_height), interpolation=cv2.INTER_AREA)
             if small_width != width else frame)
    blue, green, red = cv2.split(small)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    spread = cv2.subtract(cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red))
    neutral = (spread <= 53) & (gray >= 21) & (gray <= 99)
    matches = []
    for footer in _footer_bounds(gray, neutral, small_width):
        if (column_info := _grid_columns(neutral, footer)) is None:
            continue
        pitch, columns = column_info
        if not (rows := _grid_rows(neutral, footer, pitch)):
            continue
        if (tiles := _visible_tiles(small, gray, columns, rows)) is not None:
            matches.append((tiles, footer))
    if len(matches) != 1:
        return None

    sx, sy = width / small_width, height / small_height

    def convert(bounds):
        left, top, right, bottom = bounds
        return (round(left * sx), round(top * sy), round(right * sx), round(bottom * sy))

    tiles, footer = matches[0]
    return tuple((convert(box), convert(marker) if marker else None, state) for box, marker, state in tiles), convert(footer)
