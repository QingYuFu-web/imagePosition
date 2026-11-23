"""Verify a settings gear on the control rail of a landscape viewfinder."""

import cv2
import numpy as np

from screen_toolbar_settings import _rings


def _rail_boundary(field):
    height, width = field.shape
    pixels = field[round(.045 * height):round(.955 * height)] > 12
    positions = pixels.argmax(axis=1)[pixels.any(axis=1)]
    counts = np.bincount(positions, minlength=width)
    support = np.convolve(counts, np.ones(max(3, round(.005 * width))), mode="same")
    a, b = round(.063 * width), round(.238 * width)
    edge = a + int(support[a:b].argmax())
    return edge if support[edge] >= .215 * len(pixels) else None


def _texture(gray, left, right):
    height, width = gray.shape
    reach = round(.041 * width)
    y0, y1 = round(.073 * height), round(.92 * height)
    sample = gray[y0:y1:3, left + reach:right - reach:3]
    dark, light = np.percentile(sample, (12, 88))
    return light - dark >= 55 and sum(tile.std() >= 13 for tile in np.array_split(sample, 4, axis=1)) >= 3


def _shutter(frame, bright, rail):
    height, _ = bright.shape
    a, b = rail
    clip = bright[:, a:b]
    contours, hierarchy = cv2.findContours(np.uint8(clip >= 190), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    found = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        child = hierarchy[0, i, 2]
        if (child < 0 or hierarchy[0, i, 3] >= 0 or not .141 * height <= min(w, h) <= max(w, h) <= .29 * height
                or not .92 <= w / h <= 1.09 or not .70 <= area / (w * h) <= .83
                or not .325 * height <= y + h / 2 <= .675 * height
                or x < 1 or x + w >= clip.shape[1]):
            continue
        hx, hy, hw, hh = cv2.boundingRect(contours[child])
        if (not .57 <= cv2.contourArea(contours[child]) / area <= .89
                or not .91 <= hw / hh <= 1.11
                or max(abs(hx + hw / 2 - x - w / 2), abs(hy + hh / 2 - y - h / 2)) > .06 * w):
            continue
        cx, cy, radius = a + x + w / 2, y + h / 2, w / 2
        span = round(.39 * radius)
        core = frame[round(cy) - span:round(cy) + span, round(cx) - span:round(cx) + span].astype(np.int16)
        fill = np.median(core, axis=(0, 1))
        if max(fill) >= 103 and min(fill) <= 175 and (abs(core - fill).max(axis=2) <= 14).mean() >= .92:
            found.append((cx, cy, radius))
    return found[0] if len(found) == 1 else None


def _thumbnail(gray, strong, rail, shutter):
    a, b = rail
    cx, cy, radius = shutter
    height = gray.shape[0]
    end = max(1, round(cy - 1.69 * radius))
    ink = np.uint8(strong[:end, a:b] >= 28)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for x, y, w, h, area in stats[1:]:
        if (.045 * height <= h <= .146 * height and .75 <= w / h <= 1.68 and area >= .78 * w * h
                and x > 0 and y > 0 and x + w < b - a and y + h < end
                and abs(a + x + w / 2 - cx) <= .43 * (b - a)
                and gray[y:y + h, a + x:a + x + w].std() >= 13):
            return True
    return False


def _gear_regions(bright, rail, regions):
    height = bright.shape[0]
    a, b = rail
    cells = []
    for item in regions["candidates"]:
        x0, y0, x1, y1 = item["box"]
        w, h = x1 - x0, y1 - y0
        if (item["visibility"] == "page" and a <= x0 < x1 <= b and 0 <= y0 < y1 < .29 * height
                and .033 * height <= min(w, h) <= max(w, h) <= .245 * height and .60 <= w / h <= 1.59):
            cells.append((x0, y0, x1, y1))
    # The bright glyph can also provide a crop when the button proposal is absent.
    field = bright[:round(.23 * height), a:b]
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(field >= 183), connectivity=8)
    for x, y, w, h, area in stats[1:]:
        if (.029 * height <= h <= .135 * height and .83 <= w / h <= 1.20 and .35 <= area / (w * h) <= .75):
            pad = max(4, round(.23 * h))
            cells.append((max(a, a + x - pad), max(0, y - pad), min(b, a + x + w + pad), y + h + pad))
    return cells


def _peer_marks(bright, rail, gear):
    height = bright.shape[0]
    left, top, right, bottom = gear
    side = min(right - left, bottom - top)
    cx = (left + right) / 2
    a, b = rail
    mark = bright[max(bottom + round(.33 * side), round(.16 * height)):,
                  max(a, round(cx - .61 * side)):min(b, round(cx + .61 * side))]
    occupied = (mark >= 173).sum(axis=1) >= max(2, round(.14 * side))
    runs = np.flatnonzero(np.diff(np.r_[False, occupied, False])).reshape(-1, 2)
    return len([(first, last) for first, last in runs if last - first >= .29 * side]) >= 2


def landscape_camera_toolbar_actions(frame, regions):
    """Name the visible gear only after the viewfinder and capture rail agree."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or width < 1.43 * height or height < 200:
        return []
    ww, hh = min(640, width), round(min(640, width) * height / width)
    small = cv2.resize(frame, (ww, hh), interpolation=cv2.INTER_AREA) if width > ww else frame
    strong = cv2.max(cv2.max(small[..., 0], small[..., 1]), small[..., 2])
    left, right = _rail_boundary(strong), _rail_boundary(strong[:, ::-1])
    if left is None or right is None or abs(left - right) > .073 * ww:
        return []
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if not _texture(gray, left, ww - right):
        return []
    bright = cv2.min(cv2.min(small[..., 0], small[..., 1]), small[..., 2])
    rails = [(0, left), (ww - right, ww)]
    pairs = []
    for thumb, gear_rail in (rails, rails[::-1]):
        shutter = _shutter(small, bright, thumb)
        if shutter is not None and _thumbnail(gray, strong, thumb, shutter):
            pairs.append(gear_rail)
    if len(pairs) != 1:
        return []
    gear_rail = tuple(round(x * width / ww) for x in pairs[0])
    full_bright = cv2.min(cv2.min(frame[..., 0], frame[..., 1]), frame[..., 2])
    found = []
    for candidate in _gear_regions(full_bright, gear_rail, regions):
        box = _rings(frame, candidate)
        if box is not None and _peer_marks(full_bright, gear_rail, box):
            if not found or max(abs(a - b) for a, b in zip(box, found[0], strict=True)) > 3:
                found.append(box)
    if len(found) != 1:
        return []
    left, top, right, bottom = found[0]
    return [{"role": "settings", "function": "打开设置", "preview_text": "Settings", "box": list(found[0]),
             "click_point": [(left + right) / 2, (top + bottom) / 2]}]
