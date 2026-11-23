"""Recognize a saturated panel with a separately enclosed dismissal."""

import cv2
import numpy as np

from screen_dialog_close import _cross_footprint
from screen_foreground import _centered_cross


_ANGLES = np.arange(32) * (np.pi / 16)
_DIRECTIONS = np.stack((np.cos(_ANGLES), np.sin(_ANGLES)))


def _enclosures(image):
    height, width = image.shape[:2]
    central = (slice(round(.27 * height), round(.73 * height), 3),
               slice(round(.29 * width), round(.71 * width), 3))
    saturated = np.ptp(image[central], axis=2) >= 104
    if saturated.mean() < .46:
        return
    blue, green, red = cv2.split(image)
    strongest = cv2.max(cv2.max(blue, green), red)
    chroma = strongest - cv2.min(cv2.min(blue, green), red)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.int16)
    votes = np.bincount(hue[central][saturated], minlength=180)
    support = sum(np.roll(votes, shift) for shift in range(-4, 5))
    if support.max() < .46 * hue[central].size:
        return
    offset = abs(hue - np.argmax(support))
    nearby = np.minimum(offset, 180 - offset) < 9
    body = strongest[central][nearby[central] & saturated]
    surface = np.uint8(nearby & (chroma >= 95) & (strongest >= max(163, np.median(body) - 47)))
    solid = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    step = max(3, round(.013 * width))
    for contour in contours:
        left, top, w, h = cv2.boundingRect(contour)
        right, bottom = left + w, top + h
        if (not .49 * width <= w <= .86 * width or not .34 * height <= h <= .68 * height
                or abs(left + w / 2 - width / 2) > .065 * width
                or abs(top + h / 2 - height / 2) > .075 * height
                or min(left, width - right, top, height - bottom) < 2 * step
                or cv2.contourArea(contour) < .90 * w * h):
            continue
        # Inner and outer runs must support a real color boundary on all four sides.
        x = np.arange(left + round(.11 * w), right - round(.11 * w), 2)
        y = np.arange(top + round(.10 * h), bottom - round(.10 * h), 2)
        inner = (surface[top + step, x], surface[bottom - step, x],
                 surface[y, left + step], surface[y, right - step])
        outer = (surface[top - step, x], surface[bottom + step, x],
                 surface[y, left - step], surface[y, right + step])
        if all((i > o).mean() >= .83 for i, o in zip(inner, outer, strict=True)):
            yield left, top, right, bottom


def _ring(gray, x, y, size):
    height, width = gray.shape
    for radius in np.linspace(.92 * size, 1.33 * size, 7):
        if min(x, width - x - 1, y, height - y - 1) <= 1.32 * radius:
            continue
        measures = []
        for factor in (.74, .90, 1, 1.10, 1.28):
            cols, rows = np.rint(np.array([[x], [y]]) + factor * radius * _DIRECTIONS).astype(int)
            measures.append(gray[rows, cols].astype(np.int16))
        inner, ridge, middle, outside_ridge, outside = measures
        difference = np.maximum.reduce((ridge, middle, outside_ridge)) - np.maximum(inner, outside)
        if (difference >= 16).mean() >= .76 and np.median(difference) >= 27:
            return radius
    return None


def _dismissals(gray, panel):
    height, width = gray.shape
    left, top, right, _ = panel
    span = right - left
    start, stop = max(0, top - round(.27 * span)), top - max(2, round(.016 * span))
    for edge, inward in ((left, 1), (right, -1)):
        l, r = sorted((round(edge - .055 * span * inward), round(edge + .235 * span * inward)))
        l, r = max(0, l), min(width, r)
        tile = gray[start:stop, l:r]
        paper = float(np.percentile(tile, 79))
        if paper > 192:
            continue
        for cut in sorted({max(108, paper + 16), max(108, paper + 27),
                           max(136, paper + 38), max(148, paper + 48)}):
            _, labels, parts, _ = cv2.connectedComponentsWithStats(np.uint8(tile >= cut))
            for identity, (a, b, w, h, area) in enumerate(parts[1:], 1):
                x, y = l + a + w / 2, start + b + h / 2
                unit = (w + h) / 2
                if (not .025 * span <= min(w, h) <= max(w, h) <= .087 * span or not .78 <= w / h <= 1.25
                        or not .12 <= area / (w * h) <= .62
                        or not .036 * span <= (x - edge) * inward <= .18 * span
                        or not .051 * span <= top - y <= .165 * span):
                    continue
                radius = _ring(gray, x, y, unit)
                if radius is None:
                    continue
                cx, cy = round(x), round(y)
                core = round(.81 * radius)
                cross = labels[b:b + h, a:a + w] == identity
                if not _cross_footprint(cross) and not _centered_cross(gray[cy - core:cy + core, cx - core:cx + core]):
                    continue
                extension = round(radius + 1)
                yield (cx - extension, cy - extension, cx + extension + 1, cy + extension + 1)


def painted_layer(image, gray):
    """Require a separate round X plus a centered surface with four visible rims."""
    height, width = gray.shape
    if width < 190 or not 1.48 * width <= height <= 2.75 * width:
        return None
    choices = []
    for panel in _enclosures(image):
        unique = []
        for close in _dismissals(gray, panel):
            point = (close[0] + close[2]) / 2, (close[1] + close[3]) / 2
            if not any(a <= point[0] < c and b <= point[1] < d for a, b, c, d in unique):
                unique.append(close)
        if len(unique) == 1:
            choices.append((panel, unique[0]))
    return choices[0] if len(choices) == 1 else None
