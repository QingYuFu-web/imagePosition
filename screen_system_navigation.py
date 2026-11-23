"""Measure a three-button Android system strip separately from app sheets."""

import cv2
import numpy as np


def _left_strokes(ink, width, middle_y, icon_height):
    rail = ink[:, round(.15 * width):round(.30 * width)]
    _, _, components, _ = cv2.connectedComponentsWithStats(rail)
    bars = sorted((x, y, h) for x, y, w, h, area in components[1:]
                  if .70 * icon_height <= h <= 1.24 * icon_height and w <= .31 * h
                  and area >= .67 * w * h and abs(y + h / 2 - middle_y) <= .21 * icon_height)
    if len(bars) != 3:
        return False
    spacing = np.diff([x for x, _, _ in bars])
    return min(spacing) >= .19 * icon_height and max(spacing) <= .55 * icon_height


def _back_chevron(ink, width, middle_y, icon_height):
    rail = ink[:, round(.69 * width):round(.86 * width)]
    _, labels, components, _ = cv2.connectedComponentsWithStats(rail)
    for identity, (x, y, w, h, _) in enumerate(components[1:], start=1):
        if (not .70 * icon_height <= h <= 1.24 * icon_height or not .37 <= w / h <= .81
                or abs(y + h / 2 - middle_y) > .21 * icon_height):
            continue
        glyph = labels[y:y + h, x:x + w] == identity
        rows = np.nonzero(glyph.sum(axis=1))[0]
        if len(rows) < .91 * h:
            continue
        centers = np.array([np.flatnonzero(glyph[row]).mean() for row in rows])
        vertex = np.abs(2 * rows / (h - 1) - 1)
        if np.sqrt(np.mean((centers / (w - 1) - vertex) ** 2)) <= .24:
            return True
    return False


def _has_navigation_glyphs(ink):
    height, width = ink.shape
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    for contour, (_, _, child, parent) in zip(contours, hierarchy[0], strict=True):
        if child < 0 or parent >= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        middle_y = y + h / 2
        if (not .028 * width <= min(w, h) <= max(w, h) <= .060 * width or not .78 <= w / h <= 1.29
                or abs(x + w / 2 - width / 2) > .035 * width or not .30 * height <= middle_y <= .67 * height):
            continue
        area = cv2.contourArea(contour)
        if area < .70 * w * h or not .48 <= cv2.contourArea(contours[child]) / area <= .90:
            continue
        if _left_strokes(ink, width, middle_y, h) and _back_chevron(ink, width, middle_y, h):
            return True
    return False


def system_navigation_top(gray):
    """Find a light system bar using its full-width edge and all three glyphs."""
    height, width = gray.shape
    if not 1.35 * width <= height <= 2.90 * width or width < 160:
        return None
    start = height - round(.21 * width)
    stop = height - round(.080 * width)
    gap = max(2, round(.010 * width))
    band = gray[start - gap:stop + gap].astype(np.int16)
    changed = (band[:-2 * gap] - band[2 * gap:] >= 7).mean(axis=1)
    tone = int(np.median(gray[-gap:]))
    if not 220 <= tone <= 247:
        return None
    matches = []
    for candidate in np.flatnonzero((changed[1:] >= .88) & (changed[:-1] < .88)) + start + 1:
        roof = gray[candidate:candidate + 2 * gap]
        leads = np.flatnonzero((abs(roof.astype(np.int16) - tone) <= 4).mean(axis=1) >= .93)
        if not leads.size:
            continue
        top = int(candidate + leads[0])
        strip = gray[top:]
        if (abs(strip.astype(np.int16) - tone) <= 5).mean() >= .94 and _has_navigation_glyphs(np.uint8(strip <= tone - 26)):
            matches.append(top)
    return matches[0] if len(matches) == 1 else None
