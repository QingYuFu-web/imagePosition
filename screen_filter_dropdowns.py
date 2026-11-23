"""Preserve outlined dropdown geometry in a verified filter panel."""

import cv2
import numpy as np

from screen_game_primary import _button_box


def _down_arrow(gray, height):
    tone, peak = np.percentile(gray, (50, 99))
    if peak - tone < 43:
        return False
    matches = 0
    for level in (.60, .71, .81):
        ink = np.uint8(gray >= tone + level * (peak - tone))
        x, y, w, h = cv2.boundingRect(ink)
        if (not .32 * height <= w <= .76 * height or not .15 * height <= h <= .42 * height
                or not 1.48 * h <= w <= 2.65 * h):
            continue
        glyph = ink[y:y + h, x:x + w]
        counts = glyph.sum(axis=0)
        columns = np.flatnonzero(counts)
        if not .075 <= glyph.mean() <= .33 or len(columns) < .87 * w:
            continue
        centers = (np.arange(h) @ glyph[:, columns]) / counts[columns]
        model = 1 - abs(2 * columns / (w - 1) - 1)
        end = max(1, w // 8)
        arms = centers[:end].mean(), centers[-end:].mean()
        slope = centers[model.argmax()] - sum(arms) / 2
        residual = centers - (centers.mean() + slope * (model - model.mean()))
        if .47 * h <= slope <= .96 * h and abs(arms[0] - arms[1]) <= .21 * h and abs(residual).mean() <= .11 * h:
            matches += 1
    return matches >= 2


def _visible_value(gray, height):
    tone, peak = np.percentile(gray, (50, 99))
    if tone > 74 or peak < 136 or peak - tone < 85:
        return False
    ink = np.uint8(gray >= tone + .59 * (peak - tone))
    x, y, w, h = cv2.boundingRect(ink)
    return (.87 * height <= w <= .77 * gray.shape[1] and .29 * height <= h <= .70 * height
            and abs(y + h / 2 - gray.shape[0] / 2) <= .19 * height
            and abs(x + w / 2 - gray.shape[1] / 2) <= .36 * height and .035 <= ink.mean() <= .36)


def _heading_context(gray, box, panel):
    left, top, right, bottom = box
    height = bottom - top
    start = max(panel[0], left - round(2.72 * height))
    stop = left - round(.15 * height)
    if stop - start < .52 * height:
        return None
    tile = gray[top:bottom, start:stop]
    tone, peak = np.percentile(tile, (50, 97))
    if peak - tone < 45:
        return None
    ink = np.uint8(tile >= tone + .59 * (peak - tone))
    x, y, w, h = cv2.boundingRect(ink)
    if (.56 * height <= w <= 2.30 * height and .29 * height <= h <= .71 * height
            and abs(y + h / 2 - height / 2) <= .17 * height and .025 <= ink.mean() <= .29
            and min(x, tile.shape[1] - x - w) >= 1):
        pad = max(2, round(.13 * height))
        return [max(panel[0], start + x - pad), top, right, bottom]
    return None


def filter_dropdown_candidates(frame, panel, candidates):
    """Return the clickable outline and optional contextual heading for each verified dropdown."""
    x0, y0, x1, y1 = panel
    width, height = x1 - x0, y1 - y0
    gray = None
    results = {}
    for candidate in candidates:
        left, top, right, bottom = candidate["box"]
        w, h = right - left, bottom - top
        if not (.045 * height <= h <= .13 * height and .13 * width <= w <= .45 * width
                and 4.1 * h <= w <= 7.5 * h):
            continue
        pad = max(3, round(.18 * h))
        if min(left - x0, top - y0, x1 - right, y1 - bottom) < pad:
            continue
        if (box := _button_box(frame, candidate["box"])) is None or tuple(box) in results:
            continue
        left, top, right, bottom = box
        h = bottom - top
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tile = gray[top + round(.18 * h):bottom - round(.18 * h),
                    right - round(.90 * h):right - round(.11 * h)]
        value = gray[top + round(.13 * h):bottom - round(.13 * h),
                     left + round(.45 * h):right - round(1.05 * h)]
        if not _down_arrow(tile, h) or not _visible_value(value, h):
            continue
        row = {"box": box, "click_point": [(left + right) // 2, (top + bottom) // 2]}
        if (context := _heading_context(gray, box, panel)) is not None:
            row["context_box"] = context
        results[tuple(box)] = row
    return list(results.values())
