"""Isolate a voice transcript surface from the dimmed app and the visible IME."""

from statistics import median

import cv2
import numpy as np

from screen_component_captions import _bands


def _key_rows(gray):
    height, width = gray.shape
    start = round(.53 * height)
    white = np.uint8(gray[start:] >= 230)
    closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, _, components, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keys = [(int(x), start + int(y), int(w), int(h)) for x, y, w, h, area in components[1:]
            if .045 * width <= w <= .136 * width and .061 * width <= h <= .144 * width and area >= .75 * w * h]
    rows = []
    for y in sorted({item[1] for item in keys}):
        aligned = sorted((x, top, w, h) for x, top, w, h in keys if abs(top - y) <= .009 * width)
        if not 7 <= len(aligned) <= 12:
            continue
        centers = [x + w / 2 for x, _, w, _ in aligned]
        pitches = np.diff(centers)
        sizes = [h for _, _, _, h in aligned]
        pitch = median(pitches)
        if (not .066 * width <= pitch <= .153 * width or np.max(abs(pitches - pitch)) > .12 * pitch
                or max(sizes) - min(sizes) > .13 * median(sizes) or centers[-1] - centers[0] < .55 * width):
            continue
        if all(abs(y - prior[0]) > .020 * width for prior in rows):
            rows.append((y, pitch, median(sizes)))
    for first, second, third in zip(rows, rows[1:], rows[2:]):
        span, pitch, key_height = first
        depths = (second[0] - span, third[0] - second[0])
        if (max(abs(second[1] - pitch), abs(third[1] - pitch)) <= .16 * pitch
                and .94 * key_height <= min(depths) <= max(depths) <= 1.57 * key_height
                and max(depths) - min(depths) <= .17 * key_height):
            return span
    return None


def _keyboard_edge(gray, keys):
    width = gray.shape[1]
    pad = max(2, round(.012 * width))
    interior = gray[:, round(.07 * width):round(.93 * width)].astype(np.int16)
    start, stop = keys - round(.22 * width), keys - round(.063 * width)
    for y in range(start, stop):
        before, after = np.median(interior[y - pad:y], axis=0), np.median(interior[y:y + pad], axis=0)
        if (before >= 220).mean() >= .92 and ((before - after >= 18) & (after >= 138)).mean() >= .88:
            return y
    return None


def _waveform(working, first, last):
    width = working.shape[1]
    section = working[first:last]
    if not section.size:
        return False
    blue, green, red = cv2.split(section)
    chroma = cv2.subtract(cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red))
    bits = np.uint8(chroma >= 52)
    _, _, stats, centers = cv2.connectedComponentsWithStats(bits, connectivity=8)
    strokes = sorted((float(x), float(y), int(h), x0, y0, w) for (x0, y0, w, h, area), (x, y) in
                     zip(stats[1:], centers[1:], strict=True) if .003 * width <= w <= .022 * width
                     and max(3, .82 * w) <= h <= .15 * width and area >= .54 * w * h)
    for _, baseline, _, _, _, _ in strokes:
        row = [item for item in strokes if abs(item[1] - baseline) <= .014 * width]
        if not 16 <= len(row) <= 73 or not .30 * width <= row[-1][0] - row[0][0] <= .82 * width:
            continue
        steps = np.diff([item[0] for item in row])
        pitch = median(steps)
        heights = [item[2] for item in row]
        if (.006 * width <= pitch <= .030 * width and (np.abs(steps - pitch) <= .27 * pitch).mean() >= .79
                and max(heights) >= 1.85 * median(heights) and min(heights) <= .79 * median(heights)):
            colors = [np.median(section[y:y + h, x:x + w][bits[y:y + h, x:x + w] != 0], axis=0)
                      for _, _, h, x, y, w in row]
            paint = np.median(colors, axis=0)
            if (np.max(abs(np.asarray(colors) - paint), axis=1) <= 34).mean() >= .85:
                return True
    return False


def voice_ime_bounds(working, gray):
    """Return bounds in the working image for a proven transcript over a scrim."""
    height, width = gray.shape
    if width < 180 or not 1.62 * width <= height <= 2.74 * width:
        return None
    if (first_key := _key_rows(gray)) is None or (keyboard := _keyboard_edge(gray, first_key)) is None:
        return None
    margin, pad = round(.075 * width), max(3, round(.017 * width))
    profile = np.median(gray[:, margin:width - margin], axis=1)
    jump = np.flatnonzero(profile[2:] - profile[:-2] >= 43) + 1
    matches = []
    for y in jump:
        y = int(y)
        if not .17 * height <= y <= min(.47 * height, keyboard - .57 * width):
            continue
        before = np.median(gray[y - 2 * pad:y - pad], axis=0)
        after = np.median(gray[y + pad:y + 2 * pad], axis=0)
        marks = _bands((after >= 215) & (before <= 203) & (after - before >= 43))
        edges = [(x, end) for x, end in marks if x <= .051 * width and end >= .949 * width]
        if len(edges) != 1:
            continue
        left, right = edges[0]
        upper = gray[round(.079 * height):y - pad, margin:width - margin]
        overlay = gray[y + pad:keyboard - pad, margin:width - margin]
        if (upper >= 209).mean() > .012 or (overlay >= 219).mean() < .84:
            continue
        if _waveform(working, y + round(.19 * width), keyboard - round(.10 * width)):
            edge = next((row for row in range(y - pad, y + pad) if profile[row] >= 219), y)
            match = (left, edge, right, height)
            if match not in matches:
                matches.append(match)
    return matches[0] if len(matches) == 1 else None


def ime_search_key_actions(frame, regions, index):
    """Read the painted IME command and leave transcript text as passive content."""
    if regions["layer"] != "ime_sheet":
        return []
    from screen_caption_transcript import read_caption

    height, width = frame.shape[:2]
    results = []
    for row in regions["candidates"]:
        l, t, r, b = row["box"]
        w, h = r - l, b - t
        x, y = row["center"]
        if (row["visibility"] != "foreground" or x < .80 * width or not .84 * height <= y <= .965 * height
                or not .14 * width <= w <= .29 * width or not .078 * width <= h <= .148 * width):
            continue
        tile = frame[t:b, l:r]
        center = tile[round(.21 * h):round(.79 * h), round(.18 * w):round(.82 * w)].astype(np.int16)
        paint = np.median(center, axis=(0, 1))
        if np.ptp(paint) < 65 or (np.max(abs(center - paint), axis=2) <= 17).mean() < .75:
            continue
        silhouette = np.uint8(np.max(abs(tile.astype(np.int16) - paint), axis=2) <= 19)
        contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        surfaces = []
        for contour in contours:
            a, z, sw, sh = cv2.boundingRect(contour)
            if sw >= .83 * w and sh >= .78 * h and cv2.contourArea(contour) >= .89 * sw * sh:
                surfaces.append((l + a, t + z, l + a + sw, t + z + sh))
        if len(surfaces) != 1:
            continue
        area = (l + round(.18 * w), t + round(.21 * h), r - round(.18 * w), b - round(.21 * h))
        reading = read_caption(frame, area, index, cache={}, choice=True)
        if reading is not None and "".join(reading["text"].split()).casefold() in {"搜索", "搜尋", "search"}:
            l, t, r, b = surfaces[0]
            results.append({"role": "keyboard_search", "function": "键盘搜索", "box": [l, t, r, b],
                            "click_point": [(l + r) / 2, (t + b) / 2], "preview_text": "Search key"})
    return results if len(results) == 1 else []
