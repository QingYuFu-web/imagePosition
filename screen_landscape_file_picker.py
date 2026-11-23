"""Verify file-picker navigation, storage, and filtering from local geometry."""

import cv2
import numpy as np

from screen_candidate_glyphs import _overlap
from screen_component_captions import _bands
from screen_landscape_app_bar import _status_context


def _folder(frame, gray, depth, left, right):
    width = right - left
    paint = frame[round(.26 * depth), round((left + right) / 2)].astype(np.int16)
    y0, y1 = depth + round(.12 * depth), depth + round(.82 * depth)
    x0, x1 = left + round(.009 * width), left + round(.096 * width)
    local = frame[y0:y1, x0:x1].astype(np.int16)
    if min(local.shape[:2]) < 10:
        return None
    tone = np.median(local, axis=(0, 1))
    if abs(tone - paint).max() < 32:
        return None
    mask = np.uint8(abs(local - paint).max(axis=2) <= 14)
    shapes, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for shape in shapes:
        x, y, w, h = cv2.boundingRect(shape)
        if (not .29 * depth <= h <= .54 * depth or not 1.14 <= w / h <= 1.65
                or cv2.contourArea(shape) < .80 * w * h or not 0 < y < y + h < len(mask)):
            continue
        outline = mask[y:y + h, x:x + w]
        tab = outline[round(.075 * h):round(.13 * h)].mean()
        body = outline[round(.27 * h):round(.87 * h)].mean()
        if not .28 <= tab <= .64 or body < .94:
            continue
        upper = outline[round(.11 * h)] != 0
        if np.median(np.flatnonzero(upper)) > .44 * w:
            continue
        a, t, b, q = x0 + x, y0 + y, x0 + x + w, y0 + y + h
        label = gray[t + round(.45 * h):q, b + round(.29 * h):b + round(1.32 * h)].astype(np.int16)
        ink = np.uint8(label - np.median(label) >= 62)
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        dots = [(dx, dy, dw, dh) for dx, dy, dw, dh, area in stats[1:]
                if .051 * h <= min(dw, dh) <= max(dw, dh) <= .126 * h and area >= .59 * dw * dh]
        dots.sort()
        if (len(dots) == 2 and .12 * h <= dots[1][0] - dots[0][0] <= .32 * h
                and abs(dots[0][1] - dots[1][1]) <= .041 * h):
            found.append((a, t, b, q))
    return found[0] if len(found) == 1 else None


def _file_rows(gray, left, right, depth, lower):
    width = right - left
    y0 = lower + round(.19 * depth)
    clip = gray[y0:min(len(gray), lower + round(1.87 * depth)), left + round(.005 * width):right - round(.11 * width)]
    if not clip.size:
        return False
    paper = np.median(clip[::2, ::2])
    ink = abs(clip.astype(np.int16) - paper) >= 56
    if ink.mean() > .13:
        return False
    candidates = []
    for a, b in _bands(ink.sum(axis=1) >= .017 * width):
        x, y, w, h = cv2.boundingRect(np.uint8(ink[a:b]))
        if .111 * depth <= h <= .37 * depth and x <= .035 * width and w >= .19 * width:
            candidates.append((a + y, b))
    if len(candidates) < 2:
        return False
    pitch = np.diff([top for top, _ in candidates])
    return .45 * depth <= pitch.min() <= pitch.max() <= 1.08 * depth and np.ptp(pitch) <= .18 * depth


def _funnel(ink, bounds, shape, children):
    x, y, w, h = bounds
    if children or not .78 <= w / h <= 1.14 or not .33 <= cv2.contourArea(shape) / (w * h) <= .60:
        return False
    widths = ink[y:y + h, x:x + w].mean(axis=1)
    rim = widths[:max(2, round(.17 * h))]
    throat = widths[round(.30 * h):round(.55 * h)]
    stem = ink[y + round(.65 * h):y + h, x:x + w]
    if rim.mean() < .86 or not .32 <= throat.mean() <= .79 or not .16 <= stem.mean() <= .39:
        return False
    _, columns = stem.nonzero()
    return np.ptp(stem.mean(axis=1)) <= .102 and .40 * w <= np.median(columns) <= .58 * w


def _storage(ink, bounds, shape, children):
    x, y, w, h = bounds
    if not .64 <= w / h <= .94 or not .78 <= cv2.contourArea(shape) / (w * h) <= .94 or len(children) != 3:
        return False
    contacts = sorted(cv2.boundingRect(cut) for cut in children)
    first, middle, last = contacts
    if (not all(.074 * w <= cw <= .29 * w and .14 * h <= ch <= .33 * h
                and y + .03 * h <= cy <= y + .27 * h for _, cy, cw, ch in contacts)
            or abs((last[0] - middle[0]) - (middle[0] - first[0])) > .112 * w
            or np.ptp([cy for _, cy, _, _ in contacts]) > .074 * h):
        return False
    glyph = ink[y:y + h, x:x + w]
    high = glyph[:max(2, round(.073 * h))]
    return (high[:, :round(.26 * w)].mean() <= .44 and high[:, round(.45 * w):round(.74 * w)].mean() >= .66
            and glyph[round(.39 * h):round(.86 * h)].mean() >= .93)


def _header_marks(delta, right):
    depth = len(delta)
    x0 = max(0, right - round(1.90 * depth))
    y0 = round(.411 * depth)
    strip = delta[y0:round(.916 * depth), x0:right - 2]
    patterns = {"filter": _funnel, "storage": _storage}
    found = {key: [] for key in patterns}
    for level in (53, 77, 100):
        ink = np.uint8(strip >= level)
        shapes, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        parents = hierarchy[0, :, 3]
        for i, shape in enumerate(shapes):
            x, y, w, h = bounds = cv2.boundingRect(shape)
            if (parents[i] >= 0 or not .17 * depth <= min(w, h) <= max(w, h) <= .32 * depth
                    or not 0 < x < x + w < strip.shape[1] or not 0 < y < y + h < len(strip)
                    or not .54 * depth <= y0 + y + h / 2 <= .82 * depth):
                continue
            holes = [hole for hole, parent in zip(shapes, parents, strict=True) if parent == i]
            for family, verify in patterns.items():
                if verify(ink, bounds, shape, holes):
                    found[family].append((level, (x0 + x, y0 + y, x0 + x + w, y0 + y + h)))
    result = {}
    for key, marks in found.items():
        solid = [box for level, box in marks if any(lev != level and _overlap(box, peer) >= .78 for lev, peer in marks)]
        if solid and all(_overlap(solid[0], box) >= .70 for box in solid):
            result[key] = solid[len(solid) // 2]
    return result


def file_picker_actions(frame, gray, tone, left, right):
    """Ground header commands in a file list with an actual parent-folder row."""
    depth = len(gray)
    full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    folder = _folder(frame, full, depth, left, right)
    if folder is None or not _file_rows(full, left, right, depth, folder[3]):
        return []
    marks = []
    residual = gray.astype(np.int16) - tone
    for sign in (1, -1):
        delta = sign * residual
        icons = _header_marks(delta, right)
        if icons and _status_context(delta, left, right):
            marks.append(icons)
    if len(marks) != 1:
        return []
    a, t, b, q = folder
    box = [left, depth, right, q + round(.47 * (q - t))]
    actions = [{"role": "back", "function": "返回上一级", "preview_text": "返回上一级", "box": box,
                "click_point": [(a + b) / 2, (t + q) / 2]}]
    for key, role, text in (("filter", "filter", "筛选文件"), ("storage", "open_menu", "选择存储位置")):
        if (bounds := marks[0].get(key)) is not None:
            x, y, r, b = bounds
            actions.append({"role": role, "function": text, "preview_text": text, "box": list(bounds),
                            "click_point": [(x + r) / 2, (y + b) / 2]})
    return actions
