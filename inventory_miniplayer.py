"""Recognize an artwork-backed playback strip without using app or text names."""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

from inventory_labeled_bottom_navigation import _icon_for_label
from inventory_list_rows import _labels


def _center(box):
    left, top, right, bottom = box
    return ((left + right) / 2, (top + bottom) / 2)


def _stacked_text(nodes, width, height):
    lines = []
    for node in nodes:
        box = node.get("box")
        text = str(node.get("content") or "")
        if (node.get("kind") != "text" or not box or (node.get("score") or 0) < 86
                or sum(char.isalpha() for char in text) < 2):
            continue
        left, top, right, bottom = box
        if .67 * height <= top < bottom <= height and 0 < left < right <= width:
            lines.append(tuple(map(round, box)))
    for first, second in combinations(sorted(lines, key=lambda box: (box[1], box[0])), 2):
        h1, h2 = first[3] - first[1], second[3] - second[1]
        gap = second[1] - first[3]
        if (h1 < 6 or not .72 <= h2 / h1 <= 1.32
                or abs(first[0] - second[0]) > .30 * h1
                or not -.06 * h1 <= gap <= .62 * h1
                or min(first[2] - first[0], second[2] - second[0]) < 2.1 * h1):
            continue
        yield first, second


def _surface(lab, first, second, *, max_bottom_gap=.76, single_line=False):
    height, width = lab.shape[:2]
    local_y = round((first[1] + second[3] - 1) / 2)
    span = second[3] - first[1]
    sample = lab[:, round(.17 * width):round(.85 * width):3]
    extent = (5 if single_line else 3) * span
    lo, hi = max(0, local_y - extent), min(height, local_y + extent)
    profile = np.median(sample[lo:hi], axis=1)
    background = np.median(profile[first[1] - lo:second[3] - lo], axis=0)
    distance = np.linalg.norm(profile - background, axis=1)
    top, lower = local_y - lo, local_y - lo + 1
    if distance[top] > 5:
        return None
    while top > 0 and distance[top - 1] <= 5:
        top -= 1
    while lower < len(distance) and distance[lower] <= 5:
        lower += 1
    top, lower = top + lo, lower + lo
    thickness = lower - top
    margin = max(2, round(.09 * span))
    low, high = (2.6, 4.9) if single_line else (1.24, 2.65)
    if (not low * span <= thickness <= high * span or width < 3.6 * thickness
            or top + margin > first[1] or second[3] > lower - margin
            or height - lower > max_bottom_gap * thickness or top < round(.24 * thickness)):
        return None
    above_stop = top - max(1, round(.045 * thickness))
    above = sample[top - round(.24 * thickness):above_stop]
    boundary_contrast = 6 if single_line else 10.5
    if not above.size or np.mean(np.linalg.norm(above - background, axis=2) >= boundary_contrast) < .78:
        return None
    inside = sample[top + margin:lower - margin]
    if np.mean(np.linalg.norm(inside - background, axis=2) <= 8) < .83:
        return None
    return top, lower, background


def _artwork(lab, strip, leading_text):
    top, bottom, background = strip
    thickness = bottom - top
    sample = lab[top:bottom, :max(0, leading_text - max(1, round(.05 * thickness)))]
    if not sample.size:
        return None
    contrast = np.linalg.norm(sample - background, axis=2)
    has_art = (contrast >= 23).mean(axis=0) >= .59
    transitions = np.flatnonzero(np.diff(np.r_[False, has_art, False]))
    fits = [(int(left), top, int(right), bottom) for left, right in transitions.reshape(-1, 2)
            if left <= .48 * thickness and .73 * thickness <= right - left <= 1.28 * thickness
            and .04 * thickness <= leading_text - right <= .64 * thickness]
    return fits[0] if len(fits) == 1 else None


def _right_triangle(contour, width, height):
    area = cv2.contourArea(contour)
    if not (.63 <= width / height <= 1.06 and .32 <= area / (width * height) <= .61):
        return False
    vertices = cv2.approxPolyDP(contour, .067 * cv2.arcLength(contour, True), True).reshape(-1, 2)
    if len(vertices) != 3 or area < .86 * cv2.contourArea(cv2.convexHull(contour)):
        return False
    leading = vertices[vertices[:, 0].argsort()[:2]]
    tip = vertices[np.argmax(vertices[:, 0])]
    return (np.ptp(leading[:, 0]) <= .23 * width and np.ptp(leading[:, 1]) >= .76 * height
            and .28 * height <= tip[1] - leading[:, 1].min() <= .72 * height)


def _transport(lab, strip, trailing_text):
    top, bottom, background = strip
    thickness = bottom - top
    width = lab.shape[1]
    left = max(trailing_text + round(.35 * thickness), width - round(2.25 * thickness))
    margin = max(1, round(.10 * thickness))
    crop_top = top + margin
    patch = lab[crop_top:bottom - margin, left:width - margin]
    if not patch.size:
        return None
    mask = np.uint8(np.linalg.norm(patch - background, axis=2) >= 45)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    matches, bars = [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = (x + left, y + crop_top, x + left + w, y + crop_top + h)
        if not (.22 * thickness <= h <= .59 * thickness
                and abs(_center(box)[1] - (top + bottom) / 2) <= .16 * thickness):
            continue
        if _right_triangle(contour, w, h):
            matches.append(("play", box))
        if .16 <= w / h <= .49 and cv2.contourArea(contour) >= .59 * w * h:
            bars.append(box)
    for a, b in combinations(sorted(bars), 2):
        h1, h2 = a[3] - a[1], b[3] - b[1]
        w1, w2 = a[2] - a[0], b[2] - b[0]
        if (.83 <= h2 / h1 <= 1.20 and .7 <= w2 / w1 <= 1.40
                and abs(a[1] - b[1]) <= .13 * h1 and .14 * h1 <= b[0] - a[2] <= .52 * h1):
            matches.append(("pause", (a[0], min(a[1], b[1]), b[2], max(a[3], b[3]))))
    return matches[0] if len(matches) == 1 else None


def _matching_art(frame, art, strip_top):
    """Require a differently sized rendition to rule out generic avatars with a chevron."""
    left, top, right, bottom = art
    thumb = frame[top:bottom, left:right]
    search = frame[:strip_top - (bottom - top) // 3]
    if not search.size or cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY).std() < 12:
        return False
    shrink = min(1., 320 / frame.shape[1])
    if shrink < 1:
        search = cv2.resize(search, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    search_luma = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    for scale in np.geomspace(1.26, 3.7, 23):
        w, h = round((right - left) * shrink * scale), round((bottom - top) * shrink * scale)
        if w > search.shape[1] or h > search.shape[0] or min(w, h) < 14:
            continue
        template = cv2.resize(thumb, (w, h), interpolation=cv2.INTER_LINEAR)
        template_luma = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        correlation = cv2.matchTemplate(search_luma, template_luma, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(correlation)
        if score >= .78 and cv2.absdiff(search[y:y + h, x:x + w], template).mean() <= 14:
            return True
    return False


def _progress_row(row, background, *, flush_track):
    width = len(row)
    changes = np.flatnonzero(np.max(np.abs(np.diff(row, axis=0)), axis=1) >= 9) + 1
    runs = [(int(left), int(right)) for left, right in zip(np.r_[0, changes], np.r_[changes, width], strict=True)
            if right - left >= .042 * width]
    for (start, split), (following, end) in zip(runs, runs[1:]):
        if start > .14 * width or end < .86 * width or following - split > .016 * width:
            continue
        active = row[(start + split) // 2]
        unfilled = row[(following + end) // 2]
        if (np.linalg.norm(active - unfilled) >= 28 and np.linalg.norm(active - background) >= 33
                and (0 if flush_track else 6) <= np.linalg.norm(unfilled - background) <= 69):
            return start, split, end
    return None


def _visible_progress(lab, strip, *, flush_track=False):
    top, bottom, background = strip
    size = bottom - top
    padding = max(3, round(.058 * size))
    for edge in ((bottom,) if flush_track else (bottom, top)):
        found = []
        for y in range(max(0, edge - padding), min(len(lab), edge + padding + 1)):
            if (bounds := _progress_row(lab[y], background, flush_track=flush_track)) is not None:
                found.append((y, np.asarray(bounds)))
        if (found and found[-1][0] - found[0][0] <= max(2, round(.060 * size))
                and all(np.max(abs(bounds - found[0][1])) <= .018 * lab.shape[1] for _, bounds in found)):
            return True
    return False


def _footer_labels(nodes, width, height):
    labels = [box for box, text, score in _labels(nodes, width, height) if score >= 91 and any(c.isalpha() for c in text)
              and .011 * width <= box.height <= .048 * width and height - box.center[1] <= .085 * width]
    if not 3 <= len(labels) <= 6 or np.ptp([box.center[1] for box in labels]) > .012 * width:
        return []
    labels.sort(key=lambda box: box.center[0])
    pitch = width / len(labels)
    if any(abs(box.center[0] - (i + .5) * pitch) > .12 * pitch for i, box in enumerate(labels)):
        return []
    return labels


def _separate_footer(frame, nodes, strip, *, labels=None):
    height, width = frame.shape[:2]
    _, bottom, _ = strip
    labels = _footer_labels(nodes, width, height) if labels is None else labels
    if not labels:
        return False
    upper = min(box.top for box in labels) - round(.087 * width)
    if upper <= bottom or upper - bottom > .070 * width:
        return False
    gray = cv2.cvtColor(frame[upper:], cv2.COLOR_BGR2GRAY).astype(np.int16)
    contrast = abs(gray - np.median(gray, axis=1).astype(np.int16)[:, None])
    icons = [_icon_for_label(contrast, 0, box.top - upper - round(.087 * width),
                             (box.left, box.top - upper, box.right, box.bottom - upper)) for box in labels]
    return all(icon is not None for icon in icons) and np.ptp([box[1] + box[3] for box in icons]) <= .024 * width


def _verified_player_text(nodes, strip, first, second, transport):
    top, bottom, _ = strip
    size = bottom - top
    right = max(first[2], second[2])
    for node in nodes:
        box = node.get("box")
        if not box or node.get("kind") != "text" or (node.get("score") or 0) < 90:
            continue
        left, y0, end, y1 = box
        if (left >= first[0] - .16 * size and end <= transport[0] - .36 * size
                and top < y0 < y1 < bottom and any(abs((y0 + y1) / 2 - (b[1] + b[3]) / 2) <= .10 * size for b in (first, second))
                and sum(c.isalpha() for c in str(node.get("content") or "")) >= 2):
            right = max(right, round(end))
    return right


def _single_line_artwork(lab, strip, title_left):
    top, bottom, background = strip
    size = bottom - top
    crop = lab[top:bottom, :title_left - round(.055 * size)]
    if not crop.size:
        return None
    distance = np.linalg.norm(crop - background, axis=2)
    support = np.uint8(distance >= 9)
    contours, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    artwork = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (.52 * size <= h <= .94 * size and .78 <= w / h <= 1.22
                and x <= .4 * size and .04 * size <= title_left - x - w <= .39 * size
                and abs(y + h / 2 - size / 2) <= .13 * size
                and cv2.contourArea(contour) >= .70 * w * h
                and (distance[y:y + h, x:x + w] >= 24).mean() >= .13):
            artwork.append(x)
    return artwork[0] if len(artwork) == 1 else None


def _player_pair(body, role, glyph):
    return [
        {"role": "open_player", "function": "打开播放器", "box": body, "center": _center(body),
         "preview_text": "Player", "preview_above": True},
        {"role": role, "function": "播放" if role == "play" else "暂停", "box": glyph, "center": _center(glyph),
         "preview_text": "Play" if role == "play" else "Pause", "preview_above": True},
    ]


def _single_title_player(frame, lab, nodes):
    """Require artwork, progress, a transport and separate navigation around one title."""
    height, width = frame.shape[:2]
    footer = _footer_labels(nodes, width, height)
    if not footer:
        return []
    matches = []
    for box, text, score in _labels(nodes, width, height):
        if (score < 92 or sum(c.isalpha() for c in text) < 6
                or not .095 * width <= box.left <= .28 * width or box.width < 4.2 * box.height
                or not .77 * height <= box.center[1] <= .93 * height):
            continue
        if lab is None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        title = box.xyxy()
        strip = _surface(lab, title, title, max_bottom_gap=1.72, single_line=True)
        if strip is None:
            continue
        top, bottom, _ = strip
        if abs(box.center[1] - (top + bottom) / 2) > .11 * (bottom - top):
            continue
        transport = _transport(lab, strip, box.right)
        if transport is None or not _visible_progress(lab, strip, flush_track=True):
            continue
        start = _single_line_artwork(lab, strip, box.left)
        if start is None or not _separate_footer(frame, nodes, strip, labels=footer):
            continue
        matches.append(_player_pair((start, top, box.right, bottom), *transport))
    return matches[0] if len(matches) == 1 else []


def suggest_miniplayer(frame_bgr, nodes, *, overlay=None):
    """Return the player body and its independent visible play or pause target."""
    if overlay is not None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return []
    height, width = frame_bgr.shape[:2]
    if min(height, width) < 120:
        return []
    nodes = tuple(nodes)
    matches = []
    lab = None
    for first, second in _stacked_text(nodes, width, height):
        if lab is None:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        strip = _surface(lab, first, second, max_bottom_gap=1.72)
        if strip is None:
            continue
        top, bottom, _ = strip
        art = _artwork(lab, strip, min(first[0], second[0]))
        trailing_text = max(first[2], second[2])
        transport = _transport(lab, strip, trailing_text)
        if art is None or transport is None:
            continue
        role, glyph = transport
        near_bottom = height - bottom <= .76 * (bottom - top)
        if near_bottom and _matching_art(frame_bgr, art, top):
            split = round(_center(glyph)[0] - .52 * (bottom - top))
            if split <= trailing_text:
                continue
        elif (_visible_progress(lab, strip) and (near_bottom or _separate_footer(frame_bgr, nodes, strip))):
            split = _verified_player_text(nodes, strip, first, second, glyph)
        else:
            continue
        if split >= glyph[0]:
            continue
        body = (art[0], top, split, bottom)
        pair = _player_pair(body, role, glyph)
        if pair not in matches:
            matches.append(pair)
    if matches:
        return matches[0] if len(matches) == 1 else []
    return _single_title_player(frame_bgr, lab, nodes)
