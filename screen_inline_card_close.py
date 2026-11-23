"""Locate a circular dismissal within a visually demonstrated card corner."""

import cv2
import numpy as np

from screen_owned_corner_close import _UNIT, _circular_cross


def _candidate_pairs(regions, width, height):
    unit = min(width, height)
    cards, buttons = [], []
    for row in regions["candidates"]:
        left, top, right, bottom = row["box"]
        if row.get("visibility") != "page" or not (0 <= left < right <= width and 0 <= top < bottom <= height):
            continue
        w, h = right - left, bottom - top
        if .37 * width <= w <= .98 * width and .16 * height <= h <= .70 * height:
            cards.append(row)
        if .034 * unit <= min(w, h) <= max(w, h) <= .15 * unit and .62 <= w / h <= 1.48:
            buttons.append(row)
    for button in buttons:
        x, y = button["center"]
        a, b, c, d = button["box"]
        size = min(c - a, d - b)
        for card in cards:
            left, top, right, bottom = card["box"]
            if right - left < 5 * size or bottom - top < 3 * size or not -.90 * size <= y - top <= 1.55 * size:
                continue
            for edge, side in ((left, -1), (right, 1)):
                if .30 * size <= side * (edge - x) <= 1.50 * size:
                    yield button, edge, side, size


def _edge_candidates(strips, start, size):
    """Find long, color-consistent boundaries normal to the first axis."""
    radius = max(2, round(.043 * size))
    if not strips.size or min(strips.shape[:2]) <= 2 * radius + 2:
        return
    before = strips[:-2 * radius].astype(np.float32)
    after = strips[2 * radius:].astype(np.float32)
    difference = after - before
    step = np.median(difference, axis=1)
    contrast = np.linalg.norm(step, axis=1)
    direction = step / np.maximum(contrast[:, None], 1)
    support = (np.einsum("ijk,ik->ij", difference, direction) > 39).mean(axis=1)
    plausible = np.flatnonzero((contrast >= 52) & (support >= .73))
    positions = []
    for i in sorted(plausible, key=lambda j: (-support[j], -contrast[j], j)):
        position = i + radius + start
        if any(abs(position - other) <= 2 * radius for other in positions):
            continue
        yield position, step[i]
        positions.append(position)
        if len(positions) >= 3:
            return


def _same_direction(first, second):
    return np.dot(first, second) >= .68 * np.linalg.norm(first) * np.linalg.norm(second)


def _shared_surface(frame, top, edge, side, size):
    outside_x, inside_x = round(edge + side * .20 * size), round(edge - side * .56 * size)
    outside_y, inside_y = round(top - .19 * size), round(top + 1.17 * size)
    radius = max(1, round(.065 * size))
    height, width = frame.shape[:2]
    if min(outside_x, inside_x, width - 1 - max(outside_x, inside_x),
           outside_y, height - 1 - inside_y) < radius:
        return False
    side_paper, top_paper, fill = (
        np.median(frame[y - radius:y + radius + 1, x - radius:x + radius + 1], axis=(0, 1))
        for x, y in ((outside_x, inside_y), (inside_x, outside_y), (inside_x, inside_y))
    )
    top_gap, side_gap = fill - top_paper, fill - side_paper
    return (np.linalg.norm(side_paper - top_paper) <= 63
            and min(np.linalg.norm(top_gap), np.linalg.norm(side_gap)) >= 46
            and _same_direction(top_gap, side_gap))


def _owns_corner(frame, proposal, edge, side, size, disk, point):
    height, width = frame.shape[:2]
    x, y = proposal["center"]
    px, py = point
    disk_radius = (disk[2] - disk[0]) / 2
    left, right = sorted((round(edge - side * 3.55 * size), round(edge - side * 1.25 * size)))
    start, stop = max(0, round(y - 1.37 * size)), min(height, round(y - .17 * size))
    top_strip = frame[start:stop, max(0, left):min(width, right)]
    for top, down in _edge_candidates(top_strip, start, size):
        if not max(disk_radius, .14 * size) <= py - top <= 1.30 * size:
            continue
        left, right = max(0, round(edge - .49 * size)), min(width, round(edge + .49 * size))
        start, stop = max(0, round(top + .93 * size)), min(height, round(top + 2.66 * size))
        side_strip = frame[start:stop, left:right].transpose(1, 0, 2)
        for rim, step in _edge_candidates(side_strip, left, size):
            if (.32 * size <= side * (rim - x) <= 1.48 * size and
                    max(disk_radius, .18 * size) <= side * (rim - px) <= 1.44 * size and
                    _same_direction(down, -side * step) and _shared_surface(frame, top, rim, side, size)):
                return True
    return False


def _fitted_disks(tile, gray):
    height, width = gray.shape
    size = min(width, height)
    channels = tile.astype(np.int16)
    green = channels[..., 1]
    planes = (channels[..., 2] - green, channels[..., 0] - green, gray)
    seen = set()
    for plane in planes:
        dark, paper, light = np.percentile(plane, (20, 50, 80))
        for prominence in (dark - paper, light - paper):
            if abs(prominence) < 22:
                continue
            selected = np.sign(prominence) * (plane - paper) > .5 * abs(prominence)
            _, _, stats, centers = cv2.connectedComponentsWithStats(np.uint8(selected))
            for (left, top, w, h, area), (x, y) in zip(stats[1:], centers[1:]):
                if not (1 < left < left + w < width - 1 and 1 < top < top + h < height - 1
                        and .32 * size <= min(w, h) <= max(w, h) <= .83 * size and .80 <= w / h <= 1.25
                        and .55 <= area / (w * h) <= .90 and abs(x - width / 2) <= .24 * size
                        and abs(y - height / 2) <= .24 * size):
                    continue
                side = round((w + h) / 2)
                a, b = round(x - side / 2), round(y - side / 2)
                disk = a, b, a + side, b + side
                if disk not in seen:
                    seen.add(disk)
                    yield disk


def _adjusted_cross(tile, gray, disk):
    if (point := _circular_cross(tile, gray, disk)) is not None:
        return point
    left, top, right, bottom = disk
    size = right - left
    x, y = (left + right) / 2, (top + bottom) / 2
    height, width = gray.shape
    if min(x, width - 1 - x, y, height - 1 - y) <= .62 * size:
        return None
    cols, rows = np.rint(np.array([[x], [y]]) + .36 * size * _UNIT).astype(int)
    fill = np.uint8(np.median(tile[rows, cols], axis=0))[None, None]
    paper = int(cv2.cvtColor(fill, cv2.COLOR_BGR2GRAY)[0, 0])
    xx, yy = np.meshgrid(np.arange(left, right), np.arange(top, bottom))
    center = (xx - x) ** 2 + (yy - y) ** 2 <= (.34 * size) ** 2
    patch = gray[top:bottom, left:right] - paper
    contrast = float(np.percentile(np.abs(patch[center]), 95))
    if not 40 <= contrast < 140:
        return None
    gain = min(2.6, 140 / contrast)
    stretch = gray.copy()
    stretch[top:bottom, left:right] = np.rint(np.clip(paper + gain * patch, 0, 255))
    return _circular_cross(tile, stretch, disk)


def _button_marks(frame, button):
    left, top, right, bottom = button["box"]
    tile = frame[top:bottom, left:right]
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).astype(np.int16)
    matches = []
    for disk in _fitted_disks(tile, gray):
        if (point := _adjusted_cross(tile, gray, disk)) is not None:
            a, b, c, d = disk
            position = point[0] + left, point[1] + top
            if not any(max(abs(p - q) for p, q in zip(position, prior)) <= 3 for _, prior in matches):
                matches.append(((left + a, top + b, left + c, top + d), position))
    return matches


def _competing_light_surface(frame, point):
    """Abstain if the page close is outside a separate, enclosed bright surface."""
    height, width = frame.shape[:2]
    span = min(256, width)
    length = round(span * height / width)
    small = cv2.resize(frame, (span, length), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    x, y = point[0] * span / width, point[1] * length / height
    bright = np.uint8(gray >= 213)
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    outlines, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    margin = max(2, round(.018 * span))
    for outline in outlines:
        left, top, w, h = cv2.boundingRect(outline)
        right, bottom = left + w, top + h
        if (not .53 * span <= w <= .88 * span or not .13 * length <= h <= .66 * length
                or abs(left + right - span) > .10 * span or min(left, top, span - right, length - bottom) < 2 * margin
                or (left <= x < right and top <= y < bottom)
                or cv2.contourArea(outline) < .91 * w * h or bright[top:bottom, left:right].mean() < .83):
            continue
        rim = (gray[top - 2 * margin:top - margin, left:right], gray[bottom + margin:bottom + 2 * margin, left:right],
               gray[top:bottom, left - 2 * margin:left - margin], gray[top:bottom, right + margin:right + 2 * margin])
        if any((strip <= 195).mean() < .89 for strip in rim):
            continue
        outside = (gray[round(.085 * length):top, left:right], gray[bottom:round(.965 * length), left:right],
                   gray[:, :left], gray[:, right:])
        if all(strip.size and (strip <= 205).mean() >= .91 for strip in outside):
            return True
    return False


def inline_card_close_actions(frame, regions):
    """Return one visible page-layer dismiss control with a verified card rim."""
    height, width = frame.shape[:2]
    if (regions.get("layer") != "page" or regions.get("regions") or regions.get("detected_close")
            or width < 300 or not 1.46 * width <= height <= 2.90 * width):
        return []
    witnesses, matches = {}, []
    for button, edge, side, size in _candidate_pairs(regions, width, height):
        identity = tuple(button["box"])
        if identity not in witnesses:
            witnesses[identity] = _button_marks(frame, button)
        for box, (x, y) in witnesses[identity]:
            if any(a <= x < c and b <= y < d for a, b, c, d in regions.get("floating_controls", ())):
                continue
            if any(max(abs(x - point[0]), abs(y - point[1])) <= .14 * size for _, point in matches):
                continue
            if _owns_corner(frame, button, edge, side, size, box, (x, y)):
                matches.append((box, (x, y)))
    if len(matches) != 1:
        return []
    box, point = matches[0]
    if _competing_light_surface(frame, point):
        return []
    return [{"role": "close", "function": "关闭", "box": list(box), "click_point": list(point)}]
