"""Keep bounded visual evidence around an otherwise uncovered circular X."""

from math import ceil, floor

import cv2
import numpy as np

from screen_ai_gap_context import _mapped_box
from screen_dialog_close import _cross_footprint
from screen_page_handoff import _intersection
from screen_photo_geometry import PhotoCrops


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _paint_box(mask, matrix, box, value, screen):
    left, top, right, bottom = box
    outline = np.float32([[left, top], [right - 1, top], [right - 1, bottom - 1], [left, bottom - 1]])
    if screen is not None:
        area, outline = cv2.intersectConvexConvex(outline, screen)
        if area <= 0 or outline is None:
            return
    points = np.rint(cv2.perspectiveTransform(outline.reshape(1, -1, 2), matrix)[0]).astype(np.int32)
    cv2.fillConvexPoly(mask, points, value)


def _screen_view(frame, photo, panels, excluded, screen):
    width, height = photo.size if photo is not None else frame.shape[1::-1]
    scale = min(1., 320 / min(width, height), 900 / max(width, height))
    size = max(1, round(width * scale)), max(1, round(height * scale))
    sx, sy = size[0] / width, size[1] / height
    resize = np.diag([sx, sy, 1.])
    if photo is None:
        matrix = resize
        view = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    else:
        matrix = resize @ photo.to_screen
        view = cv2.warpPerspective(frame, matrix, size, flags=cv2.INTER_LINEAR)
    allowed = np.zeros(size[::-1], np.uint8)
    for panel in panels:
        _paint_box(allowed, matrix, panel, 255, screen)
    for box in excluded:
        _paint_box(allowed, matrix, box, 0, screen)
    if photo is not None and photo.blocked is not None:
        blocked = cv2.resize(photo.blocked, size, interpolation=cv2.INTER_NEAREST)
        allowed[blocked != 0] = 0
    return cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), allowed, (sx, sy)


def _visible_disks(gray, allowed):
    side = min(gray.shape)
    seen = set()
    # The brighter pass can separate a white disk from a light colored panel.
    for cutoff in (213, 238):
        lit = cv2.bitwise_and(cv2.inRange(gray, cutoff, 255), allowed)
        contours, _ = cv2.findContours(lit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        disks = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if (not .023 * side <= min(width, height) <= max(width, height) <= .095 * side
                    or not .85 <= width / height <= 1.18):
                continue
            area = cv2.contourArea(contour) / (width * height)
            if .66 <= area <= .85 and allowed[y:y + height, x:x + width].all():
                compactness = abs(area - .77) + abs(width - height) / max(width, height)
                disks.append((compactness, y, x, width, height))
        for _, *box in sorted(disks)[:16]:
            bounds = tuple(box)
            if bounds not in seen:
                seen.add(bounds)
                yield bounds


def _has_cross(frame, box, photo):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if min(width, height) < 12:
        return False
    scale = min(1., 112 / max(width, height))
    size = max(1, round(width * scale)), max(1, round(height * scale))
    if photo is None:
        chip = frame[top:bottom, left:right]
        if size != (width, height):
            chip = cv2.resize(chip, size, interpolation=cv2.INTER_AREA)
    else:
        sx, sy = size[0] / width, size[1] / height
        transform = np.float64([[sx, 0, -sx * left], [0, sy, -sy * top], [0, 0, 1]])
        chip = cv2.warpPerspective(frame, transform @ photo.to_screen, size, flags=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY).astype(np.int16)
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    radius = ((xx + .5 - width / 2) / (width / 2)) ** 2 + ((yy + .5 - height / 2) / (height / 2)) ** 2
    shell = gray[(radius > .33) & (radius < .76)]
    paper = int(np.median(shell))
    if paper < 184 or (abs(shell - paper) <= 14).mean() < .89:
        return False
    contrast = paper - gray
    inside = radius < .56
    peak = np.percentile(contrast[inside], 96)
    if peak < 48:
        return False
    marks = []
    for level in (.44, .56, .68):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8((contrast >= level * peak) & inside))
        if not 2 <= count <= 4:
            continue
        index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        x, y, w, h, area = stats[index]
        if (.25 * width <= w <= .65 * width and .25 * height <= h <= .65 * height and .77 <= w / h <= 1.29
                and abs(x + w / 2 - width / 2) <= .11 * width and abs(y + h / 2 - height / 2) <= .11 * height
                and .14 * w * h <= area <= .64 * w * h and _cross_footprint(labels[y:y + h, x:x + w] == index)):
            marks.append((x, y, w, h))
    return len(marks) >= 2 and np.ptp(marks, axis=0).max() <= max(2, .12 * max(width, height))


def _blocked(box, photo):
    if photo is None or photo.blocked is None:
        return False
    if (window := photo.crop_window(box)) is None:
        return True
    left, top, right, bottom = window
    width, height = photo.size
    mask = photo.blocked[max(0, top):min(bottom, height), max(0, left):min(right, width)]
    return not mask.size or bool(mask.any())


def close_context_boxes(page, frame, panels, covered, photo, *, budget, limit):
    """Return source-photo crops with a visible glyph, without assigning a click."""
    layer = page["layer"]
    if (budget < 144 or limit < 1 or not panels or layer.get("floating_boxes")
            or layer["kind"] in {"layer_uncertain", "voice_recording", "voice_stop_panel"}):
        return []
    source_height, source_width = frame.shape[:2]
    panels = [[max(0, ceil(left)), max(0, ceil(top)), min(source_width, floor(right)), min(source_height, floor(bottom))]
              for left, top, right, bottom in panels]
    panels = [panel for panel in panels if min(panel[2] - panel[0], panel[3] - panel[1]) >= 12]
    if not panels:
        return []
    if photo is None and page.get("screen_corners") is not None:
        photo = PhotoCrops(page)
    width, height = photo.size if photo is not None else frame.shape[1::-1]
    excluded = [row["box"] for row in page["elements"] if row.get("disabled")]
    excluded.extend(sample["box"] for row in page["elements"]
                    if (sample := row.get("reviewed_sample")) is not None and sample["purpose"] is None)
    screen = np.float32(page["screen_corners"]) if "screen_corners" in page else None
    gray, allowed, (sx, sy) = _screen_view(frame, photo, panels, excluded, screen)
    result = []
    for y, x, w, h in _visible_disks(gray, allowed):
        left, top = max(0, floor((x - .5) / sx)), max(0, floor((y - .5) / sy))
        right, bottom = min(width, ceil((x + w + .5) / sx)), min(height, ceil((y + h + .5) / sy))
        disk = [left, top, right, bottom]
        # The proof reads a resampling margin around the disk. Use the measured
        # footprint itself when checking that the evidence belongs to a panel.
        body = [ceil(x / sx), ceil(y / sy), floor((x + w) / sx), floor((y + h) / sy)]
        original = body if photo is None else _mapped_box(photo.to_photo, body)
        panel = next((pane for pane in panels if _intersection(original, pane) == original), None)
        if (panel is None or any(_intersection(original, box) == original for box in (*covered, *result))
                or any(_intersection(original, box) for box in excluded) or _blocked(original, photo)
                or not _has_cross(frame, disk, photo)):
            continue
        span = max(right - left, bottom - top)
        for ratio in (.84, .55, .27):
            pad = round(ratio * span)
            window = [max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad)]
            if photo is not None:
                window = _mapped_box(photo.to_photo, window)
            box = _intersection(window, panel)
            if (box is None or _intersection(box, original) != original or _area(box) > budget or _blocked(box, photo)
                    or any(_intersection(box, peer) for peer in result)):
                continue
            result.append(box)
            budget -= _area(box)
            break
        if len(result) == limit:
            break
    return result
