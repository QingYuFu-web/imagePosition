"""Recognize a light X on a dark-outlined, colored header button."""

import cv2
import numpy as np


def _cross_arms(mask):
    height, width = mask.shape
    rows, columns = np.nonzero(mask)
    horizontal = (columns - columns.mean()) / width
    vertical = (rows - rows.mean()) / height
    if not mask[height // 2, width // 2]:
        return False
    for angle in np.deg2rad((-18, -12, -6, 0, 6, 12, 18)):
        x = np.cos(angle) * horizontal - np.sin(angle) * vertical
        y = np.sin(angle) * horizontal + np.cos(angle) * vertical
        cross = np.minimum(abs(x - y), abs(x + y))
        arms = [float(((s * x > .21) & (t * y > .21)).mean()) for s in (-1, 1) for t in (-1, 1)]
        axes = ((abs(x) >= .32) & (abs(y) <= .095)) | ((abs(y) >= .32) & (abs(x) <= .095))
        if min(arms) >= .038 and min(arms) >= .34 * max(arms) and (cross <= .196).mean() >= .93 and axes.mean() <= .033:
            return True
    return False


def _outlined(ink, gray, size, limit, *, minimum):
    reach = max(1, round(size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    rim = (cv2.dilate(ink, kernel) != 0) & (ink == 0)
    return bool(rim.any() and (gray[rim] <= limit).mean() >= minimum)


def _inside_cross(gray, filled, button_width, button_height, center):
    tone = float(np.median(gray[filled != 0]))
    mark = np.uint8((gray >= max(117, tone + 45)) & (filled != 0))
    _, ids, stats, _ = cv2.connectedComponentsWithStats(mark, connectivity=8)
    for i, (left, top, w, h, area) in enumerate(stats[1:], 1):
        if (not .38 * button_width <= w <= .68 * button_width or not .38 * button_height <= h <= .68 * button_height
                or not .76 <= w / h <= 1.31 or not .28 <= area / (w * h) <= .57
                or abs(left + w / 2 - center[0]) > .11 * button_width or abs(top + h / 2 - center[1]) > .11 * button_height):
            continue
        glyph = ids == i
        if (_cross_arms(glyph[top:top + h, left:left + w])
                and _outlined(np.uint8(glyph), gray, max(2, .050 * min(w, h)), max(51, tone - 24), minimum=.55)):
            return True
    return False


def _button_faces(frame, area, panel):
    x0, y0, x1, y1 = area
    left, top, right, bottom = panel
    span = right - left
    pixels = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    blue, green, red = cv2.split(pixels.astype(np.int16))
    accent = np.uint8((red - green >= 36) & (red - blue >= 30) & (red >= 91))
    # Rectification can leave a one-pixel break between the button face and its inner cutout.
    accent = cv2.morphologyEx(accent, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(accent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = (x + w / 2, y + h / 2)
        size = min(w, h)
        if (not .066 * span <= size <= max(w, h) <= .157 * span or not .77 <= w / h <= 1.29
                or not .035 * span <= right - x0 - cx <= .169 * span
                or not .024 * span <= y0 + cy - top <= .15 * span):
            continue
        surface = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        vertices = cv2.approxPolyDP(hull, .032 * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if len(vertices) != 4 or not .53 * w * h <= surface <= .94 * w * h or surface < .86 * cv2.contourArea(hull):
            continue
        segments = vertices - np.roll(vertices, -1, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if min(lengths) < .70 * max(lengths) or any(abs(a @ b) > .27 * l1 * l2 for a, b, l1, l2 in
                zip(segments, np.roll(segments, 1, axis=0), lengths, np.roll(lengths, 1), strict=True)):
            continue
        filled = np.zeros_like(accent)
        cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
        if not _outlined(filled, gray, .046 * size, 53, minimum=.76):
            continue
        if not _inside_cross(gray, filled, w, h, (cx, cy)):
            continue
        faces.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
    return faces


def _attached(frame, panel, face):
    left, top, right, bottom = panel
    a, b, c, d = face
    size = min(c - a, d - b)
    if not (left <= a < c <= right and top <= b < d <= bottom):
        return False
    gap, width = max(2, round(.105 * size)), round(.48 * size)
    if a - gap - width < left or d + gap + width > bottom or top - gap - width < 0:
        return False
    surrounding = (
        frame[b + gap:d - gap, a - gap - width:a - gap],
        frame[d + gap:d + gap + width, a + gap:c - gap],
        frame[top - gap - width:top - gap, a + gap:c - gap],
    )
    rim_left, rim_below, outside = (float(np.median(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))) for patch in surrounding)
    return min(rim_left, rim_below) >= 103 and min(rim_left, rim_below) - outside >= 47 and abs(rim_left - rim_below) <= 40


def rim_attached_dismiss(frame, panel):
    """Return one grounded close action for the observed panel's upper-right rim."""
    height, width = frame.shape[:2]
    left, top, right, bottom = panel
    span = right - left
    if not (0 <= left < right <= width and 0 < top < bottom <= height and span >= .30 * width and span >= 140
            and bottom - top >= .42 * span):
        return None
    area = (right - round(.23 * span), max(0, top - round(.04 * span)), min(width, right + round(.021 * span)),
            min(bottom, top + round(.229 * span)))
    faces = [box for box in _button_faces(frame, area, panel) if _attached(frame, panel, box)]
    if len(faces) != 1:
        return None
    a, b, c, d = faces[0]
    pad = max(2, round(.066 * min(c - a, d - b)))
    return {"role": "close", "function": "关闭", "preview_text": "Close",
            "box": [max(left, a - pad), max(top, b - pad), min(right, c + pad), min(bottom, d + pad)],
            "click_point": [(a + c) // 2, (b + d) // 2]}
