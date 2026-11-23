"""Verify an isolated dark close X on an already established dialog."""

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_foreground import BlockingOverlay, _centered_cross


def _cross_footprint(mask):
    height, width = mask.shape
    yy, xx = np.ogrid[:height, :width]
    x = (xx + .5) / width - .5
    y = (yy + .5) / height - .5
    gap = np.minimum(abs(x - y), abs(x + y))
    if (gap[mask] > .16).mean() > .045:
        return False
    outer_arms = [np.count_nonzero(mask & (x * side > .24) & (y * edge > .24))
                  for side in (-1, 1) for edge in (-1, 1)]
    return min(outer_arms) >= .067 * mask.sum() and min(outer_arms) >= .62 * max(outer_arms)


def _alert_rim(gray, panel):
    left, top, right, bottom = panel
    span = right - left
    pad = max(2, round(.025 * span))
    trim = round(.13 * min(span, bottom - top))
    rows, columns = slice(top + trim, bottom - trim), slice(left + trim, right - trim)
    edges = ((gray[top:top + pad, columns], gray[top - pad:top, columns], 0),
             (gray[bottom - pad:bottom, columns], gray[bottom:bottom + pad, columns], 0),
             (gray[rows, left:left + pad], gray[rows, left - pad:left], 1),
             (gray[rows, right - pad:right], gray[rows, right:right + pad], 1))
    support = []
    for inside, outside, axis in edges:
        if not inside.size or not outside.size:
            return False
        inner, outer = np.median(inside, axis=axis), np.median(outside, axis=axis)
        support.append(((inner >= 215) & (inner - outer >= 32)).mean())
    # A bright part of the wallpaper can blend with one edge of the alert.
    return min(support) >= .47 and np.mean(support) >= .66


def _alert_cross(gray, panel):
    left, top, right, bottom = panel
    span = right - left
    start = right - round(.213 * span)
    stop = min(bottom, top + round(.215 * span))
    tile = gray[top:stop, start:right]
    paint = np.median(tile)
    if paint < 219 or (tile >= paint - 16).mean() < .84:
        return None
    _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(tile <= paint - 53))
    marks = []
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        a, b = start + x, top + y
        if not (.035 * span <= min(w, h) <= max(w, h) <= .111 * span and .76 <= w / h <= 1.30
                and .041 * span <= right - a - w / 2 <= .135 * span
                and .047 * span <= y + h / 2 <= .138 * span and .18 * w * h <= area <= .65 * w * h):
            continue
        if _cross_footprint(ids[y:y + h, x:x + w] == i):
            pad = round(.022 * span)
            marks.append((a - pad, b - pad, a + w + pad, b + h + pad))
    return marks[0] if len(marks) == 1 else None


def _flat_alert_boxes(ink):
    height, width = ink.shape
    interior = ink[:, round(.28 * width):round(.72 * width)]
    visible = np.uint8(interior.mean(axis=1) >= .90)
    span = max(3, round(.031 * height) | 1)
    filled = cv2.morphologyEx(visible[:, None], cv2.MORPH_CLOSE, np.ones((span, 1), np.uint8)).ravel()
    for top, bottom in _bands(filled):
        if bottom - top < .13 * height:
            continue
        inset = round(.09 * (bottom - top))
        edge = ink[top + inset:bottom - inset].mean(axis=0) >= .87
        for left, right in _bands(edge):
            if left < width / 2 < right:
                yield left, top, right - left, bottom - top


def home_light_dialog(frame):
    """Verify an opaque, closable alert over visible launcher controls."""
    height, width = frame.shape[:2]
    if width > 640:
        frame = cv2.resize(frame, (640, round(640 * height / width)), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blue, green, red = cv2.split(frame)
    chroma = cv2.subtract(cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red))
    neutral = chroma <= 27
    center = gray[round(.25 * h):round(.84 * h), round(.12 * w):round(.88 * w)]
    histogram = cv2.calcHist([center], [0], None, [256], [0, 256]).ravel()
    tone = 219 + int(np.argmax(histogram[219:]))
    levels = [(240, 255), (231, 255), (219, 255)]
    if histogram[tone] >= .025 * gray.size:
        # Isolate uniform panel paint when neighboring white icons touch it.
        levels.insert(0, (max(219, tone - 3), min(255, tone + 3)))
    for low, high in levels:
        ink = np.uint8((gray >= low) & (gray <= high) & neutral)
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = {box for contour in contours if (box := cv2.boundingRect(contour))
                 and cv2.contourArea(contour) >= .88 * box[2] * box[3]}
        if high - low <= 6:
            boxes.update(_flat_alert_boxes(ink))
        panels = []
        for x, y, span, length in boxes:
            panel = x, y, x + span, y + length
            if not (.49 * w <= span <= .90 * w and .13 * h <= length <= .53 * h
                    and abs(2 * x + span - w) <= .085 * w and .25 * h <= y <= .58 * h
                    and .36 * h <= y + length <= .84 * h):
                continue
            if (cross := _alert_cross(gray, panel)) is not None and _alert_rim(gray, panel):
                panels.append((panel, cross))
        if len(panels) == 1:
            scales = (width / w, height / h) * 2
            panel, cross = (tuple(round(value * scale) for value, scale in zip(box, scales, strict=True))
                            for box in panels[0])
            return BlockingOverlay("dialog", panel, cross)
    return None


def _header_window(panel, box):
    left, top, right, bottom = panel
    x0, y0, x1, y1 = box
    span = right - left
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if (not .042 * span <= right - cx <= .127 * span or not .042 * span <= cy - top <= .127 * span
            or not .042 * span <= min(w, h) <= max(w, h) <= .109 * span or not .71 <= w / h <= 1.40):
        return None
    reach = round(.196 * span)
    inset = max(1, round(.012 * span))
    lo, hi = right - reach, top + inset
    end_x, end_y = right - inset, min(bottom - inset, top + reach)
    if not (lo <= x0 < x1 <= end_x and hi <= y0 < y1 <= end_y):
        return None
    return lo, hi, end_x, end_y


def _header_cross(frame, panel, box):
    if (window := _header_window(panel, box)) is None:
        return False
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    lo, hi, end_x, end_y = window
    gray = cv2.cvtColor(frame[hi:end_y, lo:end_x], cv2.COLOR_BGR2GRAY)
    white = float(np.median(gray))
    if white < 229 or (gray >= white - 13).mean() < .92:
        return False
    mask = np.uint8(gray <= white - 56)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    parts = [(index, x, y, gw, gh, area) for index, (x, y, gw, gh, area) in enumerate(stats[1:], 1)
             if area >= max(4, .008 * w * h)]
    if len(parts) != 1:
        return False
    index, x, y, gw, gh, area = parts[0]
    gx, gy = x + lo, y + hi
    if (not x0 < gx < gx + gw < x1 or not y0 < gy < gy + gh < y1
            or not .31 * w <= gw <= .72 * w or not .31 * h <= gh <= .72 * h
            or not .77 <= gw / gh <= 1.30 or not .20 <= area / (gw * gh) <= .58
            or abs(gx + gw / 2 - cx) > .135 * w or abs(gy + gh / 2 - cy) > .135 * h):
        return False
    glyph = labels[y:y + gh, x:x + gw] == index
    return _cross_footprint(glyph) and _centered_cross(frame[y0:y1, x0:x1])


def _framed_header_cross(frame, panel, box):
    if (window := _header_window(panel, box)) is None:
        return False
    left, top, right, _ = panel
    lo, hi, end_x, end_y = window
    header = cv2.cvtColor(frame[hi:end_y, lo:end_x], cv2.COLOR_BGR2GRAY)
    paint = float(np.median(header))
    if paint < 219:
        return False
    # Follow the two parts of a decorative frame, keeping its attached ink
    # out of the separate close-button test.
    margin = max(2, round(.015 * (right - left)))
    if top < 2 * margin:
        return False
    ink = np.uint8(header <= paint - 56)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    for number, (x, y, w, h, _) in enumerate(stats[1:], 1):
        if x + w != ink.shape[1] or y != 0 or w < .67 * ink.shape[1] or h < .17 * ink.shape[0]:
            continue
        yy, xx = np.where(labels == number)
        if ((yy < margin) | (xx >= ink.shape[1] - margin)).mean() >= .94:
            break
    else:
        return False
    outside = frame[top - 2 * margin:top - margin,
                    left + round(.31 * (right - left)):right - round(.25 * (right - left))]
    if not outside.size or np.median(cv2.cvtColor(outside, cv2.COLOR_BGR2GRAY)) > paint - 41:
        return False

    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    inset = max(2, round(.09 * min(w, h)))
    crop = header[y0 - hi + inset:y1 - hi - inset, x0 - lo + inset:x1 - lo - inset]
    if crop.size == 0:
        return False
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    confirmed = []
    for cut in (64, 68, 72, 76):
        _, ids, parts, _ = cv2.connectedComponentsWithStats(np.uint8(crop <= paint - cut))
        marks = [(i, a, b, gw, gh, area) for i, (a, b, gw, gh, area) in enumerate(parts[1:], 1) if area >= 4]
        if len(marks) != 1:
            continue
        i, a, b, gw, gh, area = marks[0]
        cx, cy = x0 + inset + a + gw / 2, y0 + inset + b + gh / 2
        if (not .31 * w <= gw <= .72 * w or not .31 * h <= gh <= .72 * h or not .80 <= gw / gh <= 1.25
                or not .23 <= area / (gw * gh) <= .58 or abs(cx - center_x) > .13 * w or abs(cy - center_y) > .13 * h):
            continue
        quiet = crop >= paint - 13
        quiet[b:b + gh, a:a + gw] = True
        if quiet.mean() >= .96 and _cross_footprint(ids[b:b + gh, a:a + gw] == i):
            confirmed.append((a, b, a + gw, b + gh))
    return (len(confirmed) >= 2 and np.ptp(confirmed, axis=0).max() <= 2
            and _centered_cross(frame[y0:y1, x0:x1]))


def dialog_close_box(frame, regions):
    """Return one close candidate wholly on the dialog's quiet upper-right surface."""
    if regions["layer"] != "dialog" or regions.get("detected_close") or len(regions["regions"]) != 1:
        return None
    panel = tuple(regions["regions"][0])
    left, top, right, bottom = panel
    height, width = frame.shape[:2]
    if not (0 < left < right < width and 0 < top < bottom < height and right - left >= .33 * width
            and bottom - top >= .38 * (right - left)):
        return None
    matches = {tuple(item["box"]) for item in regions["candidates"] if item["visibility"] == "foreground"
               and _header_cross(frame, panel, item["box"])}
    if not matches:
        matches = {tuple(item["box"]) for item in regions["candidates"] if item["visibility"] == "foreground"
                   and _framed_header_cross(frame, panel, item["box"])}
    return next(iter(matches)) if len(matches) == 1 else None
