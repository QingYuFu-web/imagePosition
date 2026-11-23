"""Confirm raised button geometry against captions already read in this frame."""

import cv2
import numpy as np

from screen_component_captions import _rectangular_button
from screen_painted_commands import _EXPLICIT, _unclaimed
from screen_short_labels import _center, _text_key


def _owner_caption(reads, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    lines = [row for row in reads if row.get("min_margin", 0) >= .026 and row.get("max_distance", 1) <= .185
             and row.get("support", 0) >= 3 and row.get("line_box") is not None
             and left < row["line_box"][0] < row["line_box"][2] < right
             and top < row["line_box"][1] < row["line_box"][3] < bottom]
    if not 1 <= len(lines) <= 2:
        return None
    lines.sort(key=lambda row: row["line_box"][0])
    positions = [item["line_box"] for item in lines]
    x, y, r, b = (positions[0][0], min(p[1] for p in positions), positions[-1][2], max(p[3] for p in positions))
    if (abs((x + r) / 2 - (left + right) / 2) > .10 * width
            or abs((y + b) / 2 - (top + bottom) / 2) > .14 * height
            or not .16 * height <= b - y <= .56 * height
            or not .42 * (b - y) <= r - x <= 6.8 * (b - y)):
        return None
    if len(lines) == 2:
        a, z = positions
        if (not 0 <= z[0] - a[2] <= .22 * (b - y)
                or max(abs(a[1] - z[1]), abs(a[3] - z[3])) > .29 * (b - y)
                or lines[0].get("polarity") != lines[1].get("polarity")):
            return None
    caption = "".join(item["text"] for item in lines)
    key = _text_key(caption)
    if caption.isascii() and caption.isalpha():
        caption = "OK" if key == "ok" else caption.capitalize()
    purpose = _EXPLICIT.get(key)
    return (caption, purpose, (x, y, r, b)) if purpose else None


def _measured(frame, box):
    l, t, r, b = box
    radius = max(3, round(.18 * (b - t)))
    y, x = max(0, t - radius), max(0, l - radius)
    tile = frame[y:b + radius, x:r + radius]
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    fits = _rectangular_button(gray, (l - x, t - y, r - x, b - y))
    return (fits[0] + x, fits[1] + y, fits[2] + x, fits[3] + y) if fits else None


def _paint_planes(frame, box):
    """Sample the center of all four straight spans inside and outside a rounded edge."""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    pad = max(2, round(.12 * height))
    if min(left, top, frame.shape[1] - right, frame.shape[0] - bottom) < pad:
        return None
    xs = np.arange(left + round(.24 * width), right - round(.24 * width), max(2, height // 18))
    ys = np.arange(top + round(.35 * height), bottom - round(.35 * height), max(2, height // 24))
    if min(xs.size, ys.size) < 6:
        return None
    inner = (frame[top + pad, xs], frame[bottom - pad, xs],
             frame[ys, left + pad], frame[ys, right - pad])
    outer = (frame[top - pad, xs], frame[bottom + pad - 1, xs],
             frame[ys, left - pad], frame[ys, right + pad - 1])
    surface = np.median(np.concatenate(inner), axis=0)
    canvas = np.median(np.concatenate(outer), axis=0)
    if not all((np.abs(plane - canvas).max(axis=1) <= 5).mean() >= .91 for plane in outer):
        return None
    if not all((np.abs(plane - surface).max(axis=1) <= 5).mean() >= .91 for plane in inner):
        return None
    gap = np.abs(surface - canvas).max()
    if np.ptp(surface) >= 74 and gap >= 64:
        return surface, canvas, "colored"
    if np.ptp(surface) <= 10 and np.ptp(canvas) <= 10 and min(surface) >= 168 and 9 <= gap <= 29:
        return surface, canvas, "neutral"
    return None


def _curved_corners(frame, bounds, paint, canvas):
    left, top, right, bottom = bounds
    reach = max(2, round(.22 * (bottom - top)))
    tolerance = max(4., .32 * np.abs(paint - canvas).max())
    for tile in (frame[top:top + reach, left:left + reach],
                 frame[top:top + reach, right - reach:right][:, ::-1],
                 frame[bottom - reach:bottom, left:left + reach][::-1],
                 frame[bottom - reach:bottom, right - reach:right][::-1, ::-1]):
        fill = np.abs(tile.astype(np.int16) - paint).max(axis=2) <= tolerance
        if not .06 <= fill.mean() <= .91 or fill[0].mean() + fill[:, 0].mean() > .66:
            return False
    return True


def _visible_ink(frame, bounds, ink, surface, kind):
    x0, y0, x1, y1 = ink
    glyph = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.int16)
    background = int(cv2.cvtColor(np.uint8(surface[None, None]), cv2.COLOR_BGR2GRAY)[0, 0])
    delta = background - glyph
    strong = delta >= 112 if kind == "neutral" else np.abs(delta) >= 95
    if strong.mean() < .18:
        return False
    l, t, r, b = bounds
    width, height = r - l, b - t
    left, right = l + round(.12 * width), r - round(.12 * width)
    top, bottom = t + round(.16 * height), b - round(.16 * height)
    text_area = frame[top:bottom, left:right].astype(np.int16)
    visible = np.uint8(np.abs(text_area - surface).max(axis=2) >= 62)
    x, y, w, h = cv2.boundingRect(visible)
    pad = max(3, round(.08 * (ink[3] - ink[1])))
    return (w and h and max(abs(p - q) for p, q in zip((left + x, top + y, left + x + w, top + y + h), ink, strict=True))
            <= pad)


def faint_caption_actions(frame, regions, labels, known=()):
    """Return confirmed command points from verified geometry and existing text evidence."""
    height, width = frame.shape[:2]
    layer = regions["layer"]
    if layer == "page" and not regions["regions"]:
        visible, visibility = (0, 0, width, height), "page"
    elif layer in {"dialog", "bottom_sheet"} and regions["regions"]:
        visible, visibility = regions["regions"][0], "foreground"
    else:
        return []
    by_id = {}
    for label in labels:
        if label.get("line_box") is not None and label.get("text"):
            by_id.setdefault(label.get("id"), []).append(label)
    output = []
    for row in regions["candidates"]:
        if row.get("visibility") != visibility or row.get("score", 0) < .52 or row["id"] not in by_id:
            continue
        box = tuple(row["box"])
        l, t, r, b = box
        w, h = r - l, b - t
        if (not (visible[0] <= l < r <= visible[2] and visible[1] <= t < b <= visible[3])
                or not .18 * min(width, height) <= w <= .97 * width
                or not .038 * min(width, height) <= h <= .19 * min(width, height)
                or not 2.20 <= w / h <= 10.0):
            continue
        point = _center(box)
        if (not _unclaimed(box, point, (*known, *output))
                or any(x <= point[0] < r0 and y <= point[1] < b0 for x, y, r0, b0 in regions.get("floating_controls", ()))):
            continue
        read = _owner_caption(by_id[row["id"]], box)
        if read is None:
            continue
        text, purpose, ink = read
        bounds = _measured(frame, box)
        if bounds is None:
            continue
        evidence = _paint_planes(frame, bounds)
        if evidence is None:
            continue
        surface, canvas, kind = evidence
        if not _curved_corners(frame, bounds, surface, canvas) or not _visible_ink(frame, bounds, ink, surface, kind):
            continue
        role = "search" if purpose == "搜索" else "labeled_control"
        output.append({"role": role, "function": purpose, "box": list(bounds), "click_point": _center(bounds),
                       "text": text, "preview_text": text})
    return output
