"""Find the visible search box and labeled categories of landscape system settings."""

from collections import Counter
import unicodedata

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_landscape_header import _magnifier
from screen_landscape_settings_list import _SETTINGS_TITLES


def _blockers(frame, boxes, regions):
    height, width = frame.shape[:2]
    owners = []
    for control in regions.get("floating_controls", ()) or ():
        bounds = control.get("box") if isinstance(control, dict) else control
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            x, y, right, bottom = bounds
            if 0 <= x < right <= width and 0 <= y < bottom <= height:
                owners.append(tuple(bounds))
    for x, y, right, bottom in boxes:
        if (not .32 * width <= x <= .74 * width or right < .86 * width
                or not .16 * height <= y <= .58 * height or bottom - y < .24 * height):
            continue
        w, h = right - x, bottom - y
        if not 1.24 <= w / h <= 2.65:
            continue
        inside = frame[y + round(.14 * h):bottom - round(.09 * h), x + round(.12 * w):right - round(.09 * w):3]
        channels = inside.reshape(-1, 3)
        if (np.percentile(channels, 88, axis=0) - np.percentile(channels, 12, axis=0)).mean() >= 62:
            owners.append((x, y, right, bottom))
    return owners


def _visible_right(blockers, top, bottom, width):
    return min((x for x, y, r, b in blockers if x > .27 * width and min(bottom, b) > max(top, y)),
               default=width)


def _intersects(first, second):
    x, y, right, bottom = first
    a, b, c, d = second
    return min(right, c) > max(x, a) and min(bottom, d) > max(y, b)


def _bounds(mask, left=0, top=0):
    x, y, w, h = cv2.boundingRect(np.uint8(mask))
    return (left + x, top + y, left + x + w, top + y + h) if w and h else None


def _reading(frame, bounds, index, cache):
    x, y, right, bottom = bounds
    pad = max(4, round(.16 * (bottom - y)))
    box = (max(0, x - pad), max(0, y - pad), min(frame.shape[1], right + pad), min(frame.shape[0], bottom + pad))
    match = read_caption(frame, box, index, cache=cache, choice=True)
    return match["text"] if match else None


def _rounded_field(frame, gray, mark, owners):
    height, width = gray.shape
    left, top, right, bottom = mark
    ink = bottom - top
    center = (top + bottom) // 2
    if center - round(2.1 * ink) < 0 or center + round(1.8 * ink) >= height:
        return None
    safe = _visible_right(owners, center - round(1.8 * ink), center + round(1.8 * ink), width)
    sample_l, sample_r = max(right + 6 * ink, round(.78 * safe)), safe - max(6, round(.035 * safe))
    if sample_r - sample_l < 1.2 * ink:
        return None
    profile = np.median(frame[:, sample_l:sample_r:3], axis=1)
    fill = np.median(profile[center - round(.52 * ink):center + round(.52 * ink)], axis=0)
    page = np.median(profile[center - round(2.1 * ink):center - round(1.7 * ink)], axis=0)
    if not 9 <= abs(fill - page).max() <= 94:
        return None
    bands = [(y, b) for y, b in _bands(abs(profile - fill).max(axis=1) <= 5) if y <= center < b]
    if len(bands) != 1:
        return None
    y, end = bands[0]
    depth = end - y
    if (not 2.1 * ink <= depth <= 3.85 * ink or not .23 * height <= y <= .62 * height
            or abs(center - (y + end) / 2) > .115 * depth):
        return None
    surround = frame[y:end, :left + round(.34 * depth)].astype(np.int16)
    cap = abs(surround - fill).max(axis=2) <= 5
    positions = []
    for lo, hi in ((.11, .22), (.42, .58), (.78, .89)):
        columns = cap[round(lo * depth):round(hi * depth)].mean(axis=0)
        support = np.flatnonzero(columns >= .75)
        if not support.size:
            return None
        positions.append(int(support[0]))
    curve_left, field_left, curve_right = positions
    if (not .21 * depth <= left - field_left <= .76 * depth
            or min(curve_left, curve_right) - field_left < .07 * depth
            or abs(curve_left - curve_right) > .19 * depth):
        return None
    if safe == width:
        # For an unoccluded field, verify that the other cap rounds off too.
        mirror = frame[y:end, round(.57 * width):][:, ::-1].astype(np.int16)
        support = abs(mirror - fill).max(axis=2) <= 5
        edges = [np.flatnonzero(support[round(lo * depth):round(hi * depth)].mean(axis=0) >= .75)
                 for lo, hi in ((.11, .22), (.42, .58), (.78, .89))]
        if any(not edge.size for edge in edges):
            return None
        high, mid, low = (int(edge[0]) for edge in edges)
        if min(high, low) - mid < .065 * depth or mid < 2 or abs(high - low) > .19 * depth:
            return None
        safe = width - mid
    return (field_left, y, safe, end), fill, page


def _field_caption(frame, mark, field, fill):
    left, top, right, bottom = field
    depth = bottom - top
    x = mark[2] + round(.13 * depth)
    start, end = top + round(.20 * depth), bottom - round(.19 * depth)
    crop = frame[start:end, x:right - round(.18 * depth)].astype(np.int16)
    bound = _bounds(abs(crop - fill).max(axis=2) >= 42, x, start)
    if not bound:
        return None
    a, y, b, r = bound
    if (not .22 * depth <= r - y <= .51 * depth or not 1.22 * (r - y) <= b - a <= .66 * (right - x)
            or abs((y + r) / 2 - (top + bottom) / 2) > .11 * depth):
        return None
    return bound


def _title(frame, field, page):
    left, top, right, bottom = field
    depth = bottom - top
    start, end = max(0, top - round(1.65 * depth)), top - round(.255 * depth)
    x0, x1 = max(0, left - round(.14 * depth)), min(right, left + round(4.62 * depth))
    crop = frame[start:end, x0:x1].astype(np.int16)
    bound = _bounds(abs(crop - page).max(axis=2) >= 45, x0, start)
    if not bound:
        return None
    x, y, R, B = bound
    h = B - y
    if not .40 * depth <= h <= .83 * depth or not 1.45 <= (R - x) / h <= 9.2 or abs(x - left) > .42 * depth:
        return None
    return bound


def _row_parts(frame, proposal, field, page, caption, owners):
    height, width = frame.shape[:2]
    left, top, right, bottom = proposal
    a, y, b, end = field
    depth = end - y
    safe = min(right, _visible_right(owners, top, bottom, width)) - round(.045 * depth)
    lead = caption[0] - round(.20 * depth)
    if safe <= lead + 1.8 * depth:
        return None
    strip = frame[top:bottom, lead:safe].astype(np.int16)
    delta = abs(strip - page).max(axis=2)
    lines = []
    for u, v in _bands((delta >= 42).sum(axis=1) >= 3):
        if v - u >= .12 * depth:
            box = _bounds(delta[u:v] >= 42, lead, top + u)
            if box:
                lines.append(box)
    if not 1 <= len(lines) <= 2:
        return None
    first = lines[0]
    x, t, R, B = first
    if not .21 * depth <= B - t <= .37 * depth or abs(x - caption[0]) > .13 * depth:
        return None
    icons = frame[top:bottom, a + round(.13 * depth):lead].astype(np.int16)
    icon = _bounds(abs(icons - page).max(axis=2) >= 48, a + round(.13 * depth), top)
    if not icon:
        return None
    l, p, r, q = icon
    if (not .19 * depth <= q - p <= .47 * depth or not .20 * depth <= r - l <= .55 * depth
            or not -.08 * depth <= (p + q) / 2 - (t + B) / 2 <= .42 * depth):
        return None
    if len(lines) == 1 or bottom >= height - round(.08 * depth):
        return first, icon, None
    second = lines[1]
    if (not .47 * (B - t) <= second[3] - second[1] <= .96 * (B - t)
            or abs(second[0] - x) > .11 * depth or not .085 * depth <= second[1] - B <= .26 * depth
            or max(first[2], second[2]) >= safe - .21 * depth):
        return None
    hit = (max(a, left), max(top, t - round(.29 * depth)), safe, min(bottom, second[3] + round(.22 * depth)))
    return first, icon, hit


def _rows(frame, boxes, field, page, caption, owners):
    fl, top, right, bottom = field
    depth = bottom - top
    ready = []
    for box in boxes:
        left, y, end, last = box
        if (abs(left - fl) > .155 * depth or y < bottom + .14 * depth
                or not .67 * depth <= last - y <= 2.05 * depth or end - left < 3.8 * depth):
            continue
        if part := _row_parts(frame, box, field, page, caption, owners):
            ready.append((y, part))
    ready.sort()
    for i, (top, (first, icon, hit)) in enumerate(ready):
        if hit is None:
            continue
        near = [pair for j, (other, pair) in enumerate(ready) if j != i and .99 * depth <= abs(top - other) <= 2.05 * depth
                and abs(pair[0][0] - first[0]) < .09 * depth and abs(pair[1][0] - icon[0]) < .16 * depth]
        if near:
            yield first, hit


def landscape_system_settings_actions(frame, regions, index):
    """Publish only the visible search affordance and unoccluded category captions."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or height < 220 or width < 1.42 * height:
        return []
    boxes = [tuple(row["box"]) for row in regions["candidates"] if row.get("visibility") == "page"
             and 0 <= row["box"][0] < row["box"][2] <= width and 0 <= row["box"][1] < row["box"][3] <= height]
    holders = [box for box in boxes if .015 * width <= box[0] <= .15 * width
               and .23 * height <= (box[1] + box[3]) / 2 <= .70 * height
               and .048 * height <= box[3] - box[1] <= .15 * height
               and .66 <= (box[2] - box[0]) / (box[3] - box[1]) <= 1.46]
    if not holders:
        return []
    owners = _blockers(frame, boxes, regions)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cache = {}
    matches = []
    for holder in holders:
        if any(_intersects(holder, owner) for owner in owners):
            continue
        tile = gray[holder[1]:holder[3], holder[0]:holder[2]]
        mark = _magnifier(255 - gray if np.median(tile) >= 128 else gray, holder)
        if not mark or not (field_proof := _rounded_field(frame, gray, mark, owners)):
            continue
        field, fill, page = field_proof
        heading, caption = _title(frame, field, page), _field_caption(frame, mark, field, fill)
        if not heading or not caption or any(_intersects(field, owner) or _intersects(heading, owner) for owner in owners):
            continue
        if not (title := _reading(frame, heading, index, cache)):
            continue
        if unicodedata.normalize("NFKC", title).casefold() not in _SETTINGS_TITLES:
            continue
        if text := _reading(frame, caption, index, cache):
            matches.append((field, page, caption, mark, text))
    if len(matches) != 1:
        return []
    field, page, caption, mark, text = matches[0]
    actions = [{"role": "search", "function": text, "preview_text": text, "box": list(field),
                "click_point": [(mark[0] + mark[2]) // 2, (mark[1] + mark[3]) // 2]}]
    for title, hit in _rows(frame, boxes, field, page, caption, owners):
        if any(_intersects(hit, owner) for owner in owners):
            continue
        label = _reading(frame, title, index, cache)
        if label:
            actions.append({"role": "open_menu", "function": label, "preview_text": label, "box": list(hit),
                            "click_point": [(title[0] + title[2]) // 2, (hit[1] + hit[3]) // 2]})
    counts = Counter(item["function"] for item in actions)
    return [item for item in actions if counts[item["function"]] == 1]
