"""Measure whether a visible short label belongs to a drawn control."""

import cv2
import numpy as np

from inventory_toolbar_actions import _glyphs
from inventory_value_select import _looks_down
from screen_candidate_glyphs import _search


def _sizes(box):
    left, top, right, bottom = box
    return right - left, bottom - top


def _center(box):
    left, top, right, bottom = box
    return ((left + right) / 2, (top + bottom) / 2)


def _adjacent_icon(gray, bounds, text):
    left, top, right, bottom = bounds
    em = text[3] - text[1]
    ty = max(top, round(text[1] - .48 * em))
    by = min(bottom, round(text[3] + .48 * em))
    gap = max(3, round(.16 * em))
    for lx, rx in ((left + gap, text[0] - gap), (text[2] + gap, right - gap)):
        if rx - lx < .55 * em or not 0 <= lx < rx <= gray.shape[1]:
            continue
        chip = gray[ty:by, lx:rx].astype(np.int16)
        ink = np.uint8(np.abs(chip - np.median(chip)) >= 46)
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        if any(.56 * em <= min(w, h) <= max(w, h) <= 1.93 * em
               and .18 * w * h <= area <= .93 * w * h and
               abs(y + ty + h / 2 - (text[1] + text[3]) / 2) <= .45 * em
               for _, y, w, h, area in stats[1:]):
            return True
    return False


def _outlined_caption(gray, box, text):
    left, top, right, bottom = box
    w, h = _sizes(box)
    ew, em = _sizes(text)
    cx, cy = _center(box)
    _, ty = _center(text)
    if not (1.45 <= w / h <= 4.2 and .17 * h <= em <= .54 * h and .16 * w <= ew <= .69 * w
            and abs(ty - cy) <= .15 * h):
        return None
    pad = max(3, round(.10 * h))
    x0, y0 = max(0, left - pad), max(0, top - pad)
    window = gray[y0:bottom + pad, x0:right + pad]
    smoothed = cv2.GaussianBlur(window, (3, 3), 0)
    found = []
    for low, high in ((9, 23), (17, 40)):
        edge = cv2.morphologyEx(cv2.Canny(smoothed, low, high), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, sw, sh = cv2.boundingRect(contour)
            bounds = (x0 + x, y0 + y, x0 + x + sw, y0 + y + sh)
            if (.80 * w <= sw <= 1.04 * w and .72 * h <= sh <= 1.04 * h
                    and abs(x0 + x + sw / 2 - cx) <= .08 * w and abs(y0 + y + sh / 2 - cy) <= .10 * h
                    and .73 * sw * sh <= cv2.contourArea(contour) <= .95 * sw * sh
                    and bounds[0] < text[0] < text[2] < bounds[2] and bounds[1] < text[1] < text[3] < bounds[3]
                    and _adjacent_icon(gray, bounds, text)):
                found.append((low, bounds))
    return next((bounds for level, bounds in found if any(other != level and
                 max(abs(a - b) for a, b in zip(bounds, peer, strict=True)) <= .059 * h for other, peer in found)), None)


def _cell_pixels(gray, box):
    left, top, right, bottom = box
    w, h = _sizes(box)
    cx, _ = _center(box)
    x0, x1 = round(cx - .34 * w), round(cx + .34 * w)
    y0, y1, y2 = (round(top + fraction * h) for fraction in (.075, .60, .95))
    patch = gray[y0:y2, x0:x1].astype(np.int16)
    surface = float(np.percentile(patch, 80))
    flatness = np.abs(patch - surface)
    if surface < 176 or (flatness <= 17).mean() < .68:
        return None
    mask = np.uint8(flatness >= 30)
    icon = mask[:y1 - y0]
    words = mask[y1 - y0:]
    ix, iy, iw, ih = cv2.boundingRect(icon)
    tx, ty, tw, th = cv2.boundingRect(words)
    if not (.18 * w <= iw <= .65 * w and .19 * h <= ih <= .51 * h
            and .025 <= icon.mean() <= .38 and abs(ix + x0 + iw / 2 - cx) <= .18 * w
            and .20 * w <= tw <= .72 * w and .095 * h <= th <= .29 * h and .025 <= words.mean() <= .31):
        return None
    return ((ix + x0 + iw / 2, iy + y0 + ih / 2), (tx + x0, ty + y1, tx + x0 + tw, ty + y1 + th))


def _icon_row(gray, anchor, candidates, text):
    width = gray.shape[1]
    box = anchor["box"]
    w, h = _sizes(box)
    if not (.076 * width <= h <= .24 * width and .68 <= w / h <= 1.80 and text[1] >= box[1] + .50 * h):
        return None
    peers = []
    for item in candidates:
        x0, y0, x1, y1 = item["box"]
        pw, ph = x1 - x0, y1 - y0
        if (item["visibility"] != anchor["visibility"] or not .58 * w <= pw <= 1.78 * w
                or not .77 * h <= ph <= 1.29 * h or abs(y0 - box[1]) > .16 * h or abs(y1 - box[3]) > .21 * h):
            continue
        if measurement := _cell_pixels(gray, item["box"]):
            peers.append((item["id"], *measurement))
    if len(peers) < 3 or not any(item[0] == anchor["id"] for item in peers):
        return None
    # Use the measured glyph centers, where symmetric cell spacing and a shared
    # caption baseline can be checked without reading the other cell labels.
    peers.sort(key=lambda item: item[1][0])
    distinct = [peers[0]]
    for item in peers[1:]:
        if item[1][0] - distinct[-1][1][0] <= .29 * w:
            if item[0] == anchor["id"]:
                distinct[-1] = item
            continue
        distinct.append(item)
    if len(distinct) < 3:
        return None
    for count in range(3, len(distinct) + 1):
        for start in range(len(distinct) - count + 1):
            band = distinct[start:start + count]
            own = next((item[2] for item in band if item[0] == anchor["id"]), None)
            if own is None:
                continue
            xs, ys = zip(*(item[1] for item in band), strict=True)
            gaps = np.diff(xs)
            if (max(xs) - min(xs) >= .36 * width and .63 * w <= gaps.min() and gaps.max() <= 1.85 * w
                    and gaps.max() <= 1.24 * gaps.min() and max(ys) - min(ys) <= .16 * h
                    and max(item[2][3] for item in band) - min(item[2][3] for item in band) <= .16 * h
                    and max(abs(a - b) for a, b in zip(own, text, strict=True)) <= max(5, .12 * h)):
                return box
    return None


def _seam(gray, start, stop, h, width):
    step = max(3, round(.029 * h))
    start, stop = max(step, start), min(gray.shape[0] - step, stop)
    x = round(.07 * width)
    for y in range(start, stop):
        stripe = gray[y, x:width - x:3].astype(np.int16)
        before = gray[y - step, x:width - x:3].astype(np.int16)
        after = gray[y + step, x:width - x:3].astype(np.int16)
        if ((np.maximum(before, after) - stripe >= 14) & (stripe <= 240)).mean() >= .85:
            return y
    return None


def _row_pieces(gray, top, bottom, left, right):
    h = bottom - top
    y0, y1 = round(top + .24 * h), round(bottom - .23 * h)
    chip = gray[y0:y1, left:right]
    if (chip >= 237).mean() < .86:
        return []
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(chip <= 204), connectivity=8)
    parts = [(left + x, y0 + y, w, sh, labels[y:y + sh, x:x + w] == index)
             for index, (x, y, w, sh, area) in enumerate(stats[1:], 1)
             if .075 * h <= sh <= .41 * h and .022 * h <= w <= .48 * h and area >= max(6, .00065 * h * h)]
    parts.sort(key=lambda row: row[0])
    return parts


def _looks_funnel(mask):
    h, w = mask.shape
    if not (.52 <= w / h <= 1.32 and .16 <= mask.mean() <= .68):
        return False
    upper = np.where(mask[:max(1, round(.25 * h))])[1]
    lower = np.where(mask[-max(1, round(.26 * h)):])[1]
    return (upper.size >= w and lower.size > 0 and np.ptp(upper) >= .79 * w
            and .12 * w <= np.ptp(lower) <= .62 * w)


def _command_strip(gray, box):
    height, width = gray.shape
    left, top, right, bottom = box
    w, h = _sizes(box)
    if not (.38 * width <= w <= .62 * width and .049 * width <= h <= .143 * width and
            .076 * height <= top <= .72 * height and (left <= .07 * width or right >= .93 * width)):
        return None
    margin = round(.15 * h)
    y0 = _seam(gray, top - margin, top + margin, h, width)
    y1 = _seam(gray, bottom - margin, bottom + margin, h, width)
    if y0 is None or y1 is None or not .77 * h <= y1 - y0 <= 1.20 * h:
        return None
    inner = round(.10 * width)
    halves = [(inner, width // 2 - inner), (width // 2 + inner, width - inner)]
    sides = [_row_pieces(gray, y0, y1, x0, x1) for x0, x1 in halves]
    if any(len(pieces) < 4 for pieces in sides):
        return None
    if not all(.09 * (y1 - y0) <= parts[-1][0] - (parts[-2][0] + parts[-2][2]) <= .46 * (y1 - y0)
               for parts in sides):
        return None
    icons = [parts[-1] for parts in sides]
    roles = ["sort" if _looks_down(item[-1]) else "filter" if _looks_funnel(item[-1]) else None for item in icons]
    if set(roles) != {"sort", "filter"}:
        return None
    return y0, y1, roles, icons


def _paired_strip(gray, box, text, expected):
    strip = _command_strip(gray, box)
    if strip is None:
        return None
    top, bottom, roles, icons = strip
    width = gray.shape[1]
    side = int(_center(box)[0] > width / 2)
    if roles[side] != expected:
        return None
    icon = icons[side]
    _, h = _sizes(box)
    if (not .10 * h <= icon[0] - text[2] <= .52 * h
            or abs(icon[1] + icon[3] / 2 - (text[1] + text[3]) / 2) > .13 * h):
        return None
    return (0, top, width // 2, bottom) if side == 0 else (width // 2, top, width, bottom)


def supplement_command_cells(frame, regions):
    """Restore the undetected half of a rail after both operation glyphs are measured."""
    if regions["layer"] != "page":
        return regions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    width = gray.shape[1]
    previous = regions["candidates"]
    additions = []
    for row in previous:
        strip = _command_strip(gray, row["box"])
        if strip is None:
            continue
        top, bottom, _, _ = strip
        middle = (top + bottom) / 2
        for left, right in ((0, width // 2), (width // 2, width)):
            center = (left + right) / 2
            if any(abs(peer["center"][0] - center) <= .13 * width and
                   abs(peer["center"][1] - middle) <= .22 * (bottom - top) and
                   .37 * width <= _sizes(peer["box"])[0] <= .62 * width and
                   .70 * (bottom - top) <= _sizes(peer["box"])[1] <= 1.32 * (bottom - top)
                   for peer in (*previous, *additions)):
                continue
            additions.append({"id": max(item["id"] for item in previous) + 1 + len(additions),
                              "box": [left, top, right, bottom], "center": [center, middle],
                              "visibility": "page", "pass": "command_rail"})
    return {**regions, "candidates": previous + additions} if additions else regions


def _search_command(gray, anchor, candidates, text):
    height, width = gray.shape
    box = anchor["box"]
    x, y = _center(text)
    w, h = _sizes(box)
    if x < .81 * width or not .035 * height <= y <= .16 * height or abs(y - _center(box)[1]) > .19 * h:
        return None
    for row in candidates:
        left, top, right, bottom = row["box"]
        sw, sh = _sizes(row["box"])
        if (row["visibility"] != "page" or sw < .48 * width or not .78 * h <= sh <= 1.40 * h
                or not 0 <= text[0] - right <= .53 * sh or abs(y - (top + bottom) / 2) > .19 * sh):
            continue
        pad = max(3, round(.11 * sh))
        y0, x0 = max(0, top - pad), max(0, left - pad)
        crop = gray[y0:bottom + pad, x0:right + pad]
        plane = gray[top + pad:bottom - pad, round(left + .75 * sw):round(left + .90 * sw)]
        if not plane.size:
            continue
        tone = float(np.median(plane))
        above = gray[y0:top, round(left + .27 * sw):round(left + .75 * sw)]
        below = gray[bottom:bottom + pad, round(left + .27 * sw):round(left + .75 * sw)]
        if not above.size or not below.size or min(np.median(above), np.median(below)) - tone < 9:
            continue
        ink = np.uint8(np.abs(crop.astype(np.int16) - tone) <= 6)
        surfaces, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for surface in surfaces:
            sx, sy, rw, rh = cv2.boundingRect(surface)
            if not (.86 * sw <= rw <= 1.05 * sw and .72 * sh <= rh <= 1.05 * sh and
                    .78 * rw * rh <= cv2.contourArea(surface) <= .982 * rw * rh):
                continue
            lx, ty = x0 + sx, y0 + sy
            lead = (lx + round(.13 * rh), ty + round(.15 * rh), lx + round(.95 * rh), ty + round(.85 * rh))
            if any(_search(np.uint8(glyph.mask)) for glyph in _glyphs(gray, lead, round(.42 * rh))):
                return box
    return None


def owned_label_box(gray, anchor, candidates, text_box, family):
    """Return an owning region only when pixels corroborate a named affordance."""
    if anchor["visibility"] != "page":
        return None
    if family == "capsule":
        return _outlined_caption(gray, anchor["box"], text_box)
    if family == "icon_row":
        return _icon_row(gray, anchor, candidates, text_box)
    if family == "search":
        return _search_command(gray, anchor, candidates, text_box)
    return _paired_strip(gray, anchor["box"], text_box, family)


def toolbar_search_actions(frame, regions, index, known):
    """Read the separate toolbar action beside a verified wide search surface."""
    if regions["layer"] != "page":
        return []
    from screen_caption_transcript import read_caption
    from screen_component_captions import _contrast, _rectangular_button, _tight

    height, width = frame.shape[:2]
    bars = [row["box"] for row in regions["candidates"] if row["visibility"] == "page" and
            .04 * width <= row["box"][0] <= .25 * width and .71 * width <= row["box"][2] <= .895 * width and
            _sizes(row["box"])[0] >= .52 * width and .052 * width <= _sizes(row["box"])[1] <= .15 * width and
            .038 * height <= row["center"][1] <= .16 * height]
    if not bars:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    matches = []
    for bar in bars:
        measured = _rectangular_button(gray, bar)
        if measured is None:
            continue
        x0, y0, x1, y1 = measured
        w, h = _sizes(measured)
        inner = gray[y0 + round(.12 * h):y1 - round(.12 * h), x0 + round(.47 * w):x0 + round(.77 * w)]
        upper = gray[y0 - max(2, round(.085 * h)):y0 - 1, x0 + round(.47 * w):x0 + round(.77 * w)]
        if not inner.size or not upper.size or np.median(upper) - np.median(inner) < 8:
            continue
        left, right = max(bar[2] + max(2, round(.005 * width)), x1 + round(.17 * h)), width - round(.022 * width)
        top, bottom = y0 + round(.11 * h), y1 - round(.11 * h)
        if left >= right or top >= bottom:
            continue
        delta, paint = _contrast(frame[top:bottom, left:right])
        if np.min(paint) < 220 or (delta <= 16).mean() < .63:
            continue
        text = _tight(delta >= 74, left, top)
        if text is None or not .21 * h <= _sizes(text)[1] <= .61 * h or abs(_center(text)[1] - _center(measured)[1]) > .16 * h:
            continue
        tx0, ty0, tx1, ty1 = text
        nx, ny = _center(text)
        if any(item["box"][0] <= nx < item["box"][2] and item["box"][1] <= ny < item["box"][3]
               for item in known):
            continue
        pad = max(3, round(.09 * h))
        crop = (max(left, tx0 - pad), max(top, ty0 - pad), min(right, tx1 + pad), min(bottom, ty1 + pad))
        label = read_caption(frame, crop, index, cache={}, choice=True)
        if label is not None and label["text"].casefold() in {"搜索", "搜尋", "search"}:
            matches.append({"role": "search", "function": "搜索", "box": list(crop),
                            "click_point": [nx, ny], "preview_text": "Search"})
    return matches if len(matches) == 1 else []
