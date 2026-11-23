"""Verify a small leading Back control from its pixels and toolbar."""

import cv2
import numpy as np

from inventory_back_evidence import _chevrons, is_shaft_back_arrow, suggest_wide_toolbar_back
from screen_candidate_glyphs import _overlap


_HIGH_CONTRAST_CUTS = (84, 104, 126)


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _shape_masks(gray, box, short):
    left, top, right, bottom = box
    tile = gray[top:bottom, left:right].astype(np.int16)
    fill = np.median(tile)
    delta = tile - fill
    if (abs(delta) <= 13).mean() < .79:
        return []
    minimum = max(10, .017 * short)
    maximum = min(.084 * short, .93 * tile.shape[0])
    samples = []
    # The stronger cuts remove soft outlines from dark antialiased strokes.
    # Preserve the original result whenever the first set already agrees.
    contrasts = [(-delta, -delta.min()), (delta, delta.max())]
    for cuts in ((29, 40, 53, 70), _HIGH_CONTRAST_CUTS):
        for level in cuts:
            for sign, (contrast, ceiling) in enumerate(contrasts):
                if ceiling < level:
                    continue
                mask = np.uint8(contrast >= level)
                _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
                    if (not 0 < x < x + w < tile.shape[1] or not 0 < y < y + h < tile.shape[0]
                            or not minimum <= h <= maximum or not .38 <= w / h <= 1.24
                            or not .045 <= area / (w * h) <= .42
                            or abs(x + w / 2 - tile.shape[1] / 2) > .28 * tile.shape[1]
                            or abs(y + h / 2 - tile.shape[0] / 2) > .24 * tile.shape[0]):
                        continue
                    body = np.uint8(labels[y:y + h, x:x + w] == i)
                    if is_shaft_back_arrow(body) or any(_chevrons(body, h / 2, h, h)):
                        samples.append((sign, level, (left + x, top + y, left + x + w, top + y + h)))
        stable = [mark for sign, level, mark in samples if any(
            side == sign and cut != level and _overlap(mark, peer) >= .79 for side, cut, peer in samples)]
        if stable:
            return stable
    return []


def _status_band(gray, box):
    height, width = gray.shape
    _, y = _center(box)
    size = box[3] - box[1]
    bottom = round(min(.071 * height, y - .62 * size))
    if bottom < 8:
        return False
    patch = gray[:bottom:2, ::2].astype(np.int16)
    background = np.median(patch, axis=1, keepdims=True)
    ink = abs(patch - background) >= 43
    left = ink[:, round(.014 * patch.shape[1]):round(.39 * patch.shape[1])]
    right = ink[:, round(.71 * patch.shape[1]):round(.98 * patch.shape[1])]
    positions = np.flatnonzero((left.mean(axis=1) >= .08) & (right.mean(axis=1) >= .11)) * 2
    if not .13 * size <= len(positions) <= 1.12 * size:
        return False
    return .70 * size <= y - np.median(positions) <= 4.4 * size


def _inset_field(gray, mark, candidates):
    _, width = gray.shape
    x, y = _center(mark)
    size = mark[3] - mark[1]
    for row in candidates:
        left, top, right, bottom = row["box"]
        w, h = right - left, bottom - top
        if (not .52 * width <= w <= .88 * width or not 1.28 * size <= h <= 3.15 * size
                or not .019 * width <= left - x <= .108 * width or right >= .96 * width
                or abs(y - (top + bottom) / 2) > .20 * h):
            continue
        pad = max(4, round(.15 * h))
        if top <= pad or bottom + pad >= gray.shape[0]:
            continue
        core = gray[top + pad:bottom - pad, left + pad:right - pad]
        paint = np.median(core)
        if (abs(core.astype(np.int16) - paint) <= 8).mean() < .72:
            continue
        before = gray[top - pad:top, left + pad:right - pad]
        after = gray[bottom:bottom + pad, left + pad:right - pad]
        if min(abs(np.median(before) - paint), abs(np.median(after) - paint)) < 10:
            continue
        region = gray[top - pad:bottom + pad, max(0, left - pad):right + pad].astype(np.int16)
        surface = np.uint8(abs(region - paint) <= 7)
        contours, _ = cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            sx, sy, sw, sh = cv2.boundingRect(contour)
            if not (.87 * w <= sw <= 1.07 * w and .71 * h <= sh <= 1.06 * h
                    and .84 * sw * sh <= cv2.contourArea(contour) <= .994 * sw * sh):
                continue
            corner = max(3, round(.15 * sh))
            strip = surface[sy:sy + corner, sx:sx + sw]
            if max(strip[:, :corner].mean(), strip[:, -corner:].mean()) <= .77:
                return True
    return False


def _changed_toolbar(gray, mark, candidates):
    height, width = gray.shape
    _, y = _center(mark)
    size = mark[3] - mark[1]
    peers = [row for row in candidates if row["center"][0] >= .29 * width
             and abs(row["center"][1] - y) <= .47 * size
             and .023 * width <= row["box"][2] - row["box"][0] <= .20 * width
             and .70 * size <= row["box"][3] - row["box"][1] <= 2.6 * size]
    if len(peers) < 2:
        return False
    a, b = round(.14 * width), round(.88 * width)
    band = gray[round(y - .34 * size):round(y + .34 * size):2, a:b:3]
    fill = np.median(band)
    if (abs(band.astype(np.int16) - fill) <= 11).mean() < .81:
        return False
    following = gray[round(y + .64 * size):min(height, round(y + 2.2 * size)):2, a:b:3]
    return np.count_nonzero((abs(following.astype(np.int16) - fill) >= 30).mean(axis=1) >= .84) >= max(2, .09 * size)


def _heading(gray, mark, *, below):
    height, width = gray.shape
    _, y = _center(mark)
    size = mark[3] - mark[1]
    if below:
        x0, x1 = round(.026 * width), round(.61 * width)
        y0, y1 = round(y + 1.36 * size), min(height, round(y + 4.04 * size))
    else:
        x0, x1 = round(.25 * width), round(.75 * width)
        y0, y1 = round(y - .56 * size), round(y + .56 * size)
    patch = gray[y0:y1, x0:x1].astype(np.int16)
    ink = abs(patch - np.median(patch)) >= 40
    if ink.mean() > .25:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(ink), connectivity=8)
    minimum, maximum = (.41, 1.6) if below else (.29, .98)
    glyphs = [(x, y, w, h) for x, y, w, h, area in stats[1:] if minimum * size <= h <= maximum * size
              and .075 * size <= w <= 1.45 * size and area >= .027 * size * size]
    for _, top, _, h in glyphs:
        peers = [(x, x + w) for x, y, w, sh in glyphs if abs(y + sh - top - h) <= .25 * size]
        if len(peers) < 3:
            continue
        start, end = min(p[0] for p in peers), max(p[1] for p in peers)
        if not .069 * width <= end - start <= .43 * width:
            continue
        if below and x0 + start <= .13 * width:
            gap = gray[mark[3] + round(.20 * size):y0, x0:x0 + end]
            if gap.size and (abs(gap.astype(np.int16) - np.median(gap)) <= 13).mean() >= .91:
                return True
        if not below and abs(x0 + (start + end) / 2 - .5 * width) <= .081 * width:
            return True
    return False


def _unique(marks):
    distinct = []
    for mark in marks:
        if not any(_overlap(mark, other) >= .66 for other in distinct):
            distinct.append(tuple(mark))
    return distinct


def _shaft_in_box(gray, box):
    left, top, right, bottom = box
    pad = max(3, round(.14 * (bottom - top)))
    chip = gray[max(0, top - pad):bottom + pad, max(0, left - pad):right + pad].astype(np.int16)
    delta = chip - np.median(chip)
    for sign in (-1, 1):
        votes = 0
        for level in (39, 63, 86):
            mask = np.uint8(sign * delta >= level)
            x, y, w, h = cv2.boundingRect(mask)
            if is_shaft_back_arrow(mask[y:y + h, x:x + w]):
                votes += 1
        if votes >= 2:
            return True
    return False


def _title_run(tile, paper, size, *, minimum, high):
    """Find a line of aligned text components on one quiet toolbar surface."""
    ink = np.uint8(abs(tile.astype(np.int16) - paper) >= 42)
    if not .009 <= ink.mean() <= .29:
        return None
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    letters = [(x, y, w, h) for x, y, w, h, area in stats[1:]
               if .12 * size <= w <= 2.1 * size and .46 * size <= h <= high * size
               and area >= .055 * size * size and 0 < y < y + h < tile.shape[0]]
    best = None
    for _, y, _, h in letters:
        peers = sorted((x, x + w) for x, t, w, k in letters if abs(t + k - y - h) <= .41 * size)
        if len(peers) < minimum:
            continue
        # An isolated tool mark is separated from the title by a wider gap.
        end = min((i for i in range(1, len(peers)) if peers[i][0] - peers[i - 1][1] > 1.03 * size), default=len(peers))
        if end >= minimum and (best is None or end > best[0]):
            best = end, peers[0][0], peers[end - 1][1]
    return best[1:] if best is not None else None


def _toolbar_ink_groups(ink, size):
    columns = (ink.sum(axis=0) >= max(2, round(.065 * size)))
    spans = np.flatnonzero(np.diff(np.r_[False, columns, False])).reshape(-1, 2)
    groups = []
    for left, right in spans:
        if groups and left - groups[-1][1] <= .37 * size:
            groups[-1] = groups[-1][0], right
        else:
            groups.append((left, right))
    return sum(.17 * size <= right - left <= 2.05 * size and ink[:, left:right].mean() >= .11
               for left, right in groups)


def _adjacent_native_title(gray, box, has_status):
    """Corroborate a leading arrow with a left title or a separate large title."""
    height, width = gray.shape
    left, top, right, bottom = box
    x, y = _center(box)
    size = bottom - top
    if not .023 * width <= x <= .095 * width or not .027 * width <= size <= .067 * width:
        return False
    low, high = max(0, round(y - .65 * size)), min(height, round(y + .68 * size))
    strip = gray[low:high, round(.12 * width):round(.94 * width)]
    paper = float(np.median(strip))
    if (abs(strip.astype(np.int16) - paper) <= 12).mean() < .83:
        return False
    shaft = _shaft_in_box(gray, box)
    # A status row can corroborate a shorter chevron. Without it, require the
    # stronger shaft shape and a clean margin above the first viewport row.
    above = gray[:max(1, low - round(.19 * size)), round(.016 * width):round(.97 * width)]
    after = gray[high:round(y + 1.55 * size), round(.016 * width):round(.97 * width)]
    first_row = y <= .053 * height and shaft and (abs(above.astype(np.int16) - paper) <= 14).mean() >= .975
    if (has_status or first_row) and (abs(after.astype(np.int16) - paper) <= 14).mean() >= .975:
        begin = right + round(.56 * size)
        extent = _title_run(gray[low:high, begin:round(.72 * width)], paper, size, minimum=4, high=1.18)
        if extent is not None:
            a, b = extent
            if .74 * size <= begin + a - right <= 2.85 * size and 2.6 * size <= b - a <= .46 * width:
                return True
    if not has_status or not shaft:
        return False
    tools = gray[low:high, round(.74 * width):round(.983 * width)]
    if _toolbar_ink_groups(abs(tools.astype(np.int16) - paper) >= 42, size) < 2:
        return False
    first, last = round(y + 1.63 * size), min(height, round(y + 4.56 * size))
    gap = gray[high:first, round(.015 * width):round(.94 * width)]
    if (abs(gap.astype(np.int16) - paper) <= 12).mean() < .94:
        return False
    begin = round(.015 * width)
    extent = _title_run(gray[first:last, begin:round(.95 * width)], paper, size, minimum=5, high=2.13)
    return extent is not None and begin + extent[0] <= .105 * width and .35 * width <= extent[1] - extent[0] <= .88 * width


def _portrait_context(gray, box, candidates):
    if _changed_toolbar(gray, box, candidates):
        return True
    status = _status_band(gray, box)
    if status and (_inset_field(gray, box, candidates) or _heading(gray, box, below=False) or _heading(gray, box, below=True)):
        return True
    return _adjacent_native_title(gray, box, status)


def suggest_back_action(frame, regions, marks=(), *, working_view=None):
    """Find one top-toolbar Back action in original pixel coordinates."""
    if regions["layer"] != "page":
        return None
    height, width = frame.shape[:2]
    short = min(height, width)
    if short < 180:
        return None
    from screen_foreground import live_page_toolbar_back

    if (live := live_page_toolbar_back(frame, working_view=working_view)) is not None:
        return {"role": "back", "function": "返回上一级", "box": list(live), "click_point": list(_center(live))}
    candidates = [row for row in regions["candidates"] if row.get("visibility") == "page"]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def leading(box):
        left, top, right, bottom = box
        x, y = _center(box)
        return (0 <= left < right <= width and 0 <= top < bottom <= height
                and .015 * width <= x <= (.23 if width > height else .14) * width
                and (.016 if width > height else .028) * height <= y <= .145 * height)

    visible_ids = {row["id"] for row in candidates}
    found = [mark["box"] for mark in marks if mark.get("id") in visible_ids
             and mark["role"] == "back" and leading(mark["box"])]
    for row in candidates:
        box = row["box"]
        if (leading(box) and max(box[2] - box[0], box[3] - box[1]) <= .18 * short
                and min(box[2] - box[0], box[3] - box[1]) >= .026 * short):
            found.extend(mark for mark in _shape_masks(gray, box, short) if leading(mark))
    glyphs = _unique(found)
    if width > height:
        nodes = [{"kind": "icon", "box": box, "center": _center(box)} for box in glyphs]
        accepted = glyphs if len(glyphs) == 1 and suggest_wide_toolbar_back(frame, nodes) is not None else []
    else:
        accepted = [box for box in glyphs if _portrait_context(gray, box, candidates)]
        if not accepted:
            from screen_context_back import context_back

            if (action := context_back(frame, regions, gray=gray)) is not None:
                return action
            from screen_titled_back import titled_back_action

            return titled_back_action(frame, regions, gray=gray)
    if len(accepted) != 1:
        return None
    box = [int(value) for value in accepted[0]]
    return {"role": "back", "function": "返回上一级", "box": box, "click_point": list(_center(box))}
