"""Confirm a comment sheet from its plate, dismissal and visible heading."""

import math
from itertools import combinations

import cv2
import numpy as np

from screen_comment_sheet_modal import _phrase
from screen_component_captions import _bands
from screen_foreground import BlockingOverlay, _paired_sheet_corners, _working_view
from screen_rim_attached_dismiss import _cross_arms


_HEADINGS = ("comments", "评论", "評論", "条评论", "條評論", "留言", "评论区", "評論區", "全部评论")


def _surfaces(gray):
    height, width = gray.shape
    inset = max(2, round(.038 * width))
    plate = gray[:, inset:width - inset].astype(np.int16)
    tone = np.median(plate, axis=1)
    flat = (np.abs(plate - tone[:, None]) <= 7).mean(axis=1)
    pad = max(3, round(.018 * width))
    ids = np.flatnonzero((flat[1:] >= .82) & ((flat[:-1] < .82) | (abs(np.diff(tone)) >= 11))) + 1
    for top in ids:
        if not .12 * height <= top <= min(.80 * height, height - .40 * width):
            continue
        value = float(tone[top])
        if 66 < value < 214:
            continue
        face = plate[top:top + pad]
        before = plate[top - pad - 1:top - 1]
        if (np.abs(face - value) <= 9).mean() < .88 or (np.abs(before - value) >= 14).mean() < .43:
            continue
        sheet = gray[top + 2 * pad:min(height, top + round(.55 * width))]
        if not all((abs(sheet[:, side].astype(np.int16) - value) <= 10).mean() >= .84
                   for side in (slice(0, inset), slice(width - inset, width))):
            continue
        yield int(top), value


def _close(gray, top, tone, *, thresholds=(58, 86)):
    height, width = gray.shape
    left = round(.84 * width)
    bottom = min(height, top + round(.135 * width))
    crop = gray[top:bottom, left:].astype(np.float32)
    contrast = crop - tone if tone <= 66 else tone - crop
    hits = []
    for threshold in thresholds:
        _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(contrast >= threshold), connectivity=8)
        for i, (x, y, w, h, count) in enumerate(stats[1:], 1):
            if (not .016 * width <= min(w, h) <= max(w, h) <= .064 * width
                    or not .75 <= w / h <= 1.27 or not .16 <= count / (w * h) <= .62
                    or not .023 * width <= y + h / 2 <= .115 * width
                    or not .019 * width <= crop.shape[1] - x - w / 2 <= .115 * width):
                continue
            if _cross_arms(ids[y:y + h, x:x + w] == i):
                hits.append((threshold, (left + x, top + y, left + x + w, top + y + h)))
    paired = [box for threshold, box in hits if any(other != threshold and
              max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= 2 for other, peer in hits)]
    if paired and np.ptp(paired, axis=0).max() <= 2:
        return tuple(map(int, np.median(paired, axis=0)))
    return None


def _word_windows(ink):
    """Find complete heading words alongside an adjacent count or tab caption."""
    runs = _bands(ink.any(axis=0))
    if not runs:
        return []
    height = ink.shape[0]
    first, last = runs[0][0], runs[-1][1]
    widths = (1.5 * height, 8.2 * height)
    windows = [(first, last)]
    windows.extend((start, last) for start, _ in runs[1:] if widths[0] <= last - start <= widths[1])
    windows.extend((first, end) for _, end in runs[:-1] if widths[0] <= end - first <= widths[1])
    eligible = [window for window in dict.fromkeys(windows) if widths[0] <= window[1] - window[0] <= widths[1]]
    return sorted(eligible, key=lambda box: min(abs((box[1] - box[0]) / height - typical)
                                                for typical in (2., 3., 6.6)))


def _heading(frame, panel_top, index, *, dark, weak_close=None, faint_ink=False):
    height, width = frame.shape[:2]
    if weak_close is None:
        start = panel_top + round(.028 * width)
        stop = min(height, panel_top + round(.197 * width))
    else:
        close_y = (weak_close[1] + weak_close[3]) / 2
        start = max(panel_top + 2, round(close_y - .047 * width))
        stop = min(height, round(close_y + .040 * width))
    left, right = round(.037 * width), round(.770 * width)
    crop = cv2.cvtColor(frame[start:stop, left:right], cv2.COLOR_BGR2GRAY)
    if dark:
        crop = 255 - crop
    base = int(np.median(crop))
    cutoff = 42 if faint_ink else (93 if weak_close else 54)
    ink = cv2.subtract(base, crop) >= cutoff
    rows = []
    for a, b in _bands(ink.sum(axis=1) >= max(3, .013 * width)):
        if .014 * width <= b - a <= .059 * width:
            x, _, span, _ = cv2.boundingRect(np.uint8(ink[a:b]))
            rows.append((abs(left + x + span / 2 - width / 2), a, b))
    for _, top, bottom in sorted(rows):
        # Some headers place the comment count below a separate search caption.
        size = bottom - top
        pad = max(2, round(.09 * size))
        vertical = max(3, round(.25 * size))
        for a, b in _word_windows(ink[top:bottom]):
            center = (left + (a + b) / 2) / width
            if not .09 <= center <= .65:
                continue
            area = (max(0, a - pad), max(0, top - vertical),
                    min(crop.shape[1], b + pad), min(len(crop), bottom + vertical))
            if _phrase(crop, area, _HEADINGS, index):
                return True
    return False


def _content_cuts(gray):
    """Find picture strokes cut by a nearly white full-width plate."""
    height, width = gray.shape
    span = max(2, round(.010 * width))
    samples = gray.astype(np.int16)
    starts = round(.13 * height)
    stops = min(round(.75 * height), height - round(.52 * width))
    if starts + span >= stops:
        return
    before = samples[starts - span:stops - span]
    after = samples[starts:stops]
    painted = after >= 228
    changed = painted & (after - before >= 22)
    hits = (painted.mean(axis=1) >= .93) & ((before >= 223).mean(axis=1) >= .46) & (changed.mean(axis=1) >= .12)
    for first, end in _bands(hits):
        top = starts + first
        edge = changed[first:end].mean(axis=0) >= .66
        if not any(b - a >= .092 * width for a, b in _bands(edge)):
            continue
        face = gray[top:top + max(3, round(.016 * width))]
        trace = np.median(samples[top - 2 * span:top - span], axis=0)
        if (np.median(face, axis=0)[edge] - trace[edge] >= 17).mean() < .65:
            continue
        rim = gray[top:top + round(.24 * width)]
        if ((face >= 226).mean() >= .94 and
                min((rim[:, :span] >= 225).mean(), (rim[:, -span:] >= 225).mean()) >= .88):
            yield top


def _comment_rows(gray, top):
    """Require repeated avatar, text, and reaction cells below the header."""
    height, width = gray.shape
    first = top + round(.088 * width)
    body = gray[first:min(height, top + round(1.44 * width))]
    left = body[:, :round(.165 * width)]
    contours, _ = cv2.findContours(np.uint8(left <= 227), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    avatars = []
    for outline in contours:
        x, y, w, h = cv2.boundingRect(outline)
        if (.052 * width <= min(w, h) <= max(w, h) <= .125 * width and .83 <= w / h <= 1.21
                and .015 * width <= x <= .079 * width and cv2.contourArea(outline) >= .58 * w * h):
            text = body[max(0, y - round(.015 * width)):y + h + round(.045 * width),
                        max(x + w + 2, round(.139 * width)):round(.60 * width)]
            if text.size and ((text <= 187).sum(axis=1) >= .055 * width).any():
                avatars.append((x + w / 2, y + h / 2, w, h))
    if len(avatars) < 2:
        return False
    right = round(.645 * width)
    outlines, _ = cv2.findContours(np.uint8(body[:, right:] <= 223), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = [(x + w / 2 + right, y + h / 2, w, h) for contour in outlines
             for x, y, w, h in (cv2.boundingRect(contour),)
             if .024 * width <= min(w, h) <= max(w, h) <= .069 * width
             and .78 <= w / h <= 1.45 and .44 <= cv2.contourArea(contour) / (w * h) <= .87]
    for a, b in combinations(sorted(avatars, key=lambda item: item[1]), 2):
        if (abs(a[0] - b[0]) > .025 * width or not .125 * width <= b[1] - a[1] <= .56 * width
                or max(abs(a[2] - b[2]), abs(a[3] - b[3])) > .026 * width):
            continue
        for first_mark in marks:
            offset = first_mark[1] - a[1]
            if not -.038 * width <= offset <= .19 * width:
                continue
            if any(abs(m[0] - first_mark[0]) <= .022 * width and abs(m[1] - b[1] - offset) <= .047 * width
                   and max(abs(m[2] - first_mark[2]), abs(m[3] - first_mark[3])) <= .027 * width for m in marks):
                return True
    return False


def _active_comment_tab(gray, top, close, index):
    """Verify an outlined speech bubble and printed count on an active tab."""
    from inventory_list_rows import _Box
    from screen_comment_thread import _bubble_components
    from screen_counted_post_actions import _count_is_printed

    _, width = gray.shape
    close_y = (close[1] + close[3]) / 2
    y0, y1 = round(close_y), round(close_y + .115 * width)
    left_rail = gray[y0:y1, :round(.57 * width)]
    _, _, lines, _ = cv2.connectedComponentsWithStats(np.uint8(left_rail <= 117), connectivity=8)
    cache = {}
    for x, y, w, h, area in lines[1:]:
        if not (.093 * width <= w <= .35 * width and h <= .014 * width and area >= .73 * w * h
                and .043 * width <= y0 + y - close_y <= .098 * width):
            continue
        line = y0 + y
        start = max(top + 2, round(line - .119 * width))
        section = gray[start:line - max(2, round(.025 * width)), x:x + w]
        for _, (a, b, r, d) in _bubble_components(section, width, (63, 85, 108, 133)):
            bubble_h = d - b
            if not (.18 * w <= r <= .65 * w and .72 <= (r - a) / bubble_h <= 1.37
                    and abs(start + (b + d) / 2 - close_y) <= .024 * width):
                continue
            count_left = x + r + max(1, round(.16 * bubble_h))
            t = start + b - max(1, round(.12 * bubble_h))
            stop = start + d + max(1, round(.12 * bubble_h))
            tile = gray[t:stop, count_left:x + w]
            if tile.size == 0:
                continue
            u, v, cw, ch = cv2.boundingRect(np.uint8(tile <= 142))
            if .47 * bubble_h <= ch <= 1.16 * bubble_h and .21 <= cw / ch <= 3.1:
                box = _Box(count_left + u, t + v, count_left + u + cw, t + v + ch)
                if _count_is_printed(gray, box, index, cache):
                    return True
    return False


def _bright_backdrop_comment(frame, gray, index):
    height, width = frame.shape[:2]
    scale = height / len(gray)
    native = None
    found = []
    for top in _content_cuts(gray):
        if not _comment_rows(gray, top):
            continue
        if native is None:
            native = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boundary = math.floor(top * scale)
        tone = float(np.median(native[boundary:boundary + max(3, round(.018 * width))]))
        close = _close(native, boundary, tone)
        if close is not None and (_active_comment_tab(native, boundary, close, index)
                                  or _heading(frame, boundary, index, dark=False, weak_close=close)):
            found.append(BlockingOverlay("bottom_sheet", (0, boundary, width, height), close,
                                         close_style="literal", include_coarse_proposals=True, comment_sheet=True))
    return found[0] if len(found) == 1 else None


def _rounded_comment_with_faint_header(frame, gray, top, tone, index):
    """Verify low-contrast header ink when both rim corners locate the sheet."""
    height, width = frame.shape[:2]
    work_height, work_width = gray.shape
    if (tone < 229 or not _paired_sheet_corners(gray >= tone - 12, top, round(.065 * work_width))
            or not _comment_rows(gray, top)):
        return None
    native = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boundary = math.floor(top * height / work_height)
    pad = max(2, math.ceil(height / work_height))
    middle = native[boundary - pad:boundary + pad + 1, round(.10 * width):round(.90 * width)]
    leading = np.flatnonzero((middle >= tone - 7).mean(axis=1) >= .93)
    if leading.size == 0:
        return None
    boundary += int(leading[0]) - pad
    close = _close(native, boundary, tone, thresholds=(43, 61))
    if close is not None and _heading(frame, boundary, index, dark=False, weak_close=close, faint_ink=True):
        return BlockingOverlay("bottom_sheet", (0, boundary, width, height), close,
                               close_style="literal", include_coarse_proposals=True, comment_sheet=True)
    return None


def _comment_caption(gray, box, index, *, maximum_width):
    """Read a comment heading in a physically isolated title or selected tab."""
    left, top, right, bottom = box
    tile = gray[top:bottom, left:right]
    if min(tile.shape) < 6:
        return False
    contrast = cv2.subtract(int(np.median(tile)), tile)
    ink = contrast >= 46
    for a, b in _bands(ink.sum(axis=1) >= max(3, round(.011 * maximum_width))):
        if not max(5, .011 * maximum_width) <= b - a <= .055 * maximum_width:
            continue
        pad_y = max(3, round(.27 * (b - a)))
        for x, end in _word_windows(ink[a:b]):
            region = (max(0, x - 2), max(0, a - pad_y), min(tile.shape[1], end + 2), min(len(tile), b + pad_y))
            if _phrase(tile, region, _HEADINGS, index):
                return True
    return False


def _has_composer_pair(frame, panel):
    from screen_comment_composer_icons import comment_composer_icons

    return {item["role"] for item in comment_composer_icons(frame, panel)} >= {"mention", "emoji"}


def _comment_captioned_plate(frame, gray, index):
    """Find an open comment tab or a full-panel capture with verified input tools."""
    height, width = frame.shape[:2]
    work_h, work_w = gray.shape
    core = gray[:, round(.055 * work_w):round(.945 * work_w)]
    fill = (core >= 230).mean(axis=1)
    limit = max(2, round(.022 * work_w))
    candidates = [int(y) for y in np.flatnonzero((fill[1:] >= .93) & (fill[:-1] <= .53)) + 1
                  if .13 * work_h <= y <= .57 * work_h
                  and fill[y:y + limit].mean() >= .86
                  and (core[y - limit:y - 1] < 217).mean() >= .72]
    if (fill[:limit] >= .91).all():
        header = core[limit:round(.085 * work_w)]
        l, _, span, depth = cv2.boundingRect(np.uint8(header <= 160))
        if (.11 * work_w <= span <= .52 * work_w and .012 * work_w <= depth <= .057 * work_w
                and abs(l + span / 2 - core.shape[1] / 2) <= .047 * work_w):
            candidates.append(0)
    if not candidates:
        return None

    native = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for y in candidates:
        top = round(y * height / work_h)
        panel = 0, top, width, height
        close = None
        if top == 0:
            found = _comment_caption(native, (round(.27 * width), 0, round(.73 * width), round(.086 * width)),
                                      index, maximum_width=width)
        else:
            close = _close(native, top, float(np.median(core[y:y + max(2, limit)])), thresholds=(43, 61))
            if close is None:
                continue
            end = round(.29 * width)
            rail = native[top + round(.10 * width):top + end, :end]
            _, _, lines, _ = cv2.connectedComponentsWithStats(np.uint8(rail < 120))
            underline = [int(start) for x, start, w, h, area in lines[1:]
                         if x <= .060 * width and .075 * width <= w <= .27 * width
                         and h <= .014 * width and area >= .68 * w * h]
            if len(underline) != 1:
                continue
            line_y = top + round(.10 * width) + underline[0]
            found = _comment_caption(native, (0, line_y - round(.129 * width), end, line_y - 2),
                                      index, maximum_width=width)
        if found and _has_composer_pair(frame, panel):
            return BlockingOverlay("bottom_sheet", panel, close, close_style="literal" if close else None,
                                   include_coarse_proposals=True, comment_sheet=True)
    return None


def social_comment_surface(frame, font_index, *, working_view=None):
    """Return a foreground scope only when all three independent cues agree."""
    height, width = frame.shape[:2]
    if font_index is None or width < 160 or not 1.12 * width <= height <= 2.85 * width:
        return None
    _, gray = _working_view(frame) if working_view is None else working_view
    if gray is None:
        return None
    work_h, work_w = gray.shape
    scale_x, scale_y = width / work_w, height / work_h
    matches = []
    for start, tone in _surfaces(gray):
        close = _close(gray, start, tone)
        if close is None:
            if faint := _rounded_comment_with_faint_header(frame, gray, start, tone, font_index):
                matches.append(faint)
            continue
        top = math.floor(start * scale_y)
        if not _heading(frame, top, font_index, dark=tone <= 66):
            continue
        x, y, right, bottom = close
        dismiss = (math.floor(x * scale_x), math.floor(y * scale_y),
                   math.ceil(right * scale_x), math.ceil(bottom * scale_y))
        matches.append(BlockingOverlay("bottom_sheet", (0, top, width, height), dismiss, close_style="literal",
                                       comment_sheet=True))
    if matches:
        return matches[0] if len(matches) == 1 else None
    return _bright_backdrop_comment(frame, gray, font_index) or _comment_captioned_plate(frame, gray, font_index)
