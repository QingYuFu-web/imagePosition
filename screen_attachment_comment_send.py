"""Verify an enabled send button owned by a light attachment comment editor."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_comment_sheet_modal import _phrase
from screen_component_captions import _bands
from screen_dialog_close import _cross_footprint
from screen_landscape_search_field import _available


_HEADINGS = ("留言", "评论", "評論")
_WRITING = ("写留言", "写评论", "寫留言", "寫評論", "发表评论", "發表評論")
_SEND = ("发送", "發送", "发布", "發佈")


def _carrier(frame, gray, panel):
    left, top, right, bottom = panel
    width = right - left
    x, y = left + round(.60 * width), bottom - round(.196 * width)
    tile = frame[max(top, y):bottom, x:right]
    if min(tile.shape[:2], default=0) < 9:
        return None
    dark, pale = tile.min(axis=2), tile.max(axis=2)
    hue = cv2.subtract(pale, dark)
    choices = []
    for cut in (59, 83):
        mask = np.uint8((hue >= cut) & (dark <= 157))
        outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in outlines:
            l, t, w, h = cv2.boundingRect(contour)
            if (not .058 * width <= h <= .119 * width or not 1.34 <= w / h <= 3.18
                    or not .008 * width <= bottom - (y + t + h) <= .098 * width
                    or not 0 < l < l + w < tile.shape[1] or t == 0
                    or not .78 <= cv2.contourArea(contour) / (w * h) <= .98
                    or (mask[t:t + h, l:l + w] > 0).mean() < .69):
                continue
            box = x + l, y + t, x + l + w, y + t + h
            pad = max(3, round(.11 * h))
            a, b, r, d = box
            gap = max(1, round(.041 * h))
            border = gray[b - pad - gap:b - gap, a + pad:r - pad]
            if border.size and np.median(border) >= 229 and (border >= 218).mean() >= .86:
                choices.append((cut, box))
    stable = [box for cut, box in choices if any(cut != other and max(abs(a - b) for a, b in zip(box, peer, strict=True))
                                                <= max(2, .007 * width) for other, peer in choices)]
    if not stable or np.ptp(stable, axis=0).max() > max(3, .01 * width):
        return None
    return tuple(map(int, np.median(stable, axis=0)))


def _corner_remove(gray, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    x = left + round(.062 * w)
    t = top + round(.048 * h)
    for end, edge in ((.263, .30), (.276, .30), (.285, .30), (.294, .32)):
        corner = gray[t:top + round(end * h), x:left + round(edge * w)]
        for cut in (165, 183, 205):
            ink = np.uint8(corner >= cut)
            a, b, tw, th = cv2.boundingRect(ink)
            if (not .126 * h <= th <= .26 * h or not .80 <= tw / th <= 1.22
                    or not .12 <= ink.sum() / (tw * th) <= .45):
                continue
            mark = ink[b:b + th, a:a + tw] != 0
            if _cross_footprint(mark):
                return (a + tw / 2 + x - left) / w, (b + th / 2 + t - top) / h
    corner = gray[t:top + round(.293 * h), x:left + round(.325 * w)]
    for cut in (148, 169):
        lines = cv2.HoughLinesP(cv2.inRange(corner, cut, 255), 1, np.pi / 180,
                               max(4, round(.066 * h)), minLineLength=max(5, round(.13 * h)),
                               maxLineGap=max(1, round(.024 * h)))
        if lines is None:
            continue
        marks = []
        for a, b, c, d in lines[:, 0]:
            dx, dy = c - a, d - b
            if .087 * w <= abs(dx) <= .28 * w and .72 <= abs(dy / dx) <= 1.39:
                marks.append((int(np.sign(dx * dy)), (a + c) / 2, (b + d) / 2))
        for side, cx, cy in marks:
            for sign, px, py in marks:
                if sign != side and abs(px - cx) <= .045 * w and abs(py - cy) <= .046 * h:
                    return (x - left + (cx + px) / 2) / w, (t - top + (cy + py) / 2) / h
    return None


def _attachments(frame, gray, panel, button):
    left, top, right, _ = panel
    width = right - left
    start = max(top, button[1] - round(.34 * width))
    stop = button[1] - round(.024 * width)
    row = frame[start:stop, left:right]
    if min(row.shape[:2], default=0) < 15:
        return None
    mask = cv2.inRange(row.min(axis=2), 0, 192)
    outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    choices = []
    for contour in outlines:
        x, y, w, h = cv2.boundingRect(contour)
        if (not .076 * width <= h <= .166 * width or not .87 <= w / h <= 1.15
                or not 0 < x < x + w < row.shape[1] or not 0 < y < y + h < len(row)
                or not .047 * width <= button[1] - (start + y + h) <= .18 * width
                or cv2.contourArea(contour) < .72 * w * h):
            continue
        box = left + x, start + y, left + x + w, start + y + h
        mark = _corner_remove(gray, box)
        if mark is not None:
            choices.append((box, mark))
    for first, mark in choices:
        size = first[3] - first[1]
        peers = sorted(p for p, corner in choices if abs(first[1] - p[1]) <= .075 * size
                       and abs(first[3] - p[3]) <= .075 * size and abs(first[2] - first[0] - p[2] + p[0]) <= .13 * size
                       and max(abs(a - b) for a, b in zip(mark, corner, strict=True)) <= .095)
        for a, b, c in zip(peers, peers[1:], peers[2:]):
            gap, following = b[0] - a[2], c[0] - b[2]
            if .05 * size <= min(gap, following) <= max(gap, following) <= .34 * size and abs(gap - following) <= .11 * size:
                return first[1], size
    return None


def _line_groups(gray, box, contrast):
    l, t, r, b = box
    chip = gray[t:b, l:r]
    if min(chip.shape, default=0) < 8 or np.median(chip) < 215:
        return
    ink = cv2.subtract(int(np.median(chip)), chip) >= contrast
    for a, z in _bands(ink.sum(axis=1) >= max(3, round(.016 * chip.shape[1]))):
        h = z - a
        if not .21 * len(chip) <= h <= .76 * len(chip):
            continue
        line = ink[a:z]
        spans = [(x, end) for x, end in _bands(line.any(axis=0))
                 if end - x > .16 * h or np.count_nonzero(line[:, x:end].any(axis=1)) >= .63 * h]
        groups = []
        for x, end in spans:
            if groups and x - groups[-1][1] <= .35 * h:
                groups[-1] = groups[-1][0], end
            else:
                groups.append((x, end))
        pad = max(4, round(.29 * h))
        for x, end in groups:
            pad_x = max(3, round(.10 * h))
            if not 1.46 * h <= end - x <= 5.7 * h:
                continue
            yield max(l, l + x - pad_x), max(t, t + a - pad), min(r, l + end + pad_x), min(b, t + z + pad)


def _editor_geometry(gray, panel, attachments):
    left, _, right, _ = panel
    span = right - left
    top, tile = attachments
    a, b = top - round(1.37 * tile), top - round(.68 * tile)
    if a < panel[1]:
        return None
    keep = cv2.inRange(gray[a:b, left + round(.17 * span):right - round(.043 * span)], 207, 255)
    raster = gray[a:b, left + round(.17 * span):right - round(.043 * span)][:, ::3].astype(np.int16)
    step = max(2, round(.027 * tile))
    bands = (raster[:-step] - raster[step:] >= 8).mean(axis=1)
    seams = np.flatnonzero(bands >= .585)
    if len(seams) == 0 or keep.mean() < 230:
        return None
    seam = a + int(seams[-1]) + step
    # The writing lane is white above its own thumbnail rail. Search for its
    # printed prompt below the separate divider, not in the underlying list.
    return (left + round(.12 * span), seam + round(.22 * tile), right - round(.21 * span), top - round(.105 * tile))


def _send_label(gray, box, index):
    l, t, r, b = box
    h = b - t
    horizontal, vertical = round(.15 * h), round(.12 * h)
    chip = cv2.bitwise_not(gray[t + vertical:b - vertical, l + horizontal:r - horizontal])
    return _phrase(chip, (0, 0, chip.shape[1], chip.shape[0]), _SEND, index)


def _writing_prompt(frame, gray, lane, index):
    for box in reversed(list(_line_groups(gray, lane, 38))):
        if _phrase(gray, box, _WRITING, index):
            return True
        reading = read_caption(frame, box, index, cache={}, choice=True)
        if reading is not None and reading["text"] in _WRITING and reading["support"] >= 3:
            return True
    return False


def attachment_comment_send_actions(frame, regions, index):
    """Return only a visible enabled Send with independently owned attachments."""
    height, width = frame.shape[:2]
    scopes = regions.get("regions", ())
    if (index is None or regions.get("layer") != "bottom_sheet" or len(scopes) != 1
            or width < 270 or not 1.19 * width <= height <= 2.88 * width):
        return []
    panel = tuple(map(round, scopes[0]))
    left, top, right, bottom = panel
    span = right - left
    if not (0 <= left < right <= width and 0 <= top < bottom <= height and span >= .88 * width
            and bottom - top >= .58 * span and bottom >= height - .087 * span):
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    button = _carrier(frame, gray, panel)
    if button is None or not _available(button, regions):
        return []
    attached = _attachments(frame, gray, panel, button)
    if attached is None or (writing := _editor_geometry(gray, panel, attached)) is None:
        return []
    heading = (left + round(.030 * span), top + round(.072 * span),
               left + round(.31 * span), top + round(.158 * span))
    if not _available(heading, regions) or not _available(writing, regions):
        return []
    group = next(_line_groups(gray, heading, 51), None)
    if (group is None or _phrase(gray, group, _HEADINGS, index) is None
            or not _writing_prompt(frame, gray, writing, index)
            or _send_label(gray, button, index) is None):
        return []
    a, b, c, d = button
    return [{"role": "comment_send", "function": "发送评论", "preview_text": "发送",
             "box": list(button), "click_point": [(a + c) // 2, (b + d) // 2]}]
