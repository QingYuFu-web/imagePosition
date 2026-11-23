"""Read explicitly opened comment pages from their controls and text."""

import cv2
import numpy as np

from screen_comment_sheet_modal import _phrase
from screen_component_captions import _bands
from screen_landscape_search_field import _available


_PROMPTS = ("说点什么", "说点什么…", "说点什么...", "发表评论", "写评论", "添加评论", "寫評論")


def _gray_signal(tile):
    return cv2.subtract(int(np.median(tile)), tile)


def _agree(found, gap):
    matches = [box for level, box in found if any(other != level and max(abs(a - b) for a, b in
               zip(box, peer, strict=True)) <= gap for other, peer in found)]
    if matches and np.ptp(matches, axis=0).max() <= 2 * gap:
        return tuple(np.median(matches, axis=0).astype(int))
    return None


def _back(gray):
    height, width = gray.shape
    t, stop = round(.013 * height), round(.104 * height)
    tile = gray[t:stop, :round(.075 * width)]
    if np.median(tile) < 209 or (tile >= 222).mean() < .66:
        return None
    signal = _gray_signal(tile)
    found = []
    for level in (32, 48, 64, 82):
        _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(signal >= level))
        for i, (x, y, w, h, n) in enumerate(stats[1:], 1):
            if (not .030 * width <= h <= .073 * width or not .42 <= w / h <= .74
                    or not .08 <= n / (w * h) <= .37 or .9 * x + w <= .028 * width):
                continue
            yy, xx = np.nonzero(ids[y:y + h, x:x + w] == i)
            mean = np.bincount(yy, weights=xx, minlength=h) / np.maximum(1, np.bincount(yy, minlength=h))
            ideal = np.abs(np.linspace(-1, 1, h))
            if np.sqrt(np.mean((mean / max(1, w - 1) - ideal) ** 2)) < .147:
                found.append((level, (x, y + t, x + w, y + t + h)))
    return _agree(found, 2)


def _pill(gray):
    height, width = gray.shape
    start = height - round(.158 * width)
    tile = gray[start:, :round(.93 * width)]
    if np.median(gray[start:, ::4]) < 226:
        return None
    found = []
    for threshold in (244, 247, 250):
        mask = np.uint8((tile >= 216) & (tile <= threshold))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (.60 * width <= w <= .83 * width and .074 * width <= h <= .143 * width
                    and .019 * width <= x <= .083 * width and .006 * width <= len(tile) - y - h <= .047 * width
                    and .85 <= cv2.contourArea(contour) / (w * h) <= .98):
                continue
            mark = np.zeros((h, w), np.uint8)
            cv2.drawContours(mark, [contour], 0, 1, cv2.FILLED, offset=(-x, -y))
            corner = max(3, round(.19 * h))
            shapes = (mark[:corner, :corner], mark[:corner, -corner:], mark[-corner:, :corner], mark[-corner:, -corner:])
            if any(blob.mean() > .78 for blob in shapes):
                continue
            found.append((threshold, (x, start + y, x + w, start + y + h)))
    return _agree(found, max(2, round(.005 * width)))


def _no_dots(gray, box):
    x, y, r, b = box
    window = gray[y:b, x:r]
    ink = _gray_signal(window)
    mask = ink >= max(19, .34 * np.percentile(ink, 98))
    _, _, _, size = cv2.boundingRect(np.uint8(mask))
    spans = _bands(mask.any(axis=0))
    if len(spans) <= 4:
        return box
    end = spans[-3][0]
    if all(cv2.boundingRect(np.uint8(mask[:, a:z]))[3] <= .25 * size and z - a <= .25 * size
           for a, z in spans[-3:]) and end - spans[-4][1] >= .08 * size:
        return x, y, x + end, b
    return box


def _comment_count(gray, index):
    height, width = gray.shape
    x, right = round(.10 * width), round(.35 * width)
    top, bottom = round(.55 * height), round(.75 * height)
    tile = gray[top:bottom, x:right]
    ink = _gray_signal(tile)
    pad = max(3, round(.010 * width))
    marks = [((x, top + start - pad, right, top + stop + pad), start, stop)
             for start, stop in _bands((ink >= 55).sum(axis=1) >= .030 * width)
             if .026 * width <= stop - start <= .060 * width and start >= pad and stop + pad < len(tile)]
    for box, a, b in reversed(marks[-4:]):
        line = ink[a:b]
        _, _, w, h = cv2.boundingRect(np.uint8(line >= 63))
        if 2.8 * h <= w <= 6.4 * h and _phrase(gray, box, ("评论", "評論"), index, piece="suffix"):
            return box
    return None


def _header_comment(gray, frame, index):
    height, width = gray.shape
    top, end = round(.08 * height), round(.19 * height)
    tile = frame[top:end, :round(.58 * width)]
    blue, green, red = cv2.split(tile)
    vivid = (red >= 154) & (cv2.subtract(red, cv2.max(blue, green)) >= 34)
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(vivid))
    lines = [(x, top + y, w, h) for x, y, w, h, n in stats[1:]
             if .045 * width <= w <= .123 * width and .002 * width <= h <= .012 * width
             and w >= 7 * h and n >= .69 * w * h and y >= .025 * height]
    if len(lines) != 1:
        return None
    x, y, w, _ = lines[0]
    box = (max(0, x - round(.23 * w)), y - round(.88 * w),
           x + round(.78 * w), y - round(.12 * w))
    return box if _phrase(gray, box, ("评论", "評論"), index) else None


def _bubble_components(tile, width, levels):
    signal = _gray_signal(tile)
    for level in levels:
        binary = np.uint8(signal >= level)
        curves, tree = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if tree is None:
            continue
        for contour, (_, _, child, parent) in zip(curves, tree[0], strict=True):
            x, y, w, h = cv2.boundingRect(contour)
            if (parent >= 0 or child < 0 or not .020 * width <= h <= .065 * width or not .88 <= w / h <= 1.28
                    or not .63 <= cv2.contourArea(contour) / (w * h) <= .94):
                continue
            if not .43 <= cv2.contourArea(curves[child]) / cv2.contourArea(contour) <= .83:
                continue
            filled = np.zeros((h, w), np.uint8)
            cv2.drawContours(filled, [contour], 0, 1, cv2.FILLED, offset=(-x, -y))
            n = max(2, round(.13 * h))
            delta = filled[-n:]
            side = max(2, round(.20 * w))
            core = delta[:, round(.24 * w):round(.76 * w)]
            if (core.mean() < .21 or max(delta[:, :side].mean(), delta[:, -side:].mean()) > .24):
                continue
            yield level, (x, y, x + w, y + h)


def _action_footer(gray, index):
    height, width = gray.shape
    span = (round(.84 * height), round(.965 * height))
    sampled = gray[:, ::max(2, round(.019 * width))].astype(np.int16)
    gap = max(2, round(.004 * width))
    a, b = span
    rise = sampled[a + gap:b + gap] - sampled[a - gap:b - gap]
    scores = (rise >= 6).mean(axis=1)
    i = int(np.argmax(scores))
    if scores[i] < .78 or np.median(sampled[a + i + gap]) < 241:
        return None
    top = a + i + gap
    left, right = round(.418 * width), round(.493 * width)
    y, down = top + round(.014 * width), top + round(.091 * width)
    bubble = _agree(list(_bubble_components(gray[y:down, left:right], width, (44, 56, 68))), 2)
    if bubble is None:
        return None
    xx, yy, rr, bb = bubble
    scale = bb - yy
    label = (left + rr + round(.25 * scale), y + yy - round(.32 * scale),
             left + rr + round(2.88 * scale), y + bb + round(.23 * scale))
    if not _phrase(gray, label, ("评论", "評論"), index):
        x, t, r, b = label
        tile = cv2.resize(gray[t:b, x:r], None, fx=.81, fy=.81, interpolation=cv2.INTER_AREA)
        if not _phrase(tile, (0, 0, tile.shape[1], len(tile)), ("评论", "評論"), index):
            return None
    return left + xx, y + yy, label[2], y + bb


def _reply(gray, header, regions):
    height, width = gray.shape
    a, z = round(.020 * width), round(.145 * width)
    start = header[3] + round(.082 * width)
    end = round(.42 * height)
    owners = [(a, start + y, z, start + b) for y, b in
              _bands((gray[start:end, a:z] < 180).mean(axis=1) >= .31)
              if .047 * width <= b - y <= .137 * width]
    if not owners or not _available(owners[0], regions):
        return None
    x, r = round(.73 * width), round(.93 * width)
    t, b = round(.176 * height), round(.72 * height)
    tile = gray[t:b, x:r]
    found = list(_bubble_components(tile, width, (25, 27, 32, 38, 43)))
    near = []
    for cut, box in found:
        if any(other != cut and max(abs(a - z) for a, z in zip(box, peer, strict=True)) <= 2 for other, peer in found):
            if not any(max(abs(a - z) for a, z in zip(box, peer, strict=True)) <= 3 for peer in near):
                near.append(box)
    if len(near) < 3:
        return None
    near.sort(key=lambda box: box[1])
    xs, ys = np.asarray([((l + r) / 2, (t + b) / 2) for l, t, r, b in near]).T
    if np.ptp(xs) > .071 * width or not .10 * width <= np.median(np.diff(ys)) <= .34 * width:
        return None
    candidates = [(x + back, t + top, x + right, t + down) for back, top, right, down in near
                  if .027 * width <= t + (top + down) / 2 - owners[0][3] <= .098 * width]
    return candidates[0] if len(candidates) == 1 else None


def comment_thread_actions(frame, regions, index):
    """Return comment controls on an independently proven visible thread."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or width < 300 or not 1.59 * width <= height <= 2.53 * width:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    back = _back(gray)
    if back is None or not _available(back, regions):
        return []
    reply = None
    if (field := _pill(gray)) is not None:
        x, y, r, b = field
        depth = b - y
        label_box = (x + round(.20 * depth), y + round(.16 * depth),
                     x + round(.53 * (r - x)), b - round(.16 * depth))
        label_box = _no_dots(gray, label_box)
        context = _comment_count(gray, index)
        if context is None or not _available(context, regions):
            return []
        name = _phrase(gray, label_box, _PROMPTS, index) if _available(field, regions) else None
    elif (context := _header_comment(gray, frame, index)) is not None and _available(context, regions):
        field = _action_footer(gray, index)
        if field is None:
            return []
        name = "评论" if _available(field, regions) else None
        reply = _reply(gray, context, regions)
    else:
        return []
    actions = []
    for role, function, box in (("back", "返回", back),
                                ("comment_input", "输入评论", field if name else None),
                                ("reply_comment", "回复第一条评论", reply)):
        if box is None or not _available(box, regions):
            continue
        x, y, r, b = map(int, box)
        actions.append({"role": role, "function": function, "box": [x, y, r, b],
                        "click_point": [(x + r) / 2, (y + b) / 2]})
    return actions
