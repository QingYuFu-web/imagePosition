"""Match captions only inside a painted command button or a proven tile."""

import cv2
import numpy as np

from screen_candidate_glyphs import _overlap, _search
from screen_caption_transcript import read_caption
from screen_component_captions import _contrast, _rectangular_button
from screen_short_labels import _VISIBLE_VERBS, _center, _text_key


_EXPLICIT = {
    **_VISIBLE_VERBS, "next": "下一步", "下一步": "下一步", "start": "开始", "开始": "开始", "開始": "开始",
    "save": "保存", "保存": "保存", "submit": "提交", "提交": "提交",
    "apply": "应用", "应用": "应用", "應用": "应用",
    "done": "完成", "完成": "完成", "agree": "同意", "iagree": "同意", "同意": "同意", "accept": "接受", "接受": "接受",
}


def _separated_edge(inner, outer):
    difference = np.abs(inner.astype(np.int16) - outer).max(axis=1)
    return np.mean(difference >= 34) >= .78 and np.median(difference) >= 50


def _button_paint(frame, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    step = max(2, round(.075 * height))
    if min(left, top, frame.shape[1] - right, frame.shape[0] - bottom) < step:
        return False
    xs = np.arange(left + round(.19 * width), right - round(.19 * width), step)
    ys = np.arange(top + round(.20 * height), bottom - round(.20 * height), max(2, step // 2))
    # Model the fill as a coherent surface from its outer columns even when
    # its tone varies down the button. Do not compare against center text.
    first = frame[ys, left + round(.19 * width)]
    second = frame[ys, right - round(.19 * width)]
    if (np.abs(first.astype(np.int16) - second).max(axis=1) <= 11).mean() < .90:
        return False
    paint = np.median(first, axis=0)
    if np.ptp(paint) < 54:
        return False
    pairs = ((frame[top + step, xs], frame[top - step, xs]),
             (frame[bottom - step, xs], frame[bottom + step - 1, xs]),
             (frame[ys, left + step], frame[ys, left - step]),
             (frame[ys, right - step], frame[ys, right + step - 1]))
    return all(_separated_edge(inside, outside) for inside, outside in pairs)


def _command_text(frame, bounds, index, cache, *, horizontal):
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    if horizontal:
        # Locate the caption against corresponding, text-free rows, then
        # read the actual ink. This works for both a compact capsule with
        # full-width text and a large CTA with small lettering.
        x0, x1 = left + round(.14 * width), right - round(.14 * width)
        y0, y1 = top + round(.15 * height), bottom - round(.15 * height)
        tile = frame[y0:y1, x0:x1]
        surface = frame[y0:y1, left + round(.10 * width)][:, None]
        off = np.abs(tile.astype(np.int16) - surface).max(axis=2)
        a, b, w, h = cv2.boundingRect(np.uint8(off >= 53))
        if not .16 * height <= h <= .56 * height or not .43 * h <= w <= 8.6 * h:
            return None
        if abs(x0 + a + w / 2 - (left + right) / 2) > .08 * width:
            return None
        pad = max(3, round(.17 * h))
        x0, y0, x1, y1 = max(x0, x0 + a - pad), max(y0, y0 + b - pad), min(x1, x0 + a + w + pad), min(y1, y0 + b + h + pad)
    else:
        x0, y0, x1, y1 = bounds
    evidence = read_caption(frame, (x0, y0, x1, y1), index, cache=cache, choice=True,
                            sentence_case=True, join_radicals=True, join_letters=True)
    if not evidence or evidence["max_distance"] > .185 or evidence["min_margin"] < .026:
        return None
    a, b, c, d = evidence["line_box"]
    mx, my = (a + c) / 2, (b + d) / 2
    if horizontal:
        if (abs(mx - (left + right) / 2) > .08 * width or abs(my - (top + bottom) / 2) > .11 * height
                or not .16 * height <= d - b <= .56 * height):
            return None
        # Reject a visible secondary line or mark anywhere else on the mid
        # button plane. Sample corresponding rows from both edge columns so
        # gradients contribute no false ink.
        outer_left, outer_right = round(.10 * width), round(.90 * width)
        mid_top, mid_bot = top + round(.15 * height), bottom - round(.15 * height)
        tile = frame[mid_top:mid_bot, left + outer_left:right - outer_left]
        paint = frame[mid_top:mid_bot, left + outer_right][:, None]
        delta = np.abs(tile.astype(np.int16) - paint).max(axis=2) >= 58
        pad = max(2, round(.095 * (d - b)))
        delta[max(0, b - mid_top - pad):d - mid_top + pad,
              max(0, a - left - outer_left - pad):c - left - outer_left + pad] = False
        if delta.mean() > .014 or np.count_nonzero(delta) >= .20 * (c - a) * (d - b):
            return None
    return evidence["text"], (a, b, c, d)


def _painted_candidates(frame, boxes, index, known, cache):
    short = min(frame.shape[:2])
    gray = None
    for box in boxes:
        width, height = box[2] - box[0], box[3] - box[1]
        if (not .26 * short <= width <= .985 * frame.shape[1]
                or not .034 * short <= height <= .18 * short or not 2.20 <= width / height <= 12
                or not _unclaimed(box, _center(box), known)):
            continue
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bounds = _rectangular_button(gray, box)
        if bounds is None or not _button_paint(frame, bounds):
            continue
        reading = _command_text(frame, bounds, index, cache, horizontal=True)
        if reading is None:
            continue
        text, _ = reading
        name = _EXPLICIT.get(_text_key(text))
        if name:
            role = "search" if name == "搜索" else "labeled_control"
            yield {"role": role, "function": name, "box": list(bounds),
                   "click_point": _center(bounds), "text": text, "preview_text": text}


def _disk(frame, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    tile = frame[top:top + round(.68 * height), left:right]
    margin = round(.12 * width)
    sides = np.concatenate((tile[:, :margin], tile[:, -margin:]), axis=1)
    base = np.median(sides[::2, ::2], axis=(0, 1))
    if (np.abs(sides.astype(np.int16) - base).max(axis=2) <= 9).mean() < .89:
        return None
    difference = np.abs(tile.astype(np.int16) - base).max(axis=2)
    reads = []
    for level in (13, 19, 25):
        ink = np.uint8(difference >= level)
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        options = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area, arc = cv2.contourArea(contour), cv2.arcLength(contour, True)
            if (not .41 * width <= w <= .79 * width or not .38 * height <= h <= .64 * height
                    or not .87 <= w / h <= 1.15 or not .73 <= area / (w * h) <= .82
                    or abs(x + w / 2 - width / 2) > .09 * width or y > .11 * height
                    or 4 * np.pi * area < .86 * arc * arc):
                continue
            inset = round(.14 * min(w, h))
            core = tile[y + inset:y + h - inset, x + inset:x + w - inset]
            contrast, paint = _contrast(core)
            if np.abs(paint - base).max() < 19 or (contrast <= 12).mean() < .54:
                continue
            ink = np.uint8(contrast >= 62)
            a, b, dx, dy = cv2.boundingRect(ink)
            if (not .23 * w <= dx <= .62 * w or not .23 * h <= dy <= .62 * h
                    or abs(a + dx / 2 - core.shape[1] / 2) > .095 * w
                    or abs(b + dy / 2 - core.shape[0] / 2) > .095 * h):
                continue
            piece = ink[b:b + dy, a:a + dx]
            options.append(((left + x, top + y, left + x + w, top + y + h), piece, paint, base))
        if len(options) == 1:
            reads.extend(options)
    if len(reads) >= 2 and all(_overlap(reads[0][0], other[0]) >= .93 for other in reads[1:]):
        return reads[len(reads) // 2]
    return None


def _disc_rows(boxes, frame):
    height, width = frame.shape[:2]
    tiles = sorted([box for box in boxes if .12 * width <= box[2] - box[0] <= .25 * width
                    and .15 * width <= box[3] - box[1] <= .29 * width
                    and .66 <= (box[2] - box[0]) / (box[3] - box[1]) <= 1.15
                    and .07 * height <= _center(box)[1] <= .34 * height])
    visited = set()
    for box in tiles:
        _, cy = _center(box)
        unit = box[3] - box[1]
        row = tuple(peer for peer in tiles if abs(_center(peer)[1] - cy) <= .085 * unit
                    and .87 * unit <= peer[3] - peer[1] <= 1.16 * unit
                    and .87 * (box[2] - box[0]) <= peer[2] - peer[0] <= 1.15 * (box[2] - box[0]))
        if row in visited or not 3 <= len(row) <= 6:
            continue
        visited.add(row)
        step = np.diff([_center(peer)[0] for peer in row])
        if np.ptp(step) > .12 * unit or not .79 * unit <= step.min() <= step.max() <= 1.37 * unit:
            continue
        if any(a[2] > b[0] for a, b in zip(row[:-1], row[1:], strict=True)):
            continue
        if row[-1][2] - row[0][0] < .57 * width:
            continue
        yield row


def _tile_candidates(frame, boxes, index, known, cache):
    for row in _disc_rows(boxes, frame):
        discs = [_disk(frame, box) for box in row]
        if any(disk is None for disk in discs):
            continue
        disc_bounds = [disk[0] for disk in discs]
        unit = np.median([bounds[3] - bounds[1] for bounds in disc_bounds])
        shades = np.stack([disk[2] for disk in discs])
        base = np.stack([disk[3] for disk in discs])
        if (max(np.ptp(shades, axis=0)) > 9 or max(np.ptp(base, axis=0)) > 9
                or np.ptp([_center(bounds)[1] for bounds in disc_bounds]) > .09 * unit):
            continue
        if sum(_search(piece) for _, piece, _, _ in discs) != 1:
            continue
        text_rows = []
        for box, (bounds, _, _, background) in zip(row, discs, strict=True):
            width = box[2] - box[0]
            left, top, right, bottom = box[0] + round(.07 * width), bounds[3] + round(.12 * unit), box[2] - round(.07 * width), box[3] - round(.06 * unit)
            if bottom <= top:
                break
            patch = frame[top:bottom, left:right]
            values = np.abs(patch.astype(np.int16) - background).max(axis=2)
            a, b, w, h = cv2.boundingRect(np.uint8(values >= 55))
            if (not .10 * unit <= h <= .31 * unit or not .29 * unit <= w <= 1.31 * unit
                    or abs(left + a + w / 2 - _center(bounds)[0]) > .12 * unit):
                break
            text_rows.append((left, top, right, bottom, top + b + h / 2))
        if len(text_rows) != len(row) or np.ptp([sample[-1] for sample in text_rows]) > .085 * unit:
            continue
        for box, (bounds, piece, _, _), label in zip(row, discs, text_rows, strict=True):
            point = _center(bounds)
            if not _search(piece) or not _unclaimed(box, point, known):
                continue
            reading = _command_text(frame, label[:4], index, cache, horizontal=False)
            if reading is None or _EXPLICIT.get(_text_key(reading[0])) != "搜索":
                continue
            text, _ = reading
            yield {"role": "search", "function": "搜索", "box": list(box), "click_point": point,
                   "text": text, "preview_text": text}


def _unclaimed(box, point, active):
    x, y = point
    return not any(_overlap(box, previous["box"]) >= .58
                   or max(abs(a - b) for a, b in zip((x, y), previous["click_point"])) <= .28 * (box[3] - box[1])
                   for previous in active)


def painted_command_actions(frame, regions, index, known=()):
    """Offer one command per owned button while preserving confirmed actions."""
    height, width = frame.shape[:2]
    layer = regions["layer"]
    if (index is None or layer not in {"page", "dialog", "bottom_sheet"}
            or (layer == "page" and regions["regions"])):
        return []
    if layer == "page":
        scope = 0, 0, width, height
        visibility = "page"
    elif regions["regions"]:
        scope = regions["regions"][0]
        visibility = "foreground"
    else:
        return []
    boxes = [tuple(row["box"]) for row in regions["candidates"] if row.get("visibility") == visibility
             and scope[0] <= row["box"][0] < row["box"][2] <= scope[2]
             and scope[1] <= row["box"][1] < row["box"][3] <= scope[3]]
    cache, output = {}, []
    for detector in (_painted_candidates, _tile_candidates) if layer == "page" else (_painted_candidates,):
        for action in detector(frame, boxes, index, (*known, *output), cache):
            x, y = action["click_point"]
            if any(a <= x < c and b <= y < d for a, b, c, d in regions.get("floating_controls", ())):
                continue
            if _unclaimed(action["box"], action["click_point"], output):
                output.append(action)
    return output
