"""Read a visible mobile amount keypad from key carriers and printed glyphs."""

import re

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_component_captions import _bands, _tight
from screen_foreground import _centered_cross
from screen_native_numeric_dialog import _symbol
from screen_native_tab_bar import _distance
from screen_rim_attached_dismiss import _cross_arms


_SAVE = {"保存", "儲存", "储存", "保存记录", "儲存記錄", "save"}
_DIGITS = frozenset("0123456789")
_MODES = {"收入": "切换到收入", "支出": "切换到支出", "income": "切换到收入", "expense": "切换到支出"}
_HAN_CAPTION = re.compile(r"[\u3400-\u9fff]{2,5}\Z")
_ANGLE = np.linspace(0, 2 * np.pi, 72, endpoint=False)
_SIN, _COS = np.sin(_ANGLE), np.cos(_ANGLE)


def _center(box):
    left, top, right, bottom = box
    return (left + right) // 2, (top + bottom) // 2


def _action(box, purpose, role):
    return {"role": role, "function": purpose, "box": list(box), "click_point": list(_center(box))}


def _white_grid(frame):
    height, width = frame.shape[:2]
    top = height - round(1.36 * width)
    nominal = min(288, width)
    scale = nominal / width
    edge = max(3, round(.017 * width))
    jump = max(1, width // 115)
    samples = np.column_stack((frame[top::jump * 3, :edge:jump], frame[top::jump * 3, -edge::jump]))
    light = samples.mean(axis=2)
    paper = np.median(light)
    if paper < 195 or (abs(light - paper) <= 9).mean() < .87:
        return None
    stride = max(1, width // (nominal * 2))
    small = cv2.resize(frame[top::stride, ::stride], (nominal, round((height - top) * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edge = max(3, round(.017 * nominal))
    sides = np.column_stack((gray[:, :edge], gray[:, -edge:]))
    paper = int(np.median(sides))
    if paper < 196 or (np.abs(sides.astype(np.int16) - paper) <= 8).mean() < .87:
        return None
    ink = np.uint8(cv2.medianBlur(gray, 3) <= paper - 8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    keys, footers = [], []
    for x, y, w, h, pixels in stats[1:]:
        if .245 * nominal <= w <= .323 * nominal and .128 * nominal <= h <= .227 * nominal and pixels >= .84 * w * h:
            keys.append((int(x), int(y), int(w), int(h)))
        if (.59 * nominal <= w <= .923 * nominal and .061 * nominal <= h <= .137 * nominal
                and pixels >= .83 * w * h and abs(x + w / 2 - nominal / 2) <= .033 * nominal):
            footers.append((int(x), int(y), int(w), int(h)))
    if len(keys) != 12:
        return None
    keys.sort(key=lambda box: box[1])
    rows = [sorted(keys[start:start + 3]) for start in range(0, 12, 3)]
    x, y, widths, heights = np.asarray(rows).transpose(2, 0, 1)
    widths, heights = np.median(widths), np.median(heights)
    if (np.ptp(x, axis=0).max() > .013 * nominal or np.ptp(y, axis=1).max() > .012 * nominal
            or np.ptp(np.diff(x, axis=1)) > .015 * nominal or np.ptp(np.diff(y, axis=0)) > .023 * nominal
            or max(abs(w - widths) for _, _, w, _ in keys) > .04 * widths
            or max(abs(h - heights) for _, _, _, h in keys) > .09 * heights
            or abs(x[0, 0] + x[0, -1] + widths - nominal) > .034 * nominal):
        return None
    end = max(y[-1]) + heights
    footers = [box for box in footers if .018 * nominal <= box[1] - end <= .107 * nominal
               and box[1] + box[3] <= len(gray) - .027 * nominal]
    if len(footers) != 1:
        return None

    def expand(box):
        x, y, w, h = box
        return (round(x / scale), top + round(y / scale), round((x + w) / scale), top + round((y + h) / scale))

    return [[expand(box) for box in row] for row in rows], expand(footers[0])


def _key_ink(frame, box):
    x, y = _center(box)
    left, top, right, bottom = box
    dx, dy = round(.23 * (right - left)), round(.32 * (bottom - top))
    crop = cv2.cvtColor(frame[y - dy:y + dy, x - dx:x + dx], cv2.COLOR_BGR2GRAY)
    paper = int(np.median(crop[::4, ::4]))
    ink = cv2.subtract(paper, crop)
    return ink, paper


def _digit(frame, box, index, expected, reference):
    ink, paper = _key_ink(frame, box)
    x, y, w, h = cv2.boundingRect(np.uint8(ink >= 52))
    if (abs(paper - reference) > 13 or not .23 * ink.shape[0] <= h <= .52 * ink.shape[0]
            or not .28 * h <= w <= .84 * h or abs(x + w / 2 - ink.shape[1] / 2) > .14 * h
            or abs(y + h / 2 - ink.shape[0] / 2) > .16 * h):
        return False
    glyph = ink[y:y + h, x:x + w]
    return _symbol(glyph, index, _DIGITS, {"O": "0"}) == expected


def _delete(frame, box):
    x, y = _center(box)
    height = box[3] - box[1]
    dx, dy = round(.29 * height), round(.20 * height)
    patch = frame[y - dy:y + dy, x - dx:x + dx]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    paper = int(np.median(gray[::3, ::3]))
    outer = np.uint8(gray < paper - 32)
    contours, _ = cv2.findContours(outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return False
    xx, yy, w, h = cv2.boundingRect(contours[0])
    if not .155 * height <= h <= .37 * height or not 1.11 <= w / h <= 1.92:
        return False
    body = np.zeros_like(outer)
    cv2.drawContours(body, contours, 0, 1, thickness=-1)
    # A backspace glyph has a pointed left side and straight right boundary.
    lo, hi = yy + h // 7, yy + 6 * h // 7
    left = np.argmax(body[lo:hi, xx:xx + w], axis=1)
    right = np.argmax(body[lo:hi, xx:xx + w][:, ::-1], axis=1)
    edge = max(1, h // 6)
    middle = left[len(left) // 3:2 * len(left) // 3]
    if (max(right) > .12 * w or min(np.median(left[:edge]), np.median(left[-edge:])) - np.median(middle) < .092 * w
            or min(xx, yy) < 2 or xx + w > patch.shape[1] - 2 or yy + h > patch.shape[0] - 2):
        return False
    inner = gray[yy + round(.17 * h):yy + round(.85 * h), xx + round(.29 * w):xx + round(.83 * w)]
    bright = np.uint8(inner > np.median(gray[body != 0]) + 56)
    gx, gy, gw, gh = cv2.boundingRect(bright)
    return (.35 * h <= gh <= .65 * h and .75 <= gw / gh <= 1.25
            and _cross_arms(bright[gy:gy + gh, gx:gx + gw]))


def _caption(frame, area, index, allowed, cache):
    reading = read_caption(frame, area, index, cache=cache, choice=True, join_radicals=True)
    if reading is None or reading["text"].casefold() not in allowed:
        return None
    return reading["text"]


def _save(frame, footer, index, cache):
    left, top, right, bottom = footer
    x, y = _center(footer)
    h = bottom - top
    before, after = frame[y, left + h:right - h:5].min(axis=1), frame[top - max(3, h // 7), left + h:right - h:5].min(axis=1)
    if np.median(after) - np.median(before) < 42:
        return None
    area = (x - round(1.35 * h), y - round(.37 * h), x + round(1.35 * h), y + round(.37 * h))
    return _action(footer, "保存", "confirm") if _caption(frame, area, index, _SAVE, cache) else None


def _close(frame):
    height, width = frame.shape[:2]
    right, bottom = round(.19 * width), round(.081 * width)
    top = round(.017 * width)
    crop = cv2.cvtColor(frame[top:bottom, :right], cv2.COLOR_BGR2GRAY)
    paper = int(np.median(crop[::3, ::3]))
    x, y, w, h = cv2.boundingRect(np.uint8(crop <= paper - 70))
    if (not .020 * width <= h <= .052 * width or not .87 <= w / h <= 1.15
            or not .048 * width <= x + w / 2 <= .144 * width or not min(x, y) >= 3):
        return None
    pad = max(3, round(.21 * h))
    if y + h + pad >= len(crop) or x + w + pad >= right or not _centered_cross(
            crop[y - min(pad, y):y + h + pad, x - min(pad, x):x + w + pad]):
        return None
    hit = (max(0, x - h // 2), max(0, top + y - h // 2), x + w + h // 2, min(height, top + y + h + h // 2))
    return _action(hit, "关闭记账", "close")


def _mode(frame, index, cache):
    height, width = frame.shape[:2]
    top, bottom = round(.28 * width), min(height, round(.55 * width))
    crop = frame[top:bottom]
    small = cv2.resize(crop, (min(width, 288), round((bottom - top) * min(width, 288) / width)),
                       interpolation=cv2.INTER_AREA)
    span = small.shape[1]
    gray = np.min(small, axis=2)
    paper = int(np.median(gray[:, ::7]))
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(cv2.medianBlur(gray, 3) <= paper - 13))
    pills = [(x, y, w, h) for x, y, w, h, n in stats[1:]
             if .205 * span <= w <= .335 * span and .061 * span <= h <= .102 * span and n >= .80 * w * h
             and .12 * span <= abs(x + w / 2 - span / 2) <= .203 * span]
    if len(pills) != 1:
        return None
    x, y, w, h = pills[0]
    scale = width / span
    left, upper, w, h = round(x * scale), top + round(y * scale), round(w * scale), round(h * scale)
    centers = (left + w // 2, width - left - w // 2)
    texts = []
    for cx in centers:
        y0, y1 = upper + round(.16 * h), upper + round(.84 * h)
        half = round(.36 * w)
        chip = frame[y0:y1, cx - half:cx + half]
        ink = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
        paper = int(np.median(ink[::3, ::3]))
        label = _tight(ink <= paper - 33, cx - half, y0)
        if label is None:
            return None
        l, t, r, b = label
        if not .27 * h <= b - t <= .54 * h or abs(l + r - 2 * cx) > .24 * h:
            return None
        texts.append(_caption(frame, (l - 6, t - 5, r + 6, b + 5), index, _MODES, cache))
    if not all(texts) or len({_MODES[text.casefold()] for text in texts}) != 2:
        return None
    cx = centers[1]
    target = (cx - w // 2, upper, cx + w // 2, upper + h)
    return _action(target, _MODES[texts[1].casefold()], "switch_tab"), target


def _category_row(frame, paper, strength, spans, start, stop, top):
    width = frame.shape[1]
    boxes = []
    for (left, right), distance in zip(spans, strength, strict=True):
        text = _tight(distance[start:stop] >= 34, left, top + start)
        if text is None:
            return []
        x, y, rx, by = text
        if not .016 * width <= by - y <= .042 * width or not 1.43 * (by - y) <= rx - x <= 4.63 * (by - y):
            return []
        start_icon = max(0, start - round(.151 * width))
        end_icon = start - round(.007 * width)
        if start_icon >= end_icon:
            return []
        potential = cv2.medianBlur(distance[start_icon:end_icon], 5)
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(potential >= 7))
        carriers = [(cx, cy, w, h) for cx, cy, w, h, n in stats[1:]
                    if .082 * width <= min(w, h) <= max(w, h) <= .151 * width
                    and .80 <= w / h <= 1.23 and n >= .53 * w * h and abs(2 * (left + cx) + w - x - rx) <= .045 * width
                    and .006 * width <= y - (top + start_icon + cy + h) <= .033 * width]
        if len(carriers) != 1:
            return []
        cx, cy, w, h = carriers[0]
        ix, iy = int(left + cx), int(top + start_icon + cy)
        ring = False
        for ratio in (.425, .45, .475):
            xs = np.rint(ix + w / 2 + ratio * w * _COS).astype(int)
            ys = np.rint(iy + h / 2 + ratio * h * _SIN).astype(int)
            pixels = frame[ys, xs].astype(np.int16)
            if (np.max(np.abs(pixels - paper), axis=1) >= 46).mean() >= .70:
                ring = True
                break
        centers = (ix + w // 2, iy + h // 2)
        owner = (ix, iy, ix + int(w), by)
        boxes.append((text, owner, centers, ring))
    return boxes


def _categories(frame, mode, keys_top, index, cache):
    width = frame.shape[1]
    top, bottom = mode[3] + round(.007 * width), keys_top - round(.16 * width)
    if not .28 * width <= bottom - top <= .70 * width:
        return []
    tile = frame[top:bottom]
    paper = np.median(tile[::4, ::8], axis=(0, 1)).astype(np.int16)
    options = []
    for count in (4, 3, 5):
        edges = np.linspace(0, width, count + 1).round().astype(int)
        pad = round(.025 * width)
        spans = [(left + pad, right - pad) for left, right in zip(edges, edges[1:])]
        strength = [_distance(tile[:, left:right], paper) for left, right in spans]
        present = sum((dist >= 35).sum(axis=1) >= round(.009 * width) for dist in strength) >= count - 1
        joined = cv2.morphologyEx(np.uint8(present).reshape(-1, 1), cv2.MORPH_CLOSE,
                                  np.ones((max(2, round(.007 * width)), 1), np.uint8)).ravel()
        rows = [_category_row(frame, paper, strength, spans, a, b, top) for a, b in _bands(joined)
                if .017 * width <= b - a <= .040 * width]
        rows = [row for row in rows if row]
        if not 1 <= len(rows) <= 3 or (len(rows) > 1 and any(np.ptp(
                [row[col][2][0] for row in rows]) > .027 * width for col in range(count))):
            continue
        flat = [cell for row in rows for cell in row]
        if sum(cell[3] for cell in flat) == 1:
            options.append(flat)
    if len(options) != 1:
        return []
    categories = []
    for text, box, point, selected in options[0]:
        if selected:
            continue
        left, top, right, bottom = text
        reading = read_caption(frame, (left - 5, top - 7, right + 5, bottom + 7), index,
                               cache=cache, choice=True, join_radicals=True)
        if reading is None or not _HAN_CAPTION.fullmatch(reading["text"]):
            continue
        categories.append({"role": "select_option", "function": f"选择{reading['text']}", "box": list(box),
                           "click_point": list(map(int, point))})
    return categories


def _calendar_glyph(ink):
    height, width = ink.shape
    pins = ink[:round(.14 * height)].mean(axis=0) > .41
    ranges = _bands(pins)
    if len(ranges) != 2:
        return False
    (a, b), (c, d) = ranges
    body = ink[round(.23 * height):round(.40 * height)]
    tail = ink[round(.84 * height):]
    interior = ink[round(.45 * height):round(.82 * height), round(.20 * width):round(.80 * width)]
    return (.07 * width <= b - a <= .28 * width and .07 * width <= d - c <= .28 * width
            and .20 * width <= c - b <= .66 * width and body.mean() >= .82 and max(tail.mean(axis=1)) >= .82
            and .15 <= interior.mean() <= .63)


def _note_glyph(frame, border):
    left, top, right, bottom = border
    height = bottom - top
    a, b = left + round(.17 * height), left + round(.75 * height)
    u, v = top + round(.24 * height), bottom - round(.24 * height)
    ink = cv2.cvtColor(frame[u:v, a:b], cv2.COLOR_BGR2GRAY)
    paper = np.median(ink)
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(ink <= paper - 48))
    marks = [(x, y, w, h) for x, y, w, h, n in stats[1:]
             if .14 * height <= w <= .45 * height and .018 * height <= h <= .071 * height and n >= .67 * w * h]
    if len(marks) != 3:
        return False
    lefts, tops, widths, heights = np.asarray(sorted(marks, key=lambda mark: mark[1])).T
    gaps = np.diff(tops)
    return (np.ptp(lefts) <= .036 * height and np.ptp(gaps) <= .041 * height and .07 * height <= min(gaps)
            and max(gaps) <= .19 * height and np.ptp(heights) <= .028 * height
            and abs(widths[0] - widths[1]) <= .08 * heights[0] + 2 and .42 <= widths[2] / widths[0] <= .82)


def _metadata(frame, categories_bottom, keys_top, index, cache):
    width = frame.shape[1]
    upper, lower = categories_bottom + round(.026 * width), keys_top - round(.009 * width)
    if not .128 * width <= lower - upper <= .252 * width:
        return []
    crop = frame[upper:lower].astype(np.int16)
    paper = np.median(crop[::4, ::8], axis=(0, 1))
    accent = np.uint8((np.max(abs(crop - paper), axis=2) >= 38) & (np.ptp(crop, axis=2) >= 35))
    contours, _ = cv2.findContours(accent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    borders = [(x, y, w, h) for contour in contours for x, y, w, h in (cv2.boundingRect(contour),)
               if .02 * width <= x <= .124 * width and .48 * width <= w <= .72 * width
               and .09 * width <= h <= .166 * width and cv2.contourArea(contour) >= .89 * w * h]
    if len(borders) != 1:
        return []
    x, y, w, h = borders[0]
    enclosed = accent[y + round(.15 * h):y + round(.85 * h), x + round(.06 * w):x + round(.94 * w)]
    if enclosed.mean() >= .14:
        return []
    note = (x, upper + y, x + w, upper + y + h)
    icons = [(a, b, c, d) for contour in contours for a, b, c, d in (cv2.boundingRect(contour),)
             if x + w + .018 * width <= a <= x + w + .10 * width and .035 * width <= d <= .069 * width
             and .79 <= c / d <= 1.25 and abs((2 * b + d) - (2 * y + h)) <= .12 * h
             and _calendar_glyph(accent[b:b + d, a:a + c])]
    if len(icons) != 1:
        return []
    a, b, c, d = icons[0]
    left, right = a + c + round(.008 * width), min(width, a + c + round(.156 * width))
    start, stop = max(0, b - round(.03 * h)), min(len(accent), b + d + round(.05 * h))
    date = _tight(accent[start:stop, left:right], left, upper + start)
    if date is None:
        return []
    lx, ty, rx, by = date
    if not 1.7 * (by - ty) <= rx - lx <= 7.7 * (by - ty):
        return []
    reading = read_caption(frame, (lx - 4, ty - 5, rx + 4, by + 5), index, cache=cache, choice=True, numeric=True)
    if reading is None or re.fullmatch(r"(?:0?[1-9]|1[0-2])[/.-](?:0?[1-9]|[12]\d|3[01])", reading["text"]) is None:
        return []
    # The gap between the two controls belongs to neither.
    date_button = a - round(.11 * h), note[1], rx + round(.12 * h), note[3]
    result = [_action(date_button, "选择日期", "select_date")]
    if _note_glyph(frame, note):
        result.append(_action(note, "编辑备注", "input"))
    return result


def amount_entry_keypad_actions(frame, regions, font_index):
    """Expose only proved controls of a full-page numeric entry surface."""
    height, width = frame.shape[:2]
    if font_index is None or regions.get("layer") != "page" or width < 280 or height < 1.75 * width:
        return []
    grid = _white_grid(frame)
    if grid is None:
        return []
    rows, footer = grid
    blank, paper = _key_ink(frame, rows[-1][0])
    if (blank >= 39).mean() > .003:
        return []
    keys = [box for row in rows[:3] for box in row] + [rows[-1][1]]
    order = "1234567890"
    if not all(_digit(frame, box, font_index, digit, paper) for box, digit in zip(keys, order, strict=True)):
        return []
    cache = {}
    save = _save(frame, footer, font_index, cache)
    if save is None:
        return []
    actions = [_action(box, f"输入数字{digit}", "numeric_key") for box, digit in zip(keys, order, strict=True)]
    actions.append(save)
    if _delete(frame, rows[-1][2]):
        actions.append(_action(rows[-1][2], "删除一位", "numeric_backspace"))
    close = _close(frame)
    if close is not None:
        actions.append(close)
    mode = _mode(frame, font_index, cache)
    if mode is not None:
        actions.append(mode[0])
        keys_top = rows[0][0][1]
        categories = _categories(frame, mode[1], keys_top, font_index, cache)
        actions.extend(categories)
        if categories:
            actions.extend(_metadata(frame, max(row["box"][3] for row in categories), keys_top, font_index, cache))
    return actions
