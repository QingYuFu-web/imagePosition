"""Read visible digit keys and their owned controls in a foreground PIN dialog."""

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_dark_dialog_footer import _read_button
from screen_dialog_search_input import _fold
from screen_ime_completion import _check_stroke
from screen_landscape_search_field import _available
from screen_native_tab_bar import _distance


_DIGITS = frozenset("0123456789")
_PROMPTS = {"enterpin", "inputpin", "setpin", "confirmpin", "输入pin码", "输入pin", "輸入pin碼", "輸入pin"}
_PREFIXES = {"enter", "input", "set", "confirm"}
_PATTERNS = {"pattern", "图案", "圖案"}
_BIOMETRICS = {"biometrics", "biometric", "生物识别", "生物識別"}


def _symbol(chip, index, allowed, aliases=None):
    peak = float(np.percentile(chip, 98))
    if peak < 70:
        return None
    aliases = aliases or {}
    votes = {}
    for level in (.31, .47, .63):
        mask = np.uint8(chip >= level * peak)
        options = index.rank(mask, alphabet="code", limit=6)
        if not options:
            continue
        winner, distance = options[0]["text"], options[0]["distance"]
        name = aliases.get(winner, winner)
        if name not in allowed or distance > .207:
            continue
        if winner != name and not any(item["text"] == name and item["distance"] - distance <= .045 for item in options[1:]):
            continue
        next_score = next((item["distance"] for item in options if aliases.get(item["text"], item["text"]) != name), 1.)
        if next_score - distance >= .021:
            votes[name] = votes.get(name, 0) + 1
    return next((name for name, count in votes.items() if count >= 2), None) if len(votes) == 1 else None


def _pin_word(ink, index):
    columns = _bands((ink >= 56).any(axis=0))
    if len(columns) != 3:
        return False
    return all(_symbol(ink[:, a:b], index, {letter}) == letter
               for (a, b), letter in zip(columns, "PIN", strict=True))


def _tab_row(frame, panel, paper, index):
    left, top, right, bottom = panel
    width, height = right - left, bottom - top
    limits = [left + round(i * width / 3) for i in range(4)]
    start, stop = top + round(.021 * height), top + round(.072 * height)
    y0, y1 = top + round(.075 * height), top + round(.122 * height)
    marker = frame[y0:y1, left + round(.025 * width):right - round(.025 * width)]
    contrast = _distance(marker, paper)
    ink = np.uint8((contrast >= 52) & (np.ptp(marker, axis=2) >= 49))
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    underlines = [(x, y, w, h) for x, y, w, h, area in stats[1:]
                  if .04 * width <= w <= .21 * width and .003 * width <= h <= .015 * width and area >= .66 * w * h]
    if len(underlines) != 1:
        return None
    x, y, w, h = underlines[0]
    if abs(round(.025 * width) + x + w / 2 - width / 2) > .04 * width:
        return None
    accent = np.median(marker[y:y + h, x:x + w][ink[y:y + h, x:x + w] != 0], axis=0)
    captions = []
    for slot, (a, b) in enumerate(zip(limits, limits[1:])):
        inset = round(.075 * (b - a))
        crop = _distance(frame[start:stop, a + inset:b - inset], paper)
        x, y, w, h = cv2.boundingRect(np.uint8(crop >= 57))
        if (not .023 * width <= h <= .055 * width or not .036 * width <= w <= .28 * width
                or abs(x + inset + w / 2 - (b - a) / 2) > .045 * width):
            return None
        if slot == 1:
            if not _pin_word(crop[y:y + h, x:x + w], index):
                return None
            continue
        area = (a + inset + x - 4, start + y - 5, a + inset + x + w + 5, start + y + h + 5)
        reading = read_caption(frame, area, index, cache={}, choice=True, sentence_case=True)
        label = _fold(reading["text"]) if reading else ""
        purpose = "切换到图案" if label in _PATTERNS else "切换到生物识别" if label in _BIOMETRICS else None
        if purpose is None:
            return None
        box = (a + inset, top + round(.019 * height), b - inset, stop + round(.014 * height))
        captions.append(_key(box, purpose, "switch_tab"))
    return accent, captions


def _key_grid(ink, width, height):
    y0 = round(.38 * height)
    area = ink[y0:round(.87 * height)]
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(area >= 57))
    cells = []
    for x, y, w, h, n in stats[1:]:
        if not .0095 * width <= w <= .081 * width or not .028 * width <= h <= .064 * width or n < .095 * w * h:
            continue
        col = int((x + w / 2) * 3 / width)
        if abs(x + w / 2 - (col + .5) * width / 3) <= .045 * width:
            cells.append((col, (int(x), int(y) + y0, int(x + w), int(y + h) + y0)))
    rows = []
    for col, box in sorted(cells, key=lambda cell: (cell[1][1] + cell[1][3]) / 2):
        middle = (box[1] + box[3]) / 2
        if not rows or middle - rows[-1][0] > .025 * width:
            rows.append((middle, {}))
        if col in rows[-1][1]:
            return None
        rows[-1][1][col] = box
    if len(rows) != 4 or any(len(cells) != 3 for _, cells in rows[:3]) or 1 not in rows[-1][1]:
        return None
    steps = np.diff([middle for middle, _ in rows])
    pitch = np.median(steps)
    if not .115 * width <= pitch <= .237 * width or max(abs(steps - pitch)) > .11 * pitch:
        return None
    return rows, pitch


def _prompt(frame, panel, ink, first, pitch, index):
    left, top, right, bottom = panel
    width, height = right - left, bottom - top
    x0, x1 = round(.24 * width), round(.76 * width)
    y0, y1 = round(.193 * height), round(first - .82 * pitch)
    view = ink[y0:y1, x0:x1]
    active = view >= 58
    candidates = [(a, b) for a, b in _bands(active.sum(axis=1) >= max(3, round(.015 * width)))
                  if .027 * width <= b - a <= .062 * width]
    if len(candidates) != 1:
        return None
    start, stop = candidates[0]
    x, y, w, h = cv2.boundingRect(np.uint8(active[start:stop]))
    if not .13 * width <= w <= .48 * width or abs(x0 + x + w / 2 - width / 2) > .052 * width:
        return None
    col, row = x0 + x, y0 + start + y
    outer = (left + col - 4, top + row - 6, left + col + w + 5, top + row + h + 6)
    result = read_caption(frame, outer, index, cache={}, choice=True, sentence_case=True)
    if result is not None and _fold(result["text"]) in _PROMPTS:
        return row + h
    word = ink[row:row + h, col:col + w]
    bands = _bands((word >= 59).any(axis=0))
    gaps = [(a, b) for (_, a), (b, _) in zip(bands, bands[1:]) if b - a >= .36 * h]
    if len(gaps) != 1 or not _pin_word(word[:, gaps[0][1]:], index):
        return None
    prefix = (outer[0], outer[1], left + col + gaps[0][0] + 4, outer[3])
    reading = read_caption(frame, prefix, index, cache={}, choice=True, sentence_case=True)
    return row + h if reading is not None and _fold(reading["text"]) in _PREFIXES else None


def _masked_indicators(ink, after, first, pitch):
    width = ink.shape[1]
    y0, y1 = after + round(.035 * width), round(first - .41 * pitch)
    x0, x1 = round(.25 * width), round(.75 * width)
    if y1 <= y0:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(ink[y0:y1, x0:x1] >= 73))
    marks = [(x + w / 2, y + h / 2, w, h) for x, y, w, h, _ in stats[1:]
             if .011 * width <= min(w, h) <= max(w, h) <= .038 * width and .74 <= w / h <= 1.27]
    if not 3 <= len(marks) <= 8:
        return False
    marks.sort()
    xs, ys, ws, hs = zip(*marks, strict=True)
    steps = np.diff(xs)
    return (np.ptp(ys) <= .010 * width and np.ptp(ws) <= .007 * width and np.ptp(hs) <= .007 * width
            and np.ptp(steps) <= .021 * width and abs(x0 + (xs[0] + xs[-1]) / 2 - width / 2) <= .025 * width)


def _key(box, label, role):
    left, top, right, bottom = box
    return {"role": role, "function": label, "preview_text": label, "box": list(box),
            "click_point": [(left + right) // 2, (top + bottom) // 2]}


def _check(glyph, peak):
    stable = 0
    for level in (.33, .48, .63):
        mask = np.uint8(glyph >= level * peak)
        x, y, w, h = cv2.boundingRect(mask)
        if w >= 5 and h >= 5 and _check_stroke(mask[y:y + h, x:x + w]):
            stable += 1
    return stable >= 2


def _footer(frame, panel, paper, accent, last, index):
    left, top, right, bottom = panel
    width, height = right - left, bottom - top
    x0, x1 = left + round(.55 * width), right - round(.053 * width)
    y0, y1 = max(top + round(last + .080 * width), bottom - round(.118 * height)), bottom - round(.036 * height)
    if y1 <= y0:
        return None
    crop = frame[y0:y1, x0:x1]
    distances = _distance(crop, paper)
    matched = np.max(abs(crop.astype(np.int16) - accent), axis=2) <= 38
    x, y, w, h = cv2.boundingRect(np.uint8(matched & (distances >= 51)))
    if not .021 * width <= h <= .053 * width or not 1.6 * h <= w <= 5.6 * h:
        return None
    box = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
    if _read_button(frame, box, paper, index) != ("cancel", "取消"):
        return None
    dx, dy = round(.032 * width), round(.023 * width)
    return _key((box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy), "取消", "cancel")


def native_numeric_dialog_actions(frame, regions, font_index):
    """Emit recognized controls only within the foreground PIN keypad."""
    panels = regions.get("regions", ())
    if font_index is None or regions.get("layer") != "dialog" or len(panels) != 1:
        return []
    left, top, right, bottom = panel = tuple(map(int, panels[0]))
    width, height = right - left, bottom - top
    if width < 250 or not 1.25 * width <= height <= 1.96 * width:
        return []
    view = frame[top:bottom, left:right]
    center = view[round(.17 * height):round(.88 * height):7, round(.11 * width):round(.89 * width):7]
    paper = np.median(center, axis=(0, 1))
    if np.mean(paper) < 190 or (_distance(center, paper) <= 12).mean() < .955:
        return []
    ink = _distance(view, paper)
    grid = _key_grid(ink, width, height)
    if grid is None:
        return []
    rows, pitch = grid
    after = _prompt(frame, panel, ink, rows[0][0], pitch, font_index)
    if after is None or (tab := _tab_row(frame, panel, paper, font_index)) is None:
        return []
    accent, tabs = tab
    digits, actions = {}, list(tabs)
    for i, (y, cells) in enumerate(rows):
        for col, (a, b, c, d) in sorted(cells.items()):
            if i == 3 and col != 1:
                continue
            name = _symbol(ink[b:d, a:c], font_index, _DIGITS, {"O": "0"})
            if name is None or name in digits:
                return []
            x = round((col + .5) * width / 3)
            dx, dy = round(.074 * width), round(.263 * pitch)
            box = (left + x - dx, top + round(y) - dy, left + x + dx, top + round(y) + dy)
            digits[name] = (float(np.percentile(ink[b:d, a:c], 98)), box)
    if digits.keys() != _DIGITS:
        return []
    minimum = max(105, .71 * np.median([peak for peak, _ in digits.values()]))
    actions.extend(_key(box, f"输入数字{name}", "pin_digit") for name, (peak, box) in digits.items()
                   if peak >= minimum)
    if _masked_indicators(ink, after, rows[0][0], pitch):
        # An owned C and tick have this meaning only in the verified PIN mode.
        last, cells = rows[-1]
        for col, purpose, role in ((0, "删除一位PIN", "pin_backspace"), (2, "确认PIN", "confirm")):
            if col not in cells:
                continue
            a, b, c, d = cells[col]
            glyph = ink[b:d, a:c]
            peak = float(np.percentile(glyph, 98))
            if peak >= minimum and (col == 0 and _symbol(glyph, font_index, {"C"}) == "C"
                                    or col == 2 and _check(glyph, peak)):
                x = round((col + .5) * width / 3)
                dx, dy = round(.074 * width), round(.263 * pitch)
                hit = (left + x - dx, top + round(last) - dy, left + x + dx, top + round(last) + dy)
                actions.append(_key(hit, purpose, role))
    cancel = _footer(frame, panel, paper, accent, rows[-1][0], font_index)
    if cancel is not None:
        actions.append(cancel)
    return [item for item in actions if _available(item["box"], regions)]
