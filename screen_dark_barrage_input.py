"""Find dark barrage editors from their own painted controls and captions."""

import cv2
import numpy as np

from inventory_setting_switches import _stadium_mask
from screen_component_captions import _bands
from screen_landscape_paired_switches import _common_index, _fonts, _letter
from screen_live_chat_input import _PROMPTS


def _round_fields(mask, width, minimum, maximum):
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (.058 * width <= h <= .137 * width and minimum <= w / h <= maximum):
            continue
        measured = np.zeros((h, w), np.uint8)
        cv2.drawContours(measured, [contour], 0, 1, cv2.FILLED, offset=(-x, -y))
        ideal = _stadium_mask(h, w)
        if np.count_nonzero(measured & ideal) >= .89 * np.count_nonzero(measured | ideal):
            yield x, y, x + w, y + h


def _yellow(frame):
    blue, green, red = [channel.astype(np.int16) for channel in cv2.split(frame)]
    return (red >= 164) & (green >= 112) & (red - blue >= 69) & (green - blue >= 44)


def _reply_face(frame):
    h, w = frame.shape[:2]
    left = frame[:, :min(w, round(1.19 * h))]
    contours, _ = cv2.findContours(np.uint8(_yellow(left)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if (.43 * h <= min(width, height) <= max(width, height) <= .73 * h
                and .82 <= width / height <= 1.22 and .59 <= cv2.contourArea(contour) / (width * height) <= .85
                and .34 * h <= x + width / 2 <= .75 * h and abs(y + height / 2 - h / 2) <= .12 * h):
            faces.append(x + width)
    return faces[0] if len(faces) == 1 else None


def _printed(gray, index, phrases):
    contrast = np.maximum(gray.astype(np.float32) - np.median(gray), 0)
    minimum = max(31., .32 * float(np.percentile(contrast, 98)))
    ink = contrast >= minimum
    _, _, _, height = cv2.boundingRect(np.uint8(ink))
    if not 9 <= height <= .84 * len(gray):
        return None
    groups = []
    for first, last in _bands(ink.any(axis=0)):
        if groups and first - groups[-1][1] <= .082 * height and last - groups[-1][0] <= 1.03 * height:
            groups[-1] = (groups[-1][0], last)
        else:
            groups.append((first, last))
    if groups:
        a, b = groups[-1]
        if b - a <= .73 * height and cv2.boundingRect(np.uint8(ink[:, a:b]))[3] <= .48 * height:
            groups.pop()
    words = {word for word in phrases if len(word) == len(groups)}
    fonts = _fonts(index)
    if not words or not fonts:
        return None
    narrow = _common_index(fonts) if index.charset is None else index
    for pos, (a, b) in enumerate(groups):
        character = contrast[:, max(0, a - 1):b + 1]
        if not .48 * height <= b - a <= 1.21 * height:
            return None
        pixels = character.tobytes()
        reading = (_letter(narrow, character.shape, pixels, common=True)
                   or _letter(index, character.shape, pixels))
        words = {word for word in words if word[pos] == reading}
        if not words:
            return None
    return next(iter(words)) if len(words) == 1 else None


def _idle(frame, gray, index):
    height, width = gray.shape
    offset = max(0, height - round(.54 * width))
    stop = height - round(.16 * width)
    body = gray[offset:stop, :round(.65 * width)]
    base = int(np.median(body))
    if base > 63:
        return []
    options = []
    bounds = set(box for low, high in ((16, 29), (28, 48), (38, 62))
                 for box in _round_fields((body >= base + low) & (body <= base + high), width, 2.5, 5.6))
    for x, top, right, b in sorted(bounds):
        h = b - top
        t, bottom = offset + top, offset + b
        if not .015 * width <= x <= .14 * width or not .24 * width <= right - x <= .51 * width:
            continue
        tile = frame[t:bottom, x:right]
        face = _reply_face(tile)
        if face is None:
            continue
        roi = gray[t + round(.22 * h):bottom - round(.20 * h), x + face + round(.12 * h):right - round(.18 * h)]
        word = _printed(roi, index, _PROMPTS)
        if word is not None:
            options.append({"role": "comment_input", "function": "输入弹幕", "preview_text": word,
                            "box": [x, t, right, bottom], "click_point": [(x + right) // 2, (t + bottom) // 2]})
    return options if len(options) == 1 else []


def _emoji_rows(frame, pitch):
    height, width = frame.shape[:2]
    if height < 3.4 * pitch:
        return False
    yellow = _yellow(frame)
    rows = _bands(yellow.sum(axis=1) >= .105 * width)
    witnessed = 0
    for a, b in rows:
        if not .23 * pitch <= b - a <= .78 * pitch:
            continue
        columns = [(l + r) / 2 for l, r in _bands(yellow[a:b].sum(axis=0) >= max(2, round(.14 * (b - a))))
                   if .14 * pitch <= r - l <= .82 * pitch]
        if 5 <= len(columns) <= 11:
            gaps = np.diff(columns)
            if .76 * pitch <= np.median(gaps) <= 1.69 * pitch and np.count_nonzero(
                    np.abs(gaps - np.median(gaps)) <= .23 * pitch) >= len(gaps) - 1:
                witnessed += 1
    return witnessed >= 3


def _panel_rim(gray, top, size):
    start, stop = max(0, top - round(.66 * size)), top - round(.11 * size)
    reach = max(1, round(.039 * size))
    if start < reach or stop <= start:
        return None
    delta = gray[start + reach:stop + reach].astype(np.int16) - gray[start - reach:stop - reach]
    side = max(4, round(.10 * gray.shape[1]))
    rows = np.flatnonzero(((np.abs(delta) >= 6).mean(axis=1) >= .76)
                          & ((delta[:, :side] >= 6).mean(axis=1) >= .78)
                          & ((delta[:, -side:] >= 6).mean(axis=1) >= .78))
    return start + int(rows[len(rows) // 2]) if len(rows) else None


def _active_field(gray, top, right_button, bottom, panel):
    _, width = gray.shape
    size = bottom - top
    end = right_button - max(3, round(.11 * size))
    depth = max(2, round(.15 * size))
    origin = top - depth
    tile = gray[origin:bottom + depth, :end]
    paper = float(np.median(gray[panel:top, :round(.69 * width)]))
    options = []
    for left, t, right, b in _round_fields((tile >= paper + 10) & (tile <= paper + 45), width, 5, 10.5):
        if (.073 * width <= left <= .23 * width and .53 * width <= right - left <= .77 * width
                and .82 * size <= b - t <= 1.17 * size and abs(origin + t - top) <= .16 * size
                and .04 * size <= right_button - right <= .35 * size):
            options.append((left, origin + t, right, origin + b))
    return options[0] if len(options) == 1 else None


def _keyboard_switch(gray, field):
    left, top, _, bottom = field
    size = bottom - top
    chip = gray[top:bottom, max(0, left - round(1.19 * size)):left - round(.09 * size)]
    origin = max(0, left - round(1.19 * size))
    if not chip.size:
        return None
    paper = int(np.median(chip))
    contours, _ = cv2.findContours(np.uint8(chip >= paper + 30), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    choices = []
    for outline in contours:
        x, y, w, h = cv2.boundingRect(outline)
        if (not .49 * size <= min(w, h) <= max(w, h) <= .82 * size or not .85 <= w / h <= 1.17
                or not .63 <= cv2.contourArea(outline) / (w * h) <= .84
                or abs(y + h / 2 - size / 2) > .11 * size):
            continue
        x0, y0 = x + round(.15 * w), y + round(.27 * h)
        key = chip[y0:y + h - round(.27 * h), x0:x + w - round(.15 * w)] >= paper + 26
        a, t, u, v = cv2.boundingRect(np.uint8(key))
        if not .50 * w <= u <= .76 * w or not .23 * h <= v <= .43 * h:
            continue
        shaped = key[t:t + v, a:a + u]
        # A keyboard glyph has two printed key rows; the longer key on the lower row can join two columns.
        lanes = [sum(b - a >= 2 for a, b in _bands(line.any(axis=0)))
                 for line in (shaped[:round(.44 * v)], shaped[round(.59 * v):])]
        if min(lanes) >= 3:
            choices.append((origin + x, top + y, origin + x + w, top + y + h))
    return choices[0] if len(choices) == 1 else None


def _emoji_composer(frame, gray, index):
    height, width = gray.shape
    offset, stop = round(.39 * height), round(.71 * height)
    left = round(.67 * width)
    blue, green, red = [channel.astype(np.int16) for channel in cv2.split(frame[offset:stop, left:])]
    paint = (blue >= 163) & (blue - red >= 57) & (green - red >= 28)
    results = []
    for x, y, r, b in _round_fields(paint, width, 1.26, 2.60):
        x, y, r, b = x + left, y + offset, r + left, b + offset
        size = b - y
        if not .74 * width <= x < r <= .994 * width or not .10 * width <= r - x <= .239 * width:
            continue
        panel_top = _panel_rim(gray, y, size)
        if panel_top is None or not _emoji_rows(frame[b + round(.12 * size):], size):
            continue
        field = _active_field(gray, y, x, b, panel_top)
        if field is None:
            continue
        pad, trim = round(.24 * size), round(.16 * size)
        send = _printed(gray[y + pad:b - pad, x + trim:r - trim], index, {"发送", "發送"})
        if send is None:
            continue
        a, t, end, stop = field
        actions = [
            {"role": "comment_input", "function": "输入弹幕", "preview_text": "弹幕输入框",
             "box": list(field), "click_point": [(a + end) // 2, (t + stop) // 2]},
        ]
        from screen_visible_draft import has_visible_draft

        if has_visible_draft(frame, field, index, trailing_fields=.08):
            actions.append({"role": "send_comment", "function": "发送弹幕", "preview_text": send,
                            "box": [x, y, r, b], "click_point": [(x + r) // 2, (y + b) // 2]})
        if (keyboard := _keyboard_switch(gray, field)) is not None:
            kl, kt, kr, kb = keyboard
            actions.append({"role": "keyboard_switch", "function": "切换到文字键盘", "preview_text": "文字键盘",
                            "box": list(keyboard), "click_point": [(kl + kr) // 2, (kt + kb) // 2]})
        results.append(((0, panel_top, width, height), actions))
    return results[0] if len(results) == 1 else None


def dark_barrage_composer(frame, font_index):
    """Return a separately verified app-owned emoji tray and its editor actions."""
    height, width = frame.shape[:2]
    if font_index is None or width < 260 or not 1.60 * width <= height <= 2.95 * width:
        return None
    return _emoji_composer(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), font_index)


def dark_barrage_input_actions(frame, regions, font_index):
    """Return a click into a verified native dark barrage control."""
    height, width = frame.shape[:2]
    if (font_index is None or regions.get("layer") != "page" or regions.get("regions")
            or width < 260 or not 1.60 * width <= height <= 2.95 * width):
        return []
    return _idle(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), font_index)
