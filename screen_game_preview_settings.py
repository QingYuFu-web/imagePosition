"""Read an image-based choice group in a visually verified landscape settings page."""

from itertools import combinations

import cv2
import numpy as np

from inventory_setting_switches import _stadium
from screen_caption_transcript import read_caption
from screen_component_captions import _bands
from screen_game_settings_back import game_settings_back
from screen_landscape_panel_switches import _knob
from screen_wide_panel_checks import _read_glyph


_SETTINGS = {"设置", "設定", "settings", "options"}
_QUALITY = {"画面质量", "畫面品質", "画质", "圖像品質", "图像质量", "graphicsquality", "renderquality"}
_LEVELS = {"低": "低", "中": "中", "高": "高", "极高": "极高", "極高": "极高", "超高": "超高"}


def _left_selection(gray):
    height, width = gray.shape
    first, stop = round(.068 * width), round(.22 * width)
    signal = np.uint8(gray[:round(.65 * height), first:stop] >= 171)
    _, _, stats, _ = cv2.connectedComponentsWithStats(signal, connectivity=8)
    matches = []
    for x, top, span, pitch, _ in stats[1:]:
        if (not .074 * height <= pitch <= .19 * height or span > .036 * width
                or not .072 * height <= top <= .39 * height):
            continue
        middle = slice(top + pitch // 8, top + 7 * pitch // 8)
        support = signal[middle, x:x + span].mean(axis=0)
        tracks = np.flatnonzero(support >= .90)
        if tracks.size < max(2, round(.0014 * width)):
            continue
        edge = first + x + tracks[-1] + 1
        reach = max(3, round(.005 * width))
        right = gray[middle, edge:edge + reach]
        if (right >= 150).mean() <= .09 and (gray[middle, edge - reach:edge] >= 171).mean() >= .32:
            matches.append((int(edge), int(top), int(pitch)))
    return matches[0] if len(matches) == 1 else None


def _text(frame, gray, crop, index, cache):
    left, top, right, bottom = crop
    field = gray[top:bottom, left:right]
    if min(field.shape) < 8:
        return None
    contrast = max(22, float(np.percentile(field, 40)) + 49)
    x, y, w, h = cv2.boundingRect(np.uint8(field >= contrast))
    if not .24 * field.shape[0] <= h <= .93 * field.shape[0] or not .61 * h <= w <= 12 * h:
        return None
    pad = max(2, round(.12 * h))
    box = (left + max(0, x - pad), top + max(0, y - pad), min(right, left + x + w + pad),
           min(bottom, top + y + h + pad))
    reading = read_caption(frame, box, index, cache=cache, choice=True, join_radicals=True, sentence_case=True)
    return (reading["text"], box) if reading else None


def _fold(text):
    return "".join(c for c in text.casefold() if c.isalnum())


def _quality_cards(gray, edge):
    height, width = gray.shape
    x0, x1 = edge + round(.047 * width), round(.984 * width)
    y0, y1 = round(.163 * height), round(.73 * height)
    tiles = gray[y0:y1, x0:x1]
    contours, _ = cv2.findContours(np.uint8(tiles <= 24), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (.078 * width <= w <= .22 * width and .095 * height <= h <= .245 * height
                and 1.37 <= w / h <= 2.79 and cv2.contourArea(contour) >= .89 * w * h):
            boxes.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
    outer = []
    for a, t, r, b in sorted(boxes, key=lambda box: -(box[2] - box[0]) * (box[3] - box[1])):
        if not any(l - 2 <= a < r <= z + 2 and y - 2 <= t < b <= d + 2 for l, y, z, d in outer):
            outer.append((a, t, r, b))
    groups = set()
    for left, top, right, bottom in outer:
        w, h = right - left, bottom - top
        row = tuple(sorted((a, t, r, b) for a, t, r, b in outer if
                           max(abs(t - top), abs(b - bottom)) <= .038 * h and abs(r - a - w) <= .044 * w))
        if not 3 <= len(row) <= 6:
            continue
        gaps = [r[0] - l[2] for l, r in zip(row[:-1], row[1:], strict=True)]
        if .033 * w <= min(gaps) <= max(gaps) <= .27 * w and np.ptp(gaps) <= .055 * w:
            groups.add(row)
    if len(groups) != 1:
        return None
    cards = next(iter(groups))
    imagery, selected = [], []
    for i, (left, top, right, bottom) in enumerate(cards):
        w, h = right - left, bottom - top
        interior = gray[top + round(.12 * h):top + round(.74 * h), left + round(.08 * w):right - round(.08 * w)]
        preview = cv2.resize(interior, (36, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
        if preview.std() < 24:
            return None
        imagery.append(preview)
        rim = gray[top + 1:top + max(4, round(.058 * h)), left + round(.06 * w):right - round(.06 * w)]
        if np.count_nonzero((rim.std(axis=1) <= 19) & (rim.mean(axis=1) >= 119)) >= max(2, round(.013 * h)):
            bottom_rim = gray[bottom - max(4, round(.058 * h)):bottom - 1,
                              left + round(.07 * w):left + round(.67 * w)]
            if (bottom_rim.mean(axis=1).max() >= 104 and
                    (bottom_rim.mean(axis=1) >= 104).sum() >= max(2, round(.011 * h))):
                selected.append(i)
    if len(selected) != 1 or any(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).item() < .76
                                 for a, b in combinations(imagery, 2)):
        return None
    return cards, selected[0]


def _level(gray, box, index):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    chip = gray[bottom - round(.222 * h):bottom - round(.027 * h),
                left + round(.743 * w):right - round(.024 * w)].astype(np.float32)
    signal = np.maximum(0, chip - np.percentile(chip, 25))
    peak = float(np.percentile(signal, 96))
    if peak < 62:
        return None
    readings = []
    for cut in (.275, .31, .34, .38):
        _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(signal >= peak * cut), connectivity=8)
        kept = [i for i, (x, y, w, h, area) in enumerate(stats[1:], 1)
                if x > 0 and y > 0 and x + w < signal.shape[1] and area >= 2]
        mask = np.uint8(np.isin(ids, kept))
        x, y, w, h = cv2.boundingRect(mask)
        if not .58 * chip.shape[0] <= h <= .94 * chip.shape[0] or not .56 * h <= w <= 2.44 * h:
            continue
        split = [0, w]
        if w > 1.55 * h:
            cutout = mask[y:y + h, x:x + w]
            col = min(range(round(.39 * w), round(.62 * w)), key=lambda j: (cutout[:, j].sum(), abs(w / 2 - j)))
            if cutout[:, col].sum() > .25 * h:
                continue
            split.insert(1, col)
        ink = np.where(mask, signal, 0)[y:y + h, x:x + w].astype(np.float32)
        parts = [ink[:, a:b] for a, b in zip(split[:-1], split[1:], strict=True)]
        text = [_read_glyph(part, float(np.percentile(part[part > 0], 93)), index) if part.any() else None
                for part in parts]
        if all(text) and "".join(text) in _LEVELS:
            readings.append(_LEVELS["".join(text)])
    return readings[0] if len(readings) >= 2 and len(set(readings)) == 1 else None


def _switches(frame, gray, left, beneath, index, cache):
    height, width = gray.shape
    y0, y1 = beneath + round(.023 * height), round(.92 * height)
    x0 = left + round(.036 * width)
    area = gray[y0:y1, x0:]
    groups = []
    for threshold in ((13, 38), (25, 71)):
        contours, _ = cv2.findContours(cv2.Canny(area, *threshold), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (not .046 * height <= h <= .109 * height or not 1.61 <= w / h <= 2.28
                    or not .78 <= cv2.contourArea(contour) / (w * h) <= .93):
                continue
            mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask, [contour], 0, 1, cv2.FILLED, offset=(-x, -y))
            if _stadium(mask):
                boxes.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
        groups.append(boxes)
    buttons = []
    for box in sorted(groups[0]):
        if (any(max(abs(x - y) for x, y in zip(box, old, strict=True)) <= 3 for old, _ in buttons)
                or not any(max(abs(x - y) for x, y in zip(box, peer, strict=True)) <= 3 for peer in groups[1])):
            continue
        state = _knob(frame, gray, box)
        if state is None:
            continue
        x, t, r, b = box
        w, h = r - x, b - t
        core = gray[t + round(.23 * h):b - round(.23 * h), x + round((.112 if state == "off" else .63) * w):
                    x + round((.36 if state == "off" else .88) * w)]
        if float(np.median(core)) >= 117:
            buttons.append((box, state))
    if len(buttons) < 2 or not any(abs(a[0] - b[0]) <= .014 * width or abs(a[1] - b[1]) <= .014 * height
                                  for (a, _), (b, _) in combinations(buttons, 2)):
        return []
    actions = []
    for (x, t, r, b), state in buttons:
        w, h = r - x, b - t
        # Help glyphs sit beyond the left-aligned label. Keep the first line segment.
        left, right = max(x0, x - round(1.88 * w)), x - round(.44 * w)
        top, bottom = t + round(.185 * h), b - round(.16 * h)
        strip = gray[top:bottom, left:right]
        ink = strip >= float(np.median(strip)) + 47
        spans = _bands(ink.sum(axis=0) >= 2)
        if not spans or spans[0][0] > .32 * w:
            continue
        end = spans[0][1]
        for first, stop in spans[1:]:
            if first - end >= .112 * h:
                break
            end = stop
        found = _text(frame, gray, (left, top, min(right, left + end + max(2, round(.05 * h))), bottom), index, cache)
        if not found:
            continue
        label, (a, _, _, _) = found
        verb = "开启" if state == "off" else "关闭"
        actions.append(_action("toggle_setting", f"{verb}{label}", (a, t, r, b), ((x + r) // 2, (t + b) // 2),
                               setting_label=label, setting_state=state))
    return actions


def _action(role, function, box, center=None, **metadata):
    x, y, r, b = box
    return {"role": role, "function": function, "preview_text": function, "box": list(box),
            "click_point": list(center or ((x + r) // 2, (y + b) // 2)), **metadata}


def game_preview_settings(frame, overlay, index):
    """Verify a settings rail and sibling preview cards before overriding a false modal."""
    height, width = frame.shape[:2]
    if (index is None or height < 320 or not 1.57 * height <= width <= 2.78 * height
            or (overlay is not None and (overlay.kind != "dialog" or overlay.close_button is not None))):
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    selection = _left_selection(gray)
    if selection is None:
        return None
    edge, selected_top, pitch = selection
    card_group = _quality_cards(gray, edge)
    if card_group is None:
        return None
    cards, selected = card_group
    l, t, _, b = cards[0]
    r = cards[-1][2]
    if overlay is not None and max(abs(x - y) for x, y in zip(overlay.panel, (l, t, r, b), strict=True)) > .062 * (b - t):
        return None
    cache = {}
    header_end = min(selected_top, round(.16 * height))
    heading = _text(frame, gray, (round(.40 * edge), 0, round(.93 * edge), header_end), index, cache)
    quality = _text(frame, gray, (edge + round(.036 * width), t - round(.032 * height), l - round(.015 * width),
                                  t + round(.063 * height)), index, cache)
    if not heading or _fold(heading[0]) not in _SETTINGS or not quality or _fold(quality[0]) not in _QUALITY:
        return None
    switches = _switches(frame, gray, edge, b, index, cache)
    if not switches:
        return None
    back = game_settings_back(frame, heading[1], edge)
    actions = [back] if back else []
    for start in range(selected_top % pitch, round(.64 * height), pitch):
        if abs(start - selected_top) < .30 * pitch or start < round(.075 * height):
            continue
        caption = _text(frame, gray, (round(.23 * edge), start + round(.237 * pitch), round(.79 * edge),
                                      start + round(.758 * pitch)), index, cache)
        if caption:
            label, (a, y, z, d) = caption
            if abs((a + z) / 2 - .54 * edge) <= .14 * edge:
                title = f"{label}设置"
                actions.append(_action("switch_tab", f"切换{title}", (0, start, edge, start + pitch),
                                       ((a + z) // 2, (y + d) // 2), setting_label=title, setting_state="off",
                                       preview_text=label))
    options = []
    for i, box in enumerate(cards):
        if i != selected and (name := _level(gray, box, index)) is not None:
            options.append(_action("game_settings_option", f"选择{name}画质", box))
    functions = [item["function"] for item in options]
    if len(functions) < len(cards) - 2 or len(set(functions)) != len(functions):
        return None
    return (0, 0, width, height), actions + options + switches
