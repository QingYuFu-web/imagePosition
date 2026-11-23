"""Read settings with repeated boxed tick marks and literal captions."""

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_game_preview_settings import _SETTINGS, _action, _fold, _text
from screen_ime_completion import _check_stroke
from screen_wide_panel_checks import _read_glyph


def _ink(gray, crop):
    l, t, r, b = crop
    chip = gray[t:b, l:r].astype(np.float32)
    if min(chip.shape) <= 6:
        return None
    contrast = np.maximum(0, chip - np.percentile(chip, 43))
    peak = float(np.percentile(contrast, 98))
    if peak < 32:
        return None
    mask = contrast >= .31 * peak
    x, y, w, h = cv2.boundingRect(np.uint8(mask))
    if not .26 * len(chip) <= h <= .95 * len(chip) or w < .30 * h:
        return None
    return contrast[y:y + h, x:x + w], peak, (l + x, t + y, l + x + w, t + y + h)


def _native_label(frame, gray, crop, index, cache, *, units=False):
    if not units and (found := _text(frame, gray, crop, index, cache)):
        return found
    text = _ink(gray, crop)
    if not text:
        return None
    signal, peak, bounds = text
    mask = signal >= .31 * peak
    h, w = signal.shape
    if units:
        split_candidates = [i for i in range(max(2, round(w - 1.40 * h)), w - round(.62 * h))
                            if mask[:, i].sum() <= .13 * h]
        for split in sorted(split_candidates, key=lambda i: abs(w - i - h)):
            tail = _read_glyph(signal[:, split:], peak, index)
            if tail not in {"帧", "幀"}:
                continue
            if number := _digits(signal[:, :split], peak, index):
                return f"{number}帧", bounds
        return None
    count = round(w / max(1, 1.05 * h))
    if not 1 <= count <= 8:
        return None
    stops = [0]
    cols = mask.sum(axis=0)
    for i in range(1, count):
        center = i * w / count
        a, b = max(1, round(center - .16 * h)), min(w, round(center + .16 * h) + 1)
        cut = min(range(a, b), key=lambda x: (int(cols[x]), abs(x - center)))
        if cols[cut] > .19 * h:
            return None
        stops.append(cut)
    stops.append(w)
    glyphs = [_read_glyph(signal[:, l:r], peak, index) for l, r in zip(stops[:-1], stops[1:], strict=True)]
    return ("".join(glyphs), bounds) if all(glyphs) else None


def _digits(signal, peak, index):
    groups = _bands((signal >= .29 * peak).any(axis=0))
    if not 2 <= len(groups) <= 3:
        return None
    digits = []
    for left, right in groups:
        readings = []
        for cut in (.242, .32, .40, .48):
            scores = index.rank(signal[:, left:right] >= cut * peak, alphabet="code", limit=14)
            ranked = [item for item in scores if item["text"] in "0123456789"]
            if (len(ranked) >= 2 and ranked[0]["distance"] <= .176
                    and ranked[1]["distance"] - ranked[0]["distance"] >= .020):
                readings.append(ranked[0]["text"])
        if len(readings) < 2 or len(set(readings)) != 1:
            return None
        digits.append(readings[0])
    return "".join(digits)


def _nav(frame, gray, index, cache):
    height, width = gray.shape
    area = frame[:round(.40 * height):2, :round(.41 * width):2]
    mask = np.uint8((np.ptp(area, axis=2) >= 72) & (area.max(axis=2) >= 154))
    outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scopes = []
    for contour in outlines:
        x, y, w, h = (2 * v for v in cv2.boundingRect(contour))
        if (not .048 * width <= x <= .255 * width or not .078 * width <= w <= .192 * width
                or not .075 * height <= y <= .265 * height or not .067 * height <= h <= .147 * height
                or cv2.contourArea(contour) < .90 * w * h / 4):
            continue
        title = _text(frame, gray, (x + round(.27 * w), round(.08 * y), x + round(.70 * w), y - round(.21 * h)),
                      index, cache)
        if title and _fold(title[0]) in _SETTINGS:
            scopes.append((x, y, x + w, y + h))
    return scopes[0] if len(scopes) == 1 else None


def _squares(frame, gray, edge):
    height, width = gray.shape
    x0, y0 = edge + round(.039 * width), round(.128 * height)
    area = gray[y0:round(.96 * height), x0:round(.94 * width)]
    binary = cv2.Canny(area, 23, 58)
    outlines, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = set()
    for contour in outlines:
        x, y, w, h = cv2.boundingRect(contour)
        if not .026 * height <= w <= .066 * height or not .88 <= h / w <= 1.16:
            continue
        if cv2.contourArea(contour) >= .84 * w * h:
            candidates.add((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
    tiles = []
    for x, y, r, b in sorted(candidates, key=lambda box: -(box[2] - box[0]) * (box[3] - box[1])):
        w, h = r - x, b - y
        if any(a - 3 <= x < r <= c + 3 and t - 3 <= y < b <= d + 3 for (a, t, c, d), _, _ in tiles):
            continue
        dx, dy = max(2, round(.09 * w)), max(2, round(.09 * h))
        chip = frame[y + dy:b - dy, x + dx:r - dx]
        paint = np.median(chip, axis=(0, 1))
        marked = _check_stroke(chip.min(axis=2) >= max(165, paint.max() + 39))
        spread = np.max(np.abs(chip - paint), axis=2)
        if marked or (np.ptp(paint) <= 29 and 34 <= paint.mean() <= 137 and (spread <= 19).mean() >= .80):
            tiles.append(((x, y, r, b), marked, paint))
    return tiles


def _groups(tiles):
    verified = {}
    for box, marked, _ in tiles:
        if not marked:
            continue
        x, y, r, b = box
        w, h = r - x, b - y
        group = sorted(tile for tile in tiles if abs(tile[0][1] - y) <= .12 * h
                       and abs(tile[0][2] - tile[0][0] - w) < .18 * w and abs(tile[0][3] - b) <= .12 * h)
        if not 3 <= len(group) <= 7 or sum(other for _, other, _ in group) != 1:
            continue
        steps = np.diff([tile[0][0] for tile in group])
        peers = np.asarray([paint for _, other, paint in group if not other])
        if (2.68 * w <= steps.min() <= steps.max() <= 7.72 * w and np.ptp(steps) <= .24 * w
                and np.ptp(peers, axis=0).max() <= 14):
            verified[y] = group
    rows = sorted(verified.values(), key=lambda group: group[0][0][1])
    return rows if len(rows) >= 2 and np.ptp([group[0][0][0] for group in rows]) <= 4 else []


def _tab_actions(frame, gray, nav, index, cache):
    left, top, right, bottom = nav
    w, h = right - left, bottom - top
    height = gray.shape[0]
    a, strip_right = left + round(.19 * w), right - round(.19 * w)
    field = gray[bottom + round(.16 * h):round(.86 * height), a:strip_right].astype(np.float32)
    background = np.percentile(field, 25, axis=1, keepdims=True)
    lines = []
    for y, z in _bands(((field - background) >= 39).sum(axis=1) >= .059 * w):
        if lines and y - lines[-1][1] <= .085 * h:
            lines[-1] = (lines[-1][0], z)
        else:
            lines.append((y, z))
    entries = []
    y0 = bottom + round(.16 * h)
    for y, z in lines:
        if not .167 * h <= z - y <= .41 * h:
            continue
        pad = max(4, round(.12 * h))
        reading = _native_label(frame, gray, (a, y0 + y - pad, strip_right, y0 + z + pad), index, cache)
        if reading is None:
            continue
        name, bounds = reading
        if name.endswith(("设置", "設定")):
            l, t, r, b = bounds
            cy = (t + b) // 2
            entries.append(_action("switch_tab", f"切换到{name}", (left, cy - round(.51 * h), right, cy + round(.51 * h)),
                                   ((l + r) // 2, cy), setting_label=name, setting_state="off", preview_text=name))
    return entries


def _header_actions(frame, gray, nav, index, cache):
    _, top, right, _ = nav
    width = gray.shape[1]
    # Only literal commands on the settings title bar; its heading is state.
    x0, y0 = max(right, round(.46 * width)), round(.14 * top)
    y1 = round(.67 * top)
    tile = frame[y0:y1, x0:round(.94 * width)]
    lines = _bands((tile.min(axis=2) >= 104).sum(axis=0) >= 2)
    groups = []
    for a, b in lines:
        if groups and a - groups[-1][1] <= .012 * width:
            groups[-1] = (groups[-1][0], b)
        else:
            groups.append((a, b))
    roles = {"返回": "back", "返回上一级": "back", "返回登录": "back", "返回登錄": "back",
             "恢复默认设置": "reset", "恢復默認設置": "reset", "恢复默认": "reset", "重置设置": "reset"}
    actions = []
    for a, b in groups:
        if not .015 * width <= b - a <= .124 * width:
            continue
        hit = _native_label(frame, gray, (x0 + max(0, a - 5), y0, x0 + b + 5, y1), index, cache)
        if hit and hit[0] in roles:
            actions.append(_action(roles[hit[0]], hit[0], hit[1]))
    return actions


def _binary_choice(frame, gray, nav, rows, index, cache):
    height, width = gray.shape
    first = rows[0][0][0][0]
    y0, y1 = rows[0][0][0][3], rows[-1][0][0][1]
    reach = round(.022 * width)
    left = first - reach
    field = gray[y0:y1, left:first + round(.14 * width)]
    outlines, _ = cv2.findContours(cv2.Canny(field, 23, 58), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = {cv2.boundingRect(c) for c in outlines if cv2.contourArea(c) >= .003 * height**2}
    actions = []
    for x, y, w, h in boxes:
        if (abs(x - reach) > .15 * h or not .041 * height <= h <= .087 * height
                or not 2.45 <= w / h <= 3.65):
            continue
        a, t = left + x, y0 + y
        edge = a + w // 2
        inside = round(.14 * h)
        l, r = a + inside, a + w - inside
        y2, y3 = t + inside, t + h - inside
        colors = [np.median(frame[y2:y3, x1:x2], axis=(0, 1)) for x1, x2 in ((l, edge - 3), (edge + 3, r))]
        chroma = [np.ptp(color) for color in colors]
        selected = int(np.argmax(chroma))
        if chroma[selected] < 40 or chroma[1 - selected] > 24 or abs(colors[0].mean() - colors[1].mean()) < 39:
            continue
        names = [_native_label(frame, gray, (x1, y2, x2, y3), index, cache)
                 for x1, x2 in ((l, edge - 4), (edge + 4, r))]
        if any(name is None for name in names) or {name[0] for name in names} not in ({"开", "关"}, {"開", "關"}):
            continue
        heading_left = nav[2] + round(.024 * width)
        heading = _native_label(frame, gray, (heading_left, t, a - round(.031 * width), t + h), index, cache)
        if not heading:
            continue
        lo, hi = (a, edge) if selected else (edge, a + w)
        verb = "关闭" if names[1 - selected][0] in {"关", "關"} else "开启"
        actions.append(_action("select_option", f"{verb}{heading[0]}", (lo, t, hi, t + h)))
    return actions


def _limit_minus(frame, gray, nav, rows, index, cache):
    height, width = gray.shape
    y0, y1 = rows[0][0][0][3], rows[-1][0][0][1]
    left = nav[2] + round(.060 * width)
    view = gray[y0:y1, left:round(.88 * width)]
    outlines, _ = cv2.findContours(np.uint8(view >= 159), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buttons = []
    for contour in outlines:
        x, y, w, h = cv2.boundingRect(contour)
        if (not .024 * height <= h <= .059 * height or not 1.25 <= w / h <= 2.24
                or cv2.contourArea(contour) < .84 * w * h):
            continue
        chip = view[y:y + h, x:x + w]
        glyph = chip < np.median(chip) - 45
        dx, dy = max(2, round(.055 * w)), max(2, round(.075 * h))
        glyph[:, :dx] = glyph[:, -dx:] = False
        glyph[:dy] = glyph[-dy:] = False
        a, t, k, d = cv2.boundingRect(np.uint8(glyph))
        if not .31 * w <= k <= .61 * w or abs(t + d / 2 - h / 2) > .12 * h:
            continue
        stroke = glyph[t:t + d, a:a + k]
        if .09 * h <= d <= .26 * h and stroke.mean() >= .82:
            kind = "minus"
        elif .54 * h <= d <= .85 * h and glyph[:, a + k // 2].mean() >= .52 and glyph[t + d // 2].mean() >= .34:
            kind = "plus"
        else:
            continue
        buttons.append((kind, (left + x, y0 + y, left + x + w, y0 + y + h), (a, k)))
    results = []
    for kind, (x, t, r, b), _ in buttons:
        if kind != "minus":
            continue
        h = b - t
        aligned = [(box, core) for kind, box, core in buttons if kind == "plus" and
                   abs(t - box[1]) <= .12 * h and .19 * width <= box[0] - r <= .45 * width]
        if len(aligned) != 1:
            continue
        (a, _, c, _), (offset, glyph_w) = aligned[0]
        # The thumb joins the inner edge of the plus control when the slider is at its upper endpoint.
        rail = gray[t + round(.41 * h):b - round(.41 * h), r:a:3]
        if offset + glyph_w / 2 - (c - a) / 2 < .14 * h or np.median(rail) < 76 or np.std(rail) > 23:
            continue
        heading = _native_label(frame, gray, (nav[2] + round(.023 * width), t - round(2.30 * h),
                                              x + round(.25 * h), t - round(.70 * h)), index, cache)
        if heading:
            results.append(_action("decrement_setting", f"减少{heading[0]}", (x, t, r, b), setting_label=heading[0]))
    return results


def game_marked_settings(frame, overlay, index):
    """Return a readable page of boxing-based mutually exclusive settings."""
    height, width = frame.shape[:2]
    if index is None or overlay is not None or height < 310 or not 1.56 * height <= width <= 2.85 * height:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.median(gray[::10, ::10]) > 99:
        return None
    cache = {}
    nav = _nav(frame, gray, index, cache)
    if not nav:
        return None
    rows = _groups(_squares(frame, gray, nav[2]))
    if not rows:
        return None
    settings = []
    for row in rows:
        x, y, _, b = row[0][0]
        h = b - y
        start = nav[2] + round(.022 * width)
        title = _native_label(frame, gray, (start, y - round(1.62 * h), x - round(.22 * h), y - round(.47 * h)),
                              index, cache)
        step = row[1][0][0] - x
        for (left, top, right, bottom), marked, _ in row:
            if marked:
                continue
            l, t = right + round(.21 * h), top - round(.11 * h)
            crop = (l, t, min(width, left + min(round(.84 * step), round(3.05 * h))), bottom + round(.09 * h))
            found = _native_label(frame, gray, crop, index, cache)
            if not found:
                found = _native_label(frame, gray, crop, index, cache, units=True)
            if not found:
                continue
            option, (_, _, r, _) = found
            if len(option) > 6 or any(not c.isalnum() for c in option):
                continue
            purpose = f"选择{title[0]}「{option}」" if title else f"选择{option}"
            settings.append(_action("select_option", purpose, (left, top, r, bottom),
                                    ((left + right) // 2, (top + bottom) // 2)))
    if len(settings) < 2:
        return None
    actions = _tab_actions(frame, gray, nav, index, cache) + _header_actions(frame, gray, nav, index, cache)
    extras = _binary_choice(frame, gray, nav, rows, index, cache) + _limit_minus(frame, gray, nav, rows, index, cache)
    return (0, 0, width, height), actions + settings + extras
