"""Read ordered controls on a dark settings panel with a visible colored header."""

from functools import lru_cache
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

from inventory_choice_glyphs import _tick, find_choice_glyphs
from screen_dialog_close import _cross_footprint


_WORDS = tuple(dict.fromkeys((
    "Settings Preferences Options Config Controls Graphics Quality Performance Language English Minimap Map "
    "Sound Display Account Low High Medium Ultra Standard Smooth Classic Rotating Fixed Bind Unbind Link "
    "Confirm Save Apply Cancel Close Reset Restore Play Start Exit Logout Advanced Default Help Feedback "
    "About Audio Video Scale Brightness Detail Resolution FPS Effects Balance English Switch Choose"
).split() + ["North Up", "Log Out", "Sign Out", "Turn On", "Turn Off"]))


def _out(role, name, box, *, caption_box=None, **metadata):
    left, top, right, bottom = box
    if caption_box is not None:
        a, b, c, d = caption_box
        metadata["context_box"] = [int(min(left, a)), int(min(top, b)), int(max(right, c)), int(max(bottom, d))]
    return {"role": role, "function": name, "box": [int(v) for v in box],
            "click_point": [int((left + right) // 2), int((top + bottom) // 2)], **metadata}


@lru_cache(maxsize=16)
def _word_shapes(paths, vocabulary):
    words, ratios, strokes = [], [], []
    for path in paths:
        try:
            font = ImageFont.truetype(str(path), 38)
        except OSError:
            continue
        for name in vocabulary:
            for case in dict.fromkeys((name, name.upper())):
                drawn = font.getmask(case, mode="L")
                array = np.asarray(drawn, np.uint8).reshape(drawn.size[1], drawn.size[0])
                x, y, w, h = cv2.boundingRect(np.uint8(array >= 38))
                if not min(w, h):
                    continue
                words.append(name)
                ratios.append(w / h)
                strokes.append(_normalize(array[y:y + h, x:x + w] / 255.))
    return tuple(words), np.float32(ratios), np.float32(strokes)


def _normalize(ink):
    shape = cv2.resize(np.float32(ink), (80, 22), interpolation=cv2.INTER_LINEAR)
    shape = cv2.GaussianBlur(shape, (5, 3), .66).ravel()
    return shape / max(.01, np.linalg.norm(shape))


def _printed(gray, area, index, *, vocabulary=_WORDS, light=True, minimum_score=.79, minimum_margin=.049,
             with_box=False, equivalent=()):
    left, top, right, bottom = area
    left, top = max(0, left), max(0, top)
    chip = gray[top:bottom, left:right]
    if min(chip.shape) < 9:
        return None
    k = max(5, round(.235 * len(chip)) | 1)
    polarity = cv2.MORPH_TOPHAT if light else cv2.MORPH_BLACKHAT
    strokes = cv2.morphologyEx(chip, polarity, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    peak = float(np.percentile(strokes, 98))
    if peak < 38:
        return None
    paths = tuple(index.fonts or index.choice_latin_sources)
    if index.fonts is None and paths:
        folder = Path(paths[0]).parent
        paths += tuple(path for name in ("bahnschrift.ttf", "arialn.ttf") if (path := folder / name).is_file())
    if any(not word.isascii() for word in vocabulary) and index.fonts is None:
        paths = tuple(dict.fromkeys((*paths, *index.cjk_sources)))
    words, ratios, bank = _word_shapes(paths, vocabulary)
    if not words:
        return None
    if equivalent:
        names = {word: group[0] for group in equivalent for word in group}
        identities = tuple(names.get(word, word) for word in words)
    else:
        identities = words
    readings = []
    for level in (.36, .50):
        x, y, w, h = cv2.boundingRect(np.uint8(strokes >= level * peak))
        if not (.28 * chip.shape[0] <= h <= .86 * chip.shape[0] and 1.25 <= w / h <= 12
                and 0 < y < y + h < chip.shape[0] and 0 < x < x + w < chip.shape[1]):
            return None
        sample = _normalize(np.minimum(1, strokes[y:y + h, x:x + w] / peak))
        aspect = (w / h) / ratios
        eligible = np.flatnonzero((aspect >= .66) & (aspect <= 1.45))
        if not len(eligible):
            return None
        score = np.einsum("ij,j->i", bank[eligible], sample) - .15 * abs(np.log(aspect[eligible]))
        best = int(score.argmax())
        label = words[eligible[best]]
        identity = identities[eligible[best]]
        rival = max((s for i, s in zip(eligible, score, strict=True) if identities[i] != identity), default=0)
        if score[best] < minimum_score or score[best] - rival < minimum_margin:
            return None
        readings.append((label, (left + x, top + y, left + x + w, top + y + h)))
    if readings[0][0] != readings[1][0]:
        return None
    name, first = readings[0]
    if not with_box:
        return name
    left, top, right, bottom = first
    other_left, other_top, other_right, other_bottom = readings[1][1]
    return name, (min(left, other_left), min(top, other_top), max(right, other_right), max(bottom, other_bottom))


def _boxes(mask):
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask))
    return [tuple(map(int, row)) for row in stats[1:]]


def _plate_edges(gray):
    contours, _ = cv2.findContours(cv2.Canny(gray, 14, 36), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return {tuple(map(int, box)) for contour in contours if (box := cv2.boundingRect(contour))
            and cv2.contourArea(contour) >= .87 * box[2] * box[3]}


def _closer(gray, top, bottom, width):
    left = round(.82 * width)
    crop = gray[top:bottom, left:]
    seen = []
    for cutoff in (152, 182):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(crop >= cutoff))
        boxes = []
        for i, (x, y, w, h, _) in enumerate(stats[1:], 1):
            if (.32 * (bottom - top) <= min(w, h) <= max(w, h) <= .68 * (bottom - top)
                    and .80 <= w / h <= 1.24 and .20 * len(crop) <= y <= .60 * len(crop)
                    and .018 * width <= crop.shape[1] - x - w / 2 <= .083 * width
                    and _cross_footprint(labels[y:y + h, x:x + w] == i)):
                boxes.append((x + left, y + top, x + left + w, y + top + h))
        if len(boxes) != 1:
            return None
        seen.append(boxes[0])
    if max(abs(a - b) for a, b in zip(*seen, strict=True)) > 4:
        return None
    a, b, c, d = seen[0]
    pad = max(2, round(.013 * width))
    return max(0, a - pad), max(0, b - pad), min(width, c + pad), min(gray.shape[0], d + pad)


def _caption_left(gray, x, y, glyph_height, index):
    width = gray.shape[1]
    return _printed(gray, (max(0, x - round(.187 * width)), y - round(.54 * glyph_height),
                           x - round(.026 * width), y + round(.54 * glyph_height)), index)


def _account(gray, accent, index):
    h, w = gray.shape
    buttons = [(l, t, l + width, t + height) for l, t, width, height, count in _boxes(accent)
               if .123 * w <= width <= .255 * w and .041 * h <= height <= .104 * h
               and t >= .56 * h and .81 * width * height <= count and 2.7 <= width / height <= 5.2]
    pair = [(a, b) for a, b in combinations(buttons, 2) if abs(a[1] - b[1]) <= .009 * h
            and abs((a[2] - a[0]) - (b[2] - b[0])) <= .027 * w]
    if len(pair) != 1:
        return []
    verified = {"Bind": ("bind_account", "绑定账号"), "Log Out": ("logout", "退出登录"),
                "Sign Out": ("logout", "退出登录"), "Logout": ("logout", "退出登录")}
    result = []
    for l, t, r, b in pair[0]:
        padx, pady = round(.115 * (r - l)), round(.10 * (b - t))
        label = _printed(gray, (l + padx, t + pady, r - padx, b - pady), index)
        if label in verified:
            result.append(_out(*verified[label], (l, t, r, b)))
    return result


def _chevron(area, height):
    for threshold in (92, 98):
        tile = area >= threshold
        a, b, w, h = cv2.boundingRect(np.uint8(tile))
        if not (.23 * height <= w <= .55 * height and .12 * height <= h <= .36 * height and w > h):
            return False
        glyph = tile[b:b + h, a:a + w]
        arms = [part.nonzero()[0] for part in np.array_split(glyph, 3, axis=1)]
        if not all(len(cut) for cut in arms) or min(arms[1].mean() - arms[0].mean(),
                                                     arms[1].mean() - arms[2].mean()) < .14 * h:
            return False
    return True


def _language(gray, accent, edges, index):
    h, w = gray.shape
    plates = [(x, y, width, height) for x, y, width, height in edges if .34 * w <= width <= .57 * w
              and .055 * h <= height <= .115 * h and .24 * h <= y <= .65 * h]
    matches = []
    for x, y, width, height in sorted(plates):
        left, top = x + round(.83 * width), y + round(.22 * height)
        tile = accent[top:y + round(.79 * height), left:x + round(.98 * width)]
        if _chevron(tile, height) and _caption_left(gray, x, y + height // 2, round(.80 * height), index) == "Language":
            matches.append((x, y, x + width, y + height))
    if matches and max(abs(a - b) for a, b in zip(min(matches), max(matches), strict=True)) <= 3:
        return [_out("select_menu", "打开语言选择", matches[0], setting_label="语言")]
    return []


def _graphics(gray, accent, index, top):
    h, w = gray.shape
    marks = find_choice_glyphs(gray, (round(.23 * w), top, round(.89 * w), round(.64 * h)),
                              (max(11, round(.028 * h)), round(.05 * h)))
    off = [mark.box for mark in marks if mark.kind == "checkbox" and mark.state == "off"]
    shapes = _boxes(accent)
    levels = {"Low", "Medium", "High", "Ultra"}
    pairs = []
    for x, y, r, b in off:
        side = b - y
        for left, t, width, height, _ in shapes:
            if not (.82 * side <= min(width, height) <= max(width, height) <= 1.24 * side
                    and .12 * w <= x - left <= .33 * w and abs(t - y) <= .155 * side):
                continue
            tile = gray[t:t + height, left:left + width]
            ink = tile >= 169
            a, d, gw, gh = cv2.boundingRect(np.uint8(ink))
            if gw < .53 * side or gh < .39 * side or not _tick(ink[d:d + gh, a:a + gw],
                                                               tile[d:d + gh, a:a + gw], minimum_fill=.17):
                continue
            if _caption_left(gray, left, t + height // 2, round(1.61 * side), index) not in {"Graphics", "Quality"}:
                continue
            selected, available = [_printed(gray, (end + round(.16 * side), start - round(.25 * side),
                                                  end + round(3 * side), start + round(1.21 * side)),
                                            index, with_box=True) for end, start in ((left + width, t), (r, y))]
            if selected is None or available is None:
                continue
            title, caption = available
            if selected[0] in levels and title in levels and selected[0] != title:
                pairs.append(((x, y, r, b), title, caption))
    if len(pairs) == 1:
        box, title, caption = pairs[0]
        return [_out("select_option", f"选择「{title}」", box, caption_box=caption,
                     setting_label="画质", setting_display_label=title)]
    return []


def _map_choices(gray, accent, edges, index):
    h, w = gray.shape
    cards = [(l, t, l + width, t + height) for l, t, width, height in edges
             if .092 * w <= width <= .174 * w and .89 <= width / height <= 1.12 and .46 * h <= t <= .73 * h
             and np.std(gray[t + 5:t + height - 5, l + 5:l + width - 5]) >= 23]
    pairs = [(a, b) for a, b in combinations(sorted(cards), 2) if .19 * w <= b[0] - a[0] <= .42 * w
             and abs(a[1] - b[1]) <= .012 * h and abs(a[3] - b[3]) <= .012 * h
             and abs(a[2] - a[0] - b[2] + b[0]) <= .021 * w]
    if len(pairs) != 1:
        return []
    first, last = pairs[0]
    left, top, right, bottom = first
    side = bottom - top
    heading = _caption_left(gray, left, top + round(.30 * side), round(.41 * side), index)
    if heading not in {"Minimap", "Map"}:
        return []
    witnesses = []
    for l, t, r, b in (first, last):
        inset = max(2, round(.037 * side))
        edges = (accent[t:t + inset, l + inset:r - inset], accent[b - inset:b, l + inset:r - inset],
                 accent[t + inset:b - inset, l:l + inset], accent[t + inset:b - inset, r - inset:r])
        paint = [edge.any(axis=0 if i < 2 else 1).mean() for i, edge in enumerate(edges)]
        reading = _printed(gray, (l, b + max(2, round(.023 * side)), r, b + round(.285 * side)), index, with_box=True)
        if reading is None or reading[0] not in {"North Up", "Rotating", "Fixed"}:
            return []
        label, caption = reading
        witnesses.append((sum(value >= .70 for value in paint), label, (l, t, r, b), caption))
    active, inactive = sorted(witnesses, reverse=True)
    if active[0] < 3 or inactive[0] > 1 or active[1] == inactive[1]:
        return []
    _, label, box, caption = inactive
    return [_out("select_option", f"选择「{label}」", box, caption_box=caption,
                 setting_label="小地图", setting_display_label=label)]


def colored_settings_panel(frame, overlay, index):
    """Scope a framed settings overlay using measured title, dismissal and child controls."""
    h, w = frame.shape[:2]
    if index is None or h < 320 or not 1.30 * h <= w <= 2.80 * h:
        return None
    blue, green, red = cv2.split(frame)
    intensity = cv2.min(cv2.subtract(blue, red), cv2.subtract(green, red))
    accent = intensity >= 38
    bars = [(x, y, width, height) for x, y, width, height, n in _boxes(accent[:round(.135 * h)])
            if width >= .72 * w and width + x >= .81 * w and x <= .061 * w and y <= .041 * h
            and .043 * h <= height <= .106 * h and n >= .87 * width * height]
    if len(bars) != 1:
        return None
    x, y, span, depth = bars[0]
    if overlay is not None and (overlay.kind != "dialog" or overlay.panel[2] - overlay.panel[0] < .88 * w
                                or overlay.panel[1] > .12 * h or overlay.panel[3] < .87 * h):
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    below = gray[y + depth:round(.97 * h):3, round(.045 * w):round(.955 * w):3]
    if (below <= 84).mean() < .82:
        return None
    close = _closer(gray, y, y + depth, w)
    title = _printed(gray, (x + round(.024 * w), y + round(.14 * depth), x + round(.191 * w),
                            y + round(.89 * depth)), index)
    if close is None or title not in {"Settings", "Preferences", "Options", "Config"}:
        return None
    edges = _plate_edges(gray)
    actions = _account(gray, accent, index) + _language(gray, intensity, edges, index)
    actions.extend(_graphics(gray, accent, index, y + depth))
    actions.extend(_map_choices(gray, accent, edges, index))
    if len(actions) < 2:
        return None
    from screen_setting_ranges import setting_range_actions

    actions.extend(setting_range_actions(frame, (0, y + depth, w, h), index, scoped=True))
    return (0, y, w, h), [_out("close", "关闭设置", close), *actions]
