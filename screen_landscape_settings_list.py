"""Read plain navigation rows on a verified native landscape settings page."""

from collections import Counter
import unicodedata

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption


_SETTINGS_TITLES = {
    "settings", "app settings", "preferences", "options", "设置", "应用设置", "偏好设置", "設定", "設定メニュー",
    "環境設定", "환경설정", "설정", "einstellungen", "paramètres", "préférences", "ajustes", "configuración",
    "impostazioni", "preferenze", "configurações", "preferências", "настройки", "параметры", "pengaturan",
}


def _rows(boxes, width, height):
    possible = sorted((box for box in boxes if box[0] <= .13 * width
                       and .24 * width <= box[2] - box[0] <= .72 * width
                       and .063 * height <= box[3] - box[1] <= .17 * height
                       and .145 * height <= box[1] <= .85 * height), key=lambda b: b[1])
    groups = []
    for first in possible:
        if first[1] > .27 * height:
            break
        size = first[3] - first[1]
        group = [first]
        for box in possible:
            a, y, _, b = box
            if y <= first[1] or abs(a - first[0]) > .012 * width or not .84 * size <= b - y <= 1.14 * size:
                continue
            delta = y - group[-1][1]
            if delta >= 1.22 * size:
                break
            if .815 * size <= delta < 1.22 * size:
                group.append(box)
        if 4 <= len(group) <= 9:
            groups.append(group)
    if not groups:
        return None
    groups.sort(key=len, reverse=True)
    best = groups[0]
    if len(groups) > 1 and len(groups[1]) == len(best):
        return None
    if np.ptp(np.diff([box[1] for box in best])) > .081 * (best[0][3] - best[0][1]):
        return None
    return best


def _transition(frame, rows):
    height, width = frame.shape[:2]
    estimate = rows[0][1]
    inset = frame[:, round(.29 * width):round(.93 * width):4].astype(np.int16)
    gap = max(2, round(.006 * height))
    candidates = []
    for y in range(estimate - round(.022 * height), estimate + round(.026 * height)):
        before, after = inset[y - gap], inset[y + gap]
        delta = abs(before - after).max(axis=1)
        if (delta >= 14).mean() >= .94:
            candidates.append((float(np.median(delta)), -abs(estimate - y), y))
    if not candidates:
        return None
    _, _, seam = max(candidates)
    toolbar = np.median(inset[seam - 3 * gap:seam - gap], axis=(0, 1))
    paint = np.median(inset[seam + 3 * gap:rows[-1][3]], axis=(0, 1))
    preview = inset[seam + 2 * gap:rows[-1][3]]
    if (abs(toolbar - paint).max() < 17 or (abs(preview - paint).max(axis=2) <= 11).mean() < .984):
        return None
    center = round(np.median([(y + b) / 2 for _, y, _, b in rows]))
    surface = (abs(frame[center].astype(np.int16) - paint).max(axis=1) <= 14)
    left_hit = np.flatnonzero(surface)
    if not len(left_hit):
        return None
    left = int(left_hit[0])
    right = int(left_hit[-1]) + 1
    if abs(left - rows[0][0]) > .024 * width or not .83 * width <= right - left or right < .96 * width:
        return None
    return left, seam, right, toolbar, paint


def _gesture_handle(frame):
    height, width = frame.shape[:2]
    center = width // 2
    reach, depth = round(.145 * width), round(.071 * height)
    strip = cv2.cvtColor(frame[-depth:, center - reach:center + reach], cv2.COLOR_BGR2GRAY)
    surface = float(np.median(strip))
    mask = np.uint8(abs(strip.astype(np.int16) - surface) >= 57)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count != 2:
        return False
    x, y, w, h, area = stats[1]
    return (.068 * width <= w <= .205 * width and .0031 * height <= h <= .019 * height
            and abs(x + w / 2 - reach) <= .024 * width and w / h >= 9
            and 0 < y < y + h < depth and area >= .68 * w * h)


def _header_title(frame, left, seam, step, toolbar, index, cache):
    height, width = frame.shape[:2]
    top, bottom = seam - round(.94 * step), seam - round(.16 * step)
    start, end = left + round(.98 * step), min(width, left + round(4.9 * step))
    mark = frame[top:bottom, left + round(.14 * step):left + round(.91 * step)]
    x, y, w, h = cv2.boundingRect(np.uint8(abs(mark - toolbar).max(axis=2) >= 46))
    if not (.24 * step <= h <= .51 * step and .55 <= w / h <= 1.49
            and abs(y + h / 2 - (bottom - top) / 2) <= .18 * step):
        return False
    tile = frame[top:bottom, start:end]
    x, y, w, h = cv2.boundingRect(np.uint8(abs(tile - toolbar).max(axis=2) >= 46))
    if (not .17 * step <= h <= .44 * step or w > .44 * width
            or abs(y + h / 2 - (bottom - top) / 2) > .16 * step):
        return False
    pad = max(3, round(.21 * h))
    extent = (start + max(0, x - pad), max(0, top + y - pad),
              min(width, start + x + w + pad), min(height, top + y + h + pad))
    reading = read_caption(frame, extent, index, cache=cache, choice=True)
    return bool(reading and unicodedata.normalize("NFKC", reading["text"]).casefold() in _SETTINGS_TITLES)


def _entry(frame, box, left, right, paint):
    _, top, _, bottom = box
    depth = bottom - top
    x0, x1 = left + round(.18 * depth), right - round(.036 * depth)
    start, end = top + round(.13 * depth), bottom - round(.123 * depth)
    tile = frame[start:end, x0:x1].astype(np.int16)
    offset = abs(tile - paint).max(axis=2)
    if (offset <= 12).mean() < .93:
        return None
    mask = np.uint8(offset >= 45)
    x, y, w, h = cv2.boundingRect(mask)
    if (not .17 * depth <= h <= .465 * depth or not .42 * h <= w <= .37 * (right - left)
            or not .27 * depth <= x0 + x - left <= .79 * depth or abs(y + start + h / 2 - (top + bottom) / 2) > .16 * depth):
        return None
    # One caption on quiet paint excludes pills, sliders, selection marks, and summaries.
    if offset[:, x + w + round(.22 * depth):].max(initial=0) > 30:
        return None
    text = tile[y:y + h, x:x + w]
    foreground = text[mask[y:y + h, x:x + w] != 0]
    tone = np.median(foreground, axis=0)
    if abs(tone - paint).max() < 62:
        return None
    return (x0 + x, start + y, x0 + x + w, start + y + h), tone


def _bright_latin(frame, box, index):
    left, top, right, bottom = box
    gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY).astype(np.int16)
    background = float(np.median(gray))
    contrast = abs(gray - background)
    peak = float(np.percentile(contrast, 99))
    if peak < 65:
        return None
    votes = {}
    for level in (.59, .66, .73, .79):
        mask = np.uint8(contrast >= level * peak)
        spans = _bands(mask.any(axis=0))
        if not 2 <= len(spans) <= 24:
            continue
        pieces = []
        for a, b in spans:
            _, y, _, h = cv2.boundingRect(mask[:, a:b])
            glyph = mask[y:y + h, a:b]
            ranks = index.rank(glyph, alphabet="choice_latin", limit=2)
            if len(ranks) == 2:
                pieces.append((a, b, y, y + h, ranks))
        if len(pieces) != len(spans):
            continue
        caps = [(y, end) for a, b, y, end, (first, second) in pieces
                if first["text"].isupper() and first["distance"] < .152
                and second["distance"] - first["distance"] >= .040 and b - a >= .39 * (end - y)]
        if not caps:
            continue
        cap_top, base = np.median(caps, axis=0)
        height = base - cap_top
        if height < 13:
            continue
        text = []
        for a, b, y, end, (first, second) in pieces:
            character, distance = first["text"], first["distance"]
            if {character, second["text"]} == {"I", "l"} and distance <= .048:
                # A simple stem shorter than the ascender carries no reliable case evidence.
                if (b - a <= .205 * height and .65 <= cap_top - y <= .18 * height
                        and abs(end - base) <= .072 * height
                        and mask[y:end, a:b].any(axis=1).mean() >= .97):
                    character = "l"
                else:
                    break
            elif distance > .192 or second["distance"] - distance < .021 or not character.isalnum():
                break
            text.append(character)
        if len(text) != len(pieces):
            continue
        name = text[0] + "".join((" " if a - prev > .31 * height else "") + character
                                 for (_, prev), (a, _), character in zip(spans[:-1], spans[1:], text[1:], strict=True))
        votes.setdefault(name, set()).add(mask.tobytes())
    reliable = [name for name, readings in votes.items() if len(readings) >= 2]
    return reliable[0] if len(reliable) == 1 else None


def landscape_settings_list_actions(frame, regions, index):
    """Return literally named rows with no subordinate affordance."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or height < 225 or width < 1.45 * height:
        return []
    boxes = [tuple(row["box"]) for row in regions["candidates"] if row.get("visibility") == "page"
             and 0 <= row["box"][0] < row["box"][2] <= width and 0 <= row["box"][1] < row["box"][3] <= height]
    rows = _rows(boxes, width, height)
    if rows is None:
        return []
    structure = _transition(frame, rows)
    if structure is None:
        return []
    left, seam, right, toolbar, paint = structure
    if not _gesture_handle(frame):
        return []
    step = float(np.median(np.diff([box[1] for box in rows])))
    cache = {}
    if not _header_title(frame, left, seam, step, toolbar, index, cache):
        return []
    entries = [(row, detail) for row in rows if (detail := _entry(frame, row, left, right, paint))]
    if len(entries) < max(3, len(rows) - 2):
        return []
    tone = np.median([detail[1] for _, detail in entries], axis=0)
    bounds = {row[1]: (max(seam + 1, round((row[1] + row[3] - step) / 2)),
                      min(height, round((row[1] + row[3] + step) / 2)))
              for row in rows}
    actions = []
    for row, ((x, y, r, b), color) in entries:
        if abs(color - tone).max() > 35:
            continue
        pad = max(3, round(.20 * (b - y)))
        target = (max(left, x - pad), y - pad, min(right, r + pad), b + pad)
        reading = read_caption(frame, target, index, choice=True, cache=cache)
        label = reading["text"] if reading else _bright_latin(frame, target, index)
        if not label:
            continue
        top, bottom = bounds[row[1]]
        actions.append({"role": "open_menu", "function": label, "preview_text": label,
                        "box": [left, top, right, bottom], "click_point": [(x + r) // 2, (top + bottom) // 2]})
    counts = Counter(action["function"] for action in actions)
    return [action for action in actions if counts[action["function"]] == 1]
