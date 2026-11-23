"""Read the inactive tabs and footer action of an owned native property dialog."""

from collections import Counter

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_native_tab_bar import _distance


def _selection_rule(frame, panel, paint):
    left, top, right, bottom = panel
    w, h = right - left, bottom - top
    shift = top + round(.101 * h)
    clip = frame[shift:top + round(.239 * h), left + 1:right - round(.016 * w)]
    hue = np.ptp(clip, axis=2)
    ink = np.uint8(hue >= max(48, np.ptp(paint) + 41))
    raw = cv2.connectedComponentsWithStats(ink, connectivity=8)[2][1:]
    shape = [(int(x), int(y), int(width), int(height)) for x, y, width, height, area in raw
             if .162 * w <= width <= .60 * w and 1 <= height <= .015 * w and area >= .89 * width * height]
    if len(shape) != 1:
        return None
    x, y, width, height = shape[0]
    accent = np.median(clip[y:y + height, x:x + width], axis=(0, 1))
    row = (left + 1 + x, shift + y, left + 1 + x + width, shift + y + height)
    return row, accent


def _paper_ridges(tile, paint, minimum):
    distances = _distance(tile, paint)
    return _bands((distances >= minimum).mean(axis=1) >= .79)


def _property_stack(frame, panel, tabline, divider, paint):
    left, _, right, _ = panel
    w = right - left
    shift = tabline[3] + round(.029 * w)
    body = frame[shift:divider[0] - round(.023 * w), left:right]
    if body.shape[0] < .44 * w:
        return False
    inner = body[:, round(.265 * w):round(.805 * w)]
    borders = [(a + shift, b + shift) for a, b in _paper_ridges(inner, paint, 37)
               if 1 <= b - a <= .014 * w]
    layout = []
    for number, ((first, _), (last, __)) in enumerate(zip(borders[:-1], borders[1:], strict=True)):
        tall = last - first
        if not .106 * w <= tall <= .30 * w:
            continue
        y0, y1 = first + round(.12 * tall), last - round(.13 * tall)
        row = _distance(frame[y0:y1, left:right], paint) >= 29
        support = row.mean(axis=0)
        small, large = round(.038 * w), round(.171 * w)
        l = small + int(np.argmax(support[small:large]))
        r = w - large + int(np.argmax(support[w - large:w - small]))
        if min(support[l], support[r]) >= .78 and .72 * w <= r - l <= .92 * w:
            layout.append((number, l, r))
    if len(layout) < 3:
        return False
    return (np.ptp([row[1] for row in layout]) <= .018 * w and np.ptp([row[2] for row in layout]) <= .018 * w
            and all(b[0] - a[0] >= 2 for a, b in zip(layout[:-1], layout[1:], strict=True)))


def _footer_split(frame, panel, paint):
    left, top, right, bottom = panel
    h, w = bottom - top, right - left
    shift = top + round(.854 * h)
    inner = frame[shift:top + round(.947 * h), left + round(.025 * w):right - round(.025 * w)]
    hits = [(a + shift, b + shift) for a, b in _paper_ridges(inner, paint, 17)
            if 1 <= b - a <= .016 * w]
    return hits[0] if len(hits) == 1 else None


def _aligned_words(mask, left, top, cap):
    column = _bands(mask.any(axis=0))
    if not column:
        return []
    groups = [list(column[0])]
    for a, b in column[1:]:
        if a - groups[-1][1] > cap:
            groups.append([a, b])
        else:
            groups[-1][1] = b
    result = []
    for a, b in groups:
        x, y, w, h = cv2.boundingRect(np.uint8(mask[:, a:b]))
        if w and h:
            result.append((left + a + x, top + y, left + a + x + w, top + y + h))
    return result


def _accent_match(patch, accent, paper):
    ink = patch[np.ptp(patch, axis=2) >= max(40, np.ptp(paper) + 34)]
    if len(ink) < 7:
        return False
    signal = np.median(ink, axis=0) - paper
    expected = accent - paper
    return (signal @ expected) >= .906 * np.linalg.norm(signal) * np.linalg.norm(expected)


def _cap_letter(ranks, width, height):
    first = ranks[0]
    best = next((option for option in ranks if option["text"].isalpha() and option["text"].isupper()), None)
    if not best or best["distance"] > .163 or best["distance"] - first["distance"] > .048:
        return None
    name = best["text"]
    if first["text"] == name and ranks[1]["distance"] - first["distance"] >= .027:
        return name, True
    if first["text"].lower() == name.lower():
        return name, False
    if {name, first["text"]} == {"I", "l"} and width <= .24 * height:
        return name, False
    if name == "O" and first["text"] == "0" and .686 <= width / height <= .99:
        return name, False
    return None


def _uppercase_caption(frame, box, paint, index):
    left, top, right, bottom = box
    tile = frame[top:bottom, left:right]
    ink = _distance(tile, paint)
    peak = float(np.percentile(ink, 96))
    if peak < 62:
        return None
    votes = {}
    for k in (.28, .36, .44, .53):
        mask = np.uint8(ink >= k * peak)
        groups = _bands(mask.any(axis=0))
        if not 2 <= len(groups) <= 19:
            continue
        values = []
        for a, b in groups:
            x, y, w, h = cv2.boundingRect(mask[:, a:b])
            if not .83 * tile.shape[0] <= h <= tile.shape[0]:
                break
            ranks = index.rank(mask[y:y + h, a + x:a + x + w], alphabet="choice_latin", limit=7)
            if len(ranks) < 2 or not (verified := _cap_letter(ranks, w, h)):
                break
            values.append((*verified, y, y + h))
        if len(values) != len(groups):
            continue
        if (sum(row[1] for row in values) < max(1, round(.33 * len(values)))
                or max(np.ptp([row[2] for row in values]), np.ptp([row[3] for row in values])) > max(2, .078 * tile.shape[0])):
            continue
        name = values[0][0] + "".join((" " if a - previous >= .47 * tile.shape[0] else "") + value[0]
                                      for (_, previous), (a, _), value in zip(groups[:-1], groups[1:], values[1:], strict=True))
        votes.setdefault(name, set()).add(mask.tobytes())
    reliable = [name for name, cuts in votes.items() if len(cuts) >= 2]
    return reliable[0] if len(reliable) == 1 else None


def _literal(frame, box, paint, index, cache):
    if title := _uppercase_caption(frame, box, paint, index):
        return title
    l, t, r, b = box
    pad = max(3, round(.19 * (b - t)))
    limits = (max(0, l - pad), max(0, t - pad), min(frame.shape[1], r + pad), min(frame.shape[0], b + pad))
    result = read_caption(frame, limits, index, cache=cache, choice=True)
    if result and len(result["text"]) >= 2 and any(ord(c) >= 128 and c.isalpha() for c in result["text"]):
        return result["text"]
    return None


def _tab_captions(frame, panel, rule, accent, paint, index, cache):
    left, top, right, bottom = panel
    w = right - left
    x0, x1 = left + round(.024 * w), right - round(.024 * w)
    a, b = rule[1] - round(.126 * w), rule[1] - round(.031 * w)
    body = frame[a:b, x0:x1]
    words = _aligned_words(_distance(body, paint) >= 46, x0, a, round(.037 * w))
    if not 2 <= len(words) <= 4:
        return None
    x = [float((l + r) / 2) for l, _, r, _ in words]
    y = [float((t + b) / 2) for _, t, _, b in words]
    if np.ptp(y) > .012 * w or any(not .018 * w <= b - t <= .058 * w for _, t, _, b in words):
        return None
    selected = [i for i, j in enumerate(x) if rule[0] < j < rule[2]]
    if len(selected) != 1 or abs(x[selected[0]] - (rule[0] + rule[2]) / 2) > .107 * (rule[2] - rule[0]):
        return None
    choices = []
    for i, (l, t, r, b) in enumerate(words):
        physically_selected = _accent_match(frame[t:b, l:r], accent, paint)
        if physically_selected != (i == selected[0]):
            return None
        if physically_selected:
            continue
        name = _literal(frame, (l, t, r, b), paint, index, cache)
        if name:
            choices.append({"role": "dialog_tab", "function": name, "preview_text": name,
                            "box": [max(left, l - round(.057 * w)), round(y[i] - .058 * w),
                                    min(right, r + round(.057 * w)), rule[1]],
                            "click_point": [round(x[i]), round(y[i])]})
    return choices


def _confirmation(frame, panel, separator, accent, paint, index, cache):
    left, _, right, bottom = panel
    w, h = right - left, bottom - separator[1]
    x0, x1 = left + round(.048 * w), right - round(.033 * w)
    a, b = separator[1] + round(.146 * h), bottom - round(.124 * h)
    clip = frame[a:b, x0:x1]
    words = _aligned_words(_distance(clip, paint) >= 43, x0, a, round(.037 * w))
    if len(words) != 1:
        return None
    l, t, r, end = words[0]
    if (l < left + .73 * w or not .049 * w <= r - l <= .23 * w or not .16 * h <= end - t <= .38 * h
            or abs((t + end) / 2 - (separator[1] + bottom) / 2) > .115 * h
            or not _accent_match(frame[t:end, l:r], accent, paint)):
        return None
    value = _literal(frame, (l, t, r, end), paint, index, cache)
    if not value or len(value) > 14:
        return None
    pad = round(.038 * w)
    return {"role": "confirm", "function": value, "preview_text": value,
            "box": [max(left, l - pad), max(separator[1], t - pad),
                    min(right, r + pad), min(bottom, end + pad)],
            "click_point": [(l + r) // 2, (t + end) // 2]}


def native_property_dialog_actions(frame, regions, index):
    """Publish the actual unselected tab captions and one owned primary button."""
    if index is None or regions["layer"] != "dialog" or len(regions.get("regions", ())) != 1:
        return []
    height, width = frame.shape[:2]
    panel = tuple(map(int, regions["regions"][0]))
    left, top, right, bottom = panel
    w, h = right - left, bottom - top
    if (not 0 <= left < right <= width or not 0 <= top < bottom <= height
            or not .54 * width <= w <= .93 * width or not .88 * w <= h <= 2.04 * w):
        return []
    inner = frame[top + round(.22 * h):bottom - round(.13 * h):4, left + round(.14 * w):right - round(.11 * w):4]
    paint = np.median(inner, axis=(0, 1))
    if (_distance(inner, paint) <= 12).mean() < .80:
        return []
    selection = _selection_rule(frame, panel, paint)
    divider = _footer_split(frame, panel, paint)
    if not selection or not divider:
        return []
    rule, accent = selection
    if not _property_stack(frame, panel, rule, divider, paint):
        return []
    cache = {}
    tabs = _tab_captions(frame, panel, rule, accent, paint, index, cache)
    if tabs is None:
        return []
    confirmation = _confirmation(frame, panel, divider, accent, paint, index, cache)
    if confirmation:
        tabs.append(confirmation)
    counts = Counter(action["function"] for action in tabs)
    return [action for action in tabs if counts[action["function"]] == 1]
