"""Recognize measured app carriers and a system search control in a home-screen dock."""

from itertools import combinations
from functools import lru_cache

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_caption_transcript import read_caption
from screen_font_glyphs import FontGlyphIndex


_APP_NAMES = {"phone": "电话", "safari": "Safari", "messages": "信息", "music": "音乐"}


def _center(box):
    left, top, right, bottom = box
    return (left + right) // 2, (top + bottom) // 2


def _action(box, *, app=None):
    if app is None:
        return {"role": "search", "function": "搜索主屏幕", "box": list(box), "click_point": list(_center(box))}
    return {"role": "open_app", "function": f"打开{_APP_NAMES[app]}", "app_id": app,
            "box": list(box), "click_point": list(_center(box))}


def _inset(box, ratio):
    left, top, right, bottom = box
    pad = round(ratio * min(right - left, bottom - top))
    return left + pad, top + pad, right - pad, bottom - pad


@lru_cache(maxsize=1)
def _fonts():
    return FontGlyphIndex()


def _notification_badges(frame, mask, width, offset):
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (.036 * width <= min(w, h) <= max(w, h) <= .077 * width and .80 <= w / h <= 1.24
                and .64 <= cv2.contourArea(contour) / (w * h) <= .84):
            continue
        badge = (x, y + offset, x + w, y + offset + h)
        l, t, r, b = _inset(badge, .14)
        white = frame[t:b, l:r].min(axis=2)
        b0, b1 = np.percentile(white, (42, 97))
        glyph = np.uint8(white >= max(117, b0 + .41 * (b1 - b0)))
        spans = _bands(glyph.any(axis=0))
        if not 1 <= len(spans) <= 3 or b1 - b0 < 53:
            continue
        numerals = []
        for first, stop in spans:
            chip = glyph[:, first:stop]
            _, _, ww, hh = cv2.boundingRect(chip)
            ranks = _fonts().rank(chip, alphabet="code", limit=3)
            if .35 * h <= hh <= .73 * h and .20 * h <= ww <= .62 * h and ranks:
                best = ranks[0]
                if best["text"].isdigit() and best["distance"] <= .21:
                    numerals.append(best["text"])
        if numerals:
            yield badge


def _covered_corner(box, badges):
    l, t, r, b = box
    tile = min(r - l, b - t)
    return any(.27 * tile <= min(x1 - x0, y1 - y0) <= .52 * tile
               and abs((x0 + x1) / 2 - r) <= .17 * tile and abs((y0 + y1) / 2 - t) <= .16 * tile
               and x0 < r < x1 and y0 < t < y1 for x0, y0, x1, y1 in badges)


def _carrier_boxes(mask, width, offset, height, *, badges=()):
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (.11 * width <= min(w, h) <= max(w, h) <= .197 * width and .85 <= w / h <= 1.17
                and .038 * width <= height - offset - y - h <= .146 * width
                and .84 * w * h <= cv2.contourArea(contour) <= .97 * w * h):
            continue
        face = np.zeros((h, w), np.uint8)
        cv2.drawContours(face, [contour], -1, 1, cv2.FILLED, offset=(-x, -y))
        margin = max(2, round(.17 * h))
        cuts = (face[:margin, :margin], face[:margin, -margin:], face[-margin:, :margin], face[-margin:, -margin:])
        normal = min(chip.mean() for chip in cuts) >= .15 and sum(chip.mean() <= .80 for chip in cuts) >= 3
        box = x, y + offset, x + w, y + offset + h
        if normal or (all(.22 <= cuts[j].mean() <= .83 for j in (0, 2, 3))
                      and _covered_corner(box, badges)):
            yield box


def _glyph(frame, box):
    l, t, r, b = _inset(box, .12)
    chip = frame[t:b, l:r].min(axis=2)
    background, peak = np.percentile(chip, (23, 94))
    if peak < 146 or peak - background < 60:
        return None
    ink = np.uint8(chip >= background + .50 * (peak - background))
    _, ids, stats, _ = cv2.connectedComponentsWithStats(ink)
    size = min(box[2] - box[0], box[3] - box[1])
    parts = []
    for i, (x, y, w, h, n) in enumerate(stats[1:], 1):
        if (.43 * size <= min(w, h) <= max(w, h) <= .78 * size and .20 * w * h <= n <= .87 * w * h
                and abs(x + w / 2 - chip.shape[1] / 2) <= .11 * size
                and abs(y + h / 2 - chip.shape[0] / 2) <= .12 * size):
            parts.append(ids[y:y + h, x:x + w] == i)
    return parts[0] if len(parts) == 1 else None


def _handset(mask):
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    upper = mask[:round(.22 * h), :round(.40 * w)]
    lower = mask[-round(.23 * h):, -round(.44 * w):]
    quiet = mask[:round(.36 * h), -round(.37 * w):]
    return (.84 <= w / h <= 1.15 and .23 <= mask.mean() <= .47 and np.corrcoef(xs, ys)[0, 1] >= .67
            and upper.mean() >= .36 and lower.mean() >= .42 and quiet.mean() <= .06)


def _bubble(mask):
    h, w = mask.shape
    if not 1.10 <= w / h <= 1.48 or not .68 <= mask.mean() <= .88:
        return False
    upper = mask[:round(.14 * h)]
    middle = mask[round(.37 * h):round(.62 * h)]
    toe = mask[-max(2, round(.09 * h)):]
    return (np.median(middle.sum(axis=1)) >= .82 * w and upper[:, :round(.15 * w)].mean() <= .09
            and toe[:, :round(.27 * w)].any() and not toe[:, -round(.17 * w):].any())


def _notes(mask):
    h, w = mask.shape
    if not .68 <= w / h <= .92 or not .29 <= mask.mean() <= .48:
        return False
    gap, mid, right = round(.44 * w), round(.52 * w), round(.70 * w)
    middle = mask[round(.40 * h):round(.57 * h)]
    beam = mask[round(.13 * h):round(.27 * h)]
    lowest = mask[round(.83 * h):]
    second = mask[round(.66 * h):round(.84 * h)]
    tail = mask[round(.91 * h):, right:]
    return (np.max(beam.sum(axis=1)) >= .55 * w and middle[:, gap:right].mean() < .19
            and middle[:, right:].mean() > .17 and lowest[:, :mid].mean() > .38
            and second[:, right:].mean() > .35 and tail.mean() < .19)


def _compass(frame, box):
    left, top, right, bottom = _inset(box, .075)
    chip = frame[top:bottom, left:right]
    hue, sat, tone = cv2.split(cv2.cvtColor(chip, cv2.COLOR_BGR2HSV))
    blue = (hue >= 77) & (hue <= 124) & (sat >= 62) & (tone >= 102)
    if not .52 <= blue.mean() <= .87:
        return False
    h, w = blue.shape
    ink = (hue >= 148) | (hue <= 13)
    ink &= (sat >= 62) & (tone >= 117)
    x, y, a, b = cv2.boundingRect(np.uint8(ink))
    if (not .20 * w <= a <= .41 * w or not .20 * h <= b <= .41 * h or not .68 <= a / b <= 1.43
            or not .52 * w <= x + a / 2 <= .75 * w or not .21 * h <= y + b / 2 <= .48 * h):
        return False
    yy, xx = np.nonzero(ink[y:y + b, x:x + a])
    whites = chip.min(axis=2)
    mid = whites[round(.41 * h):round(.80 * h), round(.17 * w):round(.48 * w)]
    if len(xx) < .19 * a * b or np.corrcoef(xx, yy)[0, 1] >= -.38:
        return False
    # The two needle halves must share the diagonal; the pale outside bezel
    # cannot qualify because it is outside the blue disk and inner patch.
    bright = np.uint8(mid >= max(135, .69 * np.percentile(whites, 97)))
    contour, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(.105 * w <= a1 <= .36 * w and .11 * h <= b1 <= .35 * h
               and abs(a1 / b1 - 1) <= .77 and cv2.contourArea(c) >= .19 * a1 * b1
               for c in contour for _, _, a1, b1 in (cv2.boundingRect(c),))


def _dock_icons(frame):
    height, width = frame.shape[:2]
    offset = height - round(.416 * width)
    chip = frame[offset:]
    gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
    hue, sat, tone = cv2.split(cv2.cvtColor(chip, cv2.COLOR_BGR2HSV))
    greens = (hue >= 35) & (hue <= 86) & (sat >= 65) & (tone >= 98)
    reds = ((hue >= 153) | (hue <= 11)) & (sat >= 83) & (tone >= 140)
    pales = (gray >= 169) & (sat <= 64)
    choices = {}
    badges = tuple(_notification_badges(frame, reds, width, offset))
    for box in _carrier_boxes(greens, width, offset, height, badges=badges):
        if (glyph := _glyph(frame, box)) is not None:
            if _handset(glyph):
                choices.setdefault("phone", []).append(box)
            elif _bubble(glyph):
                choices.setdefault("messages", []).append(box)
    for box in _carrier_boxes(reds, width, offset, height):
        if (glyph := _glyph(frame, box)) is not None and _notes(glyph):
            choices.setdefault("music", []).append(box)
    for box in _carrier_boxes(pales, width, offset, height):
        if _compass(frame, box):
            choices.setdefault("safari", []).append(box)
    candidates = [(app, box) for app, boxes in choices.items() for box in boxes]
    if not 3 <= len(candidates) <= 8:
        return []
    aligned = []
    for count in (4, 3):
        for group in combinations(candidates, count):
            if len({app for app, _ in group}) != count:
                continue
            ordered = sorted(group, key=lambda item: item[1][0])
            centers = np.asarray([_center(box) for _, box in ordered])
            spans = np.asarray([(box[2] - box[0], box[3] - box[1]) for _, box in ordered])
            size = np.median(spans)
            gaps = np.diff(centers[:, 0])
            pitch = min(gaps)
            if not 1.20 * size <= pitch <= 1.68 * size:
                continue
            strides = gaps / pitch
            if (np.ptp(centers[:, 1]) <= .10 * size and np.ptp(spans, axis=0).max() <= .19 * size
                    and (abs(strides - np.rint(strides)) <= .10).all()
                    and sum(np.rint(strides)) <= 3):
                aligned.append(ordered)
        if aligned:
            return aligned[0] if len(aligned) == 1 else []
    return []


def _lens(mask):
    contours, tree = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if tree is None:
        return False
    holes = [c for c, parent in zip(contours, tree[0, :, 3], strict=True) if parent >= 0]
    if len(holes) != 1:
        return False
    h, w = mask.shape
    x, y, a, b = cv2.boundingRect(holes[0])
    if (not .40 * w <= a <= .72 * w or not .38 * h <= b <= .69 * h or not .73 <= a / b <= 1.34
            or not .33 * w <= x + a / 2 <= .53 * w or not .28 * h <= y + b / 2 <= .54 * h):
        return False
    yy, xx = np.nonzero(mask[-max(2, round(.14 * h)):])
    return (len(xx) >= 2 and np.median(xx) >= max(x + .61 * a, .64 * w)
            and mask[-max(2, round(.19 * h)):, :round(.25 * w)].mean() <= .43)


def _search_mark(chip, tile, center_x):
    if min(chip.shape) < 15:
        return None
    witnesses = []
    for fraction in (.070, .086):
        k = 2 * (round(fraction * tile) // 2) + 1
        strokes = cv2.morphologyEx(chip, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        for threshold in (8, 12, 16, 21, 27):
            ink = np.uint8(strokes >= threshold)
            _, ids, stats, _ = cv2.connectedComponentsWithStats(ink)
            for i, (x, y, w, h, n) in enumerate(stats[1:], 1):
                if (not .096 * tile <= min(w, h) <= max(w, h) <= .22 * tile or not .74 <= w / h <= 1.18
                        or not .34 <= n / (w * h) <= .76
                        or not center_x - .47 * tile <= x + w / 2 <= center_x - .10 * tile
                        or not _lens(np.uint8(ids[y:y + h, x:x + w] == i))):
                    continue
                xa, xb = x + w + round(.04 * tile), min(chip.shape[1], x + w + round(.53 * tile))
                ta, tb = max(0, y - round(.052 * tile)), min(len(chip), y + h + round(.052 * tile))
                label = np.uint8(strokes[ta:tb, xa:xb] >= max(8, threshold - 6))
                a, t, b, d = cv2.boundingRect(label)
                if (not .19 * tile <= b <= .50 * tile or not .107 * tile <= d <= .265 * tile
                        or not .095 <= label.mean() <= .56 or abs(ta + t + d / 2 - y - h / 2) > .078 * tile):
                    continue
                if len(_bands(label.any(axis=0))) < 2:
                    continue
                measured = x, min(y, ta + t), xa + a + b, max(y + h, ta + t + d)
                witnesses.append((threshold, measured))
    stable = [box for level, box in witnesses if any(other != level and
              max(abs(a - b) for a, b in zip(box, near, strict=True)) <= .070 * tile for other, near in witnesses)]
    if not stable or np.ptp(np.asarray(stable), axis=0).max() > .11 * tile:
        return None
    return tuple(np.rint(np.median(np.asarray(stable), axis=0)).astype(int).tolist())


def _search(frame, icons):
    boxes = [box for _, box in icons]
    cy = int(np.median([_center(box)[1] for box in boxes]))
    tile = float(np.median([box[3] - box[1] for box in boxes]))
    cx = frame.shape[1] // 2
    left, top = cx - round(.62 * tile), cy - round(1.67 * tile)
    right, bottom = cx + round(.65 * tile), cy - round(.94 * tile)
    chip = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    found = _search_mark(chip, tile, cx - left)
    scale = 1.
    if found is None and tile < 87:
        scale = 95 / tile
        enlarged = cv2.resize(chip, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        found = _search_mark(enlarged, 95., scale * (cx - left))
    if found is None:
        return None
    x, y, r, b = (round(coordinate / scale) for coordinate in found)
    return left + x, top + y, left + r, top + b


def _english_search(frame, icons):
    boxes = [box for _, box in icons]
    span = float(np.median([box[3] - box[1] for box in boxes]))
    cy = int(np.median([_center(box)[1] for box in boxes]))
    cx = frame.shape[1] // 2
    left, top = cx - round(.62 * span), cy - round(1.67 * span)
    right, bottom = cx + round(.74 * span), cy - round(.94 * span)
    chip = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    n = 2 * (round(.075 * span) // 2) + 1
    strokes = cv2.morphologyEx(chip, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n)))
    lenses = []
    for level in (16, 23, 30):
        _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(strokes >= level))
        for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if (.10 * span <= min(w, h) <= max(w, h) <= .219 * span and .77 <= w / h <= 1.18
                    and cx - .47 * span <= x + left + w / 2 <= cx - .10 * span
                    and _lens(np.uint8(ids[y:y + h, x:x + w] == i))):
                lenses.append((x + left, y + top, x + left + w, y + top + h))
    if len(lenses) < 2 or np.ptp(lenses, axis=0).max() > .068 * span:
        return None
    l, t, r, b = (round(v) for v in np.median(lenses, axis=0))
    box = r + round(.02 * span), t - round(.072 * span), r + round(.79 * span), b + round(.072 * span)
    reading = read_caption(frame, box, _fonts(), cache={}, choice=True, join_letters=True, sentence_case=True)
    if reading is None:
        return None
    text = reading["text"].casefold()
    if not (3 <= len(text) <= 12 and text.isascii() and text.isalpha()):
        return None
    x, y, end, stop = reading["line_box"]
    if not .36 * span <= end - x <= .77 * span or abs((t + b) / 2 - (y + stop) / 2) > .055 * span:
        return None
    return text, (l, min(t, y), end, max(b, stop))


def _dark_carrier(frame, label_box, icons):
    size = float(np.median([box[3] - box[1] for _, box in icons]))
    cx, cy = frame.shape[1] // 2, _center(label_box)[1]
    reach = round(.35 * size)
    t, b = cy - reach, cy + reach
    left, right = cx - round(.27 * size), cx + round(.27 * size)
    gray = cv2.cvtColor(frame[t:b, left:right], cv2.COLOR_BGR2GRAY).astype(np.int16)
    if np.median(gray[reach - round(.15 * size):reach + round(.15 * size)]) > 133:
        return False
    step = max(2, round(.032 * size))
    for y in range(reach - step, reach + step + 1):
        for radius in range(round(.17 * size), round(.29 * size)):
            if not 0 <= y - radius - step < y + radius + step < len(gray):
                continue
            upper = gray[y - radius - step] - gray[y - radius + step]
            lower = gray[y + radius + step] - gray[y + radius - step]
            if ((upper >= 10) & (lower >= 10)).mean() >= .72:
                return True
    return False


def home_dock_actions(frame, regions):
    """Find a page-bound dock from visible app glyphs, independent of slot order."""
    height, width = frame.shape[:2]
    if (regions.get("layer") != "page" or regions.get("regions")
            or width < 280 or not 1.65 * width <= height <= 2.83 * width):
        return []
    stride = max(2, round(width / 108))
    exposure = frame[::stride, ::stride]
    if (exposure.max(axis=2) >= 228).mean() < .019:
        return []
    dock = _dock_icons(frame)
    if not dock:
        return []
    actions = [_action(box, app=app) for app, box in dock]
    english = _english_search(frame, dock)
    if english is not None:
        text, search = english
        if text == "search":
            actions.append({**_action(search), "function": "打开系统搜索"})
    elif (search := _search(frame, dock)) and not _dark_carrier(frame, search, dock):
        actions.append(_action(search))
    return actions
