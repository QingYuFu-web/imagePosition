"""Resolve printed inline values and the confirm plate of a dark archive form."""

from collections import defaultdict

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_dialog_search_mode import _down_triangle
from screen_landscape_search_field import _available
from screen_native_tab_bar import _distance


_SIZE_LABELS = {"dictionary size": "选择字典大小", "word size": "选择字大小", "block size": "选择块大小"}
_MODE_VALUES = (
    frozenset({"7z", "zip", "tar", "gzip", "xz", "rar"}),
    frozenset({"lzma", "lzma2", "deflate", "bzip2", "ppmd", "zstd"}),
    frozenset({"normal", "fast", "fastest", "maximum", "ultra", "store"}),
    frozenset({"aes-128", "aes-192", "aes-256", "zipcrypto", "none", "no"}),
)
_MODE_NAMES = ("选择压缩格式", "选择压缩算法", "选择压缩级别", "选择加密方式")


def _marks(frame, panel, paint, after):
    left, _, right, bottom = panel
    w = right - left
    shift = after + round(.028 * w)
    clip = frame[shift:bottom - round(.165 * w), left + round(.25 * w):right - round(.016 * w)]
    offset = left + round(.25 * w)
    ink = _distance(clip, paint)
    matches = defaultdict(set)
    for cut in (14, 25, 36):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(ink >= cut))
        for index, (x, y, a, b, _) in enumerate(stats[1:], 1):
            if not (.023 * w <= a <= .050 * w and .011 * w <= b <= .028 * w
                    and 1.55 * b <= a <= 2.37 * b and _down_triangle(labels[y:y + b, x:x + a] == index)):
                continue
            box = tuple(map(int, (offset + x, shift + y, offset + x + a, shift + y + b)))
            key = next((old for old in matches if max(abs(n - m) for n, m in zip(old, box)) <= max(2, .004 * w)), box)
            matches[key].add(cut)
    return sorted((box for box, seen in matches.items() if len(seen) >= 2), key=lambda box: (box[1], box[0]))


def _caption(frame, clip, paint, index, cache, *, level=False):
    x, y, r, b = clip
    ink = _distance(frame[y:b, x:r], paint) >= 62
    a, t, w, h = cv2.boundingRect(np.uint8(ink))
    if not w or not h:
        return None
    pad = max(2, round(.11 * h))
    crop = (x + max(0, a - pad), y + max(0, t - pad), min(r, x + a + w + pad), min(b, y + t + h + pad))
    read = read_caption(frame, crop, index, cache=cache, choice=True, sentence_case=True, join_letters=True)
    if read is not None:
        return read["text"].casefold()
    # A narrow terminal stem in a compression level may match both I and l.
    spans = _bands(ink.any(axis=0))
    if level and len(spans) == 6:
        q, s = spans[-1]
        _, _, width, height = cv2.boundingRect(np.uint8(ink[:, q:s]))
        stem = ink[t:t + h, q:s]
        if width <= .16 * h and height >= .86 * h and stem.mean() >= .65:
            prefix = read_caption(frame, (x, y, x + q, b), index, cache=cache, choice=True)
            if prefix is not None and prefix["text"].casefold() == "norma":
                return "normal"
    return None


def _action(box, name, role, regions):
    if not _available(box, regions):
        return None
    l, t, r, b = box
    return {"role": role, "function": name, "preview_text": name, "box": list(box),
            "click_point": [(l + r) // 2, (t + b) // 2]}


def _paired_modes(frame, marks, start, paint, index, cache, w, regions):
    if len(marks) < 4:
        return []
    a, b, c, d = marks[:4]
    if (max(abs(a[1] - b[1]), abs(c[1] - d[1])) > .015 * w
            or max(abs(a[0] - c[0]), abs(b[0] - d[0])) > .016 * w
            or not .39 * w <= b[0] - a[0] <= .54 * w or not .071 * w <= c[1] - a[1] <= .135 * w):
        return []
    actions = []
    for i, mark in enumerate((a, b, c, d)):
        l, t, r, k = mark
        y = (t + k) // 2
        x = start if i % 2 == 0 else a[2] + round(.024 * w)
        label = _caption(frame, (x, y - round(.037 * w), l - round(.014 * w), y + round(.030 * w)),
                         paint, index, cache, level=i == 2)
        if label in _MODE_VALUES[i]:
            action = _action((x, y - round(.032 * w), r, y + round(.031 * w)), _MODE_NAMES[i], "dropdown", regions)
            if action is not None:
                actions.append(action)
    return actions


def _inline_value(frame, mark, start, paint, index, cache, w, regions):
    l, t, r, b = mark
    y = (t + b) // 2
    top, bottom = y - round(.042 * w), y + round(.030 * w)
    crop = frame[top:bottom, start:l - round(.016 * w)]
    if crop.size == 0:
        return None
    groups = _bands((_distance(crop, paint) >= 62).any(axis=0))
    gaps = [(c - b, b, c) for (_, b), (c, _) in zip(groups, groups[1:])]
    if not gaps:
        return None
    distance, a, v = max(gaps)
    if not .021 * w <= distance <= .085 * w:
        return None
    name = _caption(frame, (start, top, start + a + round(.006 * w), bottom), paint, index, cache)
    if name not in _SIZE_LABELS:
        return None
    value = start + v
    box = (value - round(.008 * w), top, r + round(.005 * w), bottom)
    return _action(box, _SIZE_LABELS[name], "dropdown", regions)


def _confirm_plate(frame, panel, paint, index, cache, regions):
    left, top, right, bottom = panel
    w, h = right - left, bottom - top
    x, y = left + round(.625 * w), bottom - round(.10 * h)
    clip = frame[y:bottom - round(.01 * h), x:right - round(.012 * w)]
    contours, _ = cv2.findContours(np.uint8(_distance(clip, paint) >= 7),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        a, b, c, d = cv2.boundingRect(contour)
        if (not .141 * w <= c <= .25 * w or not .056 * w <= d <= .117 * w
                or not 1.55 * d <= c <= 2.88 * d or cv2.contourArea(contour) < .94 * c * d):
            continue
        box = (x + a, y + b, x + a + c, y + b + d)
        cut = frame[y + b:y + b + d, x + a:x + a + c]
        plate = np.median(cut[::4, ::4], axis=(0, 1))
        if _caption(frame, box, plate, index, cache) == "ok":
            return _action(box, "确定", "confirm", regions)
    return None


def archive_inline_selector_actions(frame, regions, index, fields):
    if index is None or regions["layer"] != "dialog":
        return []
    left, top, right, bottom = panel = fields.panel
    w, h = right - left, bottom - top
    paint = np.median(frame[top + round(.16 * h):bottom - round(.10 * h):7,
                            left + round(.13 * w):right - round(.13 * w):9], axis=(0, 1))
    if paint.mean() > 27 or np.ptp(paint) > 22:
        return []
    owned = {item["function"]: item["box"] for item in fields.actions if item["role"] == "input"}
    if not {"输入文件名", "输入密码", "再次输入密码"} <= owned.keys():
        return []
    name, secret = owned["输入文件名"], owned["输入密码"]
    marks = _marks(frame, panel, paint, name[3])
    if len(marks) < 4:
        return []
    cache = {}
    actions = _paired_modes(frame, marks, name[0], paint, index, cache, w, regions)
    if len(actions) < 2:
        return []
    for mark in marks[4:]:
        if mark[3] < secret[1]:
            if action := _inline_value(frame, mark, name[0], paint, index, cache, w, regions):
                actions.append(action)
        elif split := owned.get("输入分卷大小"):
            l, t, r, b = mark
            y = (t + b) // 2
            if split[2] < l and abs(y - (split[1] + split[3]) / 2) <= .043 * w:
                box = (split[2] + round(.018 * w), y - round(.037 * w), r, y + round(.032 * w))
                code = _caption(frame, box, paint, index, cache)
                if code in {"kb", "mb", "gb", "kib", "mib", "gib"}:
                    if action := _action(box, "选择分卷单位", "dropdown", regions):
                        actions.append(action)
    if action := _confirm_plate(frame, panel, paint, index, cache, regions):
        actions.append(action)
    return actions
