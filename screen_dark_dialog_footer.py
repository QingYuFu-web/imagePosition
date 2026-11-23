"""Read a dark native form's paired text actions from its verified foreground."""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from screen_caption_transcript import _bands
from screen_landscape_edit_actions import _features
from screen_landscape_search_field import _available
from screen_native_tab_bar import _distance


_ACTIONS = {
    "cancel": ("cancel", "取消"),
    "ok": ("confirm", "确定"),
    "okay": ("confirm", "确定"),
    "confirm": ("confirm", "确认"),
    "apply": ("confirm", "应用"),
    "save": ("confirm", "保存"),
    "done": ("confirm", "完成"),
    "取消": ("cancel", "取消"),
    "确定": ("confirm", "确定"),
    "確定": ("confirm", "确定"),
    "确认": ("confirm", "确认"),
    "確認": ("confirm", "确认"),
    "应用": ("confirm", "应用"),
    "應用": ("confirm", "应用"),
    "保存": ("confirm", "保存"),
    "完成": ("confirm", "完成"),
}
_OTHER_WORDS = (
    "No", "Yes", "Back", "Close", "Clear", "Reset", "Next", "Open", "Create", "Keep",
    "Dismiss", "Ignore", "Delete", "Retry", "Continue", "Select", "Allow", "Deny",
    "0K", "DK", "ON", "OR", "OX", "0X", "GX", "关闭", "關閉", "返回", "拒绝", "刪除", "删除", "重置",
)


def _focused_paint(frame, panel, fields, paper):
    span = panel[2] - panel[0]
    marks = []
    for control in fields.actions:
        if control["role"] != "input":
            continue
        left, _, right, bottom = control["box"]
        if not (panel[0] <= left < right <= panel[2] and panel[1] <= bottom <= panel[3]
                and right - left >= .40 * span):
            continue
        width = right - left
        band = frame[bottom - round(.012 * span):min(panel[3], bottom + round(.004 * span)),
                     left + round(.14 * width):right - round(.14 * width):5]
        strong = (_distance(band, paper) >= 56) & (np.ptp(band, axis=2) >= 54)
        rows = strong.mean(axis=1) >= .85
        if rows.any():
            marks.append(np.median(band[rows][strong[rows]], axis=0))
    if not marks or np.ptp(marks, axis=0).max() > 12:
        return None
    return np.median(marks, axis=0)


def _footer_words(frame, panel, paper):
    left, top, right, bottom = panel
    w, h = right - left, bottom - top
    x0, x1 = left + round(.037 * w), right - round(.034 * w)
    y0, y1 = top + round(.861 * h), bottom - round(.020 * h)
    band = _distance(frame[y0:y1, x0:x1], paper)
    rows = [(a, b) for a, b in _bands((band >= 25).sum(axis=1) >= max(4, round(.014 * w)))
            if .014 * h <= b - a <= .054 * h]
    if len(rows) != 1:
        return None
    a, b = rows[0]
    a, b = max(0, a - round(.006 * h)), min(len(band), b + round(.006 * h))
    ink = np.uint8(band[a:b] >= 28)
    groups = []
    for start, end in _bands(ink.any(axis=0)):
        if groups and start - groups[-1][1] <= .024 * w:
            groups[-1][1] = end
        else:
            groups.append([start, end])
    if len(groups) != 2:
        return None
    boxes = []
    for start, end in groups:
        x, y, width, height = cv2.boundingRect(ink[:, start:end])
        boxes.append((x0 + start + x, y0 + a + y, x0 + start + x + width, y0 + a + y + height))
    first, second = boxes
    sizes = [d - b for _, b, _, d in boxes]
    if (not .016 * h <= min(sizes) <= max(sizes) <= .055 * h or min(sizes) < .77 * max(sizes)
            or first[0] - left < .38 * w or not .038 * w <= right - second[2] <= .178 * w
            or not .052 * w <= second[0] - first[2] <= .190 * w
            or abs(first[1] - second[1]) > .010 * h or abs(first[3] - second[3]) > .009 * h):
        return None
    above = frame[min(first[1], second[1]) - round(.041 * h):min(first[1], second[1]) - round(.007 * h),
                  x0:x1:3]
    below = frame[max(first[3], second[3]) + round(.010 * h):bottom - round(.018 * h), x0:x1:3]
    if any(not tile.size or (_distance(tile, paper) <= 12).mean() < .971 for tile in (above, below)):
        return None
    return boxes


@lru_cache(maxsize=6)
def _printed_shapes(latin, cjk):
    accepted = tuple(_ACTIONS) + _OTHER_WORDS
    english = tuple(dict.fromkeys(form for text in accepted if text.isascii()
                                  for form in (text.title(), text.upper())))
    chinese = tuple(text for text in accepted if not text.isascii())
    shapes = []
    for paths, words in ((latin, english), (cjk, chinese)):
        for path in paths:
            font = ImageFont.truetype(str(path), 48)
            for name in words:
                canvas = Image.new("L", (370, 74))
                ImageDraw.Draw(canvas).text((3, 0), name, font=font, fill=255)
                raster = np.asarray(canvas)
                x, y, w, h = cv2.boundingRect(raster)
                if w and h:
                    shapes.append((name.casefold(), raster[y:y + h, x:x + w]))
    return tuple(shapes)


@lru_cache(maxsize=30)
def _size_bank(latin, cjk, width, height):
    names, tiles, penalties = [], [], []
    shape = width / height
    for name, raster in _printed_shapes(latin, cjk):
        ratio = shape * raster.shape[0] / raster.shape[1]
        if not .76 <= ratio <= 1.32:
            continue
        names.append(name)
        tiles.append(cv2.resize(raster, (width, height), interpolation=cv2.INTER_AREA))
        penalties.append(.10 * abs(np.log(ratio)))
    return names, _features(np.asarray(tiles)) if tiles else None, np.asarray(penalties)


def _read_button(frame, box, paper, index):
    left, top, right, bottom = box
    margin = max(2, round((bottom - top) * .11))
    patch = frame[top - margin:bottom + margin, left - margin:right + margin]
    distances = _distance(patch, paper)
    peak = float(np.percentile(distances, 97))
    if peak < 38:
        return None
    ink = np.clip(distances.astype(np.float32) / peak, 0, 1)
    latin = index.fonts if index.fonts is not None else index.choice_latin_sources
    cjk = index.fonts if index.fonts is not None else (
        *index.cjk_sources, *(path for path in latin if path.name.casefold().startswith("deng")))
    votes = {}
    for level in (.22, .38, .56):
        x, y, w, h = cv2.boundingRect(np.uint8(ink >= level))
        if not w or not h:
            continue
        image = ink[y:y + h, x:x + w]
        if h > 36:
            w, h = max(1, round(w * 36 / h)), 36
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        names, reference, penalties = _size_bank(latin, cjk, w, h)
        if reference is None:
            continue
        query = _features(np.where(image >= level * .62, image, 0.))
        distances = np.einsum("ij,j->i", reference, query, optimize=False) - penalties
        scores = {}
        for name, score in zip(names, distances, strict=True):
            scores[name] = max(scores.get(name, 0.), score)
        ordered = sorted(scores, key=scores.get, reverse=True)
        if len(ordered) >= 2:
            winner, neighbor = ordered[:2]
            if winner in _ACTIONS and scores[winner] >= .772 and scores[winner] - scores[neighbor] >= .027:
                votes.setdefault(winner, set()).add((w, h, image.tobytes()))
    likely = [name for name, masks in votes.items() if len(masks) >= 2]
    return _ACTIONS[likely[0]] if len(likely) == 1 else None


def _uses_accent(frame, box, paper, accent):
    left, top, right, bottom = box
    crop = frame[top:bottom, left:right]
    strong = crop[_distance(crop, paper) >= .74 * np.max(abs(accent - paper))]
    return len(strong) >= 10 and (np.max(abs(strong - accent), axis=1) <= 26).mean() >= .58


def dark_dialog_footer_actions(frame, regions, font_index, *, fields):
    """Return the enabled, visibly owned pair members of a dark form footer."""
    panels = regions.get("regions", ())
    if (fields is None or font_index is None or regions.get("layer") != "dialog" or len(panels) != 1
            or tuple(panels[0]) != fields.panel):
        return []
    left, top, right, bottom = panel = fields.panel
    w, h = right - left, bottom - top
    paint = np.median(frame[top + round(.86 * h):bottom - round(.024 * h):7,
                            left + round(.08 * w):right - round(.07 * w):8], axis=(0, 1))
    if not 31 <= paint.mean() <= 111 or np.ptp(paint) > 19:
        return []
    words = _footer_words(frame, panel, paint)
    if words is None or (accent := _focused_paint(frame, panel, fields, paint)) is None:
        return []
    readings = [_read_button(frame, word, paint, font_index) for word in words]
    if any(reading is None for reading in readings) or {read[0] for read in readings} != {"cancel", "confirm"}:
        return []
    output = []
    for word, (role, label) in zip(words, readings, strict=True):
        x, y, r, b = word
        dx, dy = round(.031 * w), round(.023 * h)
        box = [max(left, x - dx), max(top, y - dy), min(right, r + dx), min(bottom, b + dy)]
        if _available(box, regions) and _uses_accent(frame, word, paint, accent):
            output.append({"role": role, "function": label, "preview_text": label, "box": box,
                           "click_point": [(x + r) // 2, (y + b) // 2]})
    return output
