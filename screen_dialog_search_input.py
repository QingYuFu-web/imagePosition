"""Verify the editable query field in a foreground native search dialog."""

from dataclasses import dataclass
import unicodedata

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_landscape_search_field import _available
from screen_native_tab_bar import _distance
from screen_pixel_modal import verified_dialog_face


_SEARCH_HEADINGS = {
    "search", "find", "搜索", "搜尋", "查找", "查詢", "查询", "搜索文件", "搜尋檔案", "查找文件",
    "搜索联系人", "搜尋聯絡人", "搜索应用", "搜尋應用程式", "查找内容", "搜尋內容",
}
_QUERY_PROMPTS = {
    "typetheword", "typetosearch", "typeakeyword", "typeaname", "searchquery", "searchterm", "searchterms",
    "searchfor", "searchfiles", "search", "enterkeywords", "enterakeyword", "entersearchterms", "enterfilename",
    "输入搜索词", "输入搜索内容", "输入关键词", "输入关键字", "输入文件名", "搜索", "搜索文件", "查找文件",
    "輸入搜尋詞", "輸入關鍵詞", "輸入關鍵字", "輸入檔案名稱", "搜尋", "搜尋檔案",
}


@dataclass(frozen=True)
class DialogSearchField:
    panel: tuple[int, int, int, int]
    title_box: tuple[int, int, int, int]
    input_box: tuple[int, int, int, int]
    caption_box: tuple[int, int, int, int]
    underline: tuple[int, int, int, int]
    caption: str


def _fold(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().strip(" .:…").split())


def _header_bounds(ink, width, height):
    x0, x1 = round(.025 * width), round(.805 * width)
    y0, y1 = round(.024 * height), round(.158 * height)
    mask = np.uint8(ink[y0:y1, x0:x1] >= 54)
    x, y, w, h = cv2.boundingRect(mask)
    if (not .026 * width <= h <= .080 * width or not 1.44 * h <= w <= .69 * width
            or x > .078 * width or y < 2 or y + h >= len(mask)):
        return None
    glyph_runs = _bands(mask[y:y + h, x:x + w].any(axis=0))
    lead = next((b for (_, b), (c, _) in zip(glyph_runs, glyph_runs[1:]) if c - b > .245 * h), w)
    title = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
    return title, min(title[2], title[0] + lead)


def _underlines(ink, title, width, height):
    _, top, _, edge = title
    depth = edge - top
    x0, x1 = round(.036 * width), round(.971 * width)
    y0, y1 = edge + round(1.54 * depth), min(len(ink), edge + round(5.6 * depth))
    if y1 - y0 < depth:
        return
    mask = np.uint8(ink[y0:y1, x0:x1] >= 44)
    _, ids, pieces, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i, (x, y, w, h, area) in enumerate(pieces[1:], 1):
        if (not .30 * width <= w <= .88 * width or not 1 <= h <= .016 * width
                or area < .78 * w * h or not 0 < y < y + h < mask.shape[0]
                or not .138 * height <= y0 + y <= .36 * height):
            continue
        body = ids[y:y + h, x:x + w] == i
        if body.mean(axis=1).max() >= .974:
            yield tuple(map(int, (x0 + x, y0 + y, x0 + x + w, y0 + y + h)))


def _owned_caption(ink, title, line):
    x0, y, x1, base = line
    depth = title[3] - title[1]
    start, stop = y - round(1.46 * depth), y - max(2, round(.104 * depth))
    if start <= title[3] or stop <= start:
        return None
    mask = np.uint8(ink[start:stop, x0:x1] >= 49)
    x, t, w, h = cv2.boundingRect(mask)
    if (not .62 * depth <= h <= 1.42 * depth or not 1.10 * h <= w <= .83 * (x1 - x0)
            or x > .41 * depth or y - (start + t + h) > .73 * depth
            or start + t - title[3] < .90 * depth):
        return None
    px = max(2, round(.14 * depth))
    quiet = ink[start:stop, x0 + x + w + px:x1]
    if not quiet.size or (quiet <= 17).mean() < .925:
        return None
    bounds = (x0 + x, start + t, x0 + x + w, start + t + h)
    return bounds, (x0, bounds[1] - round(.37 * depth), x1, base)


def _read(frame, bounds, panel, index, cache, *, xcap=None):
    a, b, c, d = bounds
    left, top, right, bottom = panel
    pad = max(2, round(.13 * (d - b)))
    limit = min(c + pad, xcap) if xcap is not None else c + pad
    area = (max(left, a - pad), max(top, b - pad), min(right, limit), min(bottom, d + pad))
    result = read_caption(frame, area, index, cache=cache, choice=True)
    return result["text"] if result else None


def dialog_search_field(frame, regions, font_index):
    """Measure a captioned query entirely inside the verified front dialog."""
    if font_index is None or regions.get("layer") != "dialog" or not regions.get("regions"):
        return None
    scope = tuple(map(int, regions["regions"][0]))
    face = verified_dialog_face(frame, scope)
    if face is None:
        return None
    left, top, right, bottom = panel = (max(face[0], scope[0]), max(face[1], scope[1]),
                                       min(face[2], scope[2]), min(face[3], scope[3]))
    width, height = right - left, bottom - top
    clip = frame[top:top + round(.42 * height), left:right]
    paint = np.median(clip[::5, round(.17 * width):round(.82 * width):5], axis=(0, 1))
    ink = _distance(clip, paint)
    if (ink[::3, ::3] <= 19).mean() < .83 or (header := _header_bounds(ink, width, height)) is None:
        return None
    title, first_word = header
    choices = [(line, owned) for line in _underlines(ink, title, width, height)
               if (owned := _owned_caption(ink, title, line)) is not None]
    if not 1 <= len(choices) <= 4:
        return None

    def original(box):
        a, b, c, d = box
        return left + a, top + b, left + c, top + d

    title = original(title)
    if not _available(title, regions):
        return None
    cache = {}
    cap = left + first_word + max(2, round(.115 * (title[3] - title[1])))
    heading = _read(frame, title, panel, font_index, cache, xcap=cap)
    if (heading is None or _fold(heading) not in _SEARCH_HEADINGS) and cap < title[2]:
        heading = _read(frame, title, panel, font_index, cache)
    if heading is None or _fold(heading) not in _SEARCH_HEADINGS:
        return None
    matches = []
    for line, (caption, field) in choices:
        caption, field, line = map(original, (caption, field, line))
        if not _available(field, regions):
            continue
        label = _read(frame, caption, panel, font_index, cache)
        if label is not None and _fold(label) in _QUERY_PROMPTS:
            matches.append(DialogSearchField(panel, title, field, caption, line, label))
    return matches[0] if len(matches) == 1 else None


_UNMEASURED_FIELD = object()


def dialog_search_input_actions(frame, regions, font_index, *, field=_UNMEASURED_FIELD):
    """Return the editable query, with its own caption and no background control."""
    if field is _UNMEASURED_FIELD:
        field = dialog_search_field(frame, regions, font_index)
    if field is None or regions.get("layer") != "dialog" or not regions.get("regions"):
        return []
    left, top, right, bottom = field.input_box
    x0, y0, x1, y1 = regions["regions"][0]
    if not (x0 <= left < right <= x1 and y0 <= top < bottom <= y1 and _available(field.input_box, regions)):
        return []
    return [{"role": "input", "function": "输入搜索词", "preview_text": field.caption,
             "box": [left, top, right, bottom],
             "click_point": [(left + right) // 2, (field.caption_box[1] + field.caption_box[3]) // 2]}]
