"""Locate a raised destination chooser beside its verified archive-name field."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_dialog_search_input import _fold
from screen_landscape_search_field import _available
from screen_native_tab_bar import _distance


_DESTINATIONS = {"destination", "destinationpath", "savepath", "outputpath", "保存路径", "输出路径", "儲存路徑", "輸出路徑"}
_ARCHIVE_NAMES = {"archive", "archivename", "filename", "压缩包", "压缩包名称", "文件名", "壓縮檔", "壓縮檔名稱", "檔案名稱"}
_ARCHIVE_CREATION = {"createarchive", "newarchive", "createziparchive", "创建压缩包", "新建压缩包", "新增封存檔"}


def _underline(frame, field):
    left, top, right, bottom = field
    width, height = right - left, bottom - top
    rows = max(3, round(.12 * height))
    start, stop = left + round(.052 * width), right - max(2, round(.037 * height))
    if stop <= start or 2 * rows >= height:
        return False
    before = frame[bottom - 2 * rows:bottom - rows, start:stop]
    paint = np.median(before[::2, ::3], axis=(0, 1))
    line = _distance(frame[bottom - rows:bottom, start:stop], paint)
    return (line >= 37).mean(axis=1).max() >= .945


def _plate(frame, panel, field):
    _, panel_top, panel_right, panel_bottom = panel
    _, top, edge, bottom = field
    depth = bottom - top
    x0 = edge + max(2, round(.061 * depth))
    x1 = min(panel_right - 1, edge + round(1.72 * depth))
    y0 = max(panel_top, top - round(.24 * depth))
    y1 = min(panel_bottom, bottom + round(.26 * depth))
    if x1 - x0 < .75 * depth or y1 - y0 < 1.07 * depth:
        return None
    sample = frame[y0:y1, x0:x1]
    pad = max(3, round(.17 * depth))
    matte = np.median(sample[:pad], axis=(0, 1))
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    base = float(np.median(gray[:pad]))
    plate_ink = (gray >= base + 13) & (gray <= base + 78)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(plate_ink), connectivity=8)
    matches = []
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        l, t, r, b = x0 + int(x), y0 + int(y), x0 + int(x + w), y0 + int(y + h)
        if (not .65 * depth <= min(w, h) <= max(w, h) <= 1.42 * depth
                or not .76 <= w / h <= 1.44 or not .90 * w * h <= area < .9995 * w * h
                or not .11 * depth <= l - edge <= .56 * depth or not .07 * depth <= panel_right - r <= .57 * depth
                or not top - .18 * depth <= t <= top + .32 * depth
                or not bottom - .23 * depth <= b <= bottom + .24 * depth
                or not 0 < x < x + w < sample.shape[1] or not 0 < y < y + h < sample.shape[0]):
            continue
        paint = np.median(sample[y:y + h, x:x + w][labels[y:y + h, x:x + w] == i], axis=0)
        if np.ptp(paint - matte) > 25 or np.mean(paint - matte) < 15:
            continue
        pad = max(2, round(.063 * depth))
        field_gap = frame[t + pad:b - pad, l - pad:l - 1]
        panel_gap = frame[t + pad:b - pad, r + 1:r + pad + 1]
        if any((_distance(gap, matte) <= 11).mean() < .73 for gap in (field_gap, panel_gap)):
            continue
        if _three_dots(frame[t:b, l:r], paint):
            matches.append(((l, t, r, b), matte))
    return matches[0] if len(matches) == 1 else None


def _three_dots(tile, paint):
    height, width = tile.shape[:2]
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    paper = int(cv2.cvtColor(np.uint8([[paint]]), cv2.COLOR_BGR2GRAY)[0, 0])
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(gray >= paper + 58), connectivity=8)
    dots = sorted((x + w / 2, y + h / 2, int(w), int(h)) for x, y, w, h, area in stats[1:]
                  if area >= 2)
    if len(dots) != 3:
        return False
    xs, ys, widths, heights = zip(*dots, strict=True)
    pitch = np.diff(xs)
    if any(not .018 * height <= min(w, h) <= max(w, h) <= .085 * height or not .64 <= w / h <= 1.57
           for w, h in zip(widths, heights, strict=True)):
        return False
    return (.065 * width <= min(pitch) <= max(pitch) <= .178 * width and max(pitch) <= 1.23 * min(pitch)
            and np.ptp(ys) <= max(1, .035 * height) and max(widths) - min(widths) <= max(1, .026 * height)
            and abs(np.mean(xs) - width / 2) <= .096 * width and .40 * height <= np.mean(ys) <= .72 * height)


def _archive_title(frame, panel, field, matte, font_index):
    left, top, right, _ = panel
    width = right - left
    x0, x1 = left + round(.033 * width), right - round(.17 * width)
    y0, y1 = top + round(.022 * width), field[1] - round(.046 * width)
    if y1 - y0 < .052 * width:
        return False
    ink = _distance(frame[y0:y1, x0:x1], matte)
    x, y, w, h = cv2.boundingRect(np.uint8(ink >= 63))
    if (not .025 * width <= h <= .078 * width or not 1.95 * h <= w <= .65 * width
            or not .015 * width <= x <= .15 * width or y + h == len(ink)):
        return False
    pad = max(2, round(.15 * h))
    box = (max(x0, x0 + x - pad), max(y0, y0 + y - pad),
           min(x1, x0 + x + w + pad), min(y1, y0 + y + h + pad))
    seen = read_caption(frame, box, font_index, cache={}, choice=True)
    return seen is not None and _fold(seen["text"]) in _ARCHIVE_CREATION


def _within(inner, outer):
    x, y, r, b = inner
    l, t, k, z = outer
    return l <= x < r <= k and t <= y < b <= z


def dialog_field_accessory_actions(frame, regions, font_index, *, fields):
    """Read one separately painted browse control owned by a measured destination input."""
    if fields is None or regions.get("layer") != "dialog" or not regions.get("regions") or font_index is None:
        return []
    owners = [action for action in fields.actions
              if action["role"] == "input" and action["function"] == "输入文件名"]
    if len(owners) != 1:
        return []
    field = owners[0]["box"]
    owner_name = _fold(owners[0]["preview_text"])
    authoritative = regions["regions"][0]
    if (owner_name not in _ARCHIVE_NAMES | _DESTINATIONS or not _within(field, fields.panel)
            or not _within(field, authoritative) or not _available(field, regions) or not _underline(frame, field)):
        return []
    measured = _plate(frame, fields.panel, field)
    if measured is None:
        return []
    button, matte = measured
    if not _within(button, authoritative) or not _available(button, regions):
        return []
    if owner_name not in _DESTINATIONS and not _archive_title(frame, fields.panel, field, matte, font_index):
        return []
    l, t, r, b = button
    return [{"role": "browse", "function": "选择保存路径", "preview_text": "选择保存路径",
             "box": list(button), "click_point": [(l + r) // 2, (t + b) // 2]}]
