"""Recognize a dark native edit form through its surface and attached captions."""

from dataclasses import dataclass

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_dialog_close import _cross_footprint
from screen_dialog_search_input import _fold
from screen_native_tab_bar import _distance


@dataclass(frozen=True)
class DarkDialogFields:
    panel: tuple[int, int, int, int]
    actions: tuple[dict, ...]


_FILE_CAPTIONS = {"filename", "archivename", "文件名", "档案名称", "檔案名稱", "压缩包名称", "壓縮檔名稱"}
_PASSWORD_CAPTIONS = {"password", "archivepassword", "密码", "密碼", "压缩密码", "壓縮密碼"}
_ARCHIVE_TITLES = {"createarchive", "newarchive", "createziparchive", "创建压缩包", "新建压缩包", "新增封存檔"}


def _panel_surface(frame):
    height, width = frame.shape[:2]
    step = max(4, round(.015 * width))
    paper = np.median(frame[round(.27 * height):round(.67 * height):step,
                            round(.24 * width):round(.76 * width):step], axis=(0, 1))
    if not 33 <= paper.mean() <= 107 or np.ptp(paper) > 17:
        return None
    ratio = min(1, 375 / width)
    work = cv2.resize(frame, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
    h, w = work.shape[:2]
    mask = np.uint8(_distance(work, paper) <= 9)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    matches = []
    for contour in contours:
        x, y, a, b = cv2.boundingRect(contour)
        r, k = x + a, y + b
        if (not .57 * w <= a <= .94 * w or not .28 * h <= b <= .68 * h
                or abs(x + r - w) > .065 * w or not .13 * h <= y <= .38 * h
                or not .63 * h <= k <= .89 * h or cv2.contourArea(contour) < .922 * a * b):
            continue
        pad = max(2, round(.012 * w))
        if min(x, y, w - r, h - k) <= 2 * pad:
            continue
        inner = mask[y + pad:k - pad, x + pad:r - pad]
        if inner.mean() < .72:
            continue
        outside = (
            work[y - 2 * pad:y - pad, x + round(.18 * a):r - round(.18 * a)],
            work[k + pad:k + 2 * pad, x + round(.18 * a):r - round(.18 * a)],
            work[y + round(.16 * b):k - round(.16 * b), x - 2 * pad:x - pad],
            work[y + round(.16 * b):k - round(.16 * b), r + pad:r + 2 * pad],
        )
        support = sorted((chip.mean(axis=2) <= paper.mean() - 15).mean() for chip in outside)
        if support[0] < .34 or support[1] < .80:
            continue
        matches.append((round(x * width / w), round(y * height / h), round(r * width / w), round(k * height / h)))
    return matches[0] if len(matches) == 1 else None


def _line_rows(ink):
    h, w = ink.shape
    _, parts, stats, _ = cv2.connectedComponentsWithStats(np.uint8(ink >= 43), connectivity=8)
    for i, (x, y, a, b, count) in enumerate(stats[1:], 1):
        if (.58 * w <= a <= .98 * w and 1 <= b <= .016 * w and .14 * h <= y <= .75 * h
                and .008 * w <= x <= .18 * w and x + a <= .99 * w
                and count >= .84 * a * b and (parts[y:y + b, x:x + a] == i).mean(axis=1).max() >= .971):
            yield tuple(map(int, (x, y, x + a, y + b)))


def _label_bounds(ink, line):
    w = ink.shape[1]
    left, y, _, _ = line
    x0, x1 = max(1, left - round(.035 * w)), left + round(.48 * w)
    a, b = y - round(.161 * w), y - round(.087 * w)
    mask = np.uint8(ink[a:b, x0:x1] >= 47)
    rows = [(u, v) for u, v in _bands(mask.sum(axis=1) >= max(2, round(.004 * w)))
            if .019 * w <= v - u <= .054 * w]
    if len(rows) != 1:
        return None
    u, v = rows[0]
    x, t, span, h = cv2.boundingRect(mask[u:v])
    t += u
    if not (.019 * w <= h <= .054 * w and 1.35 * h <= span <= .46 * w
            and abs(x0 + x - left) <= .025 * w and 0 < t < t + h < len(mask)):
        return None
    return x0 + x, a + t, x0 + x + span, a + t + h


def _text_value(ink, row, label):
    x, y, r, _ = row
    w = ink.shape[1]
    a, b = max(label[3] + round(.018 * w), y - round(.081 * w)), y - max(2, round(.006 * w))
    x0, x1 = x, x + round(.78 * (r - x))
    tile = np.uint8(ink[a:b, x0:x1] >= 43)
    start, t, span, height = cv2.boundingRect(tile)
    if not (.026 * w <= height <= .074 * w and start <= .046 * w and .045 * w <= span <= .74 * (r - x)
            and 0 < t + height <= len(tile)):
        return None
    return x0 + start, a + t, x0 + start + span, a + t + height


def _caret_matches(panel, row, label, value):
    x, y, r, b = row
    swatch = np.median(panel[y:b, x + round(.25 * (r - x)):r - round(.02 * (r - x))], axis=(0, 1))
    if np.ptp(swatch) < 44:
        return False
    w = panel.shape[1]
    view = panel[max(label[3] + round(.014 * w), y - round(.086 * w)):y - max(2, round(.004 * w)),
                 x:value[2] + round(.018 * w)]
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(_distance(view, swatch) <= 26), connectivity=8)
    return any(.0016 * w <= a <= .0105 * w and .040 * w <= b <= .083 * w and n >= .75 * a * b
               for _, _, a, b, n in stats[1:])


def _read_box(frame, panel, bounds, bank, cache):
    x, y, r, b = bounds
    left, top, _, _ = panel
    pad = max(2, round(.18 * (b - y)))
    box = (left + max(0, x - pad), top + y - pad, left + r + pad, top + b + pad)
    result = read_caption(frame, box, bank, cache=cache, choice=True)
    return result["text"] if result else None


def _creation_title(frame, panel, ink, label, index, cache):
    height, width = ink.shape
    start, end = round(.022 * height), label[1] - round(.018 * width)
    words = np.uint8(ink[start:end, :round(.83 * width)] >= 57)
    x, y, w, h = cv2.boundingRect(words)
    if not (.027 * width <= h <= .072 * width and 2.8 * h <= w <= .64 * width
            and .025 * width <= x <= .16 * width and y + h < len(words)):
        return False
    caption = _read_box(frame, panel, (x, start + y, x + w, start + y + h), index, cache)
    return caption is not None and _fold(caption) in _ARCHIVE_TITLES


def _square_span(profile, first, last, travel):
    """Refine the two opposed face edges along their uniform painted rows."""
    step = max(2, travel // 9)
    starts, ends = [], []
    for edge, marks, direction in ((first, starts, 1), (last, ends, -1)):
        for pos in range(max(step, edge - travel), min(len(profile) - step, edge + travel)):
            before, after = profile[pos - step:pos], profile[pos:pos + step]
            inside, outside = (after, before) if direction == 1 else (before, after)
            if inside.mean() >= .88 and outside.mean() <= .37:
                marks.append((inside.mean() - outside.mean(), pos))
    if starts and ends:
        return max(starts)[1], max(ends)[1]
    return None


def _deep_panel_surface(frame, paint):
    height, width = frame.shape[:2]
    if not 9 <= paint.mean() < 34 or np.ptp(paint) > 21:
        return None
    scale = min(1, 375 / width)
    view = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    h, w = view.shape[:2]
    matte = np.uint8(_distance(view, paint) <= 2)
    contours, _ = cv2.findContours(matte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels = []
    for contour in contours:
        x, y, span, deep = cv2.boundingRect(contour)
        if (not .61 * w <= span <= .91 * w or not .29 * h <= deep <= .70 * h
                or not .125 * h <= y <= .35 * h or abs(2 * x + span - w) > .067 * w
                or cv2.contourArea(contour) < .915 * span * deep):
            continue
        left, right = (round(q * width / w) for q in (x, x + span))
        top, bottom = (round(q * height / h) for q in (y, y + deep))
        inset = round(.24 * (right - left))
        column_view = frame[:, left + inset:right - inset:4]
        vertical = (_distance(column_view, paint) <= 3).mean(axis=1)
        reach = max(7, round(.069 * (right - left)))
        if (vspan := _square_span(vertical, top, bottom, reach)) is None:
            continue
        top, bottom = vspan
        inset = round(.103 * (bottom - top))
        row_view = frame[top + inset:bottom - inset:5]
        horizontal = (_distance(row_view, paint) <= 4).mean(axis=0)
        if (hspan := _square_span(horizontal, left, right, reach)) is None:
            continue
        left, right = hspan
        if not .67 * width <= right - left <= .89 * width or not .42 * height <= bottom - top <= .68 * height:
            continue
        panels.append((left, top, right, bottom))
    return panels[0] if len(panels) == 1 else None


def _flat_field_rules(ink):
    height, width = ink.shape
    pieces = np.uint8(ink >= 21)
    _, ids, stats, _ = cv2.connectedComponentsWithStats(pieces, connectivity=8)
    for i, (x, y, w, h, n) in enumerate(stats[1:], 1):
        if (.57 * width <= w <= .93 * width and 1 <= h <= .014 * width and .123 * height <= y <= .89 * height
                and .029 * width <= x <= .11 * width and n >= .73 * w * h
                and (ids[y:y + h, x:x + w] == i).mean(axis=1).max() >= .946):
            yield tuple(map(int, (x, y, x + w, y + h)))


def _placeholder(frame, panel, ink, line, index, cache):
    x, y, r, _ = line
    width = ink.shape[1]
    a, b = y - round(.084 * width), y - max(2, round(.004 * width))
    mask = np.uint8(ink[a:b, x:r] >= 47)
    left, top, w, h = cv2.boundingRect(mask)
    if (not .023 * width <= h <= .064 * width or not 2.0 * h <= w <= .75 * (r - x)
            or left > .019 * width or not .21 * h <= y - (a + top + h) <= 1.02 * h):
        return None
    box = (x + left, a + top, x + left + w, a + top + h)
    text = _read_box(frame, panel, box, index, cache)
    if text is None:
        mask = mask[top:top + h, left:left + w]
        spans = _bands(mask.any(axis=0))
        spaces = [r for (_, r), (l, _) in zip(spans, spans[1:]) if l - r >= .235 * h]
        if len(spaces) >= 2:
            prefix = (*box[:2], box[0] + spaces[1], box[3])
            text = _read_box(frame, panel, prefix, index, cache)
            if text is None and _split_length_printed(mask[:, :spaces[1]], h, index):
                text = "Split length"
    return (box, text) if text else None


def _split_length_printed(mask, height, index):
    runs = _bands(mask.any(axis=0))
    word = "splitlength"
    if len(runs) != len(word) or runs[5][0] - runs[4][1] < .25 * height:
        return False
    for (l, r), char in zip(runs, word, strict=True):
        shape = mask[:, l:r]
        x, y, w, h = cv2.boundingRect(shape)
        ranks = index.rank(shape[y:y + h, x:x + w], alphabet="choice_latin", limit=5)
        if not ranks:
            return False
        best = ranks[0]
        if char == "l" and w <= .20 * h and any(r["text"] == "l" and r["distance"] <= .112 for r in ranks[:2]):
            continue
        matches = [r for r in ranks if r["text"].casefold() == char]
        if not matches or matches[0]["distance"] > .168 or matches[0]["distance"] > best["distance"] + .016:
            return False
    return True


def _dark_header_close(ink, panel, title_y):
    left, top, _, _ = panel
    height, width = ink.shape
    x0, y0 = round(.843 * width), round(.019 * height)
    tile = ink[y0:round(.104 * height), x0:round(.965 * width)]
    evidence = []
    for cut in (32, 44, 57):
        _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(tile >= cut), connectivity=8)
        for i, (a, b, w, h, _) in enumerate(stats[1:], 1):
            if (.023 * width <= min(w, h) <= max(w, h) <= .069 * width and .79 <= w / h <= 1.22
                    and 0 < a < a + w < tile.shape[1] and 0 < b < b + h < len(tile)
                    and abs(y0 + b + h / 2 - title_y) <= .031 * width
                    and _cross_footprint(ids[b:b + h, a:a + w] == i)):
                evidence.append(tuple(map(int, (left + x0 + a, top + y0 + b, left + x0 + a + w, top + y0 + b + h))))
    if len(evidence) < 2 or np.ptp(evidence, axis=0).max() > max(2, .005 * width):
        return None
    a, b, r, k = evidence[len(evidence) // 2]
    pad = round(.011 * width)
    return {"role": "close", "function": "关闭", "preview_text": "关闭",
            "box": [a - pad, b - pad, r + pad, k + pad], "click_point": [(a + r) // 2, (b + k) // 2]}


def _dark_stack_fields(frame, panel, index):
    left, top, right, bottom = panel
    width, height = right - left, bottom - top
    if (min(width, height) < 210 or not (0 <= left < right <= frame.shape[1]
                                       and 0 <= top < bottom <= frame.shape[0])):
        return None
    tile = frame[top:bottom, left:right]
    paint = np.median(tile[::7, ::8], axis=(0, 1))
    ink = _distance(tile, paint)
    rows = sorted(_flat_field_rules(ink), key=lambda r: r[1])
    if not 3 <= len(rows) <= 6:
        return None
    header_left, header_top = round(.026 * width), round(.019 * height)
    header = np.uint8(ink[header_top:round(.092 * height), header_left:round(.62 * width)] >= 59)
    x, y, w, h = cv2.boundingRect(header)
    if not (.023 * width <= h <= .074 * width and 1.44 * h <= w <= .48 * width
            and .012 * width <= x <= .111 * width and y + h < len(header)):
        return None
    cache = {}
    title = _read_box(frame, panel, (header_left + x, header_top + y, header_left + x + w, header_top + y + h),
                      index, cache)
    if not title or _fold(title) not in _ARCHIVE_TITLES | {"archive", "压缩", "壓縮", "创建归档"}:
        return None
    actions, key_line = [], None
    for line in rows:
        result = _placeholder(frame, panel, ink, line, index, cache)
        if result is None:
            continue
        box, text = result
        name = _fold(text)
        if name in _FILE_CAPTIONS | {"enterthename", "enterfilename", "entername", "输入压缩包名称", "输入名称"}:
            function = "输入文件名"
        elif name in _PASSWORD_CAPTIONS | {"enterthekey", "enterpassword", "输入密码", "输入密钥"}:
            function, key_line = "输入密码", line
        elif name in {"re-enter", "reenter", "confirmpassword", "再次输入", "再次输入密码"}:
            if key_line is None or not (.074 * width <= line[1] - key_line[1] <= .20 * width
                                        and max(abs(line[i] - key_line[i]) for i in (0, 2)) <= .018 * width):
                continue
            function = "再次输入密码"
        elif name.startswith(("splitlength", "分卷长度", "分卷大小")):
            function = "输入分卷大小"
        else:
            continue
        x, _, r, b = line
        hit = [left + x, top + box[1] - round(.018 * width), left + r, top + b]
        actions.append({"role": "input", "function": function, "preview_text": text,
                        "box": hit, "click_point": [left + (x + r) // 2, top + (box[1] + box[3]) // 2]})
    names = {a["function"] for a in actions}
    if not {"输入文件名", "输入密码"} <= names or len(actions) != len(names):
        return None
    if close := _dark_header_close(ink, panel, header_top + y + h / 2):
        actions.append(close)
    return DarkDialogFields(tuple(panel), tuple(actions))


def dark_dialog_fields(frame, font_index, *, panel=None):
    """Measure a dark dialog only after two physically owned controls corroborate it."""
    height, width = frame.shape[:2]
    if font_index is None or width < 280 or not 1.48 * width <= height <= 2.83 * width:
        return None
    if panel is not None:
        return _dark_stack_fields(frame, panel, font_index)
    panel = _panel_surface(frame)
    if panel is None:
        stride = max(5, round(.019 * width))
        paint = np.median(frame[round(.25 * height):round(.71 * height):stride,
                                round(.25 * width):round(.75 * width):stride], axis=(0, 1))
        deep = _deep_panel_surface(frame, paint)
        return _dark_stack_fields(frame, deep, font_index) if deep is not None else None
    left, top, right, bottom = panel
    part = frame[top:bottom, left:right]
    ink = _distance(part, np.median(part[::5, ::6], axis=(0, 1)))
    rows = list(_line_rows(ink))
    if not 2 <= len(rows) <= 4:
        return None
    cache, actions = {}, []
    w = right - left
    for row in rows:
        label = _label_bounds(ink, row)
        if label is None or (value := _text_value(ink, row, label)) is None:
            continue
        caption = _read_box(frame, panel, label, font_index, cache)
        if caption is None:
            continue
        name = _fold(caption)
        if name in _PASSWORD_CAPTIONS:
            purpose = "输入密码"
        elif name in _FILE_CAPTIONS or (name in {"archive", "压缩包", "壓縮檔"} and (
                _caret_matches(part, row, label, value)
                or _creation_title(frame, panel, ink, label, font_index, cache))):
            purpose = "输入文件名"
        else:
            continue
        x, y, r, b = row
        end = r - round(.086 * w) if purpose == "输入密码" else r
        hit = [left + x, top + value[1] - round(.024 * w), left + end, top + b]
        actions.append({"role": "input", "function": purpose, "preview_text": caption,
                        "box": hit,
                        "click_point": [left + (x + end) // 2, top + (value[1] + value[3]) // 2]})
    if {item["function"] for item in actions} != {"输入文件名", "输入密码"} or len(actions) != 2:
        return None
    return DarkDialogFields(panel, tuple(actions))
