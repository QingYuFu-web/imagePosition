"""Locate labeled range controls in divided game settings rows."""

from statistics import median

import cv2
import numpy as np

from inventory_list_rows import _Box
from inventory_slider_actions import _spans
from screen_colored_settings_panel import _printed
from screen_dialog_close import _cross_footprint
from screen_landscape_settings_rail import _literal_word
from screen_range_geometry import ranged_controls
from screen_setting_ranges import range_action


_SETTINGS = ("设置", "设定", "选项", "退出", "画质", "音频", "返回")
_SECTIONS = ("镜头拖动", "辅助观察", "标记", "镜头移动", "自由视角", "瞄准", "瞄准辅助", "常规设置",
             "镜头跟随", "普通模式", "训练设置", "基础", "操作", "视角", "镜头", "武器")
_RANGE_NAMES = ("横向灵敏度", "纵向灵敏度", "水平灵敏度", "垂直灵敏度", "镜头灵敏度", "转向灵敏度",
                "瞄准灵敏度", "横向速度", "纵向速度", "移动速度", "镜头速度", "垂直速度", "水平速度")
_TABS = ("摇杆与轮盘", "陀螺仪", "镜头灵敏度", "武器灵敏度", "飞索灵敏度", "镜头与轮盘", "手柄与轮盘",
         "视角灵敏度", "转向仪", "瞄准器", "重置灵敏度", "武器特效", "滑动灵敏度", "辅助瞄准")
_FOOTER = ("恢复默认", "恢复默认值", "重置设置", "重新设置", "恢复设置", "保存设置", "应用设置", "确认设置")


def _sections(gray):
    height, width = gray.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (round(.09 * width), max(1, round(.004 * height))))
    dark = cv2.morphologyEx(cv2.compare(gray, 90, cv2.CMP_LE), cv2.MORPH_OPEN, kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    boxes = [_Box(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, count in stats[1:]
             if w >= .43 * width and .032 * height <= h <= .15 * height and count >= .86 * w * h]
    heads = [box for box in boxes if box.top <= .05 * height and box.width >= .91 * width and box.bottom <= .19 * height]
    rows = sorted((box for box in boxes if .12 * height <= box.top <= .69 * height and
                   .084 * width <= box.left <= .45 * width and .74 * width <= box.right <= .97 * width), key=lambda x: x.top)
    if len(heads) != 1 or not 2 <= len(rows) <= 5 or heads[0].bottom >= rows[0].top:
        return None
    first = rows[0]
    if any(max(abs(box.left - first.left), abs(box.right - first.right)) > .014 * width or
           abs(box.height - first.height) > .29 * first.height for box in rows[1:]):
        return None
    strides = [b.top - a.top for a, b in zip(rows, rows[1:])]
    pitch = median(strides)
    if not 2.15 * first.height <= pitch <= 6.0 * first.height or max(strides) - min(strides) > .16 * pitch:
        return None
    if sum((gray[box.bottom:box.bottom + round(.37 * pitch):3, box.left:box.right:4] >= 156).mean() >= .82
           for box in rows) != len(rows):
        return None
    return heads[0], rows, pitch


def _named(gray, box, index, names, *, light, isolated=False):
    if isolated:
        left, top, right, bottom = box
        tile = gray[top:bottom, left:right]
        if min(tile.shape) < 9:
            return None
        if not light:
            tile = cv2.bitwise_not(tile)
        for scale in (1, 1.55):
            patch = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else tile
            detected = _literal_word(patch, (0, 0, patch.shape[1], patch.shape[0]), names, index)
            if detected is not None:
                word, ink = detected
                x, y, r, b = (round(value / scale) for value in ink)
                return word, _Box(x + left, y + top, r + left, b + top)
        return None
    reading = _printed(gray, box, index, vocabulary=names, light=light, minimum_score=.82, minimum_margin=.022)
    if reading is not None:
        return reading, _Box(*box)
    return None


def _close(gray, header, bounds):
    _, width = gray.shape
    left, right = round(.70 * width), min(width, bounds.right + round(.017 * width))
    top, bottom = round(.32 * header.bottom), header.bottom - max(3, round(.064 * header.height))
    chip = gray[top:bottom, left:right]
    snapshots = []
    link = (max(1, int(.023 * header.height)) * 2 + 1)
    kernel = np.ones((link, link), np.uint8)
    levels = (142, 166) if header.height >= 50 else (128, 154)
    for threshold in levels:
        raw = cv2.compare(chip, threshold, cv2.CMP_GE)
        shapes = []
        for mask in (raw, cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)):
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            for k, (x, y, w, h, _) in enumerate(stats[1:], 1):
                if not (.24 * header.height <= min(w, h) <= max(w, h) <= .57 * header.height and .78 <= w / h <= 1.40):
                    continue
                if _cross_footprint(labels[y:y + h, x:x + w] == k):
                    shapes.append((int(left + x), int(top + y), int(left + x + w), int(top + y + h)))
        if not shapes or max(abs(a - b) for a, b in zip(min(shapes), max(shapes), strict=True)) > 2:
            return None
        snapshots.append(_Box(*min(shapes)))
    first, second = snapshots
    if max(abs(a - b) for a, b in zip(first.xyxy(), second.xyxy(), strict=True)) > max(2, round(.043 * header.height)):
        return None
    return first


def _rail_pixels(frame, raw):
    thumb, track = raw.thumb, raw.track
    xc, yc = map(round, thumb.center)
    step = max(4, round(.32 * thumb.height))
    if not step <= yc < frame.shape[0] - step:
        return None
    left, right = track.left, track.right
    row = frame[yc, left:right].astype(np.int16)
    above = frame[yc - step, left:right].astype(np.int16)
    below = frame[yc + step, left:right].astype(np.int16)
    difference = np.minimum(np.abs(row - above).max(axis=1), np.abs(row - below).max(axis=1))
    mask = cv2.morphologyEx(np.uint8((difference >= 16)[None]), cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))[0]
    # A raised thumb bridges the value rail where it interrupts the center-line samples.
    mask[max(0, thumb.left - left):min(right, thumb.right) - left] = 1
    return next((_Box(left + a, track.top, left + b, track.bottom) for a, b in _spans(mask)
                 if a <= xc - left < b and b - a >= .64 * track.width), None)


def _cells(frame, gray, rows, pitch, index):
    sliders = ranged_controls(frame)
    sliders.sort(key=lambda entry: entry.thumb.center[0])
    measured, paired = [], []
    for header in rows:
        selected = [row for row in sliders if header.bottom + .20 * pitch <= row.thumb.center[1] <= header.top + .90 * pitch
                    and header.left <= row.track.left < row.track.right <= header.right
                    and .30 * header.width <= row.track.width <= .49 * header.width]
        if not 1 <= len(selected) <= 2:
            continue
        starts = [(row.track.left - header.left) / header.width for row in selected]
        if len(selected) == 2:
            if (abs(selected[0].thumb.center[1] - selected[1].thumb.center[1]) > .055 * pitch
                    or abs(starts[1] - starts[0] - .50) > .068):
                continue
            paired.append(starts)
        measured.append((header, selected, starts))
    if len(paired) < 2:
        return []
    lanes = tuple(median(group) for group in zip(*paired, strict=True))

    labels = []
    for header, selected, starts in measured:
        height = header.height
        padding = max(1, round(.09 * height))
        name_box = (header.left + round(.98 * height), header.top + padding,
                    header.left + round(4.0 * height), header.bottom - padding)
        section = (_named(gray, name_box, index, _SECTIONS, light=True, isolated=True)
                   or _named(gray, name_box, index, _SECTIONS, light=True))
        if section is None:
            continue
        names = []
        for row, start in zip(selected, starts, strict=True):
            if min(abs(start - lane) for lane in lanes) > .035:
                continue
            l, y = row.track.left, header.bottom
            capture = (max(header.left, l - round(.20 * height)), y + max(1, round(.07 * height)),
                       min(header.right, l + round(2.83 * height)), y + round(1.12 * height))
            label = _named(gray, capture, index, _RANGE_NAMES, light=False, isolated=True)
            rail = _rail_pixels(frame, row)
            if label is not None and rail is not None:
                names.append((row.thumb, rail, *label))
        if names and len({name for _, _, name, _ in names}) == len(names):
            labels.append((*section, names))
    if (len(labels) < 2 or len({label for label, _, _ in labels}) != len(labels)
            or all(len(ranges) == 1 for _, _, ranges in labels)):
        return []
    actions = []
    for section, heading, ranges in labels:
        for thumb, rail, name, caption in ranges:
            reading = section + name
            action = range_action(thumb, rail, reading, reading, (0, 0))
            left, top, right, bottom = action["box"]
            action.update(preview_text=reading,
                          context_box=[min(left, heading.left, caption.left), min(top, heading.top, caption.top),
                                       max(right, heading.right, caption.right), max(bottom, heading.bottom, caption.bottom)])
            actions.append(action)
    return actions


def _action(role, function, box, **details):
    return {"role": role, "function": function, "click_point": list(box.center), "box": list(box.xyxy()), **details}


def _sidebar(frame, gray, rows, pitch, index):
    first = rows[0]
    unit = first.height
    left, right = round(.024 * gray.shape[1]), first.left - round(.24 * unit)
    start, stop = first.top + round(1.52 * unit), min(gray.shape[0], round(rows[-1].top + .67 * pitch))
    chip = gray[start:stop, left:right]
    if min(chip.shape) < 8:
        return []
    span = max(5, round(.22 * unit)) | 1
    contrast = cv2.morphologyEx(chip, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.compare(contrast, 25, cv2.CMP_GE))
    marks = [_Box(int(x + left), int(y + start), int(x + left + w), int(y + start + h))
             for x, y, w, h, n in stats[1:] if .132 * unit <= min(w, h) <= max(w, h) <= .37 * unit
             and .70 <= w / h <= 1.50 and .39 * w * h <= n <= .82 * w * h]
    lanes = {tuple(sorted((p for p in marks if abs(p.center[0] - point.center[0]) <= .11 * unit), key=lambda p: p.top))
             for point in marks}
    plausible = []
    for lane in lanes:
        if not 4 <= len(lane) <= 7:
            continue
        steps = [b.center[1] - a.center[1] for a, b in zip(lane, lane[1:])]
        if not 1.1 * unit <= min(steps) <= max(steps) <= 2.8 * unit or max(steps) - min(steps) >= .25 * median(steps):
            continue
        plausible.append(lane)
    if len(plausible) != 1:
        return []
    lane = plausible[0]
    colors = []
    for p in lane:
        cx, cy = map(round, p.center)
        local = frame[cy - round(.23 * unit):cy + round(.23 * unit) + 1, cx - 2:cx + round(.57 * unit)]
        blue, green, red = cv2.split(local)
        colors.append(float(np.percentile(cv2.min(cv2.subtract(red, green), cv2.subtract(red, blue)), 74)))
    order = np.argsort(colors)
    if colors[order[-1]] < max(13., colors[order[-2]] + 9):
        return []
    actions = []
    for k, p in enumerate(lane):
        if k == order[-1]:
            continue
        cx, cy = map(round, p.center)
        dx, dy = round(.32 * unit), round(.40 * unit)
        box = (cx + dx, cy - dy, min(first.left - 1, cx + round(3.50 * unit)), cy + dy + 1)
        if known := _named(gray, box, index, _TABS, light=False, isolated=True):
            name, ink = known
            actions.append(_action("switch_tab", f"切换到{name}", ink, preview_text=name,
                                   setting_label=name, setting_state="off"))
    return actions


def _footer(gray, bounds, pitch, index):
    left = bounds.right - round(.27 * bounds.width)
    top = bounds.top + round(1.15 * pitch)
    right = min(gray.shape[1], bounds.right + round(.033 * bounds.width))
    chip = gray[top:round(.955 * gray.shape[0]), left:right]
    contour, _ = cv2.findContours(cv2.Canny(chip, 22, 54), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    forms = set()
    for edge in contour:
        x, y, w, h = cv2.boundingRect(edge)
        if (.135 * bounds.width <= w <= .245 * bounds.width and .24 * pitch <= h <= .57 * pitch
                and 2.5 <= w / h <= 6.0 and abs(left + x + w - bounds.right) <= .029 * bounds.width
                and cv2.contourArea(edge) >= .86 * w * h):
            forms.add((left + x, top + y, left + x + w, top + y + h))
    if not forms or max(abs(a - b) for a, b in zip(min(forms), max(forms), strict=True)) > 5:
        return None
    p = _Box(*min(forms))
    xpad, ypad = round(.18 * p.width), round(.09 * p.height)
    close_text = _named(gray, (p.left + xpad, p.top + ypad, p.right - xpad, p.bottom - ypad), index, _FOOTER, light=True)
    if close_text is None or close_text[0] not in {"恢复默认", "恢复默认值", "重置设置"}:
        return None
    return _action("reset", close_text[0], p)


def grouped_range_grid(frame, overlay, index):
    """Return verified controls and a page scope only for the visible settings grid."""
    height, width = frame.shape[:2]
    if index is None or overlay is not None or height < 300 or not 1.55 * height <= width <= 2.88 * height:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (layout := _sections(gray)) is None:
        return None
    header, rows, pitch = layout
    title = (round(.469 * width), round(.35 * header.bottom), round(.531 * width), header.bottom - 1)
    if (caption := _named(gray, title, index, _SETTINGS, light=True)) is None or caption[0] not in _SETTINGS[:3]:
        return None
    if (end := _close(gray, header, rows[0])) is None:
        return None
    ranges = _cells(frame, gray, rows, pitch, index)
    if not ranges:
        return None
    actions = [_action("close", "关闭设置", end), *ranges, *_sidebar(frame, gray, rows, pitch, index)]
    if reset := _footer(gray, rows[-1], pitch, index):
        actions.append(reset)
    return (0, 0, width, height), actions
