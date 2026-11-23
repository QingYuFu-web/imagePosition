"""Read a landscape command footer and an independent leading return arrow."""

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_component_captions import _bands
from screen_landscape_settings_rail import _literal_word


_CREATE = {"创建方案", "新建方案", "新增方案", "创建队伍", "新建队伍", "创建小队", "新建阵容", "创建阵容"}
_EDIT = {"快速编辑", "编辑小队", "编辑方案", "调整阵容", "编辑阵容", "编辑队伍", "快速调整", "修改方案"}
_CONFIRM = {"启用方案", "应用方案", "启用小队", "保存方案", "采用方案", "应用阵容", "启用阵容", "确认选择"}
_WORDS = tuple(sorted(_CREATE | _EDIT | _CONFIRM | {
    "删除方案", "重置方案", "正在使用", "当前方案", "已经启用", "暂不可用", "返回界面",
    "方案详情", "复制方案", "分享方案", "阵容管理", "选择方案", "预览方案",
}))


def _footer_edge(gray):
    height, width = gray.shape
    top, bottom = round(.81 * height), round(.955 * height)
    sample = gray[top:bottom, round(.028 * width):round(.986 * width):4]
    ridges = (sample >= 128).mean(axis=1) >= .87
    witness = []
    for x, y in _bands(ridges):
        if not 1 <= y - x <= .013 * height:
            continue
        edge = top + y
        rows = gray[edge:min(height, edge + round(.048 * height)), round(.23 * width):round(.58 * width):3]
        if rows.size and (rows <= 37).mean() >= .95:
            witness.append(edge)
    return witness[0] if len(witness) == 1 else None


def _command_buttons(frame, start):
    height, width = frame.shape[:2]
    tile = cv2.cvtColor(frame[start:], cv2.COLOR_BGR2HSV)
    boxes = []
    for color, tone in ((58, 87), (76, 114)):
        mask = cv2.inRange(tile, (0, color, tone), (179, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (.116 * width <= w <= .235 * width and .054 * height <= h <= .114 * height
                    and 2.5 <= w / h <= 5.6 and x > 0 and x + w < width and y + h < len(tile)
                    and .84 <= cv2.contourArea(contour) / (w * h) <= .992):
                found.append((x, start + y, x + w, start + y + h))
        boxes.append(found)
    return sorted(box for box in boxes[1] if any(
        max(abs(a - b) for a, b in zip(peer, box, strict=True)) <= .105 * (box[3] - box[1])
        for peer in boxes[0]))


def _plus(gray, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    tile = gray[top + round(.13 * h):bottom - round(.10 * h),
                left + round(.105 * w):left + round(.419 * w)]
    tone, bright = np.percentile(tile, (50, 98))
    if bright - tone < 51:
        return False
    boxes = []
    for ratio in (.45, .58, .70):
        ink = np.uint8(tile >= tone + ratio * (bright - tone))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
        for i, (x, y, gw, gh, area) in enumerate(stats[1:n], 1):
            if (not .31 * h <= min(gw, gh) <= max(gw, gh) <= .64 * h
                    or not .81 <= gw / gh <= 1.24 or not .31 <= area / (gw * gh) <= .59
                    or not 0 < x < x + gw < tile.shape[1] or not 0 < y < y + gh < tile.shape[0]):
                continue
            mark = labels[y:y + gh, x:x + gw] == i
            yy, xx = np.ogrid[-1:1:complex(0, gh), -1:1:complex(0, gw)]
            cross = (abs(xx) <= .36) | (abs(yy) <= .36)
            if ((mark & cross).sum() >= .93 * area and mark[gh // 2].mean() >= .81
                    and mark[:, gw // 2].mean() >= .81):
                boxes.append((x, y, x + gw, y + gh))
    return len(boxes) >= 2 and np.max(np.ptp(boxes, axis=0)) <= .10 * h


def _button_caption(gray, box, index, *, leading=False):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    inset = round(.15 * h)
    if leading:
        area = (left + round(.49 * w), top + inset, right - round(.025 * w), bottom - inset)
        paint, roles = gray, ((_CREATE, "create"),)
    else:
        area = (left + round(.16 * w), top + inset, right - round(.16 * w), bottom - inset)
        paint, roles = cv2.bitwise_not(gray), ((_EDIT, "edit"), (_CONFIRM, "confirm"))
    parsed = _literal_word(paint, area, _WORDS, index)
    if not parsed:
        return None
    name, (a, t, r, b) = parsed
    expected = left + (.72 if leading else .50) * w
    if abs((a + r) / 2 - expected) > .096 * w or abs((t + b - top - bottom) / 2) > .17 * h:
        return None
    return next(((role, name) for names, role in roles if name in names), None)


def _ring(gray, cx, cy, radius):
    angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    height, width = gray.shape

    def sample(factor):
        x, y = np.rint(cx + factor * radius * np.cos(angle)), np.rint(cy + factor * radius * np.sin(angle))
        return gray[np.intp(np.clip(y, 0, height - 1)), np.intp(np.clip(x, 0, width - 1))].astype(np.int16)

    core = sample(.77)
    beyond = sample(1.13)
    bright = np.maximum.reduce([sample(factor) for factor in (.87, .92, .97, 1.02)])
    visible = (bright >= 72) & (bright - core >= 45) & (bright - beyond >= 43)
    return all(sector.mean() >= .64 for sector in np.split(visible, 4))


def _wide_arrow(glyph):
    if is_shaft_back_arrow(glyph):
        return True
    h, w = glyph.shape
    if not 1.12 <= w / h <= 1.60 or not .28 <= glyph.mean() <= .47:
        return False
    center = np.flatnonzero(glyph.sum(axis=1) >= .77 * w)
    if (not center.size or center.size > .33 * h or np.any(np.diff(center) != 1)
            or abs(center.mean() + .5 - h / 2) > .115 * h):
        return False
    side = round(.20 * h)
    arms = [band.nonzero()[1] for band in (glyph[:side], glyph[-side:])]
    shoulder = glyph[round(.22 * h):round(.37 * h)]
    return all(arm.size and .26 * w <= np.median(arm) <= .58 * w and arm.max() <= .66 * w
               for arm in arms) and shoulder.any() and np.median(shoulder.nonzero()[1]) <= .36 * w


def _return_button(gray, size):
    height, width = gray.shape
    roi = gray[:round(.138 * height), :round(.103 * width)]
    soft = cv2.GaussianBlur(roi, (5, 5), 1.10)
    circles = cv2.HoughCircles(soft, cv2.HOUGH_GRADIENT, 1.20, .56 * size, param1=85, param2=23,
                               minRadius=round(.44 * size), maxRadius=round(.77 * size))
    if circles is None:
        return None
    boxes = set()
    for cx, cy, radius in circles[0]:
        if cx - radius < 1 or cy - radius < -2 or cx + radius >= roi.shape[1] or cy + radius >= roi.shape[0]:
            continue
        if not _ring(gray, cx, cy, radius):
            continue
        x0, y0 = round(cx - radius), max(0, round(cy - radius))
        x1, y1 = round(cx + radius + 1), round(cy + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        core = (xx - cx) ** 2 + (yy - cy) ** 2 <= (.858 * radius) ** 2
        tile = gray[y0:y1, x0:x1]
        ink_count = 0
        for level in (191, 214, 234):
            ink = np.uint8((tile >= level) & core)
            x, y, w, h = cv2.boundingRect(ink)
            if (not .81 * radius <= h <= 1.42 * radius or not 1.05 * radius <= w <= 1.62 * radius
                    or abs(x0 + x + w / 2 - cx) > .19 * radius
                    or abs(y0 + y + h / 2 - cy) > .21 * radius):
                continue
            ink_count += _wide_arrow(ink[y:y + h, x:x + w] != 0)
        if ink_count >= 2:
            boxes.add((x0, y0, x1, y1))
    return next(iter(boxes)) if len(boxes) == 1 else None


def _action(role, function, box):
    left, top, right, bottom = box
    return {"role": role, "function": function, "preview_text": function, "box": list(box),
            "click_point": [(left + right) // 2, (top + bottom) // 2]}


def landscape_footer_command_actions(frame, overlay, index):
    """Read only printed commands in an owned landscape footer and return control."""
    height, width = frame.shape[:2]
    if index is None or overlay is not None or height < 270 or not 1.42 * height <= width <= 2.88 * height:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    start = _footer_edge(gray)
    if start is None:
        return []
    buttons = _command_buttons(frame, start)
    leading = [b for b in buttons if b[0] <= .13 * width and b[2] <= .33 * width]
    trailing = [b for b in buttons if b[0] >= .52 * width]
    if len(leading) != 1 or len(trailing) != 2:
        return []
    first = leading[0]
    a, b = trailing
    w, size = a[2] - a[0], a[3] - a[1]
    if (max(abs(a[1] - b[1]), abs(a[3] - b[3])) > .13 * size or abs((b[2] - b[0]) - w) > .12 * w
            or not .06 * w <= b[0] - a[2] <= .37 * w
            or not .80 * size <= first[3] - first[1] <= 1.20 * size
            or abs(first[3] - a[3]) > .15 * size):
        return []
    labels = []
    for box, is_leading in ((first, True), (a, False), (b, False)):
        if is_leading and not _plus(gray, box):
            continue
        if text := _button_caption(gray, box, index, leading=is_leading):
            labels.append((text, box))
    if len(labels) < 2 or len({text[0] for text, _ in labels}) != len(labels):
        return []
    result = [_action(role, text, box) for (role, text), box in labels]
    if back := _return_button(gray, size):
        result.insert(0, _action("back", "返回", back))
    return result
