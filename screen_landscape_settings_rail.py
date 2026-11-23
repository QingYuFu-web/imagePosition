"""Read a separated right-side settings rail with an accent selection."""

from functools import lru_cache

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_dialog_close import _cross_footprint
from screen_foreground import _centered_cross
from screen_game_marked_choices import _native_label
from screen_game_preview_settings import _SETTINGS
from screen_wide_panel_checks import _letter_rasters


_HEADINGS = ("设置", "设定", "設定", "選項", "选项", "菜单", "选单", "系统")
_CATEGORIES = (
    "基础", "常规", "通用", "画面", "画质", "图像", "显示", "操作", "控制", "载具",
    "车辆", "灵敏度", "灵敏", "拾取", "效果", "声音", "声效", "音效", "音量",
    "录像", "录屏", "隐私", "其他", "帮助", "按键", "网络", "账户", "下载", "系统",
    "布局", "语言", "陀螺仪", "控制器", "分辨率", "高帧率", "移动端", "摄像机",
)
_LABELS = tuple(f"{category}设置" for category in _CATEGORIES)
_GLYPH_SIZE = 22


def _accent(frame):
    height, width = frame.shape[:2]
    start = round(.76 * width)
    blue, green, red = (channel.astype(np.int16) for channel in cv2.split(frame[:, start:]))
    mask = np.uint8((red >= 150) & (red - green >= 20) & (green - blue >= 21))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [(start + int(l), int(t), start + int(l + w), int(t + h))
             for l, t, w, h, area in stats[1:]
             if (.09 * width <= w <= .22 * width and .071 * height <= h <= .144 * height
                 and area >= .88 * w * h and start + l + w >= .98 * width
                 and .12 * height <= t and t + h <= .92 * height)]
    return boxes[0] if len(boxes) == 1 else None


def _separators(gray, selection):
    left, top, right, bottom = selection
    w, pitch = right - left, bottom - top
    lane = gray[:, left + round(.10 * w):right - round(.07 * w)].astype(np.int16)
    radius = max(3, round(.115 * pitch))
    guard = max(2, round(.055 * pitch))
    center = lane[guard:-guard]
    outer = np.maximum(lane[:-2 * guard], lane[2 * guard:])
    delta = center - outer
    eligible = np.pad((delta >= 8).mean(axis=1) >= .73, (guard, guard))
    strength = np.pad(np.median(delta, axis=1), (guard, guard))
    rows = [a + int(np.argmax(strength[a:b])) for a, b in _bands(eligible)
            if b - a <= max(4, round(.097 * pitch))]
    result = [top, bottom]
    for direction in (-1, 1):
        edge = top if direction < 0 else bottom
        step = pitch
        while True:
            target = edge + direction * step
            nearest = [row for row in rows if abs(row - target) <= radius]
            if not nearest:
                break
            boundary = max(nearest, key=lambda y: strength[y] - abs(y - target))
            result.append(boundary)
            step = .50 * (step + abs(boundary - edge))
            edge = boundary
    boundaries = sorted(result)
    if (len(boundaries) >= 5 and boundaries[0] <= .87 * pitch
            and boundaries[-1] >= .67 * len(gray)):
        return boundaries
    return None


def _close(gray, rail_left, rail_right, header_bottom):
    width = rail_right - rail_left
    cell_width = min(round(.34 * width), round(1.03 * header_bottom))
    cell = rail_right - cell_width
    pad = max(2, round(.11 * header_bottom))
    left, top = cell + pad, pad
    patch = gray[top:header_bottom - pad, left:rail_right - pad]
    if min(patch.shape) < 10 or not _centered_cross(patch):
        return None
    peak = float(np.percentile(patch, 98))
    fill = float(np.median(patch))
    if peak - fill < 70:
        return None
    boxes = []
    for ratio in (.36, .44, .52, .63):
        mask = np.uint8(patch >= fill + ratio * (peak - fill))
        x, y, w, h = cv2.boundingRect(mask)
        if (.28 * header_bottom <= min(w, h) <= max(w, h) <= .61 * header_bottom
                and .77 <= w / h <= 1.28 and abs(x + w / 2 - patch.shape[1] / 2) <= .145 * cell_width
                and abs(y + h / 2 - patch.shape[0] / 2) <= .13 * header_bottom
                and _cross_footprint(mask[y:y + h, x:x + w] != 0)):
            boxes.append((left + x, top + y, left + x + w, top + y + h))
    if len(boxes) < 2 or np.max(np.ptp(boxes, axis=0)) > 2:
        return None
    return cell, boxes[0]


@lru_cache(maxsize=240)
def _outline_bank(paths, glyph):
    shapes = _letter_rasters(paths, glyph, _GLYPH_SIZE, _GLYPH_SIZE)
    if not len(shapes):
        return None
    blurred = np.stack([cv2.GaussianBlur(shape, (5, 5), .72) for shape in shapes])
    masks = np.concatenate((shapes, blurred)).reshape(2 * len(shapes), -1)
    return masks / np.sqrt(np.square(masks).sum(axis=1, keepdims=True) + .001)


@lru_cache(maxsize=512)
def _glyph_scores(paths, chars, raster):
    shape = np.frombuffer(raster, np.float32)
    shape = shape / np.sqrt(np.square(shape).sum() + .001)
    output = {}
    for char in chars:
        bank = _outline_bank(paths, char)
        output[char] = float(np.einsum("ij,j->i", bank, shape).max()) if bank is not None else 0.
    return output


def _literal_cut(signal, peak, cut, vocabulary, paths):
    x, y, width, height = cv2.boundingRect(np.uint8(signal >= cut * peak))
    count = round(width / max(1, 1.06 * height))
    options = [word for word in vocabulary if len(word) == count]
    if not 13 <= height <= 48 or len(options) < 2:
        return None
    ink = signal[y:y + height, x:x + width]
    occupied = (ink >= .44 * peak).sum(axis=0)
    bounds = [0]
    for i in range(1, count):
        expected = i * width / count
        reach = round(.16 * height)
        indices = range(max(1, round(expected) - reach), min(width, round(expected) + reach + 1))
        pivot = min(indices, key=lambda j: (occupied[j], abs(j - expected)))
        if occupied[pivot] > .20 * height:
            return None
        bounds.append(pivot)
    bounds.append(width)
    scores = []
    for i, (left, right) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        glyph = ink[:, left:right]
        local = float(np.percentile(glyph, 97))
        a, t, w, h = cv2.boundingRect(np.uint8(glyph >= .26 * local))
        if local < .48 * peak or not .62 * height <= min(w, h) <= max(w, h) <= 1.19 * height:
            return None
        stroke = np.minimum(1, glyph[t:t + h, a:a + w] / local).astype(np.float32)
        interpolation = cv2.INTER_AREA if max(w, h) > _GLYPH_SIZE else cv2.INTER_LINEAR
        raster = np.pad(cv2.resize(stroke, (_GLYPH_SIZE, _GLYPH_SIZE), interpolation=interpolation), 2)
        scores.append(_glyph_scores(paths, tuple(sorted({word[i] for word in options})), raster.tobytes()))
    ranked = []
    for word in options:
        exact = [scores[i][char] for i, char in enumerate(word)]
        ranked.append((float(np.mean(exact)), min(exact), word))
    (score, weakest, text), (next_score, _, _) = sorted(ranked, reverse=True)[:2]
    if score >= .92 and weakest >= .835 and score - next_score >= .010:
        return text, (x, y, width, height)
    return None


def _literal_word(gray, crop, vocabulary, index):
    left, top, right, bottom = crop
    tile = gray[top:bottom, left:right].astype(np.float32)
    paths = tuple(index.fonts if index.fonts is not None else index.cjk_sources)
    if min(tile.shape) < 11 or not paths:
        return None
    signal = np.maximum(0, tile - np.median(tile))
    peak = float(np.percentile(signal, 98))
    if peak < 45:
        return None
    readings = [_literal_cut(signal, peak, cut, vocabulary, paths) for cut in (.32, .40)]
    if not all(readings) or readings[0][0] != readings[1][0]:
        return None
    text, (x, y, w, h) = readings[1]
    return text, (left + x, top + y, left + x + w, top + y + h)


def _action(role, label, box, point, **metadata):
    return {"role": role, "function": label, "preview_text": label, "box": list(box), "click_point": list(point),
            **metadata}


def landscape_settings_rail_actions(frame, overlay, index):
    """Return proven inactive tabs and the close control on a settings page."""
    height, width = frame.shape[:2]
    if index is None or overlay is not None or height < 230 or not 1.50 * height <= width <= 3.04 * height:
        return []
    selection = _accent(frame)
    if selection is None:
        return []
    left, top, right, bottom = selection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boundaries = _separators(gray, selection)
    if boundaries is None:
        return []
    head = boundaries[0]
    dismissal = _close(gray, left, right, head)
    if dismissal is None:
        return []
    cell, close = dismissal
    cache = {}
    width = right - left
    title = (left + round(.15 * width), round(.15 * head), cell - 3, head - round(.16 * head))
    heading = (_literal_word(gray, title, _HEADINGS, index)
               or _native_label(frame, gray, title, index, cache))
    if heading is None or heading[0].casefold() not in _SETTINGS:
        return []
    actions = []
    inset = round(.14 * width)
    for y, b in zip(boundaries[:-1], boundaries[1:], strict=True):
        if (y, b) == (top, bottom):
            continue
        trim = round(.176 * (b - y))
        area = (left + inset, y + trim, right - inset, b - trim)
        own = _literal_word(gray, area, _LABELS, index)
        if own:
            label, (x0, y0, x1, y1) = own
            actions.append(_action("switch_tab", f"切换到「{label}」", (left, y, right, b),
                                   ((x0 + x1) // 2, (y0 + y1) // 2), preview_text=label,
                                   setting_label=label, setting_state="off"))
    if len(actions) < 2:
        return []
    a, t, b, d = close
    actions.append(_action("close", "关闭", close, ((a + b) // 2, (t + d) // 2)))
    return actions
