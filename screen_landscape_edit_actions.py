"""Verify action buttons on landscape editors from their visible controls."""

from collections import defaultdict
from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from screen_caption_transcript import read_caption


_COMMANDS = (
    "退出", "关闭", "返回", "取消", "重置", "还原", "撤销", "清空", "恢复", "保存", "应用", "完成", "确定",
    "确认", "提交", "继续", "默认", "修改", "编辑", "复制", "删除", "新增", "复位", "移动", "调整", "预览",
    "清除", "重试", "模式", "设置", "停止", "开启", "暂停", "温馨", "重装", "回复", "恢复默认",
    "保存布局", "退出编辑",
)
_LEAVE = {"退出", "返回", "关闭", "取消", "退出编辑"}
_COMMIT = {"保存", "应用", "完成", "保存布局"}


def _action(box, label):
    left, top, right, bottom = box
    return {"role": "editor_action", "function": label, "preview_text": label,
            "box": list(box), "click_point": [(left + right) // 2, (top + bottom) // 2]}


def _proposals(regions, width, height):
    return [tuple(item["box"]) for item in regions["candidates"] if item.get("visibility") == "page"
            and 0 <= item["box"][0] < item["box"][2] <= width
            and 0 <= item["box"][1] < item["box"][3] <= height]


def _header_groups(boxes, width, height):
    tiles = sorted(box for box in boxes
                   if box[1] <= .052 * height and box[3] <= .115 * height
                   and .081 * width <= box[2] - box[0] <= .17 * width
                   and .040 * height <= box[3] - box[1] <= .089 * height
                   and 2.3 <= (box[2] - box[0]) / (box[3] - box[1]) <= 6.6)
    for row in combinations(tiles, 3):
        centers = [(left + right) / 2 for left, _, right, _ in row]
        sizes = np.array([right - left for left, _, right, _ in row])
        gaps = [new[0] - old[2] for old, new in zip(row[:-1], row[1:], strict=True)]
        typical = float(np.median(sizes))
        if (abs(np.mean(centers) - width / 2) <= .092 * width and np.ptp(sizes) <= .17 * typical
                and np.ptp(np.diff(centers)) <= .14 * typical and min(gaps) >= -.022 * typical
                and max(gaps) <= .21 * typical and np.ptp([box[1] for box in row]) <= .017 * height
                and np.ptp([box[3] for box in row]) <= .015 * height):
            yield row


def _header_paint(frame, boxes):
    paints, troughs = [], []
    for left, top, right, bottom in boxes:
        w, h = right - left, bottom - top
        inset, edge = round(.14 * w), round(.30 * w)
        body = frame[top + round(.27 * h):bottom - round(.24 * h), left + inset:left + edge]
        trough = frame[bottom:bottom + max(2, round(.27 * h)), left + inset:left + edge]
        if not body.size or not trough.size:
            return False
        paint = np.median(body, axis=(0, 1))
        if (abs(body - paint).max(axis=2) <= 22).mean() < .69:
            return False
        paints.append(paint)
        troughs.append(float(np.median(trough)))
    if max(troughs) > 78 or np.ptp(troughs) > 24 or min(max(paint) for paint in paints) - max(troughs) < 34:
        return False
    return sum(np.linalg.norm(paints[a] - paints[b]) < 25 and
               min(np.linalg.norm(paints[k] - paints[a]), np.linalg.norm(paints[k] - paints[b])) >= 34
               for a, b, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0))) == 1


def _normalized(values):
    flat = values.reshape((*values.shape[:-2], -1))
    return flat / np.maximum(1e-6, np.linalg.norm(flat, axis=-1, keepdims=True))


def _features(values):
    padding = ((0, 0),) * (values.ndim - 2) + ((2, 2), (2, 2))
    values = np.pad(values.astype(np.float32, copy=False), padding)
    mean = values.mean(axis=(-2, -1), keepdims=True)
    return np.concatenate((
        .55 ** .5 * _normalized(values),
        .30 ** .5 * _normalized(values - mean),
        .075 ** .5 * _normalized(np.diff(values, axis=-2)),
        .075 ** .5 * _normalized(np.diff(values, axis=-1)),
    ), axis=-1)


@lru_cache(maxsize=12)
def _word_shapes(paths):
    shapes = []
    for path in paths:
        font = ImageFont.truetype(str(path), 44, layout_engine=ImageFont.Layout.BASIC)
        for label in _COMMANDS:
            canvas = Image.new("L", (240, 82))
            ImageDraw.Draw(canvas).text((5, 0), label, font=font, fill=255)
            image = np.asarray(canvas)
            extent = cv2.boundingRect(image)
            x, y, w, h = extent
            if w and h:
                shapes.append((label, image[y:y + h, x:x + w]))
    return tuple(shapes)


@lru_cache(maxsize=20)
def _word_bank(paths, width, height):
    labels, templates = [], []
    for label, shape in _word_shapes(paths):
        if not .64 <= width / height / (shape.shape[1] / shape.shape[0]) <= 1.40:
            continue
        for stretch_x, stretch_y in ((0, 0), (-1, 0), (0, -1), (1, 1), (1, 0), (0, 1), (-1, -1)):
            w, h = width + stretch_x, height + stretch_y
            down = cv2.resize(shape, (w, h), interpolation=cv2.INTER_AREA)
            blank = np.zeros((height + 2, width + 2), np.float32)
            x, y = (width - w) // 2 + 1, (height - h) // 2 + 1
            blank[y:y + h, x:x + w] = down
            labels.append(label)
            templates.append(blank)
    return labels, _features(np.stack(templates)) if templates else np.empty(0, np.float32)


def _tiny_title(gray, box, index):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    center = (left + right) // 2
    edge, outer = round(.16 * w), round(.33 * w)
    reach = min(round(.23 * w), round(.80 * h))
    start, end = top + round(.23 * h), bottom - round(.185 * h)
    plate = gray[start:end]
    background = np.median(np.concatenate((plate[:, left + edge:left + outer],
                                           plate[:, right - outer:right - edge]), axis=1), axis=1)[:, None]
    delta = plate[:, center - reach:center + reach + 1] - background
    peak, polarity = max((float(np.percentile(sign * delta, 97)), sign) for sign in (-1, 1))
    if peak < 32:
        return None
    ink = np.clip(delta * polarity / peak, 0, 1)
    paths = index.fonts if index.fonts is not None else index.cjk_sources
    if not paths:
        return None
    reports = defaultdict(dict)
    for cut in (.28, .47, .57):
        x, y, w, h = cv2.boundingRect(np.uint8(ink >= cut))
        if not (5 <= h <= 11 and 1.24 * h <= w <= 3.6 * h
                and abs(x + w / 2 - ink.shape[1] / 2) < .15 * ink.shape[1]):
            continue
        labels, templates = _word_bank(paths, w, h)
        if not templates.size:
            continue
        query = np.pad(ink[y:y + h, x:x + w], 1)
        raw = np.einsum("ij,j->i", templates, _features(query), optimize=False)
        scores = {}
        for label, score in zip(labels, raw, strict=True):
            scores[label] = max(scores.get(label, 0.), float(score))
        ordered = sorted(scores, key=scores.get, reverse=True)
        if len(ordered) < 2:
            continue
        winner, alternative = ordered[:2]
        score, margin = scores[winner], scores[winner] - scores[alternative]
        if score >= .863 and margin >= .026:
            reports[winner][(x, y, w, h)] = margin
    found = [label for label, readings in reports.items() if len(readings) >= 2 and max(readings.values()) >= .043]
    return found[0] if len(found) == 1 else None


def _header_actions(frame, gray, boxes, index):
    if not _header_paint(frame, boxes):
        return []
    actions = []
    for box in boxes:
        left, top, right, bottom = box
        h = bottom - top
        if h >= 22:
            center = (left + right) // 2
            cap = (center - h, top + round(.17 * h), center + h, bottom - round(.16 * h))
            reading = read_caption(frame, cap, index, cache={}, choice=True)
            label = reading["text"] if reading and reading["text"] in _COMMANDS else None
        else:
            label = _tiny_title(gray, box, index)
        if label:
            actions.append(_action(box, label))
    labels = {item["function"] for item in actions}
    return actions if labels & _LEAVE and labels & _COMMIT and len(actions) == len(labels) else []


def _toolbar_neighbors(boxes, confirm):
    left, top, right, bottom = confirm
    h = bottom - top
    midpoint = (top + bottom) / 2
    centers = sorted((a + b) / 2 for a, y, b, z in boxes if b < left - .9 * h
                     and .66 * h <= z - y <= 1.36 * h and .64 <= (b - a) / (z - y) <= 1.53
                     and abs((y + z) / 2 - midpoint) <= .24 * h)
    for first, second, third in zip(centers, centers[1:], centers[2:]):
        one, two = second - first, third - second
        if .81 * h <= min(one, two) <= max(one, two) <= 1.79 * h and abs(one - two) < .31 * h:
            return True
    return False


def _button_rims(gray, box):
    left, top, right, bottom = box
    width, depth = right - left, bottom - top
    pad = round(.19 * width)
    clip = gray[max(0, top - 2):min(len(gray), bottom + 2), left + pad:right - pad]
    for corner in (clip, clip[::-1]):
        border = corner[:round(.26 * depth)]
        fill = np.median(corner[round(.27 * depth):round(.46 * depth)], axis=0)
        if ((border - fill) >= 11).mean(axis=1).max() < .62:
            return False
    return True


def _segment_distance(x, y, ax, ay, bx, by):
    projection = np.clip(((x - ax) * (bx - ax) + (y - ay) * (by - ay)) /
                         ((bx - ax) ** 2 + (by - ay) ** 2), 0, 1)
    return np.hypot(x - ax - projection * (bx - ax), y - ay - projection * (by - ay))


def _check_shape(mask):
    yy, xx = np.nonzero(mask)
    x, y = (xx + .5) / mask.shape[1], (yy + .5) / mask.shape[0]
    across = np.minimum(_segment_distance(x, y, .10, .56, .38, .80),
                        _segment_distance(x, y, .38, .80, .91, .14))
    if (across <= .24).mean() < .88 or not .23 <= mask.mean() <= .62:
        return False
    return min((x < .28).sum(), ((x > .25) & (x < .56) & (y > .62)).sum(),
               ((x > .69) & (y < .45)).sum()) >= max(2, round(.075 * len(x)))


def _check_mark(gray, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    center, mid = (left + right) // 2, (top + bottom) // 2
    horizontal, vertical = round(.18 * w), round(.34 * h)
    patch = gray[mid - vertical:mid + vertical, center - horizontal:center + horizontal]
    base, peak = np.percentile(patch, (50, 99.5))
    if peak - base < 52:
        return False
    verified = 0
    for level in (.40, .50, .59):
        ink = np.uint8(patch >= base + level * (peak - base))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        if count < 2:
            continue
        component = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        x, y, a, b, pixels = stats[component]
        if (not .265 * h <= b <= .47 * h or not .105 * w <= a <= .235 * w or not 1.02 <= a / b <= 1.83
                or pixels < .85 * ink.sum() or abs(x + a / 2 - horizontal) > .16 * h
                or abs(y + b / 2 - vertical) > .16 * h):
            continue
        if _check_shape(labels[y:y + b, x:x + a] == component):
            verified += 1
    return verified >= 2


def landscape_edit_actions(frame, regions, index):
    """Read an editing header or the checkmark on a verified bottom toolbar."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or height < 205 or width < 1.45 * height:
        return []
    boxes = _proposals(regions, width, height)
    headers = list(_header_groups(boxes, width, height)) if index is not None else []
    bottoms = [box for box in boxes if box[0] >= .65 * width and box[3] >= .955 * height
               and .070 * width <= box[2] - box[0] <= .22 * width and .049 * height <= box[3] - box[1] <= .125 * height
               and 2.17 <= (box[2] - box[0]) / (box[3] - box[1]) <= 5.3 and _toolbar_neighbors(boxes, box)]
    if not headers and not bottoms:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    strips = [actions for row in headers if (actions := _header_actions(frame, gray, row, index))]
    footers = [box for box in bottoms if _button_rims(gray, box) and _check_mark(gray, box)]
    if len(strips) == 1 and not footers:
        return strips[0]
    if len(footers) == 1 and not strips:
        return [_action(footers[0], "完成")]
    return []
