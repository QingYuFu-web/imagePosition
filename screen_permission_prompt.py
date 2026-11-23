"""Recognize a native permission sheet from its visible request and paired actions."""

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from screen_landscape_edit_actions import _features
from screen_rim_attached_dismiss import _cross_arms


_HEADERS = {"申请", "申請", "授权", "授權", "权限", "權限", "请求", "請求"}
_DENIALS = {"拒绝", "拒絕", "不允许", "不允許"}
_APPROVALS = {"同意", "允许", "允許"}
_CANONICAL = {"申請": "申请", "授權": "授权", "權限": "权限", "請求": "请求", "拒絕": "拒绝",
              "不允許": "不允许", "允許": "允许", "確認": "确认", "關閉": "关闭", "繼續": "继续"}
_WORDS = tuple(sorted(_HEADERS | _DENIALS | _APPROVALS | {
    "取消", "确定", "確認", "确认", "完成", "保存", "返回", "关闭", "關閉", "继续", "繼續",
    "不同意", "改进", "应用", "设置", "详情", "查看", "接受", "稍后", "提示", "拒接",
}))


@dataclass(frozen=True)
class PermissionPrompt:
    panel: tuple
    controls: list
    review_only: bool


def _runs(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2).tolist()


@lru_cache(maxsize=6)
def _words(paths):
    shapes = []
    for path in paths:
        font = ImageFont.truetype(str(path), 48)
        for name in _WORDS:
            surface = Image.new("L", (215, 75))
            ImageDraw.Draw(surface).text((4, 0), name, font=font, fill=255)
            raster = np.asarray(surface)
            x, y, w, h = cv2.boundingRect(raster)
            if w and h:
                shapes.append((name, raster[y:y + h, x:x + w]))
    return tuple(shapes)


@lru_cache(maxsize=96)
def _word_bank(paths, width, height):
    names, shapes, penalties = [], [], []
    for text, raster in _words(paths):
        ratio = width / height * raster.shape[0] / raster.shape[1]
        if .76 <= ratio <= 1.30:
            names.append(text)
            shapes.append(cv2.resize(raster, (width, height), interpolation=cv2.INTER_AREA) / 255.)
            penalties.append(.14 * abs(np.log(ratio)))
    return names, _features(np.asarray(shapes)) if shapes else None, np.asarray(penalties)


def _caption(patch, paths, expected):
    if not patch.size:
        return None
    off = abs(patch.astype(np.int16) - np.median(patch))
    peak = max(float(np.percentile(off, 96)), .65 * float(off.max()))
    if peak < 36:
        return None
    votes = {}
    for level in (.27, .40, .54):
        x, y, w, h = cv2.boundingRect(np.uint8(off >= level * peak))
        if h < 6 or not 1.55 * h <= w <= 4.30 * h or x == 0 or y == 0 or x + w == patch.shape[1] or y + h == patch.shape[0]:
            continue
        crop = np.minimum(1, off[y:y + h, x:x + w] / peak)
        if h > 28:
            w, h = round(w * 28 / h), 28
            crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
        names, features, penalties = _word_bank(paths, w, h)
        if features is None:
            continue
        query = _features(np.where(crop >= .70 * level, crop, 0))
        ranking = np.einsum("ij,j->i", features, query, optimize=False) - penalties
        scores = {}
        for name, score in zip(names, ranking, strict=True):
            name = _CANONICAL.get(name, name)
            scores[name] = max(score, scores.get(name, 0.))
        ordered = sorted(scores, key=scores.get, reverse=True)
        if len(ordered) >= 2:
            word, runner_up = ordered[:2]
            if word in expected and scores[word] >= .75 and scores[word] - scores[runner_up] >= .024:
                votes[word] = votes.get(word, 0) + 1
    confirmed = [word for word, count in votes.items() if count >= 2]
    return confirmed[0] if len(confirmed) == 1 else None


def _footer(gray, frame):
    height, width = gray.shape
    start, end = max(round(.61 * height), height - round(.54 * width)), height - round(.045 * width)
    paper = float(np.median(gray[start:end:3, ::3]))
    if paper < 216:
        return None
    area = frame[start:end].astype(np.int16)
    masks = (gray[start:end] < paper - 13, np.ptp(area, axis=2) >= 65)
    measured = []
    for side, mask in zip((-1, 1), masks, strict=True):
        contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w / 2
            if (not .32 * width <= w <= .49 * width or not .073 * width <= h <= .165 * width
                    or not .113 * width <= side * (cx - .5 * width) <= .31 * width
                    or not .80 <= cv2.contourArea(contour) / (w * h) <= .985):
                continue
            a, b = x + round(.18 * w), y + round(.31 * h)
            c, d = x + round(.82 * w), y + round(.69 * h)
            fill = area[b:d, a:c]
            channel = np.median(fill, axis=(0, 1))
            stable = (abs(fill - channel).max(axis=2) <= 20).mean() >= .69
            if stable and (channel.min() >= paper - 19 if side < 0 else np.ptp(channel) >= 90):
                boxes.append((x, start + y, x + w, start + y + h))
        if len(boxes) != 1:
            return None
        measured.append(boxes[0])
    left, right = measured
    h = left[3] - left[1]
    if (not .005 * width <= right[0] - left[2] <= .055 * width
            or abs(left[0] + right[2] - width) > .046 * width
            or max(abs(left[i] - right[i]) for i in (1, 3)) > .13 * h
            or abs(left[2] - left[0] - (right[2] - right[0])) > .25 * h):
        return None
    return measured, paper


def _sheet_top(gray, buttons, paper):
    height, width = gray.shape
    reach = max(3, round(.023 * width))
    white = (gray[:, reach:width - reach:3] >= paper - 12).mean(axis=1)
    candidates = []
    start, stop = max(round(.18 * height), buttons[0][1] - round(1.65 * width)), buttons[0][1] - round(.53 * width)
    for top in range(start, stop):
        before, after = white[top - reach:top], white[top + 1:top + reach + 1]
        if before.mean() <= .69 and after.mean() >= .87 and (white[top:buttons[0][1] - reach] >= .78).mean() >= .78:
            candidates.append((float(after.mean() - before.mean()), top))
    if not candidates:
        return None
    first = candidates[0][1]
    return max((score, top) for score, top in candidates if top - first <= reach)[1]


def _close_glyph(gray, top, paper):
    width = gray.shape[1]
    a, t = round(.81 * width), top + round(.028 * width)
    b, d = width - round(.025 * width), top + round(.185 * width)
    crop = gray[t:d, a:b]
    crosses = []
    for contrast in (82, 110, 138):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(crop <= paper - contrast))
        for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if (.023 * width <= min(w, h) <= max(w, h) <= .070 * width and .71 <= w / h <= 1.40
                    and .17 <= area / (w * h) <= .52 and _cross_arms(labels[y:y + h, x:x + w] == i)):
                crosses.append((a + x, t + y, a + x + w, t + y + h))
    if len(crosses) < 2 or np.ptp(crosses, axis=0).max() > max(2, .012 * width):
        return None
    return sorted(crosses)[len(crosses) // 2]


def _request_text(gray, y0, y1, paths, paper):
    width = gray.shape[1]
    x0 = round(.13 * width)
    text_row = gray[y0:y1, x0:round(.76 * width)]
    present = (text_row < paper - 86).sum(axis=0) >= 2
    joined = cv2.morphologyEx(np.uint8(present.reshape(1, -1)), cv2.MORPH_CLOSE,
                              np.ones((1, max(2, round(.016 * width))), np.uint8))[0]
    pad = max(2, round(.012 * width))
    requests = []
    for start, end in _runs(joined != 0):
        if .049 * width <= end - start <= .225 * width:
            if word := _caption(gray[y0:y1, x0 + start - pad:x0 + end + pad], paths, _HEADERS):
                requests.append(word)
    below = gray[y1 + round(.020 * width):y1 + round(.131 * width), round(.02 * width):round(.94 * width)]
    body_words = [bottom - top for top, bottom in _runs((below <= paper - 74).sum(axis=1) >= .041 * width)]
    return len(requests) == 1 and any(.021 * width <= size <= .081 * width for size in body_words)


def _request_without_close(gray, top, paths, paper):
    width = gray.shape[1]
    first, last = top + round(.026 * width), top + round(.151 * width)
    text = gray[first:last, round(.13 * width):round(.76 * width)] < paper - 86
    rows = _runs(text.sum(axis=1) >= max(4, round(.025 * width)))
    pad = max(2, round(.016 * width))
    return sum(_request_text(gray, max(first, first + start - pad), min(last, first + end + pad), paths, paper)
               for start, end in rows if .024 * width <= end - start <= .067 * width) == 1


def _footer_name(gray, box, paths, accepted):
    a, b, c, d = box
    x, y = round(.36 * (c - a)), round(.18 * (d - b))
    patch = gray[b + y:d - y, a + x:c - x]
    if name := _caption(patch, paths, accepted):
        return name
    # Low-resolution captions lose a stroke when sampled on a fractional pixel.
    # Verify the same function at two different sampling offsets before exporting it.
    variants = [
        _caption(cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), paths, accepted)
        for scale in (.91, .86, .69, .63)
    ]
    names = set(variants) - {None}
    if len(names) == 1 and variants.count(None) <= 2:
        return names.pop()
    return None


def permission_prompt(frame, index):
    """Identify a request pane and the controls it owns, with partial reads flagged for review."""
    height, width = frame.shape[:2]
    if index is None or width < 225 or not 1.53 * width <= height <= 3.03 * width:
        return None
    scale = min(1., 440 / width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    footer = _footer(gray, small)
    if footer is None:
        return None
    buttons, paper = footer
    top = _sheet_top(gray, buttons, paper)
    if top is None:
        return None
    paths = tuple(index.fonts if index.fonts is not None else index.cjk_sources)
    close = _close_glyph(gray, top, paper)
    padding = round(.018 * gray.shape[1])
    if ((close is None or not _request_text(gray, close[1] - padding, close[3] + padding, paths, paper))
            and not _request_without_close(gray, top, paths, paper)):
        return None
    names = [_footer_name(gray, box, paths, expected) for box, expected in zip(buttons, (_DENIALS, _APPROVALS), strict=True)]
    review_only = None in names
    ratio_y, ratio_x = np.array(frame.shape[:2]) / small.shape[:2]

    def convert(box):
        return [round(q * (ratio_y if i % 2 else ratio_x)) for i, q in enumerate(box)]

    controls = [("close", "关闭授权弹窗", close)] if close is not None else []
    controls.extend((role, name, box) for role, name, box in
                    zip(("cancel", "confirm"), names, buttons, strict=True) if name is not None)
    actions = []
    for role, title, box in controls:
        a, b, c, d = convert(box)
        actions.append({"role": role, "function": title, "preview_text": title, "box": [a, b, c, d],
                        "click_point": [(a + c) // 2, (b + d) // 2],
                        **({"review_only": True} if review_only else {})})
    return PermissionPrompt(tuple(convert((0, top, small.shape[1], small.shape[0]))), actions, review_only)
