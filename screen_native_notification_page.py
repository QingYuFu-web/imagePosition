"""Verify literal Japanese captions owned by native checkbox rows."""

from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

from inventory_choice_glyphs import find_choice_glyphs
from screen_game_settings_back import _bold_left_angle
from screen_native_notification_choices import _action


_OWNED = (
    "あなたのノートに「いいね」", "あなたのノートにコメント", "あなたのコメントに「いいね」",
    "ノートやコメントであなたをメンション", "未読数を非表示", "メッセージへのリアクション", "タイムライン通知",
)
_ACTIVITY = "オープンチャットのアクティビティ"
_WORDS = (
    "通知", "設定", "通知設定", "着信", "音声", "通知音", "タイムライン", "オープンチャット", _ACTIVITY,
    "音声とポップアップで知らせる", "ポップアップで知らせる", *_OWNED,
    "あなたのコメントにコメント", "あなたのメッセージに「いいね」", "あなたのノートにメンション",
    "メッセージへのコメント", "未読数を表示", "タイムライン通知を表示", "ノートにコメント", "メンション通知",
)


@lru_cache(maxsize=1)
def _fonts():
    folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    bank = []
    for file, face in (("YuGothM.ttc", 0), ("YuGothR.ttc", 0), ("msgothic.ttc", 0), ("msgothic.ttc", 1)):
        try:
            font = ImageFont.truetype(str(folder / file), 50, index=face)
        except OSError:
            continue
        for name in _WORDS:
            mask = font.getmask(name, mode="L")
            chip = np.asarray(mask, np.uint8).reshape(mask.size[1], mask.size[0])
            x, y, w, h = cv2.boundingRect(np.uint8(chip >= 32))
            if w and h:
                bank.append((name, w / h, chip[y:y + h, x:x + w]))
    return bank


@lru_cache(maxsize=42)
def _font_bank(width):
    words, rows = [], []
    for name, aspect, image in _fonts():
        if not .79 <= 25 * aspect / width <= 1.22:
            continue
        raster = cv2.GaussianBlur(cv2.resize(image, (width, 25), interpolation=cv2.INTER_AREA), (3, 3), .55)
        row = raster.astype(np.float32).ravel()
        words.append(name)
        rows.append(row / np.linalg.norm(row))
    return words, np.asarray(rows, np.float32)


def _literal(ink, region, span):
    l, t, r, b = region
    chip = ink[t:b, l:r]
    if not chip.size:
        return None
    readings = []
    for cut in (41, 63, 85):
        x, y, w, h = cv2.boundingRect(np.uint8(chip >= cut))
        if not .49 * span <= h <= 1.22 * span or not 1.4 <= w / h <= 24:
            continue
        width = round(25 * w / h)
        words, bank = _font_bank(width)
        if len(set(words)) < 2:
            continue
        crop = np.float32(cv2.resize(chip[y:y + h, x:x + w], (width, 25), interpolation=cv2.INTER_AREA)).ravel()
        scores = np.einsum("ij,j->i", bank, crop) / np.linalg.norm(crop)
        at = int(scores.argmax())
        other = max(s for w1, s in zip(words, scores, strict=True) if w1 != words[at])
        if scores[at] >= .68 and scores[at] - other >= .115:
            readings.append((words[at], (l + x, t + y, l + x + w, t + y + h)))
    if len(readings) >= 2 and len({item[0] for item in readings}) == 1:
        return readings[len(readings) // 2]
    return None


def _spans(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def _read_rows(ink, left, right, start, stop, size, *, floor=80):
    patch = ink[start:stop, left:right]
    if not patch.size:
        return []
    lines = _spans((patch >= floor).sum(axis=1) >= max(4, round(.13 * size)))
    return [word for t, b in lines if .49 * size <= b - t <= 1.10 * size
            if (word := _literal(ink, (left, start + max(0, t - 3), right, start + min(len(patch), b + 3)), size))]


def _back_mark(mask):
    if _bold_left_angle(mask):
        return True
    h, w = mask.shape
    count = mask.sum(axis=1)
    if not .44 <= w / h <= .82 or not .11 <= mask.mean() <= .32 or not count.all():
        return False
    x = mask @ np.arange(w) / (count * w)
    tip = np.argmin(x)
    span = max(2, round(.10 * h))
    left, right = np.median(x[:span]), np.median(x[-span:])
    model = np.interp(np.arange(h), (0, tip, h - 1), (left, x[tip], right))
    return (.37 * h <= tip <= .62 * h and min(left, right) - x[tip] >= .51
            and abs(left - right) <= .18 and np.percentile(np.abs(model - x), 90) <= .12)


def notification_page_actions(frame, overlay=None):
    height, width = frame.shape[:2]
    if overlay is not None or not 1.53 * width <= height <= 2.7 * width or width < 360:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.percentile(gray[::10, ::10], 25) < 240:
        return []
    zone = (round(.80 * width), round(.165 * height), round(.984 * width), height)
    choices = [glyph for glyph in find_choice_glyphs(gray, zone, (round(.035 * width), round(.070 * width)))
               if glyph.kind == "checkbox" and glyph.confidence >= .81]
    if len(choices) < 4 or np.ptp([glyph.center[0] for glyph in choices]) > .026 * width:
        return []
    size = float(np.median([glyph.box[3] - glyph.box[1] for glyph in choices]))
    blue, green, red = cv2.split(frame)
    satur = cv2.subtract(cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red))
    dark = cv2.subtract(255, gray)
    title = _literal(dark, (round(.105 * width), round(.012 * height), round(.64 * width), round(.073 * height)), size)
    if title is None or title[0] != "通知":
        return []
    neutral = dark.copy()
    neutral[satur > 34] = 0
    accent = dark.copy()
    accent[satur <= 34] = 0
    scan = gray[:, round(.055 * width):round(.975 * width)]
    rules = [round((a + b) / 2) for a, b in _spans((scan <= 251).mean(axis=1) >= .76)
             if 1 <= b - a <= .14 * size]
    x0, x1 = round(.036 * width), min(glyph.box[0] for glyph in choices) - round(.26 * size)
    actions = []
    for glyph in sorted(choices, key=lambda glyph: glyph.center[1]):
        _, y = glyph.center
        above = max((rule for rule in rules if rule < y - .53 * size), default=-1)
        below = min((rule for rule in rules if rule > y + .53 * size), default=height + 1)
        if y - above > 4.04 * size and y - above < 6.3 * size:
            headers = _read_rows(accent, x0, x1, above + 3, round(y - .80 * size), size, floor=50)
            if len(headers) == 1 and headers[0][0] == "タイムライン":
                above = headers[0][1][3]
        if not .63 * size <= y - above <= 4.04 * size or not .63 * size <= below - y <= 3.24 * size:
            continue
        caption = _read_rows(neutral, x0, x1, above + 3, below - 3, size)
        owned = [text for text, bounds in caption if text in _OWNED]
        if len(owned) != 1 or glyph.state not in ("on", "off"):
            continue
        text = owned[0]
        verb = "取消勾选" if glyph.state == "on" else "勾选"
        actions.append(_action("checkbox_setting", text, glyph.box, f"{verb}{text}"))
    if len(actions) < 3 or len({a["preview_text"] for a in actions}) != len(actions):
        return []
    tx, ty, _, tend = title[1]
    back = dark[max(0, ty - 8):tend + 8, round(.026 * width):tx - round(.20 * size)]
    u, v, w, h = cv2.boundingRect(np.uint8(back >= 94))
    if .62 * size <= h <= 1.30 * size and _back_mark(back[v:v + h, u:u + w] >= 94):
        left, top = round(.026 * width) + u, max(0, ty - 8) + v
        actions.append(_action("back", "戻る", (left, top, left + w, top + h), "返回"))
    first = min(glyph.center[1] for glyph in choices)
    edge = max((rule for rule in rules if rule <= first - .8 * size), default=None)
    if edge is not None and edge > tend + 3 * size:
        leading = [(name, bounds) for name, bounds in
                   _read_rows(neutral, x0, x1, tend + round(.8 * size), edge - 3, size) if name == _ACTIVITY]
        subtitles = _read_rows(dark, x0, x1, tend + round(.8 * size), edge - 3, size, floor=50)
        if len(leading) == 1:
            name, (ll, tt, _, bb) = leading[0]
            for text, (a, c, e, g) in subtitles:
                if (text == "音声とポップアップで知らせる" and .14 * size <= c - bb <= .78 * size
                        and abs(ll - a) <= .24 * size and np.percentile(satur[c:g, a:e], 90) >= 39):
                    boundary = min((rule for rule in rules if rule >= g + .40 * size), default=None)
                    if boundary is not None and boundary <= g + 1.39 * size:
                        actions.append(_action("open_setting", name, (ll, tt - round(.15 * size), x1, boundary - 3),
                                               f"打开{name}"))
                    break
    return actions
