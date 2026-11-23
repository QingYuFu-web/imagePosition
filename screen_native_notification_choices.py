"""Read Japanese choices in a native lock-screen notification dialog."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

from inventory_choice_glyphs import ChoiceGlyph, find_choice_glyphs


_TITLE = "ロック画面"
_SHOW_ALL = "すべての通知内容を表示する"
_SHOW_NONE = "通知を一切表示しない"
_UNLOCKED = ("ロック解除時のみプライベートな", "内容を表示する")
_SHOW_UNLOCKED = "".join(_UNLOCKED)
_CANCEL = "キャンセル"
_WORDS = (
    _TITLE, "画面ロック", "通知の表示", "通知内容", "ロック画面の通知",
    _SHOW_ALL, "すべての通知内容を表示しない", "すべての通知を表示する", "通知の内容を非表示にする",
    _SHOW_NONE, "通知を一切表示する", "通知を非表示にする", "通知を表示しない", "通知をすべて表示する",
    *_UNLOCKED, "内容を表示しない", "通知を表示する", "ロック解除時のみ通知内容を表示する",
    _SHOW_UNLOCKED, "ロック解除時のみ内容を表示する",
    _CANCEL, "キャンペーン", "キャンプ場", "キャンセルする", "キャンセルしない", "設定を保存する",
)


@dataclass(frozen=True)
class NotificationChoices:
    panel: tuple[int, int, int, int]
    actions: tuple[dict, ...]
    options_box: tuple[int, int, int, int]


@lru_cache(maxsize=1)
def _font_words():
    folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    fonts = []
    for name, face in (("YuGothM.ttc", 0), ("YuGothR.ttc", 0), ("msgothic.ttc", 0), ("msgothic.ttc", 1)):
        if (path := folder / name).is_file():
            try:
                fonts.append(ImageFont.truetype(str(path), 50, index=face))
            except OSError:
                continue
    templates = []
    for word in _WORDS:
        for font in fonts:
            mask = font.getmask(word, mode="L")
            raw = np.asarray(mask, np.uint8).reshape(mask.size[1], mask.size[0])
            x, y, w, h = cv2.boundingRect(np.uint8(raw >= 30))
            if w and h:
                templates.append((word, w / h, raw[y:y + h, x:x + w]))
    return templates


@lru_cache(maxsize=40)
def _caption_bank(width):
    names, images = [], []
    ratio = width / 25
    for word, aspect, image in _font_words():
        if not .82 <= aspect / ratio <= 1.20:
            continue
        raster = cv2.resize(image, (width, 25), interpolation=cv2.INTER_AREA)
        sample = cv2.GaussianBlur(raster, (3, 3), .60).astype(np.float32).ravel()
        names.append(word)
        images.append(sample / np.linalg.norm(sample))
    return names, np.asarray(images, dtype=np.float32)


def _decode(ink, box, scale, *, min_margin=.14):
    left, top, right, bottom = box
    tile = ink[top:bottom, left:right]
    if not tile.size:
        return None
    peak = float(np.percentile(tile, 98))
    if peak < 60:
        return None
    decoded = []
    for cut in (.19, .30, .41):
        x, y, w, h = cv2.boundingRect(np.uint8(tile >= max(21, cut * peak)))
        if not .43 * scale <= h <= 1.17 * scale or not 2.25 * h <= w <= 19 * h:
            continue
        target = round(25 * w / h)
        names, bank = _caption_bank(target)
        if len(set(names)) < 2:
            continue
        sample = cv2.resize(tile[y:y + h, x:x + w], (target, 25), interpolation=cv2.INTER_AREA)
        sample = sample.astype(np.float32).ravel()
        values = np.einsum("ij,j->i", bank, sample) / np.linalg.norm(sample)
        best = int(values.argmax())
        second = max(score for name, score in zip(names, values, strict=True) if name != names[best])
        if values[best] >= .70 and values[best] - second >= min_margin:
            decoded.append((names[best], (left + x, top + y, left + x + w, top + y + h)))
    if len(decoded) >= 2 and len({word for word, _ in decoded}) == 1:
        return decoded[len(decoded) // 2]
    return None


def _zones(ink, area, scale):
    left, top, right, bottom = area
    tile = ink[top:bottom, left:right]
    if not tile.size:
        return ()
    active = (tile >= 48).sum(axis=1) >= max(3, round(.081 * scale))
    ranges = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
    lines = [(left, top + max(0, y - 2), right, top + min(len(tile), stop + 2)) for y, stop in ranges
             if .43 * scale <= stop - y <= 1.10 * scale]
    return lines if 1 <= len(lines) <= 2 else ()


def _accent_ring(frame, zone, peers, span):
    """Verify a colored center, quiet annulus, and segmented outer ring."""
    if len(peers) != 2 or any(glyph.state != "off" for glyph in peers):
        return None
    x0, y0, x1, y1 = zone
    blue, _, red = cv2.split(frame[y0:y1, x0:x1])
    accent = cv2.subtract(blue, red) >= 51
    centers = sorted((glyph.center for glyph in peers), key=lambda point: point[1])
    if abs(centers[0][0] - centers[1][0]) > .019 * span:
        return None
    side = round(np.median([glyph.box[3] - glyph.box[1] for glyph in peers]))
    x = round(sum(point[0] for point in centers) / 2) - x0
    a, b = (point[1] - y0 for point in centers)
    gap = b - a
    middle = [.5 * (a + b)] if .102 * span <= gap / 2 <= .176 * span else []
    if .102 * span <= gap <= .176 * span:
        middle.extend((a - gap, b + gap))
    pad = round(.64 * side)
    yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1]
    radius = np.hypot(yy, xx) / (side / 2)
    core = radius <= .40
    quiet = (.53 <= radius) & (radius <= .72)
    edge = (.82 <= radius) & (radius <= .98)
    outside = (1.09 <= radius) & (radius <= 1.25)
    theta = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    arcs = [edge & (theta >= i * np.pi / 3) & (theta < (i + 1) * np.pi / 3) for i in range(6)]
    probes = [part.astype(np.float32) / part.sum() for part in (core, quiet, edge, outside)]
    horizontal = round(.073 * side)
    vertical = round(.15 * side)
    readings = []
    for expected in middle:
        sy = round(expected) - vertical - pad
        sx = x - horizontal - pad
        ey = round(expected) + vertical + pad + 1
        ex = x + horizontal + pad + 1
        if min(sy, sx) < 0 or ey > len(accent) or ex > accent.shape[1]:
            continue
        crop = accent[sy:ey, sx:ex]
        small = crop.astype(np.float32)
        filled, separation, ring, beyond = (cv2.matchTemplate(small, mask, cv2.TM_CCORR) for mask in probes)
        keep = (filled >= .86) & (separation <= .14) & (ring >= .70) & (beyond <= .10)
        scores = filled + ring - separation - beyond
        for a, b in sorted(zip(*np.nonzero(keep), strict=True), key=lambda pt: -scores[pt]):
            patch = crop[a:a + 2 * pad + 1, b:b + 2 * pad + 1]
            if sum(patch[part].mean() >= .54 for part in arcs) < 5:
                continue
            left, top = x0 + sx + b + pad - side // 2, y0 + sy + a + pad - side // 2
            readings.append(ChoiceGlyph((left, top, left + side, top + side), "radio", "on", 85., .86))
            break
    return readings[0] if len(readings) == 1 else None


def _radios(frame, gray, panel):
    left, top, right, bottom = panel
    span = right - left
    zone = (left + round(.035 * span), top + round(.160 * span),
            left + round(.169 * span), bottom - round(.126 * span))
    size = (round(.039 * span), round(.077 * span))
    rings = [ring for ring in find_choice_glyphs(gray, zone, size)
             if ring.kind == "radio" and ring.confidence >= .78]
    if not any(ring.state == "on" for ring in rings):
        if ring := _accent_ring(frame, zone, rings, span):
            rings.append(ring)
    if len(rings) != 3 or sum(ring.state == "on" for ring in rings) != 1:
        return ()
    rings.sort(key=lambda ring: ring.center[1])
    x, y = np.asarray([ring.center for ring in rings]).T
    sizes = np.asarray([ring.box[3] - ring.box[1] for ring in rings])
    pitch = np.diff(y)
    if (np.ptp(x) > .019 * span or np.ptp(sizes) > .22 * np.median(sizes)
            or not .104 * span <= pitch.min() <= pitch.max() <= .175 * span or np.ptp(pitch) > .15 * pitch.min()):
        return ()
    return rings


def _action(role, text, bounds, purpose):
    left, top, right, bottom = map(int, bounds)
    return {"role": role, "function": purpose, "preview_text": text, "box": [left, top, right, bottom],
            "click_point": [(left + right) // 2, (top + bottom) // 2]}


def native_notification_choices(frame, overlay):
    """Recognize only two inactive choices and Cancel within a measured dialog."""
    if overlay is None or overlay.kind != "dialog":
        return None
    height, width = frame.shape[:2]
    left, top, right, bottom = panel = tuple(overlay.panel)
    span = right - left
    if (not .67 * width <= span <= .94 * width or not .52 * span <= bottom - top <= .95 * span
            or not 0 < left < right < width or not 0 < top < bottom < height):
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    paint = round(float(np.median(gray[top:bottom:5, left:right:5])))
    if paint < 216:
        return None
    rings = _radios(frame, gray, panel)
    if not rings:
        return None
    dark = cv2.subtract(np.full_like(gray, paint), gray)
    centers = [ring.center[1] for ring in rings]
    pitch = float(np.median(np.diff(centers)))
    scale = float(np.median([ring.box[3] - ring.box[1] for ring in rings]))
    text_right = right - round(.048 * span)
    title = _decode(dark, (left + round(.038 * span), top + round(.035 * span), text_right,
                           rings[0].box[1] - round(.20 * pitch)), scale)
    if title is None or title[0] != _TITLE:
        return None
    text_left = max(ring.box[2] for ring in rings) + round(.049 * span)
    readings = []
    for ring in rings:
        _, y = ring.center
        area = (text_left, round(y - .48 * pitch), text_right, round(y + .48 * pitch))
        zones = _zones(dark, area, scale)
        lines = [_decode(dark, line, scale, min_margin=.10 if len(zones) == 2 else .14) for line in zones]
        readings.append((ring, lines))
    def caption(lines):
        if lines and all(lines):
            return "".join(item[0] for item in lines)
        return None

    verified = [(ring, caption(lines)) for ring, lines in readings]
    if not any(word == _SHOW_UNLOCKED for _, word in verified):
        return None
    y0, y1 = round(centers[0] - .49 * pitch), round(centers[-1] + .49 * pitch)
    actions = []
    for ring, word in verified:
        if ring.state == "off" and word in (_SHOW_ALL, _SHOW_NONE, _SHOW_UNLOCKED):
            _, y = ring.center
            area = (left + round(.05 * span), round(y - .41 * pitch), text_right, round(y + .41 * pitch))
            actions.append(_action("select_option", word, area, f"选择「{word}」"))
    end = bottom - round(.027 * span)
    start = max(y1 + round(.024 * span), bottom - round(.158 * span))
    x0 = left + round(.56 * span)
    footer = frame[start:end, x0:text_right]
    if footer.size:
        accent = cv2.subtract(footer.max(axis=2), footer.min(axis=2))
        cancel = (_decode(dark, (x0, start, text_right, end), scale)
                  if np.percentile(accent, 98) >= 52 else None)
        if cancel is not None and cancel[0] == _CANCEL:
            l, t, r, b = cancel[1]
            pad = round(.023 * span)
            actions.append(_action("cancel", _CANCEL, (l - pad, t - pad, r + pad, b + pad), "取消"))
    return NotificationChoices(panel, tuple(actions), (left, y0, right, y1)) if actions else None
