"""Measure a visible light comment sheet and read its owned controls."""

from collections import defaultdict
from functools import lru_cache
from math import sqrt
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from screen_component_captions import _bands
from screen_dialog_close import _cross_footprint
from screen_foreground import (_comment_sheet_surfaces, _rounded_light_sheet, _working_view, account_comment_field,
                               BlockingOverlay)
from screen_landscape_search_field import _available
from screen_sheet_actions import _comment_heading


_HEADS = ("Comments", "comments", "评论", "評論", "评论区", "留言", "全部评论")
_PROMPTS = ("Add a comment", "Write a comment", "Comment as", "添加评论", "写评论", "发表评论", "写下评论",
            "写下你的评论", "说点什么", "新增留言", "寫評論", "發表評論", "添加評論")
_CLOSE = ("关闭", "關閉", "取消", "Close", "Cancel")
_RIVALS = ("Comment", "Commented", "Contents", "Community", "Add an answer", "Read more", "See all comments",
           "View replies", "Add a description", "暂无评论", "查看评论", "隐藏评论", "评论已关闭", "说点别的")


@lru_cache(maxsize=3)
def _font_paths(fonts):
    if fonts:
        return tuple(map(Path, fonts))
    root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    names = ("arial.ttf", "arialbd.ttf", "segoeui.ttf", "segoeuib.ttf", "tahoma.ttf", "msyh.ttc", "msyhbd.ttc", "simhei.ttf")
    return tuple(root / name for name in names if (root / name).is_file())


@lru_cache(maxsize=512)
def _bitmap(path, text):
    font = ImageFont.truetype(str(path), 76)
    l, t, r, b = font.getbbox(text)
    tile = Image.new("L", (r - l, b - t))
    ImageDraw.Draw(tile).text((-l, -t), text, font=font, fill=255)
    ink = np.asarray(tile)
    x, y, w, h = cv2.boundingRect(np.uint8(ink >= 34))
    return ink[y:y + h, x:x + w]


@lru_cache(maxsize=4200)
def _templates(paths, label, width, height):
    result = []
    for name in paths:
        if not label.isascii() and name.name.lower() in {"arial.ttf", "arialbd.ttf", "segoeui.ttf", "segoeuib.ttf", "tahoma.ttf"}:
            continue
        ink = _bitmap(name, label)
        if ink.size and .74 <= ink.shape[1] * height / (ink.shape[0] * width) <= 1.33:
            result.append(cv2.resize(ink, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
    return result


def _words(mask, size):
    spans = _bands(mask.any(axis=0))
    if len(spans) < 3:
        return []
    gaps = [right[0] - left[1] for left, right in zip(spans[:-1], spans[1:], strict=True)]
    cut = max(.19 * size, np.median(gaps) * 1.65)
    return [(left[1], right[0]) for left, right in zip(spans[:-1], spans[1:], strict=True) if right[0] - left[1] >= cut]


def _match(signal, candidates, paths, box):
    x, y, w, h = box
    if min(w, h) < 5:
        return None
    active = signal[y:y + h, x:x + w]
    count = w * h
    summed = float(active.sum(dtype=np.float64))
    squared = float(np.einsum("ij,ij->", active, active, dtype=np.float64))
    variance = max(0., squared - summed * summed / count)

    def correlation(glyph):
        mean = float(glyph.sum(dtype=np.float64))
        norm = float(np.einsum("ij,ij->", glyph, glyph, dtype=np.float64))
        variation = max(0., norm - mean * mean / count)
        product = float(np.einsum("ij,ij->", active, glyph, dtype=np.float64))
        centered = 1. if variation < 1e-12 else (
            (product - summed * mean / count) / sqrt(variance * variation) if variance else 0.)
        uncentered = product / sqrt(squared * norm) if squared and norm else 0.
        return .54 * centered + .46 * uncentered

    def reference(glyph):
        return (.54 * cv2.matchTemplate(active, glyph, cv2.TM_CCOEFF_NORMED)[0, 0]
                + .46 * cv2.matchTemplate(active, glyph, cv2.TM_CCORR_NORMED)[0, 0])

    def rank(score):
        scores = sorted(((max(map(score, _templates(paths, label, w, h)), default=0.), label)
                         for label in candidates), reverse=True)
        value, label = scores[0]
        rival = next(other for other, name in scores[1:] if name.casefold() != label.casefold())
        return value, value - rival, label

    value, margin, label = rank(correlation)
    # The two arrays have the same extent, so each comparison is a single dot
    # product. Use the original implementation at floating-point decision edges.
    if abs(value - .754) <= .0002 or abs(margin - .103) <= .0002:
        value, margin, label = rank(reference)
    if value >= .754 and margin >= .103:
        return label
    return None


def _phrase(gray, area, names, index, *, piece=None):
    l, t, r, b = area
    tile = gray[t:b, l:r]
    if min(tile.shape) < 6:
        return None
    ink = cv2.subtract(int(np.median(tile)), tile)
    peak = float(np.percentile(ink, 98))
    if peak < 39:
        return None
    signal = ink.astype(np.float32) / peak
    paths = _font_paths(index.fonts)
    reports = defaultdict(set)
    rivals = tuple(name for name in _RIVALS if name not in names)
    seen = set()
    for level in (.23, .33, .44, .54, .62):
        mask = np.uint8(ink >= level * peak)
        dx, dy, w, h = cv2.boundingRect(mask)
        if not 5 <= h <= .76 * len(tile) or not 1.48 * h <= w <= 26 * h:
            continue
        identities = [(mask, 0)]
        gaps = _words(mask, h)
        if piece == "suffix" and gaps:
            offset = gaps[-1][1]
            identities.append((mask[:, offset:], offset))
        if piece == "account" and len(gaps) >= 2:
            end = gaps[1][0]
            identities.append((mask[:, :end], 0))
        for chosen, offset in identities:
            identity = chosen.shape, chosen.tobytes(), offset
            if identity in seen:
                continue
            seen.add(identity)
            x, y, width, height = cv2.boundingRect(chosen)
            label = _match(signal, (*names, *rivals), paths, (x + offset, y, width, height))
            if label not in names or (label == "Comment as" and chosen is mask):
                continue
            verified_name = next(name for name in names if name.casefold() == label.casefold())
            reports[verified_name].add(identity)
    verified = [label for label, masks in reports.items() if len(masks) >= 2]
    return verified[0] if len(verified) == 1 else None


def _outlined_field(gray, panel):
    left, top, right, bottom = panel
    width = right - left
    start = max(top, bottom - round(.21 * width))
    mouth = gray[start:bottom, left:right]
    if np.median(mouth) < 245:
        return None
    contours_by_level = []
    for level in (6, 9, 11):
        ring = np.uint8(mouth.astype(np.int16) <= 255 - level)
        contours, _ = cv2.findContours(ring, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (.62 * width <= w <= .91 * width and .067 * width <= h <= .159 * width
                    and .071 * width <= x <= .24 * width and 0 <= bottom - (start + y + h) <= .066 * width
                    and .88 <= cv2.contourArea(contour) / (w * h) <= .98):
                continue
            mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask, [contour], 0, 1, cv2.FILLED, offset=(-x, -y))
            corner = max(2, round(.19 * h))
            corners = (mask[:corner, :corner], mask[:corner, -corner:], mask[-corner:, :corner], mask[-corner:, -corner:])
            if min(piece.mean() for piece in corners) >= .97 or max(piece.mean() for piece in corners) >= 1:
                continue
            fill = mouth[y + round(.22 * h):y + round(.78 * h), x + round(.11 * w):x + round(.84 * w)]
            if (fill >= 245).mean() < .88:
                continue
            contours_by_level.append((level, (left + x, start + y, left + x + w, start + y + h)))
    consensus = [box for level, box in contours_by_level if any(other != level and
                 max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= max(2, .011 * width)
                 for other, peer in contours_by_level)]
    if not consensus or np.ptp(consensus, axis=0).max() > max(3, .017 * width):
        return None
    return tuple(np.median(consensus, axis=0).astype(int))


def _dismiss(gray, panel, index):
    left, top, right, bottom = panel
    width = right - left
    a, t = right - round(.16 * width), top + round(.046 * width)
    box = a, t, right - round(.022 * width), min(bottom, top + round(.153 * width))
    area = gray[t:box[3], a:box[2]]
    if min(area.shape) < 6:
        return None
    if label := _phrase(gray, box, _CLOSE, index):
        return ("取消" if label in {"取消", "Cancel"} else "关闭"), box
    gray_ink = cv2.subtract(int(np.median(area)), area)
    observed = []
    for threshold in (43, 61, 85):
        _, labels, parts, _ = cv2.connectedComponentsWithStats(np.uint8(gray_ink >= threshold))
        for i, (x, y, w, h, count) in enumerate(parts[1:], 1):
            if (.024 * width <= min(w, h) <= max(w, h) <= .076 * width and .83 <= w / h <= 1.20
                    and .18 <= count / (w * h) <= .44 and _cross_footprint(labels[y:y + h, x:x + w] == i)):
                observed.append((threshold, (a + x, t + y, a + x + w, t + y + h)))
    matches = [mark for cut, mark in observed if any(other != cut and max(abs(x - y) for x, y in
               zip(mark, peer, strict=True)) <= 2 for other, peer in observed)]
    if matches and np.ptp(matches, axis=0).max() <= 2:
        return "关闭评论", tuple(np.median(matches, axis=0).astype(int))
    return None


def comment_sheet_modal(frame, index, *, overlay=None, regions=None, working_view=None, surfaces=None):
    """Return foreground scope and explicitly verified input/dismissal controls."""
    height, width = frame.shape[:2]
    if (index is None or width < 174 or not 1.55 * width <= height <= 2.73 * width
            or (overlay is not None and overlay.kind != "bottom_sheet")):
        return None
    _, gray = _working_view(frame) if working_view is None else working_view
    panels = list(_comment_sheet_surfaces(gray) if surfaces is None else surfaces)
    if not panels and (candidate := _rounded_light_sheet(gray)):
        panels.append(candidate)
    if len(panels) != 1:
        return None
    small = panels[0]
    field = account_comment_field(gray, small) or _outlined_field(gray, small)
    if field is None:
        return None
    factor = (width / gray.shape[1], height / len(gray)) * 2

    def project(box):
        return tuple(round(value * fraction) for value, fraction in zip(box, factor, strict=True))

    panel = project(small)
    if overlay is not None and max(abs(a - b) for a, b in zip(overlay.panel, panel, strict=True)) > .038 * width:
        return None
    field = project(field)
    l, t, r, b = panel
    w = r - l
    heading = l + round(.165 * w), t + round(.064 * w), r - round(.165 * w), t + round(.147 * w)
    native = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (regions is not None and not _available(heading, regions)) or not (
            _phrase(native, heading, _HEADS, index, piece="suffix") or _comment_heading(frame, panel, index)):
        return None
    x, y, end, d = field
    size = d - y
    prompt = (x + round(.14 * size), y + round(.19 * size),
              min(end - round(.92 * size), x + round(.68 * (end - x))), d - round(.17 * size))
    actions = []
    if (regions is None or _available(field, regions)) and (name := _phrase(native, prompt, _PROMPTS, index, piece="account")):
        box = [x, y, end - round(.83 * size), d]
        actions.append({"role": "comment_input", "function": "输入评论", "preview_text": name,
                        "box": box, "click_point": [(box[0] + box[2]) / 2, (y + d) / 2]})
    if (close := _dismiss(native, panel, index)) and (regions is None or _available(close[1], regions)):
        name, (x0, y0, x1, y1) = close
        actions.append({"role": "close", "function": name, "box": [x0, y0, x1, y1],
                        "click_point": [(x0 + x1) / 2, (y0 + y1) / 2]})
    return BlockingOverlay("bottom_sheet", panel), actions
