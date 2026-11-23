"""Locate a populated search field and its clear button on a visible sheet."""

import cv2
import numpy as np

from inventory_toolbar_actions import _holes
from screen_candidate_glyphs import _overlap, _search
from screen_foreground import _centered_cross


def _short_handle(ink):
    if _search(ink):
        return True
    h, w = ink.shape
    holes = _holes(ink)
    if len(holes) != 1 or not .83 <= w / h <= 1.18 or not .12 <= ink.mean() <= .28:
        return False
    hole = holes[0]
    x, y, a, b = cv2.boundingRect(hole)
    if not (.85 <= a / b <= 1.18 and .39 <= cv2.contourArea(hole) / (w * h) <= .58):
        return False
    ys, xs = np.nonzero(ink)
    dx, dy = xs - x - (a - 1) / 2, ys - y - (b - 1) / 2
    tail = np.hypot(dx, dy) >= 1.36 * (a + b) / 4
    if not .045 <= tail.mean() <= .20:
        return False
    return bool(((dx[tail] > 0) & (dy[tail] > 0)).mean() >= .92
                and .63 <= dx[tail].mean() / max(1, dy[tail].mean()) <= 1.59)


def _glyph_box(gray, anchor, kind):
    left, top, right, bottom = anchor
    chip = gray[top:bottom, left:right]
    surface = float(np.median(chip))
    if surface < 217:
        return None
    sightings = []
    for contrast in (21, 34, 47, 58):
        ink = np.uint8(chip <= surface - contrast)
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (not .36 * chip.shape[1] <= w <= .88 * chip.shape[1]
                    or not .31 * chip.shape[0] <= h <= .88 * chip.shape[0]
                    or not 0 < x < x + w < chip.shape[1] or not 0 < y < y + h < chip.shape[0]):
                continue
            if kind == "lens":
                if not _short_handle(ink[y:y + h, x:x + w]):
                    continue
                proved = True
            else:
                if not .89 <= w / h <= 1.12 or not .66 <= cv2.contourArea(contour) / (w * h) <= .83:
                    continue
                dx, dy = max(2, round(.21 * w)), max(2, round(.21 * h))
                center = chip[y + dy:y + h - dy, x + dx:x + w - dx]
                proved = center.size > 0 and np.ptp(center) >= 35 and _centered_cross(
                    cv2.normalize(center, None, 0, 255, cv2.NORM_MINMAX))
            sightings.append((contrast, (left + x, top + y, left + x + w, top + y + h), proved))
    stable = [(box, proved) for cut, box, proved in sightings if any(
        other != cut and _overlap(box, peer) >= .80 for other, peer, _ in sightings)]
    if stable and any(proved for _, proved in stable) and all(_overlap(stable[0][0], box) >= .80 for box, _ in stable):
        return stable[len(stable) // 2][0]
    return None


def _center(box):
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def _field(gray, panel, lens, clear):
    left, top, right, bottom = panel
    width = right - left
    x, y = _center(lens)
    cx, cy = _center(clear)
    if abs(cy - y) > .021 * width:
        return None
    reach = round(.092 * width)
    a, b = max(top, round(y) - reach), min(bottom, round(y) + reach)
    chip = gray[a:b, left:right]
    mid = slice(round(.29 * width), round(.71 * width))
    paper = float(np.median(chip[[0, -1], mid]))
    if paper < 235:
        return None
    ink = np.uint8((chip >= paper - 39) & (chip <= paper - 4))
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for outline in contours:
        u, v, w, h = cv2.boundingRect(outline)
        l, t, r, s = left + u, a + v, left + u + w, a + v + h
        if (not .76 * width <= w <= .97 * width or not .052 * width <= h <= .145 * width
                or not .94 <= cv2.contourArea(outline) / (w * h) <= .994
                or abs((l + r) - (left + right)) > .05 * width or not t < y < s
                or not .21 * h <= x - l <= .90 * h or not .21 * h <= r - cx <= .90 * h):
            continue
        text_left, text_right = lens[2] + round(.20 * h), round(l + .60 * w)
        text_top, text_bottom = t + round(.18 * h), s - round(.18 * h)
        tx, ty, tw, th = cv2.boundingRect(np.uint8(gray[text_top:text_bottom, text_left:text_right] <= paper - 65))
        if .20 * h <= th <= .63 * h and .24 * h <= tw <= .47 * w and tx <= .48 * h:
            return (l, t, clear[0] - round(.14 * h), s), (text_left + tx + tw / 2, text_top + ty + th / 2)
    return None


def sheet_search_actions(frame, regions):
    """Verify the lens, editable content, painted field and distinct clear glyph."""
    if regions["layer"] != "bottom_sheet" or not regions["regions"]:
        return []
    panel = regions["regions"][0]
    left, top, right, _ = panel
    width = right - left
    near = [row for row in regions["candidates"] if row["visibility"] == "foreground"
            and .12 * width <= row["center"][1] - top <= .37 * width
            and .035 * width <= row["box"][3] - row["box"][1] <= .135 * width
            and .038 * width <= row["box"][2] - row["box"][0] <= .15 * width]
    leading = [row["box"] for row in near if row["center"][0] - left <= .155 * width]
    trailing = [row["box"] for row in near if right - row["center"][0] <= .155 * width]
    if not leading or not trailing:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lenses = {box for anchor in leading if (box := _glyph_box(gray, anchor, "lens")) is not None}
    clears = {box for anchor in trailing if (box := _glyph_box(gray, anchor, "clear")) is not None}
    fields = [(result, clear) for lens in lenses for clear in clears
              if (result := _field(gray, panel, lens, clear)) is not None]
    if len(fields) != 1:
        return []
    (search, point), clear = fields[0]
    return [{"role": "search_input", "function": "编辑搜索内容", "box": list(search),
             "click_point": list(point), "preview_text": "Search"},
            {"role": "clear_search", "function": "清除搜索内容", "box": list(clear),
             "click_point": list(_center(clear)), "preview_text": "Clear search"}]
