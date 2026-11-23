"""Read the inactive label in a paired foreground tab strip under a search field."""

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_game_shop_tabs import _caption
from screen_sheet_search import sheet_search_actions


def _divider(gray, panel, field_bottom):
    left, _, right, bottom = panel
    width = right - left
    inset = round(.025 * width)
    start = field_bottom + round(.084 * width)
    stop = min(bottom, field_bottom + round(.163 * width))
    tile = gray[start:stop, left + inset:right - inset].astype(np.int16)
    if not tile.size:
        return None
    paper = int(np.percentile(tile, 78))
    if paper < 232:
        return None
    faint = (tile >= paper - 43) & (tile <= paper - 8)
    lines = [(a, b) for a, b in _bands(faint.mean(axis=1) >= .72) if b - a <= .007 * width]
    if len(lines) != 1:
        return None
    a, b = lines[0]
    pad = max(2, round(.013 * width))
    upper, lower = start + a - pad, start + b + pad
    if lower >= bottom or upper <= field_bottom:
        return None
    ends = gray[[upper, lower], left + inset:right - inset] >= paper - 6
    return (start + a, start + b, paper) if ends.mean(axis=1).min() >= .83 else None


def _captions(frame, gray, panel, top, end, paper):
    left, _, right, _ = panel
    width = right - left
    midpoint = (left + right) // 2
    inset = round(.07 * width)
    names = []
    for a, b in ((left + inset, midpoint), (midpoint, right - inset)):
        tile = gray[top:end, a:b]
        ink = tile <= paper - 58
        x, y, w, h = cv2.boundingRect(np.uint8(ink))
        if (not .020 * width <= h <= .055 * width or not 1.75 <= w / h <= 8.0
                or not 0 < x < x + w < b - a or not 0 < y < y + h < end - top):
            return []
        pixels = frame[top:end, a:b][ink]
        if np.percentile(np.ptp(pixels, axis=1), 85) > 19:
            return []
        names.append(((x + a, y + top, x + a + w, y + top + h), paper - float(np.median(tile[ink]))))
    first, second = [box for box, _ in names]
    fh, sh = first[3] - first[1], second[3] - second[1]
    fx, sx = [(a + b) / 2 for a, _, b, _ in (first, second)]
    if (abs(first[1] - second[1]) > .23 * min(fh, sh) or not .83 <= fh / sh <= 1.20
            or not .24 * width <= sx - fx <= .64 * width or abs(fx + sx - left - right) > .12 * width):
        return []
    return names


def _selected(frame, gray, panel, divider, labels):
    left, _, right, _ = panel
    width = right - left
    edge, stop, paper = divider
    depth = max(2, round(.010 * width))
    inset = round(.025 * width)
    ink = gray[edge - depth:stop, left + inset:right - inset] <= paper - 122
    spans = _bands(ink.sum(axis=0) >= max(2, round(.0024 * width)))
    if len(spans) != 1:
        return None
    x, end = spans[0]
    if not .085 * width <= end - x <= .31 * width or ink[depth:, x:end].mean() < .76:
        return None
    painted = frame[edge - depth:stop, left + inset + x:left + inset + end][ink[:, x:end]]
    if np.percentile(np.ptp(painted, axis=1), 85) > 19:
        return None
    marked = []
    bar_center = left + inset + (x + end) / 2
    for i, ((a, y, b, bottom), _) in enumerate(labels):
        near = abs((a + b) / 2 - bar_center) <= .18 * (b - a)
        if near and 1.01 * (b - a) <= end - x <= 1.78 * (b - a) and .21 <= (edge - bottom) / (bottom - y) <= 1.29:
            marked.append(i)
    if len(marked) != 1:
        return None
    selected = marked[0]
    active, other = labels[selected][1], labels[1 - selected][1]
    return selected if active - other >= 30 and other >= max(91, .42 * active) else None


def sheet_paired_tab_actions(frame, regions, index, *, search_actions=None):
    """Return the literal sibling of one underlined tab inside a proven search sheet."""
    if index is None or regions.get("layer") != "bottom_sheet" or not regions.get("regions"):
        return []
    panel = tuple(regions["regions"][0])
    left, top, right, bottom = panel
    height, frame_width = frame.shape[:2]
    width = right - left
    if not (0 <= left < right <= frame_width and 0 <= top < bottom <= height
            and width >= .78 * frame_width and bottom - top >= .52 * width):
        return []
    controls = sheet_search_actions(frame, regions) if search_actions is None else search_actions
    fields = [item["box"] for item in controls if item["role"] == "search_input"]
    if len(fields) != 1:
        return []
    a, t, b, end = fields[0]
    if not (left <= a < b <= right and top < t < end < bottom and .68 * width <= b - a <= .94 * width):
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (divider := _divider(gray, panel, end)) is None:
        return []
    edge, stop, paper = divider
    tab_top = max(end + round(.022 * width), edge - round(.091 * width))
    labels = _captions(frame, gray, panel, tab_top, edge - round(.008 * width), paper)
    if len(labels) != 2 or (selected := _selected(frame, gray, panel, divider, labels)) is None:
        return []
    x, y, r, base = labels[1 - selected][0]
    pad_x, pad_y = max(3, round(.24 * (base - y))), max(4, round(.37 * (base - y)))
    tile = 255 - gray[y - pad_y:min(edge - 1, base + pad_y), x - pad_x:r + pad_x]
    bounds = 0, 0, tile.shape[1], tile.shape[0]
    cache = {}
    # The font matcher verifies real glyphs from two scales and does not rely on tab names.
    if (name := _caption(tile, bounds, index, cache)) is None:
        return []
    # An unusual glyph must be independently strong at both sampled sizes.
    if name.encode("gb2312", errors="ignore").decode("gb2312") != name and _caption(
            tile, bounds, index, cache, min_glyph_margin=.017) != name:
        return []
    split = (labels[0][0][2] + labels[1][0][0]) // 2
    span = (split, right - round(.035 * width)) if selected == 0 else (left + round(.035 * width), split)
    return [{"role": "switch_tab", "function": f"切换到{name}", "preview_text": name,
             "box": [span[0], tab_top, span[1], stop], "click_point": [(x + r) // 2, (y + base) // 2]}]
