"""Identify strongly corroborated toolbar glyphs in screenshot pixels."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


_FUNCTIONS = {"drawer": "打开导航菜单", "search": "搜索", "sort": "排序",
              "refresh": "刷新", "more": "更多选项"}


@dataclass(frozen=True)
class _Glyph:
    box: tuple[int, int, int, int]
    mask: np.ndarray

    @property
    def center(self):
        left, top, right, bottom = self.box
        return (left + right) / 2, (top + bottom) / 2


def _spans(active):
    cuts = np.flatnonzero(np.diff(np.r_[False, active, False]))
    return list(zip(cuts[::2], cuts[1::2]))


def _title_boxes(nodes, width, height):
    short = min(width, height)
    options = []
    for node in nodes:
        label = str(node.get("content") or "").strip()
        score = node.get("score") or 0
        if (node.get("kind") != "text" or score < 88 or not label or len(label) > 36
                or not label[0].isalpha() or sum(char.isalpha() for char in label) < 2):
            continue
        x0, y0, x1, y1 = map(round, node["box"])
        h = y1 - y0
        if (max(9, .020 * short) <= h <= .11 * short and x1 - x0 >= .94 * h
                and 1.15 * h <= x0 < x1 < width - h
                and .024 * height <= (y0 + y1) / 2 <= .18 * height):
            options.append((x0, y0, x1, y1))
    return sorted(options, key=lambda box: box[1])


def _glyphs(gray, box, title_height):
    left, top, right, bottom = box
    roi = gray[top:bottom, left:right]
    if min(roi.shape) < 6:
        return []
    background = float(np.median(roi))
    deviation = np.abs(roi.astype(np.float32) - background)
    noise = np.median(deviation)
    mask = deviation >= max(37, 4 * noise + 11)
    if mask.mean() > .23:
        return []
    ink = mask.sum(axis=0) >= max(2, round(title_height * .04))
    spans = []
    for a, b in _spans(ink):
        if spans and a - spans[-1][1] <= max(1, round(.14 * title_height)):
            spans[-1] = spans[-1][0], b
        else:
            spans.append((a, b))
    result = []
    for a, b in spans:
        ys, xs = np.nonzero(mask[:, a:b])
        ink_left, ink_right = a + xs.min(), a + xs.max() + 1
        ink_top, ink_bottom = ys.min(), ys.max() + 1
        w, h = ink_right - ink_left, ink_bottom - ink_top
        if (.10 * title_height <= w <= 1.40 * title_height and .29 * title_height <= h <= 1.35 * title_height
                and ink_top > 0 and ink_bottom < roi.shape[0]):
            bounds = (left + int(ink_left), top + int(ink_top), left + int(ink_right), top + int(ink_bottom))
            result.append(_Glyph(bounds, mask[ink_top:ink_bottom, ink_left:ink_right]))
    return result


def _bars(mask):
    """Classify three solid horizontal stripes with matching or tapering length."""
    h, w = mask.shape
    if w < 1.08 * h:
        return None
    bands = _spans(mask.sum(axis=1) >= max(2, round(.24 * w)))
    if len(bands) != 3:
        return None
    bars = []
    for top, bottom in bands:
        ys, xs = np.nonzero(mask[top:bottom])
        left, right = xs.min(), xs.max() + 1
        length, thick = right - left, bottom - top
        if length < 2.25 * thick or len(xs) < .69 * length * thick:
            return None
        bars.append((left, right, length, thick))
    gaps = [bands[i + 1][0] - bands[i][1] for i in range(2)]
    if min(gaps) < 1 or max(gaps) > 1.85 * min(gaps):
        return None
    lengths = [bar[2] for bar in bars]
    if min(lengths) >= .82 * max(lengths):
        if np.ptp([bar[0] for bar in bars]) <= .15 * w and np.ptp([bar[1] for bar in bars]) <= .15 * w:
            return "drawer"
        return None
    first, middle, last = lengths
    if .47 <= middle / first <= .82 and .20 <= last / first <= .52 and last <= .78 * middle:
        if np.ptp([bar[0] for bar in bars]) <= .14 * w:
            return "sort"
    return None


def _is_more(mask):
    h, w = mask.shape
    if h < 2.35 * w:
        return False
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    rows = sorted(zip(stats[1:], centers[1:]), key=lambda row: row[1][1])
    if len(rows) != 3:
        return False
    sizes = []
    for (_, _, width, height, count), _ in rows:
        if not (.67 <= width / height <= 1.5 and .53 <= count / (width * height) <= 1):
            return False
        sizes.append((width + height) / 2)
    size = np.mean(sizes)
    xs = [center[0] for _, center in rows]
    gaps = np.diff([center[1] for _, center in rows])
    return bool(max(sizes) <= 1.45 * min(sizes) and np.ptp(xs) <= .35 * size
                and 1.05 * size <= min(gaps) <= max(gaps) <= 2.45 * size
                and np.ptp(gaps) <= .43 * size)


def _holes(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    return [contour for contour, indices in zip(contours, hierarchy[0], strict=True) if indices[3] >= 0]


def _is_search(mask, holes):
    h, w = mask.shape
    if not .73 <= w / h <= 1.40 or len(holes) != 1 or min(w, h) < 9:
        return False
    hole = holes[0]
    x0, y0, width, height = cv2.boundingRect(hole)
    if not (.69 <= width / height <= 1.44 and .12 * w * h <= cv2.contourArea(hole) <= .42 * w * h):
        return False
    cx, cy = x0 + (width - 1) / 2, y0 + (height - 1) / 2
    ys, xs = np.nonzero(mask)
    dx, dy = xs - cx, ys - cy
    radial = np.hypot(dx, dy)
    radius = (width + height) / 4
    far = radial > 1.58 * radius
    if not .07 <= np.mean(far) <= .38:
        return False
    ax, ay = np.mean(dx[far]), np.mean(dy[far])
    if not .37 <= abs(ax) / max(1e-6, abs(ay)) <= 2.7:
        return False
    directed = ((dx[far] * ax + dy[far] * ay) / radial[far] / np.hypot(ax, ay) >= .80).mean()
    around = (radial >= .66 * radius) & (radial <= 1.57 * radius)
    directions = np.floor((np.arctan2(dy[around], dx[around]) + np.pi) / (2 * np.pi) * 20).astype(int) % 20
    return directed >= .9 and len(set(directions)) >= 17


_ANGLES = np.arange(72) * 2 * np.pi / 72


def _polar(mask, radii):
    radii = np.asarray(radii)
    x = np.rint(31 + 31 * np.cos(_ANGLES)[:, None] * radii).astype(int)
    y = np.rint(31 + 31 * np.sin(_ANGLES)[:, None] * radii).astype(int)
    return mask[y, x].mean(axis=1)


def _is_refresh(mask, holes):
    h, w = mask.shape
    if holes or not .78 <= w / h <= 1.26 or min(w, h) < 9 or not .23 <= mask.mean() <= .59:
        return False
    shape = cv2.resize(mask.astype(np.uint8), (63, 63), interpolation=cv2.INTER_NEAREST)
    outer = _polar(shape, np.arange(.60, 1.01, .065)) >= .24
    gaps = _spans(np.tile(~outer, 2))
    if not gaps:
        return False
    start, stop = max(gaps, key=lambda interval: interval[1] - interval[0])
    missing = 72 - int(outer.sum())
    if not 3 <= stop - start <= 16 or missing > 21 or stop - start < .60 * missing:
        return False
    core = _polar(shape, (.12, .18, .24))
    if core.mean() > .08:
        return False
    inner = _polar(shape, np.arange(.36, .65, .05)) >= .35
    if not 3 <= inner.sum() <= 19:
        return False
    before = np.arange(start - 10, start) % 72
    after = np.arange(stop, stop + 10) % 72
    first, second = inner[before].sum(), inner[after].sum()
    # The inward triangle is concentrated at one end of the arc, unlike a C.
    return max(first, second) >= .65 * inner.sum() and min(first, second) <= .25 * inner.sum()


def _right_role(glyph):
    mask = glyph.mask
    if _is_more(mask):
        return "more"
    if _bars(mask) == "sort":
        return "sort"
    holes = _holes(mask)
    single_component, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if single_component != 2:
        return None
    if _is_search(mask, holes):
        return "search"
    if _is_refresh(mask, holes):
        return "refresh"
    return None


def _row_actions(gray, title):
    image_h, image_w = gray.shape
    x0, y0, x1, y1 = title
    h = y1 - y0
    top, bottom = max(0, round(y0 - .27 * h)), min(image_h, round(y1 + .37 * h))
    left = _glyphs(gray, (0, top, round(x0 - .20 * h), bottom), h)
    right = _glyphs(gray, (round(x1 + .20 * h), top, image_w, bottom), h)
    drawers = [glyph for glyph in left if _bars(glyph.mask) == "drawer"]
    if len(drawers) != 1:
        return []
    drawer = drawers[0]
    candidates = [(role, glyph) for glyph in right if abs(glyph.center[1] - drawer.center[1]) <= .25 * h
                  and (role := _right_role(glyph)) is not None]
    more = [glyph for role, glyph in candidates if role == "more"]
    if len(more) != 1:
        return []
    # A compact app bar is corroborated by the left drawer, its title, and
    # a separate overflow at the trailing end of the same visible lane.
    nodes = [("drawer", drawer), *((role, glyph) for role, glyph in candidates
                                   if sum(other == role for other, _ in candidates) == 1
                                   and glyph.center[0] <= more[0].center[0])]
    actions = []
    for role, glyph in nodes:
        left, top, right, bottom = glyph.box
        cx, cy = glyph.center
        pad = max(.22 * h, .18 * max(right - left, bottom - top))
        half_w, half_h = max((right - left) / 2 + pad, .63 * h), max((bottom - top) / 2 + pad, .66 * h)
        actions.append({"role": role, "function": _FUNCTIONS[role], "center": (cx, cy),
                        "box": (max(0, round(cx - half_w)), max(0, round(cy - half_h)),
                                min(image_w, round(cx + half_w)), min(image_h, round(cy + half_h)))})
    return actions


def suggest_toolbar_actions(frame_bgr, nodes, *, overlay=None) -> list[dict]:
    """Find app-bar actions with OCR and glyph evidence; ignore blocked bars."""
    if overlay is not None:
        return []
    height, width = frame_bgr.shape[:2]
    titles = _title_boxes(nodes, width, height)
    if not titles:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    for title in titles:
        actions = _row_actions(gray, title)
        if actions:
            return actions
    return []
