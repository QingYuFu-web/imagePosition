"""Detect visually separated blocking overlays and isolate their active controls."""

from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import product
import math
import unicodedata

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_artwork import is_captioned_artwork_row
from screen_composer import nested_composer_bounds
from screen_rim_attached_dismiss import _cross_arms


COMMENT_SHEET_TITLES = {"comments", "评论", "留言", "評論"}


def _folded(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def is_comment_sheet_heading(value):
    title = _folded(value)
    if title in COMMENT_SHEET_TITLES:
        return True
    prefix = title.removesuffix("comments").strip()
    return (prefix != title and 1 <= len(prefix) <= 24
            and 1 <= len(prefix.split()) <= 2
            and all(character.isalpha() or character.isspace() for character in prefix))


def is_account_comment_placeholder(value):
    text = _folded(value).rstrip(" .…")
    intro, sep, account = text.partition(" as ")
    if (intro, sep) != ("comment", " as "):
        compact = "".join(text.split())
        if not compact.startswith("commentas"):
            return False
        account = compact.removeprefix("commentas")
    return (2 <= len(account) <= 32 and any(character.isalpha() for character in account)
            and all(character.isalnum() or character in " ._-" for character in account))


@dataclass(frozen=True)
class BlockingOverlay:
    kind: str
    panel: tuple[int, int, int, int]
    close_button: tuple[int, int, int, int] | None = None
    header_bottom: int | None = None
    close_style: str | None = None
    recording_stop: tuple[int, int, int, int] | None = None
    share_icons: tuple[tuple[int, int, int, int], ...] = ()
    include_coarse_proposals: bool = False
    comment_sheet: bool = False

    @property
    def regions(self):
        return (self.panel,) if self.close_button is None else (self.panel, self.close_button)

    @property
    def attached_close(self):
        if self.close_button is None:
            return None
        left, top, right, bottom = self.close_button
        panel_left, panel_top, panel_right, panel_bottom = self.panel
        return self.close_button if (panel_left <= left < right <= panel_right
                                     and panel_top <= top < bottom <= panel_bottom) else None

    def _region_at(self, point):
        x, y = point
        return next((region for region in self.regions
                     if region[0] <= x < region[2] and region[1] <= y < region[3]), None)

    def contains(self, point):
        return self._region_at(point) is not None

    def filter_elements(self, elements):
        accepted = []
        for element in elements:
            polygon = np.asarray(element["polygon"])
            region = self._region_at(polygon.mean(axis=0))
            if region is None:
                continue
            lower, upper = polygon.min(axis=0), polygon.max(axis=0)
            overlap = np.maximum(0, np.minimum(upper, region[2:]) - np.maximum(lower, region[:2]))
            if overlap.prod() >= .6 * (upper - lower).prod():
                accepted.append(element)
        return accepted

    def filter_controls(self, controls):
        controls = self.filter_elements(controls)
        if self.kind != "dialog":
            return controls
        left, top, right, bottom = self.panel
        close = self.close_button
        attached = self.attached_close
        boxes = [np.array([np.min(item["polygon"], axis=0), np.max(item["polygon"], axis=0)]) for item in controls]
        accepted = []
        for item, box in zip(controls, boxes):
            x0, y0 = box[0]
            x1, y1 = box[1]
            center = box.mean(axis=0)
            if close is not None and (center >= close[:2]).all() and (center < close[2:]).all():
                if attached is None:
                    accepted.append({**item, "label": "关闭", "source": "modal_close"})
                continue
            # A broad panel decoration can contain a separate, smaller button.
            spans_panel = (x0 <= left + .04 * (right - left) and x1 >= right - .04 * (right - left)
                           and y1 - y0 >= .35 * (bottom - top))
            if spans_panel and any(
                    (child[0] >= box[0]).all() and (child[1] <= box[1]).all()
                    and (child[1] - child[0]).prod() <= .4 * (box[1] - box[0]).prod()
                    for child in boxes):
                continue
            accepted.append(item)
        return accepted

    def isolate(self, frame):
        result = np.zeros_like(frame)
        for left, top, right, bottom in self.regions:
            result[top:bottom, left:right] = frame[top:bottom, left:right]
        return result

    def contains_fraction(self, point, scale):
        return point is not None and self.contains(np.asarray(point) * scale)

    def describe(self, scale):
        description = {"kind": self.kind, "regions_fraction": [
            np.minimum(1., np.asarray(region).reshape(2, 2) / scale).round(5).ravel().tolist()
            for region in self.regions
        ]}
        if self.header_bottom is not None:
            description["header_bottom_fraction"] = round(self.header_bottom / scale[1], 5)
        return description


def _has_background_above_sheet(gray, top):
    height, width = gray.shape
    if top >= .19 * height:
        return True
    margin = round(.09 * width)
    content = gray[round(.067 * height):top - 2, margin:width - margin]
    if content.size == 0:
        return False
    low, median, high = np.percentile(content, (12, 50, 92))
    return median >= 19 or high - low >= 17


def _live_toolbar_above(gray, top):
    """Recognize an undimmed toolbar meeting a normal light page."""
    height, width = gray.shape
    if top > .21 * height:
        return None
    first = max(round(.027 * height), top - round(.145 * width))
    end = top - max(2, round(.01 * width))
    band = gray[first:end]
    if len(band) < .065 * width:
        return None
    tone = np.median(band)
    if not 14 <= tone <= 112 or np.mean(np.abs(band.astype(np.int16) - tone) <= 13) < .87:
        return None
    left = band[:, :round(.17 * width)]
    threshold = max(115, round(tone) + 72)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(left >= threshold))
    for index, (x, y, w, h, _) in enumerate(stats[1:], 1):
        if not (.023 * width <= min(w, h) <= max(w, h) <= .089 * width
                and .030 * width <= x + w / 2 <= .126 * width):
            continue
        glyph = labels[y:y + h, x:x + w] == index
        if (np.max(left[y:y + h, x:x + w][glyph]) < 216 or not is_shaft_back_arrow(glyph)
                or not .038 * width <= len(band) - (y + h / 2) <= .108 * width):
            continue
        # Several bright, separate marks share the Back arrow's toolbar row.
        limit = round(.39 * width)
        row_ink = band[y:y + h, limit:] >= 193
        occupied = np.uint8(row_ink.any(axis=0))[None, :]
        connected = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, np.ones((1, max(3, round(.016 * width))), np.uint8))
        _, _, segments, _ = cv2.connectedComponentsWithStats(connected)
        peers = sum(.012 * width <= span <= .14 * width and row_ink[:, start:start + span].sum() >= 8
                    for start, _, span, _, _ in segments[1:])
        title_left, title_right = round(.16 * width), round(.70 * width)
        title = band[max(0, y - 2):y + h + 2, title_left:title_right] >= 196
        runs = np.flatnonzero(np.diff(np.r_[False, title.any(axis=0), False])).reshape(-1, 2)
        title_row = (len(runs) >= 4 and runs[0, 0] <= .066 * width and
                     .135 * width <= runs[-1, 1] - runs[0, 0] <= .49 * width and
                     max(np.diff(runs.flatten())[1::2], default=0) <= .027 * width)
        if peers >= 2 or title_row:
            return int(x), first + int(y), int(x + w), first + int(y + h)
    return None


def _full_width_light_edges(gray, limit=.88, *, rail_inset=0):
    height, width = gray.shape
    margin = round(.07 * width)
    columns = np.r_[rail_inset:margin, width - margin:width - rail_inset]
    # A sheet has a wide light surface newly covering a dimmed upper layer.
    for y in range(round(.075 * height), round(limit * height)):
        if gray[y, width // 2] < 170:
            continue
        before = gray[y - 5:y - 1]
        after = gray[y:y + 4]
        if (before < 100).mean() < .81 or (after > 195).mean() < .81:
            continue
        above = np.median(before, axis=0)
        below = np.median(after, axis=0)
        edge = (above < 100) & (below > 195) & (below - above > 80)
        if (edge.mean() >= .84 and (gray[round(.015 * height):y - 3, columns] < 95).mean() >= .85
                and _has_background_above_sheet(gray, y)):
            yield y


def _bottom_sheet(gray, *, rail_inset=0):
    height, width = gray.shape
    for y in _full_width_light_edges(gray, rail_inset=rail_inset):
        if _live_toolbar_above(gray, y) is None or _sheet_header_control(gray, y, right_only=True):
            return 0, y, width, height
    return None


def live_page_toolbar_back(frame, *, working_view=None):
    """Project the verified Back arrow of a live toolbar to its original box."""
    _, gray = _working_view(frame) if working_view is None else working_view
    if gray is None:
        return None
    boxes = {back for y in _full_width_light_edges(gray, limit=.21)
             if (back := _live_toolbar_above(gray, y)) is not None and not _sheet_header_control(gray, y, right_only=True)}
    if len(boxes) != 1:
        return None
    x0, y0, x1, y1 = boxes.pop()
    sx, sy = frame.shape[1] / gray.shape[1], frame.shape[0] / gray.shape[0]
    return math.floor(x0 * sx), math.floor(y0 * sy), math.ceil(x1 * sx), math.ceil(y1 * sy)


def _rounded_light_sheet(gray, *, heading_y=None):
    height, width = gray.shape
    trim = round(.13 * width)
    middle = gray[:, trim:width - trim]
    white = (middle >= 222).mean(axis=1)
    inner = max(2, round(.010 * width))
    halo = max(4, round(.041 * width))
    edge = round(.026 * width)
    flank = max(2, round(.023 * width))
    grip_half_width = round(.15 * width)
    center_dark = (middle <= 145).mean(axis=1)
    bands = np.flatnonzero((white[inner:] >= .93) & (center_dark[:-inner] >= .90)) + inner
    for y in bands:
        if not .22 * height <= y <= .86 * height or y + 3 * halo >= height:
            continue
        if (white[y:y + halo].mean() < .93
                or center_dark[y - halo:y - inner].mean() < .87):
            continue
        shoulders = (slice(edge, edge + flank), slice(width - edge - flank, width - edge))
        if any((gray[y:y + inner, side] <= 163).mean() < .60
               or (gray[y + halo:y + 2 * halo, side] >= 222).mean() < .85 for side in shoulders):
            continue
        if any((gray[round(.025 * height):y - halo, side] < 122).mean() < .71 for side in shoulders):
            continue
        if heading_y is not None:
            if .067 <= (heading_y - y) / (height - y) <= .18:
                return 0, int(y), width, height
            continue
        mid = width // 2
        top = y + max(2, round(.018 * width))
        bottom = y + round(.066 * width)
        grip_zone = gray[top:bottom, mid - grip_half_width:mid + grip_half_width]
        dark_stroke = np.uint8(grip_zone <= 182)
        _, _, components, _ = cv2.connectedComponentsWithStats(dark_stroke, connectivity=8)
        grips = [(x, yy, w, h) for x, yy, w, h, area in components[1:]
                 if .046 * width <= w <= .15 * width and h <= .018 * width
                 and abs(x + w / 2 - grip_half_width) <= .023 * width
                 and area >= .58 * w * h]
        if len(grips) == 1 and (grip_zone >= 225).mean() >= .88:
            return 0, int(y), width, height
    return None


def _has_panel_surface(salient):
    if salient.mean() >= .55:
        return True
    # A thin dashed callout can become a closed contour only at the coarser scale.
    fine = cv2.morphologyEx(salient, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(fine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(contour) >= .70 * salient.size for contour in contours)


def _center_panel(frame, gray):
    height, width = gray.shape
    if frame.dtype == np.uint8 and frame.shape[2] == 3:
        salient = np.uint8(cv2.inRange(frame, (0, 0, 0), (134, 134, 134)) == 0)
    else:
        salient = np.uint8(frame.max(axis=2) >= 135)
    solid = cv2.morphologyEx(salient, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gap, band = 2, round(.028 * width)
    row_band = round(.016 * height)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        left, top, w, h = cv2.boundingRect(contour)
        right, bottom = left + w, top + h
        if not (.25 * width <= w <= .91 * width and .14 * height <= h <= .80 * height
                and .30 * width <= left + w / 2 <= .70 * width
                and .04 * width <= left < right <= .96 * width
                and row_band + gap < top < bottom < height - row_band - gap
                and cv2.contourArea(contour) >= .85 * w * h):
            continue
        perimeter = (
            gray[top:bottom, left - band - gap:left - gap],
            gray[top:bottom, right + gap:right + band + gap],
            gray[top - row_band - gap:top - gap, left:right],
            gray[bottom + gap:bottom + row_band + gap, left:right],
        )
        if not all((edge < 100).mean() >= .85 for edge in perimeter) or not _has_panel_surface(salient[top:bottom, left:right]):
            continue
        panel = left, top, right, bottom
        if is_captioned_artwork_row(frame, panel) and not _has_artwork_dismissal(gray, panel):
            continue
        return panel
    return None


def _extend_dialog_header(frame, gray, bounds):
    """Include a painted header when both physical side edges continue from the body."""
    left, top, right, bottom = bounds
    height, width = gray.shape
    span = right - left
    edge = max(3, round(.025 * span))
    inner = max(2, round(.012 * span))
    if left < edge + 2 or right + edge + 2 >= width or top < edge or span < .42 * width:
        return bounds
    if np.median(gray[top:top + edge, left + edge:right - edge]) < 207:
        return bounds
    start = max(round(.07 * height), top - round(min(.35 * (bottom - top), .43 * span)))
    if top - start < .12 * span:
        return bounds
    sample = frame[start:top]
    central = np.median(sample[:, left + round(.25 * span):right - round(.25 * span)], axis=1)
    valid = np.ptp(central, axis=1) >= 67
    for inside, outside in ((slice(left + inner, left + inner + edge), slice(left - edge - 2, left - 2)),
                            (slice(right - inner - edge, right - inner), slice(right + 2, right + edge + 2))):
        surface = np.median(sample[:, inside], axis=1)
        backdrop = np.median(sample[:, outside], axis=1)
        valid &= ((np.abs(surface - central).sum(axis=1) <= 31)
                  & (np.ptp(backdrop, axis=1) <= 46)
                  & (np.abs(surface - backdrop).sum(axis=1) >= 77))
    matches = np.flatnonzero(valid)
    if not matches.size or len(valid) - matches[-1] > max(3, round(.014 * span)):
        return bounds
    end = int(matches[-1])
    beginning = max(np.flatnonzero(~valid[:end]), default=-1) + 1
    if beginning == 0 or end - beginning < .115 * span:
        return bounds

    # At a rounded top, the colored center can appear a few rows sooner than
    # the straight side edges.
    color = central[beginning]
    for _ in range(edge):
        if beginning == 0 or np.abs(central[beginning - 1] - color).sum() > 34:
            break
        beginning -= 1
    return left, start + int(beginning), right, bottom


def _has_close_mark(crop):
    small = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)
    lines = cv2.HoughLinesP(cv2.Canny(small, 30, 90), 1, np.pi / 180, 12,
                            minLineLength=14, maxLineGap=5)
    if lines is None:
        return False
    directions = set()
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        dx, dy = int(x1) - int(x0), int(y1) - int(y0)
        if (dx and .55 <= abs(dy / dx) <= 1.65 and
                abs((x0 + x1) / 2 - 32) <= 15 and abs((y0 + y1) / 2 - 32) <= 15):
            directions.add(dx * dy > 0)
    return len(directions) == 2


def _centered_cross(crop):
    gray = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(cv2.Canny(gray, 30, 90), 1, np.pi / 180, 12,
                           minLineLength=14, maxLineGap=5)
    if lines is None:
        return False
    diagonals = ([], [])
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        dx, dy = int(x1) - int(x0), int(y1) - int(y0)
        if dx == 0 or not .55 <= abs(dy / dx) <= 1.65:
            continue
        slope = dy / dx
        diagonals[int(slope > 0)].append((slope, y0 - slope * x0, min(x0, x1), max(x0, x1)))
    for first, second in product(*diagonals):
        x = (second[1] - first[1]) / (first[0] - second[0])
        y = first[0] * x + first[1]
        if 17 <= x <= 47 and 17 <= y <= 47 and all(
                left + .2 * (right - left) <= x <= right - .2 * (right - left)
                for _, _, left, right in (first, second)):
            return True
    return False


def _dark_rim_lines(delta, width):
    joined = cv2.morphologyEx(np.uint8(delta >= 14), cv2.MORPH_CLOSE,
                              np.ones((1, max(5, round(.033 * width))), np.uint8))
    lines = cv2.morphologyEx(joined, cv2.MORPH_OPEN,
                             np.ones((1, round(.21 * width)), np.uint8))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        start, row, length, thickness = cv2.boundingRect(contour)
        if .21 * width <= length <= 1.22 * width and thickness <= .028 * width:
            yield row + 2 + thickness // 2, start, start + length


def _rim_evidence(gray, panel):
    left, top, right, bottom = panel
    inset = max(4, round(.07 * min(right - left, bottom - top)))
    columns = slice(left + inset, right - inset)
    rows = slice(top + inset, bottom - inset)
    edges = ((gray[top - 3, columns], gray[top + 3, columns]),
             (gray[bottom + 3, columns], gray[bottom - 3, columns]),
             (gray[rows, left - 3], gray[rows, left + 3]),
             (gray[rows, right + 3], gray[rows, right - 3]))
    support = [np.mean(outside - inside >= 13) for outside, inside in edges]
    if min(support) < .55 or sum(ratio >= .74 for ratio in support) < 3:
        return False
    # The same varying scene persists across the boundary through a dark tint.
    texture = sum(outside.std() >= 12 and inside.std() >= 4 and np.corrcoef(outside, inside)[0, 1] >= .66
                  for outside, inside in edges)
    return texture >= 2


def _bright_crosses(gray):
    height, width = gray.shape
    x_base, y_base = round(.46 * width), round(.14 * height)
    patch = gray[y_base:round(.68 * height), x_base:round(.97 * width)]
    hits = []
    for threshold in (175, 135):
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(patch >= threshold))
        for x, y, w, h, area in stats[1:]:
            if not (.013 * width <= min(w, h) <= max(w, h) <= .075 * width
                    and .70 <= w / h <= 1.42 and .16 <= area / (w * h) <= .62):
                continue
            x, y = int(x) + x_base, int(y) + y_base
            cx, cy = x + w / 2, y + h / 2
            if any(abs(cx - px) <= 2 and abs(cy - py) <= 2 for px, py, _, _ in hits):
                continue
            pad = max(2, round(.35 * max(w, h)))
            if _centered_cross(gray[y - pad:y + h + pad, x - pad:x + w + pad]):
                hits.append((cx, cy, int(w), int(h)))
    return hits


def _rim_close(cross, panel):
    cx, cy, w, h = cross
    left, top, right, bottom = panel
    span = right - left
    depth = min(.22 * span, .18 * (bottom - top))
    if not (.027 * span <= min(w, h) <= max(w, h) <= .09 * span
            and .20 * depth <= cy - top <= .65 * depth
            and .20 * depth <= right - cx <= .68 * depth):
        return None
    extent = max(3, round(.55 * max(w, h)))
    return (max(left, round(cx - w / 2) - extent), max(top, round(cy - h / 2) - extent),
            min(right, round(cx + w / 2) + extent), min(bottom, round(cy + h / 2) + extent))


def _translucent_dialog(gray):
    """Find an enclosed dark overlay with a close X attached to its corner."""
    source_height, source_width = gray.shape
    width = max(320, source_width)
    height = round(source_height * width / source_width)
    if width > source_width:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    crosses = _bright_crosses(gray)
    if not crosses:
        return None
    luma = gray.astype(np.int16)
    row_diff = luma[4:] - luma[:-4]
    tops = list(_dark_rim_lines(-row_diff, width))
    bottoms = list(_dark_rim_lines(row_diff, width))
    if not tops or not bottoms:
        return None
    column_diff = luma[:, 4:].T - luma[:, :-4].T
    lefts = list(_dark_rim_lines(-column_diff, width))
    rights = list(_dark_rim_lines(column_diff, width))
    margin = .09 * width
    matches = set()
    for (top, start, end), (bottom, base_start, base_end) in product(tops, bottoms):
        panel_height = bottom - top
        if not (.24 * width <= panel_height <= 1.16 * width and .29 * height <= (top + bottom) / 2 <= .71 * height
                and top >= 4 and bottom < height - 4
                and max(abs(start - base_start), abs(end - base_end)) <= margin):
            continue
        near_left = [x for x in lefts if max(abs(x[0] - start), abs(x[0] - base_start)) <= margin]
        near_right = [x for x in rights if max(abs(x[0] - end), abs(x[0] - base_end)) <= margin]
        for (left, left_top, left_bottom), (right, right_top, right_bottom) in product(near_left, near_right):
            span = right - left
            if not (.29 * width <= span <= .83 * width and .32 * width <= (left + right) / 2 <= .68 * width
                    and left >= 4 and right < width - 4
                    and min(left_bottom, bottom) - max(left_top, top) >= .61 * panel_height
                    and min(right_bottom, bottom) - max(right_top, top) >= .61 * panel_height):
                continue
            panel = left, top, right, bottom
            close = [box for cross in crosses if (box := _rim_close(cross, panel)) is not None]
            if len(close) == 1 and _rim_evidence(luma, panel):
                matches.add((panel, close[0]))
    if len(matches) != 1:
        return None
    panel, close = next(iter(matches))
    scale_x, scale_y = source_width / width, source_height / height

    def project(box):
        left, top, right, bottom = box
        return (math.floor(left * scale_x), math.floor(top * scale_y),
                math.ceil(right * scale_x), math.ceil(bottom * scale_y))

    return project(panel), project(close)


def _detached_close(gray, panel, *, small_frame=None):
    height, width = gray.shape
    left, top, right, bottom = panel
    small_input = small_frame is not None
    bright = np.uint8(gray >= (75 if small_input else 88))
    bright[top:bottom, left:right] = 0
    kernel = 7 if small_input else 3
    contours, _ = cv2.findContours(cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8)),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = []
    verify = _centered_cross if small_input else _has_close_mark
    mark_frame = small_frame if small_input else gray
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (.04 * width <= min(w, h) <= max(w, h) <= .18 * width and .65 <= w / h <= 1.5):
            continue
        if small_input and not (.055 * width <= min(w, h) <= max(w, h) <= .125 * width):
            continue
        cx, cy = x + w / 2, y + h / 2
        dx, dy = max(left - cx, cx - right, 0), max(top - cy, cy - bottom, 0)
        near_panel_side = (dx == 0 and 0 < dy <= .07 * height) or (dy == 0 and 0 < dx <= .09 * width)
        if not near_panel_side:
            continue
        x0, y0, x1, y1 = max(0, x - 2), max(0, y - 2), min(width, x + w + 2), min(height, y + h + 2)
        if verify(mark_frame[y0:y1, x0:x1]):
            marks.append((dx + dy, (x0, y0, x1, y1)))
    if small_input and len(marks) != 1:
        return None
    return min(marks)[1] if marks else None


def _has_light_header_close(gray, panel):
    left, top, right, bottom = panel
    width = right - left
    depth = min(round(.14 * width), bottom - top)
    reach = round(.15 * width)
    matches = 0
    for start in (left, right - reach):
        crop = gray[top:top + depth, start:start + reach]
        if crop.size == 0:
            continue
        ink = np.uint8(np.median(crop) - crop.astype(np.float32) >= 17)
        _, _, components, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        for x, y, w, h, area in components[1:]:
            if not (max(4, .012 * width) <= min(w, h) <= max(w, h) <= .066 * width
                    and .72 <= w / h <= 1.38 and .15 <= area / (w * h) <= .74):
                continue
            glyph = crop[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2]
            if _centered_cross(cv2.normalize(glyph, None, 0, 255, cv2.NORM_MINMAX)):
                matches += 1
    return matches == 1


def _has_artwork_dismissal(gray, panel):
    """Give a verified close mark precedence over a matching artwork row."""
    return (_detached_close(gray, panel) is not None or _has_light_header_close(gray, panel)
            or any(_rim_close(cross, panel) is not None for cross in _bright_crosses(gray)))


def _light_panel_rim(gray, bounds, *, wide, attached):
    """Require matching boundaries against a darker surface outside the panel."""
    left, top, right, bottom = bounds
    width = gray.shape[1]
    pad = max(2, round(.014 * width))
    trim_x, trim_y = round(.075 * (right - left)), round(.09 * (bottom - top))
    vertical = slice(top + trim_y, bottom - trim_y)
    horizontal = slice(left + trim_x, right - trim_x)
    rims = [(gray[top + 1:top + pad + 1, horizontal], gray[max(0, top - pad - 1):top - 1, horizontal], 0)]
    if not attached:
        rims.append((gray[bottom - pad - 1:bottom - 1, horizontal], gray[bottom + 1:bottom + pad + 1, horizontal], 0))
    if not wide:
        rims.extend(((gray[vertical, left + 1:left + pad + 1], gray[vertical, left - pad - 1:left - 1], 1),
                     (gray[vertical, right - pad - 1:right - 1], gray[vertical, right + 1:right + pad + 1], 1)))
    for inner, outer, axis in rims:
        if min(inner.shape) == 0 or min(outer.shape) == 0:
            return False
        inside, outside = np.median(inner, axis=axis), np.median(outer, axis=axis)
        support = (inside >= 211) & (outside <= 181) & (inside - outside >= 42)
        # Dark artwork and text can sit on a scrim. Its lighter background
        # should still form a coherent band along the edge.
        shade, upper = np.percentile(outside, (50, 90))
        if support.mean() < .86 or upper - shade > 48:
            return False
    return True


def _rounded_sheet_top(bright, top):
    width = bright.shape[1]
    shoulder = max(3, round(.033 * width))
    start, stop = top + round(.065 * width), top + round(.095 * width)
    cap = bright[top:top + max(2, round(.013 * width))]
    body = bright[start:stop]
    return bool(body.size and all(cap[:, edge].mean() <= .30 and body[:, edge].mean() >= .88
                                  for edge in (slice(None, shoulder), slice(-shoulder, None))))


def _paired_sheet_corners(bright, top, reach):
    """Follow the bright side edges down from two rounded top corners."""
    width = bright.shape[1]
    cap = bright[top:top + reach]
    if cap.shape[0] != reach:
        return False
    first_gaps = []
    for profile in (cap[:, :reach], cap[:, -reach:][:, ::-1]):
        gap = np.where(profile.any(axis=1), profile.argmax(axis=1), reach)
        if (not max(2, round(.006 * width)) <= gap[0] < reach or gap[-2:].max() > 1
                or np.count_nonzero(np.diff(gap) < 0) < 2 or np.any(np.diff(gap) > 2)):
            return False
        first_gaps.append(int(gap[0]))
    return abs(first_gaps[0] - first_gaps[1]) <= max(3, round(.024 * width))


def _attached_light_sheet(bright, gray, contour, bounds):
    """Find rounded, full-width surfaces that can start close to the bottom."""
    height, width = gray.shape
    left, top, right, bottom = bounds
    if (left > .012 * width or right < .988 * width or bottom < .985 * height
            or not .24 * height <= top <= .889 * height
            or cv2.contourArea(contour) < .91 * (right - left) * (bottom - top)):
        return None

    inset = round(.105 * width)
    inner = slice(inset, width - inset)
    edge = max(4, round(.078 * width))
    search = max(3, round(.022 * width))
    leading = np.flatnonzero(bright[top:top + search, inner].mean(axis=1) >= .94)
    if not leading.size:
        return None
    top += int(leading[0])
    surface = bright[top:bottom, inner]
    if surface.mean() < .80 or surface[-edge:].mean() < .87:
        return None

    # Follow both corners inwards from the viewport edge. A mere horizontal
    # transition from a hero image to a light details section has no such arc.
    if not _paired_sheet_corners(bright, top, edge):
        return None

    pad = max(2, round(.014 * width))
    before = np.median(gray[top - pad - 1:top - 1, inner], axis=0)
    after = np.median(gray[top + 1:top + pad + 1, inner], axis=0)
    if ((after >= 216) & (before <= 202) & (after - before >= 42)).mean() < .90:
        return None
    # An occluded strip remains visible above the panel. This keeps ordinary
    # light cards over a bright page out even when their corners are rounded.
    backdrop = gray[top - round(.105 * height):top - pad, inner]
    if (backdrop <= 202).mean() < .88:
        return None
    return 0, top, width, height


def _dark_cross_core(chip, component):
    paper = float(np.median(chip[~component]))
    contrast = paper - float(chip.min())
    if paper < 209 or contrast < 45:
        return False
    dark = np.uint8(component & (chip <= paper - .45 * contrast))
    x, y, width, height = cv2.boundingRect(dark)
    return min(width, height) >= 6 and _cross_arms(dark[y:y + height, x:x + width])


def _sheet_header_control(gray, top, *, right_only=False, extra_depth=0, corroborate_core=False):
    """Recognize an isolated close X or the left down chevron of a preview rail."""
    width = gray.shape[1]
    reach = round(.17 * width)
    depth = round(.113 * width) + extra_depth
    for left_side, start in ((True, 0), (False, width - reach)):
        if left_side and right_only:
            continue
        chip = gray[top:top + depth, start:start + reach]
        ink = np.uint8(np.median(chip) - chip.astype(np.float32) >= 17)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if (not .016 * width <= w <= .065 * width or not .010 * width <= h <= .058 * width
                    or not .17 <= area / (w * h) <= .78
                    or min(x, y, reach - x - w, depth - y - h) < 2):
                continue
            # A letter surrounded by caption text is not an independent glyph.
            side = max(2, round(.46 * w))
            neighbors = (ink[y:y + h, max(0, x - side):x], ink[y:y + h, x + w:x + w + side])
            if any(segment.mean() > .14 for segment in neighbors):
                continue
            if not left_side and .72 <= w / h <= 1.38:
                glyph = chip[y - 2:y + h + 2, x - 2:x + w + 2]
                isolated = labels[y - 2:y + h + 2, x - 2:x + w + 2] == index
                if (_centered_cross(cv2.normalize(glyph, None, 0, 255, cv2.NORM_MINMAX))
                        or (corroborate_core and _dark_cross_core(glyph, isolated))):
                    return start + x, top + y, start + x + w, top + y + h
            if left_side and x + w / 2 <= .090 * width and 1.25 <= w / h <= 3.1:
                glyph = labels[y:y + h, x:x + w] == index
                third = int(w) // 3
                portions = [np.nonzero(part)[0] for part in (glyph[:, :third], glyph[:, third:-third], glyph[:, -third:])]
                if all(len(part) for part in portions):
                    start_y, tip_y, end_y = map(np.median, portions)
                    if tip_y - max(start_y, end_y) >= .19 * h and abs(start_y - end_y) <= .33 * h:
                        return x, top + y, x + w, top + y + h
    return None


def _centered_sheet_title(gray, mark):
    width = gray.shape[1]
    y = (mark[1] + mark[3]) / 2
    left, right = round(.28 * width), round(.72 * width)
    start = round(y - .035 * width)
    crop = gray[start:round(y + .035 * width), left:right]
    ink = np.uint8(crop <= 163)
    x, t, w, h = cv2.boundingRect(ink)
    return (.077 * width <= w <= .40 * width and .016 * width <= h <= .062 * width
            and 0 < t < t + h < crop.shape[0] and .07 <= ink[t:t + h, x:x + w].mean() <= .71
            and abs(left + x + w / 2 - width / 2) <= .049 * width
            and abs(start + t + h / 2 - y) <= .015 * width)


def _header_marked_sheet(bright, gray, *, early_header=True):
    """Confirm a modal header even if its content below is visually fragmented."""
    height, width = gray.shape
    pad = max(2, round(.014 * width))
    side = round(.085 * width)
    middle = slice(side, width - side)
    fill = bright[:, middle].mean(axis=1)
    depth = round(.114 * width)
    backdrop_start = round(.05 * height)
    earliest = min(.22 * height, backdrop_start + .15 * width) if early_header else .22 * height
    matches = []
    for top in np.flatnonzero((fill[1:] >= .93) & (fill[:-1] <= .43)) + 1:
        if not earliest <= top <= .87 * height or top + depth >= height:
            continue
        if fill[top:top + pad + 1].mean() < .94 or fill[top:top + depth].mean() < .79:
            continue
        before = np.median(gray[top - pad - 1:top - 1, middle], axis=0)
        after = np.median(gray[top + 1:top + pad + 1, middle], axis=0)
        if ((after >= 216) & (before <= 195) & (after - before >= 44)).mean() < .90:
            continue
        upper = gray[backdrop_start:top - pad]
        flanks = (upper[:, 3:side], upper[:, -side:-3])
        if any((flank <= 174).mean() < .96 for flank in flanks):
            continue
        high_header = top < .22 * height
        panel = (0, int(top), width, height)
        if _sheet_header_control(gray, int(top), right_only=high_header, corroborate_core=True):
            matches.append((panel, None))
        elif (high_header and height >= 1.52 * width and top >= .12 * height
              and _paired_sheet_corners(bright, int(top), round(.068 * width))
              and (mark := _sheet_header_control(gray, int(top))) is not None
              and mark[2] < .13 * width and _centered_sheet_title(gray, mark)):
            matches.append((panel, mark))
    return matches[0] if len(matches) == 1 else None


def _near_top_rounded_sheet(bright, gray):
    """Recognize a nearly full-height sheet below a short visible backdrop."""
    height, width = gray.shape
    inset = round(.093 * width)
    center = slice(inset, width - inset)
    face = bright[:, center].mean(axis=1)
    reach = max(4, round(.066 * width))
    pad = max(2, round(.019 * width))
    possible = []
    for y in np.flatnonzero((face[1:] >= .92) & (face[:-1] <= .55)) + 1:
        if not .030 * height <= y <= .15 * height or y <= 2 * pad:
            continue
        if (face[y:y + reach].mean() < .94 or face[y:].mean() < .77 or
                (gray[y - 2 * pad:y - 1, center] <= 158).mean() < .92 or
                not _paired_sheet_corners(bright, int(y), reach)):
            continue
        # A separate, measured X is required on this short backdrop.
        if _sheet_header_control(gray, int(y), right_only=True, extra_depth=2):
            possible.append((0, int(y), width, height))
    return possible[0] if len(possible) == 1 else None


def _preview_sheet_from_side_edges(bright, gray):
    """Find a near-full-height preview with photo content between its light sides."""
    height, width = gray.shape
    if height < 1.35 * width:
        return None
    rail_start, rail_end = round(.034 * width), round(.108 * width)
    rails = (slice(rail_start, rail_end), slice(width - rail_end, width - rail_start))
    rail_fill = np.minimum(*(bright[:, rail].mean(axis=1) for rail in rails))
    edge = max(4, round(.078 * width))
    upper = max(round(.047 * height), edge)
    lower = round(.25 * height)
    tail = slice(round(.93 * height), round(.987 * height))
    # Inset footer buttons can cover a wide rail without crossing the sheet edge.
    tip = max(3, round(.022 * width))
    if min(bright[tail, :tip].mean(), bright[tail, -tip:].mean()) < .88:
        return None
    matches = []
    for top in np.flatnonzero((rail_fill[1:] >= .90) & (rail_fill[:-1] <= .43)) + 1:
        if not upper <= top <= lower or not _paired_sheet_corners(bright, top, edge):
            continue
        starts = slice(top + 1, top + max(3, round(.023 * width)))
        above = slice(top - max(3, round(.021 * width)), top - 1)
        if any(bright[starts, rail].mean() < .90 or (gray[above, rail] <= 190).mean() < .90 for rail in rails):
            continue
        # Check the visible band of the original page, below status-bar icons.
        pad = max(3, round(.025 * width))
        background = gray[top - pad - 1:top - 1, rail_end:width - rail_end]
        low, middle, high = np.percentile(background, (15, 50, 93))
        if not (23 <= middle <= 166 and high <= 196 and high - low >= 13):
            continue
        leading = bright[top:top + edge, rail_end:width - rail_end]
        if leading.mean() >= .83 or not _sheet_header_control(gray, int(top), right_only=True):
            continue
        matches.append((0, int(top), width, height))
    return matches[0] if len(matches) == 1 else None


def _light_overlay(frame, gray):
    """Find an opaque light surface with a continuous dimmed enclosure."""
    height, width = gray.shape
    blue, green, red = cv2.split(frame)
    high = cv2.max(cv2.max(blue, green), red)
    low = cv2.min(cv2.min(blue, green), red)
    bright = np.uint8((gray >= 218) & (cv2.subtract(high, low) <= 32))
    mask = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    matches = []
    for contour in contours:
        left, top, span, length = cv2.boundingRect(contour)
        right, bottom = left + span, top + length
        late_sheet = _attached_light_sheet(bright, gray, contour, (left, top, right, bottom))
        if late_sheet is not None:
            matches.append(("bottom_sheet", late_sheet))
            continue
        if not (.48 * width <= span and .12 * height <= length <= .78 * height
                and .085 * height <= top <= .76 * height
                and abs(left + right - width) <= .08 * width
                and cv2.contourArea(contour) >= .89 * span * length):
            continue
        wide = left <= .015 * width and right >= .985 * width
        attached = wide and bottom >= .985 * height
        if not wide and (left < .028 * width or right > .972 * width):
            continue
        if not attached and bottom > .945 * height:
            continue
        inset = round(.065 * span)
        surface = bright[top:bottom, left + inset:right - inset].mean(axis=1)
        search = max(3, round(.05 * width))
        leading = np.flatnonzero(surface[:search] >= .82)
        trailing = np.flatnonzero(surface[-search:] >= .82)
        if not leading.size or (not attached and not trailing.size) or surface.mean() < .72:
            continue
        top += int(leading[0])
        if not attached:
            bottom -= len(surface[-search:]) - int(trailing[-1]) - 1
        bounds = (0 if wide else left, top, width if wide else right, height if attached else bottom)
        if not _light_panel_rim(gray, bounds, wide=wide, attached=attached):
            continue
        if wide:
            canopy = gray[max(0, top - round(.13 * height)):top - 2]
            rounded = attached and length >= .23 * height and _rounded_sheet_top(bright, top)
            if (canopy <= 188).mean() < .70 or not (rounded or _has_light_header_close(gray, bounds)):
                continue
        matches.append(("bottom_sheet" if attached else "dialog", bounds))
    if not matches and (header := _header_marked_sheet(bright, gray)) is not None:
        sheet, close = header
        return "bottom_sheet", sheet, close
    if not matches and (sheet := _near_top_rounded_sheet(bright, gray)) is not None:
        return "bottom_sheet", sheet, None
    if not matches and (sheet := _preview_sheet_from_side_edges(bright, gray)) is not None:
        return "bottom_sheet", sheet, None
    return (*matches[0], None) if len(matches) == 1 else None


def _working_frame(frame):
    height, width = frame.shape[:2]
    interpolation = cv2.INTER_AREA
    if min(height, width) < 160:
        if not 128 <= width < height:
            return None
        # Keep the panel and separator kernels at comparable working sizes
        # on small portrait inputs. Projection still uses the original frame.
        resized_width = width * 2
        interpolation = cv2.INTER_CUBIC
    else:
        resized_width = min(width, 320)
    resized_height = round(height * resized_width / width)
    if resized_height < 64:
        return None
    if (resized_width, resized_height) == (width, height):
        return frame
    return cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)


def _working_view(frame):
    """Return the same overview for color and lightness measurements."""
    small = _working_frame(frame)
    return small, None if small is None else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _comment_sheet_surfaces(gray):
    """Find rounded light sheets with a physically visible center grip."""
    height, width = gray.shape
    middle = width // 2
    trim = round(.14 * width)
    center = gray[:, trim:width - trim]
    white = (center >= 222).mean(axis=1)
    inner = max(3, round(.013 * width))
    overhang = round(.022 * width)
    start, stop = round(.031 * width), round(.066 * width)
    half = round(.14 * width)
    for y in np.flatnonzero((white[1:] >= .96) & (white[:-1] < .90)) + 1:
        if not .23 * height <= y <= .78 * height or y + stop >= height:
            continue
        if (center[y - inner:y - 1] <= 179).mean() < .82 or white[y:y + stop].mean() < .93:
            continue
        columns = (gray[y + start:y + stop] >= 222).mean(axis=0) >= .91
        if not columns[middle]:
            continue
        left = int(max(np.flatnonzero(~columns[:middle]), default=-1)) + 1
        right = middle + int(min(np.flatnonzero(~columns[middle:]), default=width - middle))
        if right - left < .87 * width or abs((left + right) / 2 - middle) > .035 * width:
            continue
        corners = (gray[y:y + inner, left:left + overhang], gray[y:y + inner, right - overhang:right])
        if any((corner < 205).mean() < .60 for corner in corners):
            continue
        grip = gray[y + max(1, round(.01 * width)):y + round(.072 * width), middle - half:middle + half]
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(grip <= 219), connectivity=8)
        bars = [(x, w) for x, _, w, h, area in stats[1:]
                if .048 * width <= w <= .14 * width
                and .004 * width <= h <= .024 * width and area >= .48 * w * h
                and abs(x + w / 2 - half) <= .024 * width]
        if len(bars) != 1 or (grip >= 224).mean() < .89:
            continue

        # A detached bottom border can end above a phone bezel or screen edge.
        lower = round(.89 * height)
        levels = np.median(center[lower - inner:], axis=1)
        bottom = next((lower + i - inner for i in range(inner, len(levels) - 1)
                       if levels[i:i + 2].max() <= 228 and np.median(levels[i - inner:i]) >= 248),
                      height)
        if white[y + stop:bottom].mean() >= .90 and (bottom != height or white[-inner:].mean() >= .91):
            yield left, int(y), right, int(bottom)


def _comment_header_end(gray, panel):
    left, top, right, _ = panel
    span = right - left
    trim = round(.056 * span)
    strip = gray[:, left + trim:right - trim]
    start, stop = round(top + .115 * span), round(top + .208 * span)
    matched = []
    for y in range(start, stop):
        white = np.median(np.concatenate((strip[y - 3:y - 1], strip[y + 2:y + 4])), axis=0)
        line = strip[y]
        if ((white - line >= 7) & (line >= 180) & (white >= 247)).mean() >= .86:
            matched.append(y)
    if matched and matched[-1] - matched[0] + 1 == len(matched):
        return matched[-1] + 1
    # Without a divider, inspect only the narrow band where header controls belong.
    return round(top + .135 * span)


def account_comment_field(gray, panel):
    """Measure the rounded, light-gray account composer independently of its text."""
    left, top, right, bottom = panel
    span = right - left
    first = round(top + .76 * (bottom - top))
    crop = gray[first:bottom, left:right]
    mask = np.uint8((crop >= 214) & (crop <= 248))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if not (.55 * span <= w <= .87 * span and .073 * span <= h <= .16 * span
                and .065 * span <= x <= .25 * span and x + w <= .96 * span
                and area >= .71 * w * h and 0 <= bottom - (first + y + h) <= .09 * span):
            continue
        component = labels[y:y + h, x:x + w] == index
        side, rim = max(1, round(.19 * h)), max(1, round(.12 * h))
        corners = (component[:rim, :side], component[:rim, -side:],
                   component[-rim:, :side], component[-rim:, -side:])
        if any(corner.mean() >= .65 for corner in corners):
            continue
        if min(component[h // 2, :side].mean(), component[h // 2, -side:].mean()) < .70:
            continue
        boxes.append((left + x, first + y, left + x + w, first + y + h))
    return boxes[0] if len(boxes) == 1 else None


def detect_comment_sheet_from_heading(frame, texts):
    """Confirm a physical comment sheet from foreground text and surface evidence."""
    height, width = frame.shape[:2]
    plain = []
    headings = []
    composer = []
    for item in texts:
        if item["score"] < .95:
            continue
        polygon = np.asarray(item["polygon"])
        x, y = polygon.mean(axis=0)
        if (is_comment_sheet_heading(item["text"])
                and .36 * width <= x <= .64 * width and .25 * height <= y <= .92 * height
                and .025 * width <= np.ptp(polygon[:, 1]) <= .105 * width):
            headings.append(y)
            if _folded(item["text"]) in COMMENT_SHEET_TITLES:
                plain.append(y)
        if y >= .81 * height and is_account_comment_placeholder(item["text"]):
            composer.append(polygon)
    if not (headings or composer) or (work := _working_frame(frame)) is None:
        return None
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    work_h, work_w = gray.shape
    if len(plain) == 1:
        sheet = _rounded_light_sheet(gray, heading_y=plain[0] * work_h / height)
        if sheet is not None:
            return BlockingOverlay("bottom_sheet", (0, math.floor(sheet[1] * height / work_h), width, height))
    if not 1.55 * width <= height <= 2.65 * width:
        return None
    panels = []
    for panel in _comment_sheet_surfaces(gray):
        _, top, _, bottom = panel
        matching = [y for y in headings if .040 <= (y * work_h / height - top) / (bottom - top) <= .18]
        if len(matching) == 1:
            panels.append(panel)
        elif not matching and composer and (field := account_comment_field(gray, panel)):
            x0, y0, x1, y1 = field
            scale = np.asarray([work_w / width, work_h / height])
            contained = 0
            for polygon in composer:
                minimum, maximum = polygon.min(axis=0) * scale, polygon.max(axis=0) * scale
                contained += (x0 < minimum[0] < maximum[0] < x1
                              and y0 < minimum[1] < maximum[1] < y1)
            if contained == 1:
                panels.append(panel)
    if len(panels) != 1:
        return None
    panel = panels[0]
    left, top, right, bottom = panel
    header = _comment_header_end(gray, panel)
    return BlockingOverlay("bottom_sheet", (math.floor(left * width / work_w), math.floor(top * height / work_h),
                                           math.ceil(right * width / work_w), math.ceil(bottom * height / work_h)),
                           header_bottom=math.ceil(header * height / work_h))


def detect_dark_dialog_from_dismissal(frame, texts):
    """Pair an explicit detached dismissal with a dark raised panel and primary button."""
    height, width = frame.shape[:2]
    dismissals = []
    for item in texts:
        if item["score"] < .94 or _folded(item["text"]) not in {"dismiss", "close", "关闭", "取消"}:
            continue
        polygon = np.asarray(item["polygon"])
        x, y = polygon.mean(axis=0)
        if (.42 * width <= x <= .58 * width and .28 * height <= y <= .78 * height
                and .05 * width <= np.ptp(polygon[:, 0]) <= .26 * width):
            dismissals.append(polygon)
    if len(dismissals) != 1 or (work := _working_frame(frame)) is None:
        return None
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    work_h, work_w = gray.shape
    sx, sy = work_w / width, work_h / height
    dismiss_top = dismissals[0][:, 1].min() * sy
    kernel = np.ones((5, 5), np.uint8)
    for threshold in (64, 48, 36, 30, 25):
        mask = cv2.morphologyEx(np.uint8(gray >= threshold), cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        panels = []
        for contour in contours:
            left, top, span, length = cv2.boundingRect(contour)
            right, bottom = left + span, top + length
            if not (.58 * work_w <= span <= .94 * work_w and .13 * work_h <= length <= .52 * work_h
                    and abs(left + right - work_w) <= .06 * work_w and .10 * work_h <= top
                    and .025 * work_w <= dismiss_top - bottom <= .16 * work_w
                    and cv2.contourArea(contour) >= .89 * span * length):
                continue
            panel = gray[top:bottom, left:right]
            ink = float(np.median(panel))
            if not 26 <= ink <= 106 or (np.abs(panel.astype(np.float32) - ink) <= 7).mean() < .53:
                continue
            trim_x, trim_y = round(.10 * span), round(.12 * length)
            rims = (gray[top - 4:top - 1, left + trim_x:right - trim_x],
                    gray[top + trim_y:bottom - trim_y, left - 4:left - 1],
                    gray[top + trim_y:bottom - trim_y, right + 1:right + 4])
            if any(rim.size == 0 or (rim <= ink - 13).mean() < .90 for rim in rims):
                continue
            low_part = panel[round(.43 * length):]
            _, _, components, _ = cv2.connectedComponentsWithStats(np.uint8(low_part >= 170), connectivity=8)
            if not any(.54 * span <= w <= .92 * span and .09 * length <= h <= .29 * length
                       and abs(2 * x + w - span) <= .09 * span and area >= .74 * w * h
                       for x, _, w, h, area in components[1:]):
                continue
            panels.append((math.floor(left / sx), math.floor(top / sy), math.ceil(right / sx), math.ceil(bottom / sy)))
        if len(panels) != 1:
            continue
        glyph = dismissals[0]
        pad = max(2, round(.008 * width))
        x0, y0 = np.floor(glyph.min(axis=0)).astype(int)
        x1, y1 = np.ceil(glyph.max(axis=0)).astype(int)
        close = tuple(map(int, (max(0, x0 - pad), y0 - pad, min(width, x1 + pad), min(height, y1 + pad))))
        return BlockingOverlay("dialog", panels[0], close)
    return None


def foreground_frame_digest(frame):
    return sha256(np.ascontiguousarray(frame)).hexdigest()


def overlay_for_observation(frame, observation):
    """Use text-confirmed bounds only for the exact image they came from."""
    checksum = observation.get("active_layer_frame_sha256")
    layer = observation.get("active_layer") or {}
    if not checksum or layer.get("kind") not in {"dialog", "bottom_sheet"} or checksum != foreground_frame_digest(frame):
        return detect_blocking_overlay(frame)
    height, width = frame.shape[:2]
    regions = layer.get("regions_fraction")
    if not isinstance(regions, list) or not 1 <= len(regions) <= 2:
        return detect_blocking_overlay(frame)
    boxes = []
    for region in regions:
        if (not isinstance(region, list) or len(region) != 4
                or not all(type(value) in (int, float) and 0 <= value <= 1 for value in region)):
            return detect_blocking_overlay(frame)
        x0, y0, x1, y1 = (round(value * (extent - 1)) if value < 1 or index < 2 else extent
                          for index, (value, extent) in enumerate(zip(region, (width, height, width, height))))
        if x0 >= x1 or y0 >= y1:
            return detect_blocking_overlay(frame)
        boxes.append((x0, y0, x1, y1))
    header = layer.get("header_bottom_fraction")
    return BlockingOverlay(layer["kind"], boxes[0], boxes[1] if len(boxes) == 2 else None,
                           round(header * (height - 1)) if isinstance(header, (int, float)) and 0 <= header <= 1 else None)


def detect_blocking_overlay(frame, *, working_view=None, camera=False, label_index=None):
    """Find a blocking sheet or dialog from its visible enclosure."""
    small, gray = _working_view(frame) if working_view is None else working_view
    if small is None:
        return None
    overlay = _detect_blocking_overlay(frame, small, gray, camera=camera, label_index=label_index)
    if overlay is None or overlay.kind != "bottom_sheet" or overlay.panel[3] < .95 * frame.shape[0]:
        return overlay
    from screen_system_navigation import system_navigation_top

    if (navigation := system_navigation_top(gray)) is None:
        return overlay
    left, top, right, bottom = overlay.panel
    edge = min(bottom, round(navigation * frame.shape[0] / gray.shape[0]))
    if edge - top >= .18 * frame.shape[0]:
        return replace(overlay, panel=(left, top, right, edge))
    return overlay


def _detect_blocking_overlay(frame, small, gray, *, camera, label_index):
    height, width = frame.shape[:2]
    resized_height, resized_width = small.shape[:2]
    # The composer can sit over another sheet and owns clicks before it.
    if (composer := nested_composer_bounds(frame, small, gray)) is not None:
        return BlockingOverlay("composer", composer)

    def to_frame(box):
        left, top, right, bottom = box
        ratio_x, ratio_y = width / resized_width, height / resized_height
        return (math.floor(left * ratio_x), math.floor(top * ratio_y),
                min(width, math.ceil(right * ratio_x)), min(height, math.ceil(bottom * ratio_y)))

    from screen_voice_recording import voice_recording_bounds
    from screen_voice_stop import voice_stop_surface

    if (gesture := voice_recording_bounds(gray, source=frame)) is not None:
        return BlockingOverlay("voice_recording", to_frame(gesture))
    if (recorder := voice_stop_surface(frame, small, gray)) is not None:
        panel, stop = recorder
        return BlockingOverlay("voice_stop_panel", panel, recording_stop=stop)
    sheet = _bottom_sheet(gray) or _rounded_light_sheet(gray)
    if sheet is not None:
        return BlockingOverlay("bottom_sheet", to_frame(sheet))
    panel = _center_panel(small, gray)
    if panel is not None:
        panel = _extend_dialog_header(small, gray, panel)
        close_button = _detached_close(gray, panel, small_frame=small if min(height, width) < 160 else None)
        if close_button is None:
            from screen_painted_layer import painted_layer

            if (painted := painted_layer(small, gray)) is not None:
                observed, dismiss = painted
                if max(abs(a - b) for a, b in zip(panel, observed, strict=True)) <= .025 * resized_width:
                    close_button = dismiss
        return BlockingOverlay("dialog", to_frame(panel), None if close_button is None else to_frame(close_button))
    if (light := _light_overlay(small, gray)) is not None:
        kind, panel, close = light
        if kind == "dialog":
            panel = _extend_dialog_header(small, gray, panel)
        if close is not None:
            return BlockingOverlay(kind, to_frame(panel), to_frame(close), close_style="sheet_down")
        return BlockingOverlay(kind, to_frame(panel))
    if (translucent := _translucent_dialog(gray)) is not None:
        panel, close_button = translucent
        return BlockingOverlay("dialog", to_frame(panel), to_frame(close_button))
    from screen_ime_sheet import voice_ime_bounds
    from screen_side_drawer import side_drawer_bounds
    from screen_wide_panel import wide_panel_bounds

    if (ime := voice_ime_bounds(small, gray)) is not None:
        return BlockingOverlay("ime_sheet", to_frame(ime))
    if (drawer := side_drawer_bounds(frame)) is not None:
        return BlockingOverlay("side_drawer", *drawer)
    if (wide := wide_panel_bounds(frame)) is not None:
        return BlockingOverlay("wide_panel", *wide)
    from screen_painted_layer import painted_layer

    if (painted := painted_layer(small, gray)) is not None:
        panel, close = painted
        return BlockingOverlay("dialog", to_frame(panel), to_frame(close))
    if camera and 1.55 * width <= height <= 2.73 * width:
        # The rectified glass may retain a narrow reflective rim on both sides.
        # Keep the full-width transition test and sample the inner rails for dimming.
        if (sheet := _bottom_sheet(gray, rail_inset=max(1, round(.022 * resized_width)))) is not None:
            return BlockingOverlay("bottom_sheet", to_frame(sheet))
    from screen_contrast_confirmation import detect_contrast_confirmation

    if (confirmation := detect_contrast_confirmation(frame, working_view=(small, gray))) is not None:
        return confirmation
    if label_index is not None:
        from screen_social_comment_surface import social_comment_surface

        comment = social_comment_surface(frame, label_index, working_view=(small, gray))
        if comment is not None:
            return comment
    from screen_share_surface import share_sheet_surface

    if (shared := share_sheet_surface(small, gray)) is not None:
        panel, close, icons = shared
        return BlockingOverlay("bottom_sheet", to_frame(panel), None if close is None else to_frame(close),
                               close_style=None if close is None else "literal",
                               share_icons=tuple(map(to_frame, icons)))
    from screen_legacy_share_surface import legacy_share_sheet_surface

    return legacy_share_sheet_surface(frame, label_index, working_view=(small, gray))
