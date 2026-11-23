"""Locate dropdown fields and foreground choices from independent visual evidence."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import re
from statistics import median
import unicodedata

import cv2
import numpy as np

from inventory_compact_selector import _outside_label, find_compact_named_selector
from inventory_compact_selector_text import read_compact_selector_value
from inventory_dark_selector_fields import dark_neutral_enclosures
from inventory_file_tab_titles import _title_views
from inventory_grouped_selector_text import read_grouped_selector_text
from inventory_grouped_selectors import grouped_selector_rows
from inventory_list_rows import _Box, _labels
from inventory_merged_selector_readback import recapture_merged_selectors
from inventory_selector_pair_text import recapture_selector_pair_text
from inventory_selector_rail_fields import selector_rail_fields
from inventory_selector_rail_readback import read_rail_selectors
from inventory_selector_side_titles import _stepper_arrow, recover_side_titles
from inventory_selector_text import RESTART_NOTE, recapture_dimension_text, restore_select_text
from inventory_selector_text_completion import complete_selector_text
from inventory_selector_value_peers import mark_unlisted_value_peers


_PLACEHOLDER = re.compile(r"^(?:[-—\s]+|(?:please )?(?:select|choose)(?:\s+an?)?\s+(?:option|value))$", re.I)
_COMMAND_LABEL = re.compile(
    r"^((?:edit|manage|configure|customize|add|browse)(?:\s+[^.…]+)?|"
    r"(?:编辑|管理|配置|自定义|添加|浏览)[^.…]*)(?:\.{2,3}|…)$", re.I,
)
_EDGE_CLOSE = np.ones((3, 3), np.uint8)
_UNPREPARED = object()


@dataclass(frozen=True)
class _Line:
    box: _Box
    text: str
    score: float = 100


@dataclass(frozen=True)
class _Arrow:
    box: _Box
    up: bool


@dataclass(frozen=True)
class _ValueSelectEvidence:
    gray: np.ndarray
    lines: tuple[_Line, ...]
    rectangles: tuple[_Box, ...]
    arrows: dict[_Box, _Arrow]
    closed: dict[_Box, _Box]
    compact: dict[_Box, tuple[_Line, _Line]]
    strict_closed: dict[_Box, _Box]


@dataclass(frozen=True)
class _Attachment:
    box: _Box
    lines: tuple[_Line, ...]
    below: bool


@dataclass(frozen=True)
class ExpandedValueSelect:
    panel: tuple[int, int, int, int]
    field: tuple[int, int, int, int]
    setting_label: str | None
    actions: tuple[dict, ...]


def _rectangles(gray):
    height, width = gray.shape
    dx = cv2.absdiff(gray[:, 1:], gray[:, :-1])
    dy = cv2.absdiff(gray[1:, :], gray[:-1, :])
    boxes = set()

    def collect(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            left, top, w, h = cv2.boundingRect(contour)
            if (w >= 27 and 11 <= h <= .92 * height and w >= .5 * h
                    and 0 < left < left + w < width and 0 < top < top + h < height
                    and cv2.contourArea(contour) >= .77 * w * h):
                boxes.add(_Box(left, top, left + w, top + h))

    for threshold in (7, 15):
        edges = np.zeros_like(gray)
        edges[:, 1:] = dx >= threshold
        edges[1:, :] |= dy >= threshold
        # Keep narrow gaps between adjacent fields when closing joins their rims.
        collect(edges)
        collect(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _EDGE_CLOSE))

    # A filled panel can stay rectangular when JPEG breaks its edge into pieces.
    sample = gray[::2, ::2]
    bins = np.bincount(sample.ravel(), minlength=256)
    tones = []
    for raw in np.argsort(bins)[::-1]:
        if bins[raw] < .015 * sample.size or len(tones) == 4:
            break
        tone = int(raw)
        if any(abs(tone - previous) < 12 for previous in tones):
            continue
        tones.append(tone)
        collect(cv2.inRange(gray, max(0, tone - 8), min(255, tone + 8)))
    return sorted(boxes, key=lambda box: (-box.width * box.height, box.top, box.left))


def _looks_down(glyph, *, relaxed=False):
    height, width = glyph.shape
    if not (width >= 4 and height >= 3 and 1.15 <= width / height <= 3.45
            and .15 <= glyph.mean() <= .83):
        return False
    # A filled triangle has a broad upper edge and narrows toward the tip.
    widths = glyph.sum(axis=1)
    tip = np.flatnonzero(glyph[-1])
    shoulder = widths[1:] if relaxed and height >= 5 and widths[1] > widths[0] else widths
    if (widths[0] >= .62 * width and widths[-1] <= .51 * width
            and .29 * width <= tip.mean() <= .71 * width and (np.diff(shoulder) <= 1).all()):
        return True
    yy, xx = glyph.nonzero()
    visible = np.bincount(xx, minlength=width)
    if (visible > 0).mean() < .87:
        return False
    center_y = np.bincount(xx, weights=yy, minlength=width) / np.maximum(1, visible)
    start, end = max(1, width // 4), width - max(1, width // 4)
    if center_y[start:end].mean() - max(center_y[:start].mean(), center_y[end:].mean()) < .23 * height:
        return False
    shape = 1 - np.abs(np.linspace(-1, 1, width))
    return bool(np.sqrt(np.mean((center_y / max(1, height - 1) - shape) ** 2)) <= .23)


def _arrow(gray, box, *, relaxed=False, thresholds=(46, 33), left_limit=0):
    h = box.height
    left = max(left_limit, box.left + round(.55 * box.width), box.right - round(1.10 * h))
    right = box.right - max(2, round(.09 * h))
    top = box.top + max(2, round(.16 * h))
    bottom = box.bottom - max(2, round(.16 * h))
    field = gray[top:bottom, left:right].astype(np.int16)
    if min(field.shape, default=0) < 6:
        return None
    baseline = np.median(field, axis=1)
    contrast = np.abs(field - baseline[:, None])
    for threshold in thresholds:
        mask = np.uint8(contrast >= threshold)
        _, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        marks = [(index, x, y, w, height) for index, (x, y, w, height, area) in enumerate(stats[1:], 1)
                 if area >= 4 and w >= 2 and height >= 2
                 and 0 < x < x + w < field.shape[1] and 0 < y < y + height < field.shape[0]]
        if not relaxed and len(marks) != 1:
            continue
        minimum_width, minimum_height = (.145, .075) if relaxed else (.20, .10)
        matches = []
        for identifier, x, y, w, height in marks:
            if not (max(3, minimum_width * h) <= w <= .70 * h and minimum_height * h <= height <= .39 * h
                    and abs(top + y + height / 2 - box.center[1]) <= .17 * h
                    and .11 * h <= box.right - left - x - w / 2 <= .82 * h):
                continue
            glyph = components[y:y + height, x:x + w] == identifier
            arrow = _Box(left + x, top + y, left + x + w, top + y + height)
            if _looks_down(glyph, relaxed=relaxed):
                matches.append(_Arrow(arrow, up=False))
            elif _looks_down(glyph[::-1, :], relaxed=relaxed):
                matches.append(_Arrow(arrow, up=True))
        if len(matches) == 1:
            return matches[0]
    return None


def _ink_strength(gray, line):
    box = line.box
    roi = gray[box.top:box.bottom, box.left:box.right].astype(np.int16)
    return float(np.quantile(np.abs(roi - np.median(roi)), .90))


def _readable_value_field(gray, field, arrow, title, value):
    heading = _ink_strength(gray, title)
    selected = _ink_strength(gray, value)
    if selected < max(39, .48 * heading):
        return False
    if selected >= 74 or heading >= 130:
        return True
    left = max(field.left, field.right - round(.85 * field.height))
    strip = gray[arrow.top:arrow.bottom, left:field.right - 1].astype(np.int16)
    glyph = gray[arrow.top:arrow.bottom, arrow.left:arrow.right].astype(np.int16)
    # Dim type and a dim arrow together are common when a field is disabled.
    if not strip.size or not glyph.size:
        return False
    strength = np.quantile(np.abs(glyph - np.median(strip, axis=1)[:, None]), .75)
    return bool(strength >= 144)


def _same_horizontal_surface(gray, label, field):
    left, right = label.right, field.left
    if right - left <= 1.8 * label.height:
        return True
    y = round(label.center[1])
    half = max(2, round(.31 * label.height))
    strip = gray[y - half:y + half + 1, left + 2:right - 2].astype(np.int16)
    tone = np.median(strip, axis=0)
    return bool((np.abs(np.diff(tone)) <= 9).all() and np.ptp(tone) <= 12)


def _inline_title(gray, field, lines):
    candidates = [line for line in lines
                  if 0 <= field.left - line.box.right <= .77 * gray.shape[1]
                  and abs(line.box.center[1] - field.center[1]) <= .23 * field.height
                  and .34 * field.height <= line.box.height <= 1.18 * field.height]
    for line in sorted(candidates, key=lambda item: item.box.right, reverse=True):
        if _same_horizontal_surface(gray, line.box, field) and RESTART_NOTE.fullmatch(line.text) is None:
            return line
    return None


def _paragraph(text):
    return len(text) > 69 or text.endswith((".", "。", "!", "！", "?", "？"))


def _above_title(field, lines):
    h = field.height
    candidates = [line for line in lines
                  if abs(line.box.left - field.left) <= .42 * h
                  and 0 <= field.top - line.box.bottom <= 4.25 * h
                  and .35 * h <= line.box.height <= .85 * h]
    candidates.sort(key=lambda line: line.box.bottom, reverse=True)
    for line in candidates:
        if _paragraph(line.text):
            continue
        if field.top - line.box.bottom <= 1.30 * h or any(
                neighbour.box.top >= line.box.bottom and _paragraph(neighbour.text) for neighbour in candidates
        ):
            return line
        break
    return None


def _compact_caption_row(title, lines, gap_ratio):
    box = title.box
    peers = [line.box for line in lines if line is not title
             and .68 * box.height <= line.box.height <= 1.44 * box.height
             and abs(line.box.center[1] - box.center[1]) <= .24 * box.height]
    gap = gap_ratio * box.height
    return (any(0 <= box.left - peer.right <= gap for peer in peers)
            and any(0 <= peer.left - box.right <= gap for peer in peers))


def _separated_toolbar_title(gray, field, title, lines):
    first, stop = title.box.bottom + 1, field.top
    if first >= stop or not _compact_caption_row(title, lines, .90):
        return False
    height = field.height
    margin, reach = round(.30 * height), round(3.10 * height)
    left = max(0, field.left - reach), field.left - margin
    right = field.right + margin, min(gray.shape[1], field.right + reach)
    # A field's own outline cannot account for the same rail on both sides.
    support = []
    for start, end in (left, right):
        if end - start < 1.15 * height:
            return False
        stripe = gray[first - 3:stop + 2, start:end].astype(np.int16)
        earlier, middle, later = stripe[2:-3], stripe[3:-2], stripe[4:-1]
        peak = np.minimum(middle - earlier, middle - later) >= 7
        trough = np.minimum(earlier - middle, later - middle) >= 7
        # When resampling fades a thin rule, both toolbar planes can still be visible.
        delta = later + stripe[5:] - stripe[:-5] - stripe[1:-4]
        steady = (np.abs(stripe[:-5] - stripe[1:-4]) <= 8) & (np.abs(later - stripe[5:]) <= 8)
        support.append(np.array([mask.mean(axis=1) for mask in
                                 (peak, trough, steady & (delta >= 16), steady & (delta <= -16))]) >= .84)
    return bool(np.any(support[0] & support[1]))


def _shown_value(gray, field, arrow, lines):
    h = field.height
    names = [line for line in lines
             if field.left - .17 * h <= line.box.left <= field.left + 1.12 * h
             and line.box.right < arrow.left - .13 * h
             and .35 * h <= line.box.height <= 1.20 * h
             and abs(line.box.center[1] - field.center[1]) <= .21 * h]
    if len(names) == 1:
        return True, names[0]
    if names:
        return False, None
    inner = gray[field.top + round(.18 * h):field.bottom - round(.18 * h),
                 field.left + round(.14 * h):arrow.left - max(1, round(.12 * h))].astype(np.int16)
    # A genuinely empty field differs from a missed or clipped OCR value.
    if inner.size and (np.abs(inner - np.median(inner, axis=1)[:, None]) > 14).sum() < 3:
        return True, None
    return False, None


def _boundary_value(gray, field, arrow, lines):
    """Recover a visible value whose OCR bounds cross a thin field inset."""
    h = field.height
    candidates = [line for line in lines
                  if field.left - .42 * h <= line.box.left <= field.left + 1.12 * h
                  and line.box.right <= arrow.left - max(1, round(.04 * h))
                  and .35 * h <= line.box.height <= 1.20 * h
                  and abs(line.box.center[1] - field.center[1]) <= .21 * h]
    if len(candidates) != 1:
        return None
    value = candidates[0]
    left = max(field.left + max(2, round(.10 * h)), value.box.left)
    box = value.box
    body = gray[max(field.top + 2, box.top):min(field.bottom - 2, box.bottom),
                left:box.right].astype(np.int16)
    if body.size:
        foreground = np.count_nonzero(np.abs(body - np.median(body)) >= 39)
        if foreground >= max(4, .05 * body.size):
            return value
    return None


def _attached_panels(field, rectangles, selects, lines):
    h = field.height
    tolerance = max(3, .26 * h)
    aligned_boxes = [box for box in rectangles if box.height >= .24 * h
                     and abs(box.left - field.left) <= tolerance and abs(box.right - field.right) <= tolerance
                     and not any(_overlap(box, control) for control in selects)]
    bands = [box for box in aligned_boxes if box.height >= .55 * h]
    for direction in (-1, 1):
        if direction > 0:
            intervals = sorted((box.top, box.bottom) for box in bands if box.top >= field.bottom - tolerance)
            boundary = field.bottom
        else:
            intervals = sorted((-box.bottom, -box.top) for box in bands if box.bottom <= field.top + tolerance)
            boundary = -field.top
        far = boundary
        for start, stop in intervals:
            if start > far + tolerance:
                break
            far = max(far, stop)
        if far - boundary < 1.50 * h:
            continue
        aligned = [line for line in lines if field.left < line.box.left <= field.left + .64 * h
                   and line.box.right <= field.right and boundary + .17 * h <= direction * line.box.center[1] <= far]
        ys = sorted(line.box.center[1] for line in aligned)
        if any(.52 * h <= next_y - y <= 1.45 * h for y, next_y in zip(ys, ys[1:])):
            # A final clipped row may provide a real rim without a full label.
            ends = [(box.top, box.bottom) if direction > 0 else (-box.bottom, -box.top)
                    for box in aligned_boxes if box.height < .55 * h]
            far = max(far, max((stop for start, stop in ends if abs(start - far) <= tolerance), default=far))
            top, bottom = (boundary, far) if direction > 0 else (-far, -boundary)
            yield _Attachment(_Box(field.left, top, field.right, bottom), tuple(aligned), below=direction > 0)


def _wide_panels(field, rectangles, selects, lines):
    h = field.height
    tolerance = max(3, .27 * h)
    for box in rectangles:
        if (box.height < 2.25 * h or not 1.16 * field.width <= box.width <= 3.85 * field.width
                or min(abs(box.left - field.left), abs(box.right - field.right)) > tolerance
                or any(_overlap(box, control) for control in selects)):
            continue
        below = abs(box.top - field.bottom) <= tolerance
        if not below and abs(box.bottom - field.top) > tolerance:
            continue
        inset = min(1.75 * h, .33 * box.width)
        aligned = tuple(line for line in lines if box.left < line.box.left <= box.left + inset
                        and line.box.right < box.right and box.top < line.box.center[1] < box.bottom)
        if len(aligned) >= 3:
            yield _Attachment(box, aligned, below)


def _vertical_edge(gray, x, top, bottom, search, pitch):
    left, right = max(0, x - search - 1), min(gray.shape[1], x + search + 1)
    if right - left < 3:
        return None
    patch = gray[top:bottom, left:right]
    edges = cv2.absdiff(patch[:, 1:], patch[:, :-1]) >= 7
    core = edges[-max(4, round(.63 * pitch)):]
    continuity = core.mean(axis=0)
    column = int(np.argmax(continuity))
    if continuity[column] < .66:
        return None
    return left + column + 1, edges[:, column]


def _open_field_above(gray, panel, pitch):
    # Floating labels can interrupt the upper rim. Both actual side edges must
    # still run upward from the independently observed panel boundary.
    search = max(3, round(.10 * pitch))
    top = max(0, panel.top - round(1.48 * pitch))
    bottom = panel.top - max(2, round(.16 * pitch))
    if bottom - top < .74 * pitch:
        return None
    left = _vertical_edge(gray, panel.left, top, bottom, search, pitch)
    right = _vertical_edge(gray, panel.right, top, bottom, search, pitch)
    if left is None or right is None or right[0] - left[0] < 1.7 * pitch:
        return None
    common = left[1] & right[1]
    sustained = np.convolve(common.astype(np.uint8), np.ones(max(3, round(.12 * pitch)), dtype=np.uint8), "valid")
    start = np.flatnonzero(sustained >= .67 * max(3, round(.12 * pitch)))
    if not start.size:
        return None
    field = _Box(left[0], top + int(start[0]), right[0], panel.top)
    if not .70 * pitch <= field.height <= 1.46 * pitch:
        return None
    return field


def _interrupted_fields(gray, rectangles, lines, known):
    seen = []
    for seed in rectangles:
        if seed.width < 65 or seed.top < 21:
            continue
        horizon = seed.top + max(135, .34 * gray.shape[0])
        near = sorted((line for line in lines if seed.left < line.box.left <= seed.left + .28 * seed.width
                       and line.box.right < seed.right and seed.top <= line.box.center[1] <= horizon),
                      key=lambda line: line.box.center[1])
        if len(near) < 2:
            continue
        first = near[0]
        pairs = [line for line in near[1:] if abs(line.box.left - first.box.left) <= .43 * first.box.height
                 and .70 <= line.box.height / first.box.height <= 1.42
                 and 1.18 * first.box.height <= line.box.center[1] - first.box.center[1] <= 4.2 * first.box.height]
        if not pairs:
            continue
        pitch = pairs[0].box.center[1] - first.box.center[1]
        if not .28 * pitch <= first.box.center[1] - seed.top <= .82 * pitch:
            continue
        field = _open_field_above(gray, seed, pitch)
        if field is None or any(_overlap(field, previous) for previous in (*known, *seen)):
            continue
        arrow = _arrow(gray, field, relaxed=True)
        if arrow is not None and arrow.up:
            seen.append(field)
            yield field, arrow


def _overlap(first, second):
    x = max(0, min(first.right, second.right) - max(first.left, second.left))
    y = max(0, min(first.bottom, second.bottom) - max(first.top, second.top))
    return x * y >= .80 * min(first.width * first.height, second.width * second.height)


def _merged_value(field, arrow, lines):
    h = field.height
    matches = [line for line in lines if line.box.left <= field.left - 1.40 * h
               and field.left + .49 * h <= line.box.right <= field.right + .16 * h
               and .35 * h <= line.box.height <= 1.24 * h
               and abs(line.box.center[1] - field.center[1]) <= .19 * h]
    if len(matches) != 1:
        return None
    line = matches[0]
    x0 = field.left + max(2, round(.10 * h))
    x1 = min(arrow.left - max(1, round(.12 * h)), line.box.right + 2)
    if x1 - x0 < .47 * h:
        return None
    title_box = _Box(line.box.left, line.box.top, field.left - 2, line.box.bottom)
    # Keep a short value at the observed text height for the ink-strength check.
    value_box = _Box(x0, max(field.top + 2, line.box.top - 1),
                     x1, min(field.bottom - 2, line.box.bottom + 1))
    return line.text, title_box, value_box


def _key(text):
    return "".join(text.casefold().split())


def _option_rows(field, attachment):
    panel = attachment.box
    boundary = panel.top if attachment.below else panel.bottom
    sign = 1 if attachment.below else -1
    h = field.height
    ordered = sorted(attachment.lines, key=lambda line: sign * line.box.center[1])
    first = ordered[0]
    y0 = sign * (first.box.center[1] - boundary)
    if not .21 * h <= y0 <= 1.12 * h:
        return []
    lane = [line for line in ordered if abs(line.box.left - first.box.left) <= .43 * first.box.height
            and .61 <= line.box.height / first.box.height <= 1.47]
    gaps = [sign * (second.box.center[1] - first.box.center[1]) for first, second in zip(lane, lane[1:])]
    fits = [gap for gap in gaps[:4] if .53 * h <= gap <= 1.48 * h]
    if not fits:
        return []
    pitch = median(fits)
    rows = []
    last = first.box.center[1]
    for line in lane:
        y = line.box.center[1]
        if rows and not .76 * pitch <= sign * (y - last) <= 1.22 * pitch:
            break
        top, bottom = round(y - .49 * pitch), round(y + .49 * pitch)
        margin = max(2, .11 * pitch)
        if top < panel.top - margin or bottom > panel.bottom + margin:
            break
        row = _Box(panel.left, max(top, panel.top), panel.right, min(bottom, panel.bottom))
        if not (row.top <= line.box.top <= y <= line.box.bottom <= row.bottom
                and line.box.right < row.right - .11 * h and line.box.height <= .91 * pitch):
            break
        rows.append((line, row))
        last = y
    return sorted(rows, key=lambda pair: pair[1].top)


def _checked(gray, panel, row, line):
    pitch = row.height
    middle = round(row.center[1])
    half = max(3, round(.42 * pitch))
    y0, y1 = middle - half, middle + half
    margins = ((panel.left + 2, line.box.left - 3),
               (max(line.box.right + 3, panel.right - round(1.6 * pitch)), panel.right - 3))
    for left, right in margins:
        roi = gray[y0:y1, left:right].astype(np.int16)
        if min(roi.shape, default=0) < 5:
            continue
        ink = np.uint8(np.abs(roi - np.median(roi)) >= 42)
        _, components, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if not (max(4, .18 * pitch) <= w <= .50 * pitch and .10 * pitch <= h <= .42 * pitch
                    and 1.13 <= w / h <= 2.45 and .13 <= area / (w * h) <= .58
                    and 0 < x < x + w < roi.shape[1] and 0 < y < y + h < roi.shape[0]):
                continue
            glyph = components[y:y + h, x:x + w] == index
            yy, xx = glyph.nonzero()
            count = np.bincount(xx, minlength=w)
            if not count.all():
                continue
            axes = np.bincount(xx, weights=yy, minlength=w) / count
            kink = int(np.argmax(axes))
            if (.16 * w <= kink <= .55 * w and axes[kink] - axes[0] >= .25 * h
                    and axes[kink] - axes[-1] >= .66 * h and axes[0] - axes[-1] >= .17 * h):
                return True
    return False


def _highlighted(frame, rows, index):
    if len(rows) < 3:
        return False
    colors = []
    for _, row in rows:
        x0, x1 = row.left + round(.32 * row.width), row.right - round(.15 * row.width)
        _, cy = row.center
        offset = max(1, round(.15 * row.height))
        pixels = frame[round(cy) - offset:round(cy) + offset + 1, x0:x1]
        colors.append(np.median(pixels.reshape(-1, 3), axis=0))
    peers = np.asarray([value for i, value in enumerate(colors) if i != index])
    base = np.median(peers, axis=0)
    agrees = (np.max(np.abs(peers - base), axis=1) <= 5).sum()
    return agrees >= max(2, len(peers) - 1) and np.max(np.abs(colors[index] - base)) >= 9


def _command_after_divider(gray, rows, index, panel):
    line, row = rows[index]
    if index == 0 or index != len(rows) - 1:
        return None
    label = _COMMAND_LABEL.fullmatch(line.text)
    if label is None:
        return None
    previous, _ = rows[index - 1]
    gap_middle = round((previous.box.center[1] + line.box.center[1]) / 2)
    tolerance = max(2, round(.16 * row.height))
    start = max(previous.box.bottom + 2, gap_middle - tolerance)
    stop = min(line.box.top - 2, gap_middle + tolerance + 1)
    margin = round(.10 * panel.width)
    section = gray[start - 2:stop + 3, panel.left + margin:panel.right - margin].astype(np.int16)
    for offset in range(2, stop - start + 2):
        above = np.median(section[offset - 2:offset], axis=0)
        for thickness in (1, 2):
            stripe = np.median(section[offset:offset + thickness], axis=0)
            below = np.median(section[offset + thickness:offset + thickness + 2], axis=0)
            contrast = np.maximum(np.minimum(above - stripe, below - stripe),
                                  np.minimum(stripe - above, stripe - below))
            if np.mean(contrast >= 3) >= .78:
                return label[1].rstrip() + "…"
    return None


def _expanded_context(gray, field, arrow, lines, *, wider):
    title = None if wider else _inline_title(gray, field, lines)
    h = field.height
    inside = [line for line in lines if field.left <= line.box.left <= field.left + .79 * h
              and line.box.right <= min(field.right, arrow.right + .18 * h)
              and field.top < line.box.center[1] < field.bottom
              and line.box.height <= .79 * h]
    value = max(inside, key=lambda item: item.box.center[1], default=None)
    if value is not None and (split := re.fullmatch(r"([^:：]+)[:：]\s*(.+)", value.text)) is not None:
        key, shown = (part.strip() for part in split.groups())
        if any(c.isalpha() for c in key) and shown:
            return _Line(value.box, key, value.score), shown
    if title is None:
        candidates = [line for line in lines
                      if -.17 * h <= line.box.left - field.left <= .59 * h
                      and field.top - 3.9 * h <= line.box.center[1] <= field.top + .41 * h
                      and .23 * h <= line.box.height <= .89 * h and line != value
                      and not _paragraph(line.text)]
        title = max(candidates, key=lambda item: item.box.center[1], default=None)
    return title, value.text if value is not None else None


def _expanded_actions(frame, evidence, field, arrow, attachment, rows, line_reader):
    title, value = _expanded_context(evidence.gray, field, arrow.box, evidence.lines,
                                      wider=attachment.box.width >= 1.16 * field.width)
    label = title.text.strip(" :：") if title else None
    if title is not None and title.score < 97 and line_reader is not None:
        try:
            recognized = line_reader.recognize(frame, [{"index": 0, "box": title.box.xyxy()}])
        except RuntimeError:
            recognized = []
        if recognized and (candidate := recognized[0].get("text")):
            label = candidate.strip(" :：")
    if label:
        label = restore_select_text(evidence.gray, title.box.xyxy(), label)
    strengths = [_ink_strength(evidence.gray, line) for line, _ in rows]
    threshold = max(34, .56 * max(strengths, default=0))
    matches = [i for i, (line, _) in enumerate(rows) if value and _key(line.text) == _key(value)]
    selected = None
    if len(matches) == 1:
        index = matches[0]
        line, row = rows[index]
        if strengths[index] >= threshold and (_checked(evidence.gray, attachment.box, row, line)
                                               or _highlighted(frame, rows, index)):
            selected = index
    actions = []
    for index, (line, box) in enumerate(rows):
        if strengths[index] < threshold:
            continue
        if line.text.endswith(("..", "…")):
            command = _command_after_divider(evidence.gray, rows, index, attachment.box)
            if command is not None:
                title = f"{label}中的" if label else ""
                actions.append({"role": "invoke_dropdown_command", "function": f"执行{title}{command}",
                                "setting_label": label, "menu_command_label": command, "confidence": line.score / 100,
                                "box": box.xyxy(), "center": line.box.center})
            continue
        option = restore_select_text(evidence.gray, line.box.xyxy(), line.text) if " " not in line.text else line.text
        function = f"选择{label}中的{option}" if label else f"选择{option}"
        action = {"role": "select_setting_option", "function": function,
                  "setting_label": label, "setting_option": option, "confidence": line.score / 100,
                  "box": box.xyxy(), "center": line.box.center}
        if selected == index:
            action["setting_option_selected"] = True
            action["function"] += "（当前选项）"
        actions.append(action)
    return ExpandedValueSelect(attachment.box.xyxy(), field.xyxy(), label, tuple(actions))


def find_expanded_value_select(frame, nodes, *, line_reader=None, evidence=_UNPREPARED):
    """Return the one independently confirmed menu, excluding obscured controls."""
    if evidence is _UNPREPARED:
        evidence = prepare_value_select(frame, nodes)
    if evidence is None:
        return None
    candidates = []
    for field, arrow in evidence.arrows.items():
        groups = list(_attached_panels(field, evidence.rectangles, evidence.arrows, evidence.lines))
        groups.extend(_wide_panels(field, evidence.rectangles, evidence.arrows, evidence.lines))
        for attachment in groups:
            rows = _option_rows(field, attachment)
            if len(rows) >= 2:
                candidates.append((field, arrow, attachment, rows))
    candidates.sort(key=lambda entry: (len(entry[3]), entry[0].height * entry[0].width, entry[2].box.height),
                    reverse=True)
    distinct = []
    for candidate in candidates:
        field, _, panel, _ = candidate
        if any(_overlap(field, previous[0]) and _overlap(panel.box, previous[2].box) for previous in distinct):
            continue
        distinct.append(candidate)
    if len(distinct) != 1:
        return None
    return _expanded_actions(frame, evidence, *distinct[0], line_reader)


def _merged_text_key(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def _merged_views(frame, box):
    color, contrast = _title_views(frame, box, box, box.right)
    return color, cv2.resize(contrast, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def _read_merged(frame, merged, line_reader):
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    if not merged or not callable(recognize):
        return {}
    try:
        crops = [crop for _, (_, title, value) in merged for box in (title, value)
                 for crop in _merged_views(frame, box)]
        readings = recognize(crops)
    except (RuntimeError, ValueError, cv2.error):
        return {}
    if len(readings) != len(crops):
        return {}
    accepted = {}
    for index, (field, (combined, title_box, value_box)) in enumerate(merged):
        title, alt_title, value, alt_value = readings[4 * index:4 * index + 4]
        title_text, value_text = title["text"], value["text"]
        title_score = min(title["score"], alt_title["score"])
        value_score = min(value["score"], alt_value["score"])
        if (title_text and value_text and title_text == alt_title["text"] and value_text == alt_value["text"]
                and min(title_score, value_score) >= 92
                and _merged_text_key(title_text + value_text) == _merged_text_key(combined)):
            accepted[field] = (_Line(title_box, title_text, title_score), _Line(value_box, value_text, value_score))
    return accepted


def _read_legacy_merged(frame, merged, line_reader):
    recognize = getattr(line_reader, "recognize", None)
    if not merged or not callable(recognize):
        return {}
    # Preserve the original full-height value crop for readers and controls it supports.
    legacy = [(field, combined, title, _Box(value.left, field.top + 2, value.right, field.bottom - 2))
              for field, (combined, title, value) in merged]
    crops = [{"index": 2 * i + offset, "box": box.xyxy()}
             for i, (_, _, title, value) in enumerate(legacy) for offset, box in enumerate((title, value))]
    try:
        results = {entry["index"]: entry.get("text") for entry in recognize(frame, crops)}
    except RuntimeError:
        return {}
    accepted = {}
    for i, (field, combined, title_box, value_box) in enumerate(legacy):
        title, value = results.get(2 * i), results.get(2 * i + 1)
        if title and value and _key(title + value) == _key(combined):
            accepted[field] = (_Line(title_box, title), _Line(value_box, value))
    return accepted


def _recaptured_toolbar_fields(gray, lines):
    """Recover closed fields whose fill is distinctive within the toolbar band."""
    height, width = gray.shape
    titles = sorted(lines, key=lambda line: line.box.bottom)
    bottoms = [line.box.bottom for line in titles]
    matched = {}
    for value in lines:
        box = value.box
        h = box.height
        if value.score < 80 or not 8 <= h <= .11 * min(height, width) or _paragraph(value.text):
            continue
        first = bisect_left(bottoms, box.top - .92 * h)
        stop = bisect_right(bottoms, box.top)
        candidates = [line for line in titles[first:stop] if abs(line.box.left - box.left) <= .59 * h
                      and .54 * h <= line.box.height <= 1.08 * h]
        title = max(candidates, key=lambda item: item.box.bottom, default=None)
        near = _Box(max(0, box.left - round(.24 * h)), box.top,
                    min(width, box.right + round(1.4 * h)), min(height, box.bottom + 2))
        if title is None or not _separated_toolbar_title(gray, near, title, lines) or _ink_strength(gray, value) < 74:
            continue
        x0, y0 = max(0, box.left - round(.58 * h)), max(0, box.top - round(.38 * h))
        x1, y1 = min(width, box.right + 3 * h), min(height, box.bottom + round(.40 * h))
        for local in _rectangles(gray[y0:y1, x0:x1]):
            field = _Box(x0 + local.left, y0 + local.top, x0 + local.right, y0 + local.bottom)
            if (not 11 <= field.height <= .11 * min(height, width) or field.width < 1.70 * field.height
                    or any(_overlap(field, known) for known in matched)
                    or _above_title(field, lines) != title or not _separated_toolbar_title(gray, field, title, lines)):
                continue
            arrow = _arrow(gray, field)
            if arrow is not None and not arrow.up and _shown_value(gray, field, arrow.box, lines) == (True, value):
                matched[field] = arrow.box
    return matched


def _adjacent_fields(first, second):
    left, right = (first, second) if first.left < second.left else (second, first)
    h = min(left.height, right.height)
    return (abs(left.top - right.top) <= .17 * h and abs(left.bottom - right.bottom) <= .17 * h
            and -.10 * h <= right.left - left.right <= .40 * h)


def _in_value_lane(field, line):
    box, h = line.box, field.height
    return (field.left - .17 * h <= box.left <= field.left + 1.12 * h and box.right <= field.right + .14 * h
            and .35 * h <= box.height <= 1.20 * h and abs(box.center[1] - field.center[1]) <= .21 * h)


def _peer_arrow(gray, field, anchor, mark):
    scale = field.height / anchor.height
    x = field.right - (anchor.right - mark.center[0]) * scale
    y = field.top + (mark.center[1] - anchor.top) * scale
    arrow = _arrow(gray, field, left_limit=round(x - .25 * field.height))
    if arrow is None:
        # A strong arrow core can exclude disconnected compression fringes.
        arrow = _arrow(gray, field, thresholds=(64,))
        if arrow != _arrow(gray, field, thresholds=(70,)):
            return None
    if (arrow is None or arrow.up or abs(arrow.box.center[0] - x) > .17 * field.height
            or abs(arrow.box.center[1] - y) > .13 * field.height
            or not .65 * mark.width * scale <= arrow.box.width <= 1.55 * mark.width * scale):
        return None
    return arrow.box


def _local_peer_rectangles(gray, anchor, lines):
    height, width = gray.shape
    for line in lines:
        box, h = line.box, line.box.height
        if (not 0 <= box.left - anchor.right <= 1.45 * anchor.height or h < 8
                or abs(box.center[1] - anchor.center[1]) > .21 * anchor.height or line.score < 80):
            continue
        x0, x1 = max(0, box.left - round(.58 * h)), min(width, box.right + 3 * h)
        y0, y1 = max(0, box.top - round(.44 * h)), min(height, box.bottom + round(.52 * h))
        for local in _rectangles(gray[y0:y1, x0:x1]):
            yield _Box(x0 + local.left, y0 + local.top, x0 + local.right, y0 + local.bottom)


def _recover_peer_fields(gray, lines, rectangles, closed):
    """Confirm a separate adjacent rectangle using the established selector's arrow lane."""
    found = {}
    anchors = []
    for anchor, mark in closed.items():
        if any(_overlap(anchor, earlier) for earlier in anchors):
            continue
        valid, value = _shown_value(gray, anchor, mark, lines)
        if not valid or value is None or _ink_strength(gray, value) < 74:
            continue
        anchors.append(anchor)
        candidates = rectangles if anchor in rectangles else (*rectangles, *_local_peer_rectangles(gray, anchor, lines))
        for field in candidates:
            if (not _adjacent_fields(anchor, field) or not 1.70 <= field.width / field.height <= 33
                    or any(_overlap(field, known) for known in (*closed, *found))
                    or not any(_in_value_lane(field, line) for line in lines)):
                continue
            arrow = _peer_arrow(gray, field, anchor, mark)
            if arrow is not None:
                found[field] = arrow
    return found


def _field_body_tone(gray, field):
    margin = max(2, round(.14 * field.height))
    pixels = gray[field.top + margin:field.top + 2 * margin,
                  field.left + margin:field.right - margin].astype(np.int16)
    tone = float(np.median(pixels))
    return tone if (np.abs(pixels - tone) <= 5).mean() >= .68 else None


def _neighbor_value_boxes(gray, field, arrow, rectangles, occupied, lines):
    boxes = [other for other in occupied if _adjacent_fields(field, other)]
    if boxes or field not in occupied:
        return boxes
    # A separate matching enclosure still owns its text if its chevron is absent.
    # Only use that enclosure as context for a field with its own strict chevron.
    possible = [other for other in rectangles if _adjacent_fields(field, other)
                and other.width >= 1.70 * other.height and any(_in_value_lane(other, line) for line in lines)]
    if not possible or _arrow(gray, field) != _Arrow(arrow, up=False):
        return []
    tone = _field_body_tone(gray, field)
    return [] if tone is None else [other for other in possible if (peer := _field_body_tone(gray, other)) is not None
                                   and abs(peer - tone) <= 6]


def _dark_named_fields(frame, gray, lines, known_fields):
    """Require a new tinted-panel enclosure, one down arrow and a readable named value."""
    recovered = {}
    marks = set(known_fields.values())
    for field in dark_neutral_enclosures(frame, gray):
        if any(.78 <= field.height / known.height <= 1.28 and _overlap(field, known) for known in known_fields):
            continue
        arrow = _arrow(gray, field)
        if arrow is None or arrow.up or arrow.box in marks:
            continue
        title = _inline_title(gray, field, lines)
        if title is None:
            continue
        valid, value = _shown_value(gray, field, arrow.box, lines)
        if (not valid or value is None or min(title.score, value.score) < 90
                or not _readable_value_field(gray, field, arrow.box, title, value)):
            continue
        recovered[field] = arrow
        marks.add(arrow.box)
    return recovered


def prepare_value_select(frame, nodes):
    """Compute the common field, arrow and text evidence once per frame."""
    height, width = frame.shape[:2]
    span = min(height, width)
    if span < 24:
        return None
    max_text_height = .56 if span < 180 else .11
    lines = tuple(_Line(box, label, score) for box, label, score in
                  _labels(nodes, width, height, max_height_fraction=max_text_height))
    if len(lines) < 2:
        return None
    compact_only = span < 180 or len(lines) < 3
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rectangles = tuple(_rectangles(gray))
    arrows, closed, compact, strict_closed = {}, {}, {}, {}
    max_field_height = .82 * height if compact_only else .22 * span
    for field in rectangles:
        if not (max(11, .009 * span) <= field.height <= max_field_height
                and 1.70 <= field.width / field.height <= 33):
            continue
        strict = _arrow(gray, field)
        if strict is not None and not strict.up:
            strict_closed[field] = strict.box
            pair = find_compact_named_selector(frame, gray, field, strict.box, lines)
            if pair is not None:
                compact[field] = pair
                closed[field] = strict.box
        if compact_only:
            if field in compact:
                arrows[field] = strict
            continue
        arrow = strict or _arrow(gray, field, relaxed=True)
        if arrow is None and field.height <= .11 * span and field.width >= 2.2 * field.height:
            weak = _arrow(gray, field, thresholds=(25,))
            if (weak is not None and not weak.up and weak == _arrow(gray, field, thresholds=(22,))
                    and _merged_value(field, weak.box, lines) is not None):
                strict = arrow = weak
        if arrow is None:
            continue
        arrows[field] = arrow
        if strict is not None and not strict.up and field.height <= .11 * span:
            closed[field] = strict.box
    if not compact_only:
        if not any(next(_attached_panels(field, rectangles, arrows, lines), None) is not None
                   or next(_wide_panels(field, rectangles, arrows, lines), None) is not None for field in arrows):
            arrows.update(_interrupted_fields(gray, rectangles, lines, tuple(arrows)))
        if not closed:
            closed.update(_recaptured_toolbar_fields(gray, lines))
        closed.update(_recover_peer_fields(gray, lines, rectangles, closed))
    dark_fields = _dark_named_fields(frame, gray, lines, closed | strict_closed)
    if compact_only and not compact and not dark_fields:
        return None
    arrows.update(dark_fields)
    dark_marks = {field: arrow.box for field, arrow in dark_fields.items()}
    closed.update(dark_marks)
    strict_closed.update(dark_marks)
    return _ValueSelectEvidence(gray, lines, rectangles, arrows, closed, compact, strict_closed) if arrows or closed else None


def _peer_value_crop(gray, field, arrow, lines):
    candidates = [line for line in lines if _in_value_lane(field, line) and line.score >= 85]
    if len(candidates) != 1 or _arrow(gray, field) != _Arrow(arrow, up=False):
        return None
    line = candidates[0]
    h = field.height
    # Use independent body ink before the arrow when the OCR polygon is too wide.
    box = _Box(max(field.left + 2, line.box.left - 1), max(field.top + 2, line.box.top - 1),
               arrow.left - max(1, round(.07 * h)), min(field.bottom - 2, line.box.bottom + 1))
    if box.width < .80 * h or box.height < .37 * h or _ink_strength(gray, _Line(box, line.text)) < 74:
        return None
    return field, line, box


def _read_peer_values(frame, requests, line_reader):
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    if not requests or recognize is None:
        return {}
    crops = []
    for _, _, box in requests:
        color = frame[box.top:box.bottom, box.left:box.right]
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.int16)
        ground = np.median(gray)
        delta = gray - ground if ground <= 127 else ground - gray
        contrast = np.clip(1.75 * delta, 0, 255).astype(np.uint8)
        crops.extend((color, cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)))
    try:
        readings = recognize(crops)
    except (RuntimeError, ValueError):
        return {}
    if len(readings) != len(crops):
        return {}
    return {field: (None, _Line(box, first["text"], min(first["score"], second["score"])))
            for (field, original, box), first, second in zip(requests, readings[::2], readings[1::2], strict=True)
            if first["text"] == second["text"] and _key(first["text"]) == _key(original.text)
            and min(first["score"], second["score"]) >= 90}


def _closed_selector_action(field, arrow, label, shown, *, display_label=None):
    if label is None:
        function = f"展开下拉选项（当前值 {shown}）"
    else:
        function = f"选择{label}" if shown is None else f"选择{label}（当前值 {shown}）"
    action = {"role": "choose_setting", "function": function, "setting_label": label,
              "setting_value": shown, "box": field.xyxy(), "center": arrow.center}
    if display_label and display_label != label:
        action["setting_display_label"] = display_label
    return action


def _rail_title(gray, field, lines):
    title = _inline_title(gray, field, lines)
    if title is not None:
        return title
    h = field.height
    crossing = [line for line in lines if line.score >= 90
                and .35 * h <= line.box.height <= 1.12 * h
                and abs(line.box.center[1] - field.center[1]) <= .20 * h
                and line.box.left <= field.left - .80 * h
                and 0 < line.box.right - field.left <= .40 * h]
    return _outside_label(gray, field, crossing[0]) if len(crossing) == 1 else None


def _closed_rail_actions(frame, evidence, rows, line_reader):
    if not callable(getattr(line_reader, "recognize_prepared_lines", None)):
        return []
    gray, lines, rectangles = evidence.gray, evidence.lines, evidence.rectangles
    marks = set(evidence.strict_closed.values())
    pending = {}
    for field in selector_rail_fields(gray, evidence.strict_closed):
        if any(_overlap(field, other) for other, _ in rows):
            continue
        mark = _arrow(gray, field)
        if mark is None or mark.up or mark.box in marks:
            continue
        title = _rail_title(gray, field, lines)
        if title is None or title.score < 90 or _stepper_arrow(gray, field, mark.box, (*rectangles, field)):
            continue
        contours = evidence.arrows | {field: mark}
        if (next(_attached_panels(field, rectangles, contours, lines), None) is not None
                or next(_wide_panels(field, rectangles, contours, lines), None) is not None):
            continue
        pending[field] = mark.box, title
        marks.add(mark.box)
    actions = []
    for field, value in read_rail_selectors(frame, gray, pending, line_reader).items():
        arrow, title = pending[field]
        value = _Line(*value)
        if not _readable_value_field(gray, field, arrow, title, value):
            continue
        displayed = restore_select_text(gray, title.box.xyxy(), title.text).strip(" :：")
        label = RESTART_NOTE.sub("", displayed).strip(" :：")
        if label and label.casefold() != value.text.casefold():
            shown = None if _PLACEHOLDER.fullmatch(value.text) is not None else value.text
            actions.append((field, _closed_selector_action(field, arrow, label, shown, display_label=displayed)))
    return actions


def _blank_grouped_peer(gray, row, strict_fields, lines):
    reference = row.title_reference
    if reference is None:
        return None
    height = row.field.height
    candidates = {}
    for field, arrow in strict_fields.items():
        if (abs(field.center[1] - reference.center[1]) <= .22 * height
                and abs(field.left - row.field.left) <= .18 * height
                and abs(field.right - row.field.right) <= .18 * height
                and .82 <= field.height / height <= 1.22
                and _shown_value(gray, field, arrow, lines) == (True, None)):
            candidates.setdefault(arrow, field)
    if len(candidates) == 1:
        arrow, field = next(iter(candidates.items()))
        return field, arrow
    return None


def suggest_value_selects(frame, nodes, *, overlay=None, line_reader=None, evidence=_UNPREPARED):
    """Return one click per evidenced closed selector; withhold while a list is attached."""
    if overlay is not None:
        return []
    if evidence is _UNPREPARED:
        evidence = prepare_value_select(frame, nodes)
    if evidence is None:
        return []
    gray, lines, rectangles, selects = evidence.gray, evidence.lines, evidence.rectangles, evidence.closed
    if any(next(_attached_panels(field, rectangles, evidence.arrows, lines), None) is not None
           or next(_wide_panels(field, rectangles, evidence.arrows, lines), None) is not None
           for field in evidence.arrows):
        return []
    grouped = []
    if getattr(line_reader, "recognize_prepared_lines", None) is not None:
        grouped = grouped_selector_rows(gray, evidence.strict_closed, lines, read_arrow=_arrow)
    readings = read_grouped_selector_text(frame, grouped, line_reader) if grouped else {}
    rows = []
    for row in grouped:
        if row.field not in readings:
            continue
        label, shown = readings[row.field]
        label = label.strip(" :：")
        if _PLACEHOLDER.fullmatch(shown):
            shown = None
        if label and not any(_overlap(row.field, prior) for prior, _ in rows):
            rows.append((row.field, _closed_selector_action(row.field, row.arrow, label, shown)))
            peer = _blank_grouped_peer(gray, row, evidence.strict_closed, lines)
            if peer is not None and not any(_overlap(peer[0], prior) for prior, _ in rows):
                field, arrow = peer
                rows.append((field, _closed_selector_action(field, arrow, label, None)))
    prepared, merged, peer_crops = {}, [], []
    occupied = {field for field in selects if any(_in_value_lane(field, line) for line in lines)}
    for field, arrow in selects.items():
        if any(_overlap(field, verified) for verified, _ in rows):
            continue
        pair = evidence.compact.get(field)
        if pair is None:
            valid, value = _shown_value(gray, field, arrow, lines)
            inline = _inline_title(gray, field, lines)
        else:
            inline, value = pair
            valid = True
        peers = _neighbor_value_boxes(gray, field, arrow, rectangles, occupied, lines)
        peer_value = inline is not None and any(other.right <= field.left + .10 * field.height
                                                and _in_value_lane(other, inline) for other in peers)
        if peer_value:
            inline = None
        if not valid and inline is not None:
            value = _boundary_value(gray, field, arrow, lines)
            valid = value is not None
        title = inline or _above_title(field, lines)
        if inline is None and title is not None and any(
            other.bottom <= field.top and _in_value_lane(other, title)
            and _shown_value(gray, other, mark, lines) == (True, title)
            for other, mark in evidence.strict_closed.items()
        ):
            title = None
        separated = valid and title is not None and inline is None and _separated_toolbar_title(gray, field, title, lines)
        # A value in a neighboring field cannot label this one, and the menu row
        # above two adjacent fields does not supply independent field names.
        if (separated or (title is None and peers)
                or (peer_value and title is not None and _compact_caption_row(title, lines, 1.15))):
            if valid and value is not None and _ink_strength(gray, value) >= 74:
                prepared[field] = (None, value)
            elif not valid and peers and not any(_overlap(field, entry[0]) for entry in peer_crops):
                crop = _peer_value_crop(gray, field, arrow, lines)
                if crop is not None:
                    peer_crops.append(crop)
            continue
        if not valid:
            candidate = _merged_value(field, arrow, lines)
            if candidate is not None and not any(_overlap(field, prior) for prior, _ in merged):
                merged.append((field, candidate))
            continue
        if title is None or (value is None and inline is None):
            continue
        prepared[field] = (title, value)
    legacy_merged = {field: pair for field, pair in _read_legacy_merged(frame, merged, line_reader).items()
                     if _readable_value_field(gray, field, selects[field], *pair)}
    pending_merged = [(field, candidate) for field, candidate in merged if field not in legacy_merged]
    confirmed_merged = _read_merged(frame, pending_merged, line_reader)
    bounded = [(field, candidate) for field, candidate in pending_merged
               if field not in confirmed_merged and evidence.strict_closed.get(field) == selects[field]]
    recovered_merged = recapture_merged_selectors(frame, gray, bounded, evidence.strict_closed, rectangles, line_reader)
    confirmed_merged.update({field: (_Line(*title), _Line(*value)) for field, (title, value) in recovered_merged.items()})
    prepared.update(legacy_merged)
    prepared.update(confirmed_merged)
    prepared.update(_read_peer_values(frame, peer_crops, line_reader))
    side_titles, paired = {}, {}
    if callable(getattr(line_reader, "recognize_prepared_lines", None)):
        named = [field for field, _ in rows]
        named.extend(field for field, (title, _) in prepared.items() if title is not None)
        pending = {}
        for field, arrow in selects.items():
            if evidence.strict_closed.get(field) != arrow or any(_overlap(field, known) for known in named):
                continue
            valid, value = _shown_value(gray, field, arrow, lines)
            if valid and value is not None:
                pending[field] = value
        if pending:
            side_titles = recover_side_titles(frame, gray, rectangles, evidence.strict_closed, pending, lines, line_reader)
            if side_titles:
                # A verified title takes precedence over an overlapping anonymous value.
                prepared = ({field: (title, pending[field]) for field, title in side_titles.items()}
                            | {field: pair for field, pair in prepared.items() if field not in side_titles})
        paired = recapture_selector_pair_text(frame, evidence, (*named, *side_titles), line_reader)
        if paired:
            prepared = ({field: (_Line(*title), _Line(*value)) for field, (title, value) in paired.items()}
                        | {field: pair for field, pair in prepared.items() if field not in paired})
    confirmed_text = confirmed_merged.keys() | paired.keys()
    completed_text = complete_selector_text(frame, gray, (
        (field, selects[field], None if field in side_titles else title, value)
        for field, (title, value) in prepared.items()
        if field not in confirmed_text and evidence.strict_closed.get(field) == selects[field]
    ), line_reader)
    for field, (title, value) in prepared.items():
        if any(_overlap(field, box) for box, _ in rows):
            continue
        if title is not None and value is not None and not _readable_value_field(gray, field, selects[field], title, value):
            continue
        repaired_title, repaired_value = completed_text.get(field, (None, None))
        if repaired_title is not None:
            displayed = repaired_title
        elif field in confirmed_text or field in side_titles:
            displayed = title.text
        else:
            displayed = restore_select_text(gray, title.box.xyxy(), title.text) if title is not None else None
        label = RESTART_NOTE.sub("", displayed).strip(" :：") if displayed is not None else None
        checked = (read_compact_selector_value(frame, field, selects[field], value, line_reader)
                   if value is not None and evidence.compact.get(field) == (title, value) else None)
        if repaired_value is not None:
            shown = repaired_value
        elif field in confirmed_text:
            shown = value.text
        elif checked is not None:
            shown = checked[0]
        else:
            shown = restore_select_text(gray, value.box.xyxy(), value.text) if value is not None else None
            if value is not None:
                shown = recapture_dimension_text(frame, field, selects[field], value.box, shown, line_reader)
        if title is not None and (not label or shown is not None and shown.casefold() == label.casefold()):
            continue
        shown = shown if shown is not None and _PLACEHOLDER.fullmatch(shown) is None else None
        if label is not None or shown is not None:
            rows.append((field, _closed_selector_action(field, selects[field], label, shown,
                                                        display_label=displayed.strip(" :：") if displayed else None)))
    rows.extend(_closed_rail_actions(frame, evidence, rows, line_reader))
    mark_unlisted_value_peers(evidence, rows)
    return [row for _, row in sorted(rows, key=lambda item: (item[0].top, item[0].left))]
