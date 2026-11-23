"""Recover settings navigation from a confirmed page and its icon-aligned rows."""

from __future__ import annotations

import re
from statistics import median
import unicodedata

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from inventory_list_rows import _labels, _runs
from inventory_toolbar_actions import _glyphs, _spans


_HEADINGS = {"settings", "preferences", "设置", "設定", "設置", "偏好设置", "偏好設定"}
_CLOCK = re.compile(r"(?:[01]?\d|2[0-3]):[0-5]\d")


def _normalized(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def settings_back(frame, nodes, header_box):
    """Read the glyph beside or above an already supported settings heading."""
    height, width = frame.shape[:2]
    left, top, right, bottom = map(round, header_box)
    scale = bottom - top
    if (not 0 <= left < right <= width or not max(10, .018 * width) <= scale <= .11 * width
            or not .035 * height <= (top + bottom) / 2 <= .23 * height):
        return None
    clock_bottom = max((box.bottom for box, label, _ in _labels(nodes, width, height)
                        if box.center[1] < .06 * height and _CLOCK.fullmatch(label)), default=0)
    safe_top = max(round(.037 * height), clock_bottom + round(.20 * scale))
    if left >= 1.45 * scale:
        # A title beside the arrow, on the same toolbar row.
        cap = min(round(.21 * width), round(left - .18 * scale))
        y = (top + bottom) / 2
        region = (0, max(safe_top, round(y - .86 * scale)), cap, min(height, round(y + .86 * scale)))
    elif left <= .80 * scale:
        # An expanded title below the toolbar.
        cap = min(round(.21 * width), round(2.45 * scale))
        region = (0, max(safe_top, round(top - 2.35 * scale)), cap, round(top - .15 * scale))
    else:
        return None
    if region[3] <= region[1]:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    arrows = [glyph for glyph in _glyphs(gray, region, scale)
              if .023 * width <= glyph.center[0] <= .12 * width and is_shaft_back_arrow(glyph.mask)]
    if len(arrows) != 1:
        return None
    glyph = arrows[0]
    return {"role": "back", "function": "返回上一页", "center": glyph.center, "box": glyph.box}


def _leading_marks(gray, label, pitch, *, minimum):
    """Measure an isolated icon on the same flat surface as its title."""
    h = label.height
    _, cy = label.center
    left, top = max(0, round(label.left - 3.6 * h)), max(0, round(cy - .43 * pitch))
    right, bottom = round(label.left - .45 * h), min(gray.shape[0], round(cy + .43 * pitch))
    crop = gray[top:bottom, left:right].astype(np.int16)
    if min(crop.shape) < 6:
        return []
    fill = np.median(crop)
    deviation = np.abs(crop - fill)
    if (deviation <= 4).mean() < .82:
        return []
    mask = deviation >= minimum
    spans = []
    for x0, x1 in _spans(mask.any(axis=0)):
        if spans and x0 - spans[-1][1] <= round(.12 * h):
            spans[-1] = spans[-1][0], x1
        else:
            spans.append((x0, x1))
    boxes = []
    for x0, x1 in spans:
        rows, _ = np.nonzero(mask[:, x0:x1])
        y0, y1 = rows.min(), rows.max() + 1
        w, glyph_h = x1 - x0, y1 - y0
        glyph = mask[y0:y1, x0:x1]
        if (.42 * h <= w <= 1.55 * h and .50 * h <= glyph_h <= 1.53 * h
                and .55 <= w / glyph_h <= 1.65 and .12 <= glyph.mean() <= .89
                and x0 > 0 and x1 < crop.shape[1] and y0 > 0 and y1 < crop.shape[0]
                and abs((y0 + y1) / 2 + top - cy) <= .54 * h):
            boxes.append((left + int(x0), top + int(y0), left + int(x1), top + int(y1)))
    return boxes


def _center_x(box):
    return (box[0] + box[2]) / 2


def _row_marks(gray, lines, pitch):
    scale = median(box.height for box, _, _ in lines)
    candidates = [_leading_marks(gray, box, pitch, minimum=34) for box, _, _ in lines]
    anchors = [mark for options in candidates for mark in options]
    if len(anchors) < 4:
        return []

    def near(options, x):
        matches = [mark for mark in options if abs(_center_x(mark) - x) <= .24 * scale]
        return matches[0] if len(matches) == 1 else None

    alignments = [[near(options, _center_x(anchor)) for options in candidates] for anchor in anchors]
    valid = max(alignments, key=lambda marks: sum(mark is not None for mark in marks))
    present = [mark for mark in valid if mark is not None]
    if len(present) < max(4, len(lines) - 1):
        return []
    x = median(map(_center_x, present))
    # A low-contrast icon needs corroboration from the same row spacing and an
    # independently established leading lane.
    for i, (box, _, _) in enumerate(lines):
        if valid[i] is None:
            valid[i] = near(_leading_marks(gray, box, pitch, minimum=6), x)
    return valid


def _visible_spaces(gray, box, label):
    """Restore a missing word space only when the text pixels resolve the break."""
    words = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=\S)(?=&)|(?<=&)(?=\S)", " ", label).split()
    if len(words) == len(label.split()) or not all(
            word == "&" or (word.isascii() and word.isalpha()) for word in words):
        return label
    roi = gray[box.top:box.bottom, box.left:box.right].astype(np.int16)
    deviation = np.abs(roi - np.median(roi))
    strokes = _spans((deviation >= 48).any(axis=0))
    if len(strokes) < len(words) * 2:
        return label
    gaps_between_strokes = [(left, right) for (_, left), (right, _) in zip(strokes[:-1], strokes[1:], strict=True)]
    limit = max(.16 * box.height, 1.9 * median(right - left for left, right in gaps_between_strokes))
    first, last = strokes[0][0], strokes[-1][1]
    gaps = np.asarray([((left + right) / 2 - first) / (last - first)
                       for left, right in gaps_between_strokes if right - left >= limit])
    ends = np.cumsum([len(word) for word in words])
    expected = ends[:-1] / ends[-1]
    if len(gaps) == len(expected) and np.max(np.abs(gaps - expected)) <= .10:
        return " ".join(words)
    return label


def _names(frame, gray, lines, line_reader):
    names = [_visible_spaces(gray, box, label) for box, label, _ in lines]
    ambiguous = [
        {"index": index, "box": box.xyxy(), "content": names[index], "score": score}
        for index, (box, _, score) in enumerate(lines)
        if re.search(r"[a-z][A-Z]|\S&|&\S", names[index])
    ]
    if ambiguous and line_reader is not None:
        try:
            results = line_reader.recognize(frame, ambiguous)
        except RuntimeError:
            results = []
        for result in results:
            index, title = result["index"], result.get("text")
            if title and _normalized(title) == _normalized(names[index]):
                names[index] = _visible_spaces(gray, lines[index][0], title)
    return names


def suggest_settings_navigation(frame, nodes, line_reader=None):
    """Read repeated, icon-led rows beneath the visible settings title."""
    height, width = frame.shape[:2]
    if width >= height:
        return []
    labels = list(_labels(nodes, width, height))
    headings = [box for box, title, score in labels
                if score >= 89 and _normalized(title) in _HEADINGS
                and .038 * height <= box.center[1] <= .23 * height]
    if len(headings) != 1:
        return []
    title = headings[0]
    below = [entry for entry in labels if title.bottom < entry[0].top and entry[0].left <= .38 * width]
    runs = _runs(below)
    if not runs:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for lines in runs:
        pitch = median(np.diff([box.center[1] for box, _, _ in lines]))
        if lines[0][0].center[1] - title.bottom > 1.65 * pitch:
            continue
        marks = _row_marks(gray, lines, pitch)
        if not marks:
            continue
        names = _names(frame, gray, lines, line_reader)
        selected = {box for box, _, _ in lines}
        actions = []
        for (box, _, _), label, mark in zip(lines, names, marks, strict=True):
            if mark is None:
                continue
            _, y = box.center
            inset = max(0, round(mark[0] - .50 * box.height))
            lower = round(y + .44 * pitch)
            summary_bottom = max((other.bottom for other, _, _ in below
                                  if other not in selected and abs(other.left - box.left) <= .31 * box.height
                                  and 0 <= other.top - box.bottom <= .40 * box.height
                                  and other.height <= .95 * box.height), default=0)
            actions.append({"role": "open_setting", "function": f"打开{label}", "setting_label": label,
                            "center": box.center,
                            "box": (inset, round(y - .44 * pitch), width - inset,
                                    min(height, max(lower, summary_bottom + round(.14 * pitch))))})
        back = settings_back(frame, nodes, title.xyxy())
        return [back, *actions] if back else actions
    return []
