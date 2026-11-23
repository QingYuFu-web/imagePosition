"""Measure labeled pill switches inside an already verified landscape modal."""

import cv2
import numpy as np

from inventory_setting_switches import _stadium
from screen_component_captions import _bands, _tight
from screen_landscape_side_panel import _divider, _rule
from screen_wide_panel_checks import _soft_letter


def _form_columns(gray, header, footer, end):
    """Require two independently ruled columns with a common gutter."""
    left = round(.012 * end)
    field = gray[header:footer, left:end].astype(np.int16)
    ridge = field[2:-2] - np.maximum(field[:-4], field[4:])
    rows = np.flatnonzero((ridge >= 3).mean(axis=1) >= .86)
    columns = []
    for y in rows:
        if y > .61 * (footer - header):
            continue
        lines = []
        for a, b in _bands(ridge[y] >= 3):
            if lines and a - lines[-1][1] <= .0045 * end:
                lines[-1] = (lines[-1][0], b)
            else:
                lines.append((a, b))
        pair = [(left + a, left + b) for a, b in lines if .35 * end <= b - a <= .59 * end]
        if len(pair) == 2 and .008 * end <= pair[1][0] - pair[0][1] <= .048 * end:
            columns.append((y, pair))
    if len(columns) < 2 or columns[-1][0] - columns[0][0] < .12 * (footer - header):
        return []
    _, lanes = columns[0]
    if any(max(abs(a - b) for pair, other in zip(lanes, measured, strict=True)
               for a, b in zip(pair, other, strict=True)) > .025 * end for _, measured in columns):
        return []
    return lanes


def _knob(frame, gray, box):
    x, y, r, b = box
    width, height = r - x, b - y
    chip = gray[y:b, x:r]
    witnesses = {"on": 0, "off": 0}
    for low, high in ((10, 32), (19, 46), (28, 62)):
        contours, _ = cv2.findContours(cv2.Canny(chip, low, high), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        seen = set()
        for contour in contours:
            a, t, w, h = cv2.boundingRect(contour)
            cx, cy = a + w / 2, t + h / 2
            if (not .44 * height <= min(w, h) <= max(w, h) <= .79 * height or not .81 <= w / h <= 1.23
                    or not .57 <= cv2.contourArea(contour) / (w * h) <= .83
                    or abs(cy - height / 2) > .105 * height):
                continue
            if .16 * width <= cx <= .32 * width:
                seen.add("off")
            if .66 * width <= cx <= .83 * width:
                seen.add("on")
        for state in seen:
            witnesses[state] += 1
    states = [state for state, count in witnesses.items() if count >= 2]
    if len(states) != 1:
        return None
    state = states[0]
    far = round(x + (.28 if state == "on" else .73) * width)
    mid = round((y + b) / 2)
    paint = np.median(frame[mid - 3:mid + 4, far - 3:far + 4].reshape(-1, 3), axis=0)
    if (state == "on" and np.ptp(paint) < 55) or (state == "off" and np.ptp(paint) > 42):
        return None
    return state


def _pills(frame, gray, header, footer, lanes):
    height = gray.shape[0]
    x0, x1 = lanes[0][0], lanes[-1][1]
    y0, y1 = header + round(.085 * height), footer - max(2, round(.008 * height))
    field = gray[y0:y1, x0:x1]
    shapes = []
    for cut in ((9, 30), (20, 58)):
        edges = cv2.Canny(field, *cut)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            a, t, w, h = cv2.boundingRect(contour)
            if (not .056 * height <= h <= .112 * height or not 1.69 <= w / h <= 2.22
                    or cv2.contourArea(contour) <= .79 * w * h):
                continue
            fill = cv2.drawContours(np.zeros((h, w), np.uint8), [contour], 0, 1, cv2.FILLED, offset=(-a, -t))
            if _stadium(fill):
                shapes.append((cut, (x0 + a, y0 + t, x0 + a + w, y0 + t + h)))
    found = []
    for cut, box in sorted(shapes, key=lambda item: -(item[1][2] - item[1][0]) * (item[1][3] - item[1][1])):
        a, t, r, b = box
        if not any(other != cut and max(abs(v - x) for v, x in zip(box, peer, strict=True)) <= 3
                   for other, peer in shapes):
            continue
        lane = next((pair for pair in lanes
                     if pair[0] <= a < r <= pair[1] and pair[1] - r <= .072 * (pair[1] - pair[0])), None)
        if lane is None or any(max(abs(v - x) for v, x in zip(box, old, strict=True)) <= 4 for old, _, _ in found):
            continue
        state = _knob(frame, gray, box)
        if state is not None:
            found.append((box, state, lane))
    return found


def _heading_box(gray, pill, lane):
    _, top, _, bottom = pill
    size = bottom - top
    left, right = lane
    x0, x1 = left + round(.008 * (right - left)), left + round(.43 * (right - left))
    y0, y1 = top - round(.43 * size), top + round(.12 * size)
    field = gray[y0:y1, x0:x1]
    # Settings descriptions and masked game lettering are fainter than the heading.
    ink = field >= max(122, float(np.median(field)) + 84)
    groups = []
    for a, b in _bands(ink.sum(axis=0) >= 2):
        if groups and a - groups[-1][1] <= .10 * size:
            groups[-1] = groups[-1][0], b
        else:
            groups.append((a, b))
    if not groups:
        return None
    a, b = groups[0]
    box = _tight(ink[:, a:b], x0 + a, y0)
    if box is None:
        return None
    x, y, r, z = box
    h = z - y
    if x - left > .40 * size or not .25 * size <= h <= .50 * size or not 2.9 * h <= r - x <= 9 * h:
        return None
    return box


def _glyph(glyph, peak, index):
    first, second = (glyph >= cut * peak for cut in (.22, .29))
    ranks = [index.rank(mask, alphabet="cjk", limit=8) for mask in (first, second)]
    if (not np.array_equal(first, second) and all(len(row) >= 2 for row in ranks)
            and ranks[0][0]["text"] == ranks[1][0]["text"] != "內"
            and min(row[1]["distance"] - row[0]["distance"] for row in ranks) >= .030
            and max(row[0]["distance"] for row in ranks) <= .116):
        return ranks[0][0]["text"]
    ranks.append(index.rank(glyph >= .40 * peak, alphabet="cjk", limit=8))
    if (all(len(row) >= 2 for row in ranks) and len({row[0]["text"] for row in ranks}) == 1
            and min(row[1]["distance"] - row[0]["distance"] for row in ranks) >= .025
            and max(row[0]["distance"] for row in ranks) <= .145):
        return ranks[0][0]["text"]
    ranks.extend(index.rank(glyph >= cut * peak, alphabet="cjk", limit=8) for cut in (.53, .62, .69, .75))
    options = tuple(sorted({item["text"] for row in ranks for item in row}))
    paths = tuple(index.fonts if index.fonts is not None else index.cjk_sources)
    matches = {}
    for level in (.25, .39):
        x, y, w, h = cv2.boundingRect(np.uint8(glyph >= level * peak))
        if not 10 <= h <= 40 or not .63 <= w / h <= 1.36:
            continue
        key = x, y, w, h
        if key not in matches:
            shaped = np.float32(np.pad(np.minimum(1, glyph[y:y + h, x:x + w] / peak), 2))
            matches[key] = _soft_letter(paths, options, shaped.tobytes(), w, h)
    words = {v for v in matches.values() if v is not None}
    return next(iter(words)) if len(words) == 1 else None


def _heading(gray, box, index):
    x, y, r, b = box
    field = gray[y - 3:b + 4, x - 2:r + 4].astype(np.float32)
    contrast = np.maximum(field - np.median(field), 0)
    peak = float(np.percentile(contrast, 99))
    x, t, w, h = cv2.boundingRect(np.uint8(contrast >= .30 * peak))
    count = round(w / max(1, 1.075 * h))
    if not 3 <= count <= 8 or not 12 <= h <= 34:
        return None
    source = contrast[t:t + h, x:x + w]
    columns = (source >= .30 * peak).sum(axis=0)
    boundaries = [0]
    for i in range(1, count):
        center = i * w / count
        reach = .17 * h
        edge = min(range(round(center - reach), round(center + reach) + 1),
                   key=lambda j: (int(columns[j]), abs(center - j)))
        if columns[edge] > .15 * h:
            return None
        boundaries.append(edge)
    boundaries.append(w)
    read = [_glyph(source[:, a:b], peak, index) for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)]
    return "".join(read) if all(read) else None


def landscape_panel_switch_actions(frame, panel, index):
    """Publish the opposite setting state only for owned and literally read switches."""
    if index is None or panel is None:
        return []
    left, top, right, bottom = map(int, panel)
    if (left, top, right, bottom) != (0, 0, frame.shape[1], frame.shape[0]):
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    header = _rule(gray, .075 * height, .176 * height)
    footer = _rule(gray, .775 * height, .931 * height)
    if header is None or footer is None or (edge := _divider(gray, header)) is None:
        return []
    lanes = _form_columns(gray, header, footer, edge[0])
    if not lanes:
        return []
    pills = _pills(frame, gray, header, footer, lanes)
    if len(pills) < 2 or len({state for _, state, _ in pills}) != 2:
        return []
    actions = []
    for pill, state, lane in pills:
        heading = _heading_box(gray, pill, lane)
        if heading is None or (label := _heading(gray, heading, index)) is None:
            continue
        x, y, r, b = pill
        a, t, _, _ = heading
        verb = "关闭" if state == "on" else "开启"
        actions.append({"role": "toggle_setting", "function": f"{verb}{label}", "preview_text": label,
                        "box": [a, t, r, b], "click_point": [(x + r) // 2, (y + b) // 2]})
    return actions
