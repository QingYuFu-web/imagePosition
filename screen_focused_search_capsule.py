"""Verify one focused search field from its rounded toolbar and editable text."""

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_candidate_glyphs import _search
from screen_caption_transcript import _bands, read_caption
from screen_keyboard_footer_geometry import keyboard_footer_bounds
from screen_landscape_search_field import _available
from screen_native_tab_bar import _distance
from screen_wide_panel import _diagonal_cross


def _rounded_bounds(surface, minimum_fill):
    depth, width = surface.shape
    rows = [np.flatnonzero(surface[round(ratio * (depth - 1))]) for ratio in (.10, .50, .90)]
    if any(row.size == 0 for row in rows):
        return None
    first, center, last = rows
    left, right = int(center[0]), int(center[-1]) + 1
    if (right - left < .69 * width or min(left, width - right) < .065 * depth
            or max(left, width - right) > .15 * width or abs(left - (width - right)) > .07 * width):
        return None
    retreats = (first[0] - left, last[0] - left, right - first[-1] - 1, right - last[-1] - 1)
    if (min(retreats) < .064 * depth or max(retreats) > .38 * depth
            or max(retreats) - min(retreats) > .125 * depth):
        return None
    if surface[:, left:right].mean() < minimum_fill:
        return None
    return left, right


def _painted_rows(frame, start, stop, threshold):
    width = frame.shape[1]
    profile = np.median(frame[start:stop, round(.27 * width):round(.66 * width):max(3, width // 160)], axis=1)
    changes = np.flatnonzero(np.max(abs(np.diff(profile, axis=0)), axis=1) >= threshold) + 1
    return profile, np.r_[0, changes, len(profile)]


def _rounded_surfaces(frame, start, stop, tops, minimum_fill):
    width = frame.shape[1]
    profile, edges = _painted_rows(frame, start, stop, 8)
    for a, b in zip(edges, edges[1:]):
        top, bottom = int(a), int(b)
        depth = bottom - top
        if not .071 * width <= depth <= .178 * width or not tops[0] <= top + start <= tops[1]:
            continue
        pad = max(3, round(.10 * depth))
        if top < pad or bottom + pad > len(profile):
            continue
        paint = np.median(profile[top:bottom], axis=0)
        surrounds = (np.median(profile[top - pad:top], axis=0), np.median(profile[bottom:bottom + pad], axis=0))
        if any(abs(paint - other).max() < 14 for other in surrounds) or abs(surrounds[0] - surrounds[1]).max() > 17:
            continue
        delta = _distance(frame[top + start:bottom + start], paint)
        if (bounds := _rounded_bounds(delta <= 10, minimum_fill)) is not None:
            yield (bounds[0], top + start, bounds[1], bottom + start), delta[:, bounds[0]:bounds[1]]


def _toolbar_surfaces(frame):
    height, width = frame.shape[:2]
    limit = min(round(.23 * height), round(.55 * width))
    return _rounded_surfaces(frame, 0, limit, (.020 * height, .134 * height), .86)


def _center_y(mark):
    return (mark[1] + mark[3]) / 2


def _leading_arrow(labels, marks, depth):
    arrows = []
    for index, (l, t, r, b) in marks:
        w, h = r - l, b - t
        if (.26 * depth <= l <= .58 * depth and .23 * depth <= h <= .47 * depth
                and .79 <= w / h <= 1.31 and abs((t + b) / 2 - depth / 2) <= .13 * depth
                and is_shaft_back_arrow(labels[t:b, l:r] == index)):
            arrows.append((l, t, r, b))
    return arrows[0] if len(arrows) == 1 else None


def _filter_left(lines, depth):
    filters = []
    for top in lines:
        tw = top[2] - top[0]
        if not .26 * depth <= tw <= .52 * depth:
            continue
        for axis in (lambda mark: mark[0], lambda mark: (mark[0] + mark[2]) / 2):
            aligned = [mark for mark in lines if abs(axis(mark) - axis(top)) <= .046 * depth]
            first_y = _center_y(top)
            middle = [mark for mark in aligned if .46 * tw <= mark[2] - mark[0] <= .78 * tw
                      and .064 * depth <= _center_y(mark) - first_y <= .148 * depth]
            if len(middle) != 1:
                continue
            mw = middle[0][2] - middle[0][0]
            expected_y = 2 * _center_y(middle[0]) - first_y
            if any(.25 * mw <= mark[2] - mark[0] <= .69 * mw
                   and abs(_center_y(mark) - expected_y) <= .026 * depth for mark in aligned):
                filters.append(top[0])
                break
    return filters


def _trailing_start(labels, marks, depth):
    span = labels.shape[1]
    tail = [(index, mark) for index, mark in marks
            if mark[0] >= .51 * span and mark[2] <= span - .22 * depth
            and .21 * depth <= mark[1] < mark[3] <= .78 * depth]
    lenses = []
    lines = []
    for index, (l, t, r, b) in tail:
        w, h = r - l, b - t
        if .21 * depth <= h <= .47 * depth and .79 <= w / h <= 1.28:
            if _search(np.uint8(labels[t:b, l:r] == index)):
                lenses.append(l)
        if .017 * depth <= h <= .083 * depth and .056 * depth <= w <= .52 * depth and w >= 1.38 * h:
            lines.append((l, t, r, b))
    filters = sorted(set(_filter_left(lines, depth)))
    if len(filters) >= 2 and filters[1] - filters[0] >= .48 * depth:
        lenses.append(filters[0])
    return min(lenses, default=None)


def _focused_input(delta, left, right):
    depth = len(delta)
    x, r = left + round(.19 * depth), right - round(.16 * depth)
    if r - x < 1.38 * depth:
        return None
    mask = delta[:, x:r] >= 48
    mid = mask[round(.31 * depth):round(.68 * depth)]
    stems = _bands(mid.mean(axis=0) >= .93)
    matches = []
    for a, b in stems:
        if not .018 * depth <= b - a <= .083 * depth or a < .136 * depth or r - x - b < .53 * depth:
            continue
        column = (a + b) // 2
        runs = _bands(mask[:, column])
        stem = next(((t, e) for t, e in runs if .15 * depth <= t <= .325 * depth
                     and e - t >= .42 * depth and e >= .696 * depth), None)
        if stem is None:
            continue
        top = stem[0]
        band_top, band_bottom = round(.23 * depth), round(.76 * depth)
        caption = mask[band_top:band_bottom, :a]
        px, y, w, h = cv2.boundingRect(np.uint8(caption))
        if (not .177 * depth <= h <= .44 * depth or not .136 * depth <= w
                or a - (px + w) > .18 * depth or band_top + y - top < .055 * depth
                or not .34 * depth <= band_top + y + h / 2 <= .65 * depth):
            continue
        blank = mask[band_top:band_bottom, b + round(.03 * depth):]
        if blank.mean() <= .014:
            owner = (x, round(.056 * depth), r, depth - round(.067 * depth))
            caption_box = (x + px, band_top + y, x + px + w, band_top + y + h)
            matches.append((owner, caption_box))
    return matches[0] if len(matches) == 1 else None


def _toolbar_actions(field, delta):
    left, top, right, bottom = field
    depth = bottom - top
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(delta >= 48), connectivity=8)
    marks = [(index, tuple(map(int, (x, y, x + w, y + h))))
             for index, (x, y, w, h, _) in enumerate(stats[1:], 1)]
    arrow = _leading_arrow(labels, marks, depth)
    trailing = _trailing_start(labels, marks, depth)
    if arrow is None or trailing is None:
        return []
    edit = _focused_input(delta, arrow[2], trailing)
    if edit is None:
        return []
    a, b, c, d = arrow
    pad = round(.16 * depth)
    back = [left + a - pad, top + b - pad, left + c + pad, top + d + pad]
    x, y, r, k = edit[0]
    target = [left + x, top + y, left + r, top + k]
    return [
        {"role": "back", "function": "返回", "preview_text": "返回", "box": back,
         "click_point": [left + (a + c) // 2, top + (b + d) // 2]},
        {"role": "search", "function": "搜索", "preview_text": "编辑搜索内容", "box": target,
         "click_point": [left + (x + r) // 2, top + depth // 2]},
    ]


def _outlined_surfaces(frame):
    height, width = frame.shape[:2]
    limit = min(round(.27 * height), round(.57 * width))
    profile, edges = _painted_rows(frame, 0, limit, 33)
    for a, b in zip(edges, edges[1:]):
        start, end = int(a), int(b)
        depth = end - start
        if not .017 * height <= start <= .14 * height or not .075 * width <= depth <= .177 * width:
            continue
        margin = round(.091 * depth)
        if start < margin or end + margin > limit:
            continue
        paint = profile[(start + end) // 2]
        upper = profile[start - margin:start]
        lower = profile[end:end + margin]
        tint = upper[np.argmax(abs(upper - paint).max(axis=1))]
        before = _bands(abs(upper - tint).max(axis=1) <= 24)
        after = _bands(abs(lower - tint).max(axis=1) <= 27)
        if (abs(tint - paint).max() < 63 or len(before) != 1 or len(after) != 1
                or before[0][1] < .69 * margin or after[0][0] > .31 * margin):
            continue
        a0, a1 = before[0]
        b0, b1 = after[0]
        if not all(.012 * depth <= stroke <= .10 * depth for stroke in (a1 - a0, b1 - b0)):
            continue
        top, bottom = start - margin + a0, end + b1
        line_y = start - margin + (a0 + a1) // 2
        ridge = _distance(frame[line_y:line_y + 1], tint)[0]
        chords = [(l, r) for l, r in _bands(ridge <= 21) if .69 * width <= r - l <= .95 * width]
        if len(chords) != 1:
            continue
        left, right = chords[0]
        if abs(left - (width - right)) > .074 * width:
            continue
        stripe = max(2, round(.021 * depth))
        sides = (frame[start:end, left:left + stripe], frame[start:end, right - stripe:right])
        lower_line = frame[end + b0:bottom, left:right]
        if any((_distance(pixels, tint) <= 27).mean() < .75 for pixels in (*sides, lower_line)):
            continue
        delta = _distance(frame[top:bottom, left:right], paint)
        yield (left, top, right, bottom), delta


def _clearable_editor(delta):
    depth, span = delta.shape
    ink = np.uint8(delta >= 48)
    _, ids, components, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    crosses, buttons = [], []
    for index, (x, y, w, h, area) in enumerate(components[1:], 1):
        if abs(y + h / 2 - depth / 2) > .12 * depth:
            continue
        if (.72 * span <= x and x + w <= span - .28 * depth and .195 * depth <= h <= .47 * depth
                and .79 <= w / h <= 1.22 and _diagonal_cross(ids[y:y+h, x:x+w] == index)):
            crosses.append(int(x))
        if .63 * span <= x and .33 * depth <= h <= .84 * depth and .82 <= w / h <= 1.55 and area >= .56 * w * h:
            buttons.append((int(x), int(x + w)))
    if len(crosses) != 1:
        return None
    close_x = crosses[0]
    peers = [x for x, r in buttons if .105 * depth <= close_x - r <= .83 * depth]
    right = min(peers, default=close_x)
    start = round(.27 * depth)
    middle = ink[round(.22 * depth):round(.77 * depth), start:round(.38 * span)]
    entry = _bands(middle.mean(axis=0) >= .13)
    if not entry:
        return None
    left, end = entry[0]
    if left > .35 * depth or not .19 * depth <= end - left <= 2.05 * depth:
        return None
    return _focused_input(delta, start + end, right)


def _printed_search_guide(frame, field, index):
    left, top, right, bottom = field
    depth = bottom - top
    x, r = left + round(.385 * depth), right - round(.30 * depth)
    y, b = bottom + round(.20 * depth), bottom + round(.61 * depth)
    chip = frame[y:b, x:r]
    paint = np.median(chip[::4, ::5], axis=(0, 1))
    ink = _distance(chip, paint) >= 41
    _, ink_top, _, ink_height = cv2.boundingRect(np.uint8(ink))
    if not .10 * depth <= ink_height <= .32 * depth or ink.mean() > .24:
        return False
    words = []
    for a, z in _bands(ink.any(axis=0)):
        if words and a - words[-1][1] <= .265 * ink_height:
            words[-1] = (words[-1][0], z)
        else:
            words.append((a, z))
    cache = {}
    for a, z in words[:5]:
        if not 1.2 * ink_height <= z - a <= 5.7 * ink_height:
            continue
        pad = max(2, round(.085 * ink_height))
        box = (x + max(0, a - pad), y + max(0, ink_top - pad), x + min(r - x, z + pad),
               y + min(b - y, ink_top + ink_height + pad))
        read = read_caption(frame, box, index, cache=cache, choice=True)
        if read is not None and read["text"].casefold().strip(" :：") in {"search", "find", "搜索", "搜尋", "查找", "查询", "查詢"}:
            return True
    return False


def _word_glyphs(mask):
    glyphs = []
    for l, r in _bands(mask.any(axis=0)):
        _, t, _, h = cv2.boundingRect(np.uint8(mask[:, l:r]))
        if h >= 7:
            glyphs.append(mask[t:t + h, l:r])
    return glyphs


def _glyph_matches(a, b, *, minimum_overlap=.54):
    if abs(np.log((a.shape[1] / a.shape[0]) / (b.shape[1] / b.shape[0]))) > .51:
        return False
    u, v = (cv2.resize(g.astype(np.float32), (23, 29), interpolation=cv2.INTER_AREA) >= .46 for g in (a, b))
    if (u & v).sum() < minimum_overlap * (u | v).sum():
        return False
    kernel = np.ones((3, 3), np.uint8)
    return all((first & (cv2.dilate(np.uint8(second), kernel) == 0)).sum() <= .145 * first.sum()
               for first, second in ((u, v), (v, u)))


def _linked_result_sheet(frame, field, text):
    left, top, right, bottom = field
    depth, span = bottom - top, right - left
    row = top - round(.53 * depth)
    paint = np.median(frame[row, left + round(.24 * span):right - round(.20 * span):5], axis=0)
    width = frame.shape[1]
    surface = (_distance(frame[row:row + 1], paint)[0] <= 11)
    edges = [bounds for bounds in _bands(surface) if bounds[1] - bounds[0] >= .66 * width]
    if len(edges) != 1 or max(abs(edges[0][0] - left), abs(edges[0][1] - right)) > .22 * depth:
        return False
    query = text >= 48
    # Composition underlines are drawn across the word, outside the character shapes.
    query[query.mean(axis=1) >= .67] = False
    word = _word_glyphs(query)
    if not 3 <= len(word) <= 19 or sum(g.shape[1] >= .24 * g.shape[0] for g in word) < 2:
        return False
    a, b = max(round(.052 * frame.shape[0]), top - round(11 * depth)), top - round(.24 * depth)
    start, stop = left + round(1.48 * depth), right - round(.31 * span)
    result = _distance(frame[a:b, start:stop], paint) >= 53
    if result.mean() > .23:
        return False
    rows = []
    for y, k in _bands(result.sum(axis=1) >= max(2, round(.035 * depth))):
        if .17 * depth <= k - y <= .58 * depth:
            glyphs = _word_glyphs(result[y:k])
            if len(glyphs) >= 3:
                rows.append(glyphs)
    return len(rows) >= 3 and any(len(row) >= len(word) and all(_glyph_matches(a, b) for a, b in zip(word, row))
                                   for row in rows)


def _action_in_field(field, editor):
    left, top, _, _ = field
    x, y, r, b = editor
    return {"role": "search", "function": "搜索", "preview_text": "编辑搜索内容",
            "box": [left + x, top + y, left + r, top + b],
            "click_point": [left + (x + r) // 2, top + (y + b) // 2]}


def _clearable_searches(frame, index):
    height = frame.shape[0]
    if index is not None:
        for field, delta in _outlined_surfaces(frame):
            if (editor := _clearable_editor(delta)) is not None and _printed_search_guide(frame, field, index):
                yield _action_in_field(field, editor[0]), field, False
    start, stop = round(.37 * height), round(.77 * height)
    for field, delta in _rounded_surfaces(frame, start, stop, (start, stop), .71):
        if (editor := _clearable_editor(delta)) is None:
            continue
        x, y, r, b = editor[1]
        if _linked_result_sheet(frame, field, delta[y:b, x:r]):
            yield _action_in_field(field, editor[0]), field, True


def focused_search_capsule_actions(frame, regions, *, get_keyboard=None, font_index=None):
    """Read the editor and its Back affordance once a docked keyboard corroborates focus."""
    height, width = frame.shape[:2]
    if regions.get("layer") != "page" or width < 250 or not 1.52 * width <= height <= 2.85 * width:
        return []
    matches = [(actions, field, False) for field, delta in _toolbar_surfaces(frame)
               if (actions := _toolbar_actions(field, delta))]
    matches.extend(([action], field, footer) for action, field, footer in _clearable_searches(frame, font_index))
    if len(matches) != 1:
        return []
    keyboard = keyboard_footer_bounds(frame) if get_keyboard is None else get_keyboard()
    if keyboard is None:
        return []
    actions, (_, top, _, bottom), footer = matches[0]
    gap = keyboard[1] - bottom
    if (footer and not .08 * (bottom - top) <= gap <= .69 * (bottom - top)) or (not footer and gap < .27 * height):
        return []
    return [action for action in actions if _available(action["box"], regions)]
