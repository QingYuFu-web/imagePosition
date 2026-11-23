"""Verify search that retains its caret and results after the keyboard is hidden."""

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_caption_transcript import _bands
from screen_focused_search_capsule import _action_in_field, _focused_input, _glyph_matches, _painted_rows, _word_glyphs
from screen_landscape_search_field import _available
from screen_native_tab_bar import _distance
from screen_wide_panel import _diagonal_cross


def _search_surfaces(frame):
    height, width = frame.shape[:2]
    limit = min(round(.23 * height), round(.52 * width))
    profile, cuts = _painted_rows(frame, 0, limit, 3)
    for a, b in zip(cuts, cuts[1:]):
        top, bottom = int(a), int(b)
        depth = bottom - top
        if not .076 * width <= depth <= .165 * width or not .027 * height <= top <= .132 * height:
            continue
        pad = max(3, round(.15 * depth))
        if top < pad or bottom + pad > limit:
            continue
        paint = np.median(profile[top:bottom], axis=0)
        above, below = np.median(profile[top - pad:top], axis=0), np.median(profile[bottom:bottom + pad], axis=0)
        inset = min(abs(paint - above).max(), abs(paint - below).max())
        if not 5 <= inset <= 68 or abs(above - below).max() > .57 * inset:
            continue
        delta = _distance(frame[top:bottom], paint)
        surface = delta <= max(2, min(12, .42 * inset))
        middle = surface[round(.29 * depth):round(.70 * depth)]
        spans = [(a, b) for a, b in _bands(middle.mean(axis=0) >= .76) if b - a >= .028 * depth]
        if not spans:
            continue
        left, right = spans[0][0], spans[-1][1]
        if not (.66 * width <= right - left <= .955 * width and .015 * width <= left <= .12 * width
                and width - right <= .245 * width):
            continue
        edges = [np.flatnonzero(surface[round(p * (depth - 1)), left:right]) for p in (.025, .975)]
        if any(not row.size for row in edges):
            continue
        first, last = edges
        span = right - left
        rounded = (first[0], last[0], span - first[-1] - 1, span - last[-1] - 1)
        if min(rounded) < max(1, .015 * depth) or max(rounded) > .45 * depth:
            continue
        if max(rounded) - min(rounded) > .15 * depth or surface[:, left:right].mean() < .86:
            continue
        yield (left, top, right, bottom), delta[:, left:right], below


def _end_marks(delta):
    depth, span = delta.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(delta >= 48), connectivity=8)
    leading, trailing = [], []
    for i, (x, y, w, h, _) in enumerate(stats[1:], 1):
        if (not .205 * depth <= h <= .48 * depth or not .79 <= w / h <= 1.27
                or abs(y + h / 2 - depth / 2) > .105 * depth):
            continue
        mark = tuple(map(int, (x, y, x + w, y + h)))
        mask = labels[y:y + h, x:x + w] == i
        if .19 * depth <= x <= .69 * depth and is_shaft_back_arrow(mask):
            leading.append(mark)
        if .72 * span <= x and .16 * depth <= span - x - w <= .77 * depth and _diagonal_cross(mask):
            trailing.append(mark)
    if len(leading) == len(trailing) == 1:
        return leading[0], trailing[0]
    return None


def _same_narrow_glyph(first, second):
    ratios = [glyph.shape[1] / glyph.shape[0] for glyph in (first, second)]
    if not .056 <= min(ratios) <= max(ratios) <= .24 or max(ratios) > 2.25 * min(ratios):
        return False
    profiles = [cv2.resize(np.float32(glyph.mean(axis=1)), (1, 32), interpolation=cv2.INTER_AREA)[:, 0] >= .22
                for glyph in (first, second)]
    a, b = profiles
    return len(_bands(a)) == len(_bands(b)) in (1, 2) and (a & b).sum() >= .82 * (a | b).sum()


def _same_word(first, second):
    if not len(first) <= len(second) <= len(first) + 5:
        return False
    return all(_glyph_matches(a, b, minimum_overlap=.655) or _same_narrow_glyph(a, b)
               for a, b in zip(first, second))


def _row_mentions_query(ink, query, depth):
    for a, b in _bands(ink.sum(axis=1) >= max(2, round(.032 * depth))):
        height = b - a
        if not .16 * depth <= height <= .59 * depth:
            continue
        line = ink[a:b]
        words = []
        for x, r in _bands(line.any(axis=0)):
            if words and x - words[-1][1] <= .285 * height:
                words[-1] = (words[-1][0], r)
            else:
                words.append((x, r))
        if any(.52 <= (r - x) / height <= 13 and _same_word(query, _word_glyphs(line[:, x:r])) for x, r in words):
            return True
    return False


def _linked_results(frame, field, query_delta, paint):
    left, top, _, bottom = field
    height, width = frame.shape[:2]
    depth = bottom - top
    query = _word_glyphs(query_delta >= 48)
    if not 4 <= len(query) <= 18 or sum(g.shape[1] >= .25 * g.shape[0] for g in query) < 3:
        return False
    start, stop = bottom + round(.27 * depth), min(round(.81 * height), bottom + round(12.3 * depth))
    x, r = max(0, left - round(.09 * depth)), left + round(1.08 * depth)
    rail = _distance(frame[start:stop, x:r], paint) >= 49
    if rail.mean() >= .48:
        return False
    rows = []
    for a, b in _bands(rail.mean(axis=1) >= .16):
        if rows and a - rows[-1][1] <= .095 * depth:
            rows[-1] = (rows[-1][0], b)
        else:
            rows.append((a, b))
    items = [(a, b) for a, b in rows if .54 * depth <= b - a <= 1.22 * depth and rail[a:b].mean() >= .32]
    if len(items) < 3 or items[0][0] > 2.45 * depth:
        return False
    text_x, text_right = r + round(.06 * depth), width - round(.058 * width)
    votes = 0
    for i, (a, b) in enumerate(items[:8]):
        lo = start + max(0, a - round(.21 * depth))
        following = items[i + 1][0] if i + 1 < len(items) else stop - start
        hi = start + min(following - round(.09 * depth), b + round(.29 * depth))
        contrast = _distance(frame[lo:hi, text_x:text_right], paint)
        candidates = (contrast >= cutoff for cutoff in (58, 40))
        if any(ink.mean() <= .24 and _row_mentions_query(ink, query, depth) for ink in candidates):
            votes += 1
            if votes == 2:
                return True
    return False


def _mark_action(field, mark, role, name):
    left, top, _, bottom = field
    pad = round(.13 * (bottom - top))
    x, y, r, b = mark
    return {"role": role, "function": name, "preview_text": name,
            "box": [left + x - pad, top + y - pad, left + r + pad, top + b + pad],
            "click_point": [left + (x + r) // 2, top + (y + b) // 2]}


def retained_search_focus_actions(frame, regions):
    """Read a retained focus toolbar once the query matches independent list results."""
    height, width = frame.shape[:2]
    if regions.get("layer") != "page" or width < 265 or not 1.53 * width <= height <= 2.85 * width:
        return []
    readings = []
    for field, delta, paint in _search_surfaces(frame):
        marks = _end_marks(delta)
        if marks is None:
            continue
        arrow, clear = marks
        edit = _focused_input(delta, arrow[2], clear[0])
        if edit is None:
            continue
        owner, (x, y, r, b) = edit
        if _linked_results(frame, field, delta[y:b, x:r], paint):
            readings.append([_mark_action(field, arrow, "back", "返回"),
                             _action_in_field(field, owner),
                             _mark_action(field, clear, "clear_input", "清除搜索内容")])
    return [action for action in readings[0] if _available(action["box"], regions)] if len(readings) == 1 else []
