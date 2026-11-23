"""Find outlined native input fields with visibly attached captions."""

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_caption_transcript import _bands, _layout, _metrics, read_caption
from screen_dialog_close import _cross_footprint
from screen_landscape_search_field import _available
from screen_native_tab_bar import _distance


def _outlines(ink, width):
    strokes = np.uint8(ink >= 63)
    contours = cv2.findContours(strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    fields = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (not .77 * width <= w <= .965 * width or not .095 * width <= h <= .51 * width
                or x < .024 * width or abs(x - (width - x - w)) > .044 * width
                or cv2.contourArea(contour) < .93 * w * h):
            continue
        a, b, c, d = x, y, x + w, y + h
        n, gap = max(2, round(.007 * width)), max(5, round(.020 * width))
        horizontal = (strokes[b:b + n, a + gap:c - gap], strokes[d - n:d, a + gap:c - gap])
        vertical = (strokes[b + gap:d - gap, a:a + n], strokes[b + gap:d - gap, c - n:c])
        if min(part.any(axis=0).mean() for part in horizontal) < .94:
            continue
        if min(part.any(axis=1).mean() for part in vertical) < .91:
            continue
        fields.append((a, b, c, d))
    return sorted(fields, key=lambda field: field[1])


def _editor_header(frame, ink, paint, first):
    width = frame.shape[1]
    limit = first[1] - round(.067 * width)
    start = max(0, first[1] - round(.267 * width))
    reach = round(.109 * width)
    rail = np.uint8(ink[start:limit, :reach] >= 73)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(rail, connectivity=8)
    for k, (x, y, w, h, _) in enumerate(stats[1:], 1):
        if not (.025 * width <= h <= .053 * width and .73 <= w / h <= 1.26
                and .027 * width <= x <= .085 * width and y > 0 and y + h < len(rail)
                and is_shaft_back_arrow(labels[y:y + h, x:x + w] == k)):
            continue
        y += start
        top, bottom = max(0, y - round(.38 * h)), y + round(1.37 * h)
        left_title = ink[top:bottom, x + w + round(.52 * h):round(.37 * width)] >= 73
        right_start = round(.76 * width)
        right = frame[top:bottom, right_start:width - round(.032 * width)]
        accent = (_distance(right, paint) >= 78) & (np.ptp(right, axis=2) >= np.ptp(paint) + 23)
        title = cv2.boundingRect(np.uint8(left_title))
        button = cv2.boundingRect(np.uint8(accent))
        if (.65 * h <= title[3] <= 1.31 * h and 1.26 * h <= title[2] <= 6.2 * h
                and .49 * h <= button[3] <= 1.14 * h and 1.22 * h <= button[2] <= 5.6 * h
                and abs(title[1] + title[3] / 2 - button[1] - button[3] / 2) <= .30 * h
                and width - right_start - button[0] - button[2] <= .10 * width):
            a, b, c, d = map(int, button)
            return (tuple(map(int, (x, y, x + w, y + h))),
                    (right_start + a, int(top) + b, right_start + a + c, int(top) + b + d))
    return None


def _caption_box(ink, field):
    left, top, right, _ = field
    w = right - left
    y0, y1 = top - round(.093 * w), top - round(.020 * w)
    x1 = left + round(.56 * w)
    mask = ink[y0:y1, left:x1] >= 45
    x, y, width, height = cv2.boundingRect(np.uint8(mask))
    if (not .013 * w <= height <= .042 * w or not 1.7 * height <= width <= .51 * w
            or x > .046 * w or not .020 * w <= top - (y0 + y + height) <= .066 * w
            or len(_bands(mask.any(axis=0))) < 3):
        return None
    return max(left, x + left - 3), y + y0 - 2, x + left + width + 3, y + y0 + height + 2


def _caption_cut(cut, index, paths):
    x, y, w, h = cv2.boundingRect(cut)
    if not 9 <= h <= 27:
        return None
    cut = cut[y:y + h, x:x + w]
    parts, rankings = [], []
    for l, r in _bands(cut.any(axis=0)):
        dx, top, width, height = cv2.boundingRect(cut[:, l:r])
        if not .48 * h <= height <= h or not .068 * h <= width <= 1.32 * h:
            return None
        glyph = cut[top:top + height, l + dx:l + dx + width]
        ranks = index.rank(glyph, alphabet="choice_latin", limit=8)
        if len(ranks) < 2 or any(row.get("font") not in paths for row in ranks):
            return None
        parts.append((l, top, r, top + height, glyph))
        rankings.append(ranks)
    if not 3 <= len(parts) <= 32 or not (layout := _layout(parts, rankings, paths, h)):
        return None
    scale, baseline = layout
    gaps = [b[0] - a[2] for a, b in zip(parts[:-1], parts[1:], strict=True)]
    space = max(.25 * h, np.median(gaps) + .125 * h)
    letters = []
    for position, ((left, top, right, bottom, _), ranks) in enumerate(zip(parts, rankings, strict=True)):
        scores = []
        for rank in ranks:
            char = rank["text"]
            a, b = _metrics(paths[rank["font"]], char)
            penalty = .49 * (abs(top - baseline - a * scale) + abs(bottom - baseline - b * scale)) / h
            scores.append((rank["distance"] + penalty, char))
        scores.sort()
        best, char = scores[0]
        other, alternate = scores[1]
        if position and gaps[position - 1] >= space:
            letters.append(" ")
        if {char, alternate} == {"I", "l"} and right - left <= .19 * h and other - best < .034:
            char = "|"
            other = scores[2][0] if len(scores) > 2 else 0
        if best > .231 or other - best < .024 or not (char.isascii() and (char.isalpha() or char == "|")):
            return None
        letters.append(char)
    words = "".join(letters).split()
    if any("|" in word and (word[0] == "|" or sum(c.islower() for c in word) < 3
                            or any(c.isupper() for c in word[1:])) for word in words):
        return None
    return " ".join(words).replace("|", "l"), cut.tobytes()


def _caption(frame, ink, box, index, cache):
    l, t, r, b = box
    glyphs = ink[t:b, l:r]
    peak = float(np.percentile(glyphs, 97))
    if peak < 64:
        return None
    sources = index.fonts if index.fonts is not None else index.choice_latin_sources
    paths = {path.name: path for path in sources}
    reports = {}
    for fraction in (.179, .213, .246, .278, .313, .358):
        reading = _caption_cut(np.uint8(glyphs >= peak * fraction), index, paths)
        if reading:
            label, raster = reading
            reports.setdefault(label, set()).add(raster)
    matches = [label for label, votes in reports.items() if len(votes) >= 2]
    if len(matches) == 1:
        return matches[0]
    if not matches and (standard := read_caption(frame, box, index, cache=cache, choice=True)):
        return standard["text"]
    return None


def _typed_rows(ink, field):
    l, t, r, b = field
    w, h = r - l, b - t
    dx, dy = round(.024 * w), round(.074 * h)
    sample = ink[t + dy:b - dy, l + dx:r - round(.16 * w)]
    if (sample < 17).mean() < .86:
        return []
    return [(a + t + dy, z + t + dy) for a, z in _bands((sample >= 66).sum(axis=1) >= max(3, round(.008 * w)))
            if .017 * w <= z - a <= .063 * w]


def _clear_icon(ink, field):
    l, t, r, b = field
    w, h = r - l, b - t
    center = (t + b) / 2
    x0, y0 = r - round(.128 * w), round(center - .078 * w)
    right, bottom = r - round(.017 * w), round(center + .078 * w)
    patch = np.uint8(ink[y0:bottom, x0:right] >= 66)
    if not patch.size or patch.mean() >= .16:
        return None
    contours, tree = cv2.findContours(patch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if tree is None:
        return None
    marks = []
    for i, outer in enumerate(contours):
        x, y, a, d = cv2.boundingRect(outer)
        child = tree[0, i, 2]
        if (child < 0 or tree[0, i, 3] >= 0 or not .035 * w <= a <= .077 * w
                or not .84 <= a / d <= 1.19 or not .65 <= cv2.contourArea(outer) / (a * d) <= .82
                or abs(y0 + y + d / 2 - center) > .029 * w):
            continue
        u, v, e, f = cv2.boundingRect(contours[child])
        if not (.68 * a <= e <= .94 * a and .68 * d <= f <= .94 * d):
            continue
        inner = cv2.erode(cv2.drawContours(np.zeros_like(patch), contours, child, 1, cv2.FILLED),
                          np.ones((3, 3), dtype=np.uint8)) & patch
        gx, gy, gw, gh = cv2.boundingRect(inner)
        if (.40 * a <= gw <= .68 * a and .40 * d <= gh <= .68 * d
                and abs(gx + gw / 2 - x - a / 2) <= .093 * a and abs(gy + gh / 2 - y - d / 2) <= .093 * d
                and _cross_footprint(inner[gy:gy + gh, gx:gx + gw].astype(bool))):
            pad = round(.40 * min(a, d))
            hit = [max(l, x0 + x - pad), max(t, y0 + y - pad), min(r, x0 + x + a + pad), min(b, y0 + y + d + pad)]
            marks.append((hit, [x0 + x + a // 2, y0 + y + d // 2]))
    return marks[0] if len(marks) == 1 else None


def _editor_geometry(frame):
    height, width = frame.shape[:2]
    if width < 255 or height < 1.58 * width:
        return None
    limit = round(.56 * height)
    paint = np.median(frame[round(.075 * height):limit:5, ::5], axis=(0, 1))
    if 77 < np.mean(paint) < 193:
        return None
    ink = _distance(frame[:limit], paint)
    if (ink[::4, ::4] <= 13).mean() < .83:
        return None
    fields = _outlines(ink, width)
    if not 2 <= len(fields) <= 6:
        return None
    first = fields[0]
    if not .195 * width <= first[1] <= .50 * width or not (header := _editor_header(frame, ink, paint, first)):
        return None
    entries = []
    for k, field in enumerate(fields):
        l, t, r, _ = field
        if (abs(l - first[0]) > .016 * width or abs(r - first[2]) > .016 * width
                or (k and not .068 * width <= t - fields[k - 1][3] <= .174 * width)):
            return None
        caption = _caption_box(ink, field)
        rows = _typed_rows(ink, field)
        if caption and rows:
            entries.append((field, caption, rows))
    return (ink, entries, header) if len(entries) >= 2 else None


def outlined_editor_fields(frame):
    """Return physical field bounds on a verified editor without reading text."""
    geometry = _editor_geometry(frame)
    return [field for field, _, _ in geometry[1]] if geometry else []


def _toolbar_actions(frame, ink, header, regions, index, cache):
    back, trailing = header
    width = frame.shape[1]
    actions = []
    for role, glyph in (("back", back), ("toolbar_action", trailing)):
        l, t, r, b = glyph
        depth = b - t
        dx, dy = round(.61 * depth), round(.74 * depth)
        hit = [max(0, l - dx), max(0, t - dy), min(width, r + dx), b + dy]
        if not _available(hit, regions):
            continue
        label = "返回"
        if role == "toolbar_action":
            padding = max(2, round(.15 * depth))
            caption = (max(0, l - padding), max(0, t - padding), min(width, r + padding), b + padding)
            label = _caption(frame, ink, caption, index, cache)
            if not label:
                continue
        actions.append({"role": role, "function": label, "preview_text": label,
                        "box": hit, "click_point": [(l + r) // 2, (t + b) // 2]})
    return actions


def outlined_editor_actions(frame, regions, index):
    """Read a portrait editor's toolbar, outlined fields, and attached clear control."""
    if index is None or regions.get("layer") != "page":
        return []
    geometry = _editor_geometry(frame)
    if geometry is None:
        return []
    ink, entries, header = geometry
    cache = {}
    actions = _toolbar_actions(frame, ink, header, regions, index, cache)
    for field, caption, rows in entries:
        if not _available(field, regions) or not _available(caption, regions):
            continue
        label = _caption(frame, ink, caption, index, cache)
        if label is None:
            continue
        l, t, r, b = field
        actions.append({"role": "input", "function": f"编辑{label}", "preview_text": label, "box": list(field),
                        "click_point": [round(l + .53 * (r - l)), round(t + .53 * (b - t))]})
        if len(rows) >= 2 and (clear := _clear_icon(ink, field)):
            box, point = clear
            actions.append({"role": "clear_input", "function": "清空输入", "preview_text": label,
                            "box": box, "click_point": point})
    return actions
