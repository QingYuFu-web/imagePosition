"""Verify short command buttons from their painted front layer and aligned peers."""

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_faint_commands import _measured
from screen_painted_commands import _EXPLICIT, _unclaimed
from screen_short_labels import _center, _text_key
from screen_rendered_labels import rank_rendered_label


_TERMS = {
    **_EXPLICIT, "gotologin": "登录", "login": "登录", "signin": "登录",
    "signupwithemail": "使用邮箱注册", "continuewithapple": "通过 Apple 继续",
    "continuewithgoogle": "通过 Google 继续", "continuewithfacebook": "通过 Facebook 继续",
    "continuewithmicrosoft": "通过 Microsoft 继续", "deny": "拒绝", "denyaccess": "拒绝",
    "dontallow": "不允许", "don'tallow": "不允许", "allow": "允许",
    "allowonlywhileusingtheapp": "仅在使用中允许", "whileusingtheapp": "仅在使用中允许",
    "allowonce": "仅此一次允许", "onlythistime": "仅此一次允许", "once": "仅此一次允许",
}
_CONTEXT_DEPENDENT = {"once"}
_PERMISSION_WORDS = {
    "deny", "denyaccess", "dontallow", "don'tallow", "allow", "allowonlywhileusingtheapp",
    "whileusingtheapp", "allowonce", "onlythistime",
}


def _surface_fill(frame, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    bottom_edge, right_edge = frame.shape[:2]
    dx, dy = max(3, round(.17 * h)), max(3, round(.17 * h))
    x0, y0 = max(0, left - dx), max(0, top - dy)
    patch = frame[y0:min(bottom_edge, bottom + dy), x0:min(right_edge, right + dx)]
    sample = frame[top + h // 3:bottom - h // 3:2,
                   left + round(.13 * w):left + round(.24 * w):3]
    if not sample.size:
        return None
    plane = np.median(sample, axis=(0, 1))
    mask = cv2.inRange(patch, np.uint8(np.maximum(0, plane - 6)), np.uint8(np.minimum(255, plane + 6)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals = []
    for contour in contours:
        a, b, sw, sh = cv2.boundingRect(contour)
        x, y, r, z = x0 + a, y0 + b, x0 + a + sw, y0 + b + sh
        if (not .84 * w <= sw <= 1.08 * w or not .79 * h <= sh <= 1.12 * h
                or cv2.contourArea(contour) < .82 * sw * sh
                or max(abs(x - left), abs(r - right), abs(y - top), abs(z - bottom)) > .20 * h):
            continue
        radius = max(3, round(.075 * sh))
        if min(x, y, right_edge - r, bottom_edge - z) < radius + 2:
            continue
        xs = np.arange(x + round(.25 * sw), r - round(.25 * sw), max(2, sh // 16))
        ys = np.arange(y + round(.38 * sh), z - round(.38 * sh), max(2, sh // 22))
        if min(xs.size, ys.size) < 3:
            continue
        outside = (frame[y - radius, xs], frame[z + radius, xs],
                   frame[ys, x - radius], frame[ys, r + radius])
        inside = (frame[y + radius, xs], frame[z - radius, xs],
                  frame[ys, x + radius], frame[ys, r - radius])
        if not all((np.abs(pixels - plane).max(axis=1) >= 10).mean() >= .87 for pixels in outside):
            continue
        if not all((np.abs(pixels - plane).max(axis=1) <= 6).mean() >= .87 for pixels in inside):
            continue
        proposals.append(((x, y, r, z), plane, "fill"))
    return proposals[0] if len(proposals) == 1 else None


def _outline_owner(frame, box):
    bounds = _measured(frame, box)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    w, h = right - left, bottom - top
    pad = max(4, round(.091 * h))
    if min(left, top, frame.shape[1] - right, frame.shape[0] - bottom) <= pad:
        return None
    x = np.arange(left + round(.25 * w), right - round(.25 * w), max(2, h // 17))
    y = np.arange(top + round(.34 * h), bottom - round(.34 * h), max(2, h // 25))
    if min(x.size, y.size) < 4:
        return None
    borders = (frame[top - pad:top + pad + 1, x], frame[bottom - pad:bottom + pad + 1, x],
               frame[y, left - pad:left + pad + 1].swapaxes(0, 1),
               frame[y, right - pad:right + pad + 1].swapaxes(0, 1))
    paints = [np.median(np.concatenate((line[:2], line[-2:])), axis=(0, 1)) for line in borders]
    for band, paint in zip(borders, paints, strict=True):
        if np.max(np.abs(paint - paints[0])) > 8:
            return None
        difference = np.abs(band - paint).max(axis=2)
        if (difference[[0, -1]] <= 7).mean() < .88:
            return None
        interior = difference[2:-2] >= 24
        if interior.any(axis=0).mean() < .87 or interior.sum(axis=0).mean() > .51 * len(band):
            return None
    return bounds, paints[0], "outline"


def _visible_label(frame, bounds, plane, index):
    left, top, right, bottom = bounds
    w, h = right - left, bottom - top
    x0, x1 = left + round(.095 * w), right - round(.095 * w)
    y0, y1 = top + round(.18 * h), bottom - round(.18 * h)
    tile = frame[y0:y1, x0:x1]
    if not tile.size:
        return None
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).astype(np.int16)
    paper = int(cv2.cvtColor(np.uint8(plane.reshape(1, 1, 3)), cv2.COLOR_BGR2GRAY)[0, 0])
    distance = np.abs(gray - paper)
    ink = np.uint8(distance >= 84)
    a, b, tw, th = cv2.boundingRect(ink)
    if (not .18 * h <= th <= .57 * h or not .055 * w <= tw <= .82 * w or
            abs(x0 + a + tw / 2 - (left + right) / 2) > .105 * w or
            abs(y0 + b + th / 2 - (top + bottom) / 2) > .135 * h):
        return None
    glyph = distance[b:b + th, a:a + tw]
    # A low-contrast label can be legible while its activation is uncertain.
    if (glyph >= 110).mean() < .155 or np.percentile(glyph, 86) < 118:
        return None
    spans = _bands(ink[b:b + th].any(axis=0))
    if not 2 <= len(spans) <= 35:
        return None
    start = a
    if len(spans) >= 7:
        first, end = spans[0]
        separator = spans[1][0] - end
        symbol = ink[b:b + th, first:end]
        _, _, _, symbol_height = cv2.boundingRect(symbol)
        if (.44 * th <= end - first <= 1.31 * th and symbol_height >= .65 * th
                and separator >= .39 * th and symbol.mean() >= .31):
            start = spans[1][0]
    caption = _matching_caption(distance[b:b + th, start:a + tw], index)
    if caption is None or _text_key(caption) not in _TERMS:
        return None
    if start != a and not _text_key(caption).startswith(("continuewith", "signupwith", "signinwith")):
        return None
    return caption


def _matching_caption(distance, index, *, maximum=.235):
    reads = [rank_rendered_label(np.uint8(distance >= level), index, limit=2) for level in (74, 98)]
    if not all(len(ranked) == 2 and ranked[0][1] <= maximum and ranked[1][1] - ranked[0][1] >= .08
               for ranked in reads):
        return None
    return reads[0][0][0] if reads[0][0][0] == reads[1][0][0] else None


def _similar_geometry(first, second):
    a, t, b, d = first
    l, y, r, z = second
    h = max(d - t, z - y)
    return (0.77 <= (d - t) / (z - y) <= 1.30
            and max(abs(a - l), abs(b - r)) <= .32 * h)


def _groups(boxes):
    identified = set()
    runs = []
    for a in sorted(boxes, key=lambda box: box[1]):
        if a in identified:
            continue
        matched = [a]
        end = a
        for b in sorted(boxes, key=lambda box: box[1]):
            if b in identified or b[1] <= end[1] or not _similar_geometry(a, b):
                continue
            h = end[3] - end[1]
            if -.09 * h <= b[1] - end[3] <= .43 * h:
                matched.append(b)
                end = b
        if 2 <= len(matched) <= 6:
            identified.update(matched)
            runs.append(tuple(matched))
    return runs


def scoped_row_actions(frame, regions, index, known=(), *, diagnostic=False):
    """Return scoped command names with separate layout and crisp-ink evidence."""
    height, width = frame.shape[:2]
    layer = regions.get("layer")
    panels = regions.get("regions")
    if index is None or layer not in {"dialog", "bottom_sheet"} or not panels or len(panels) > 2:
        return []
    visible = panels[0]
    if visible[2] - visible[0] < .57 * width or not 1.43 * width <= height <= 2.94 * width:
        return []
    boxes = []
    for row in regions["candidates"]:
        l, t, r, b = box = tuple(row["box"])
        if (row.get("visibility") != "foreground" or row.get("score", 0) < .61 or
                not (visible[0] <= l < r <= visible[2] and visible[1] <= t < b <= visible[3]) or
                not .49 * width <= r - l <= .97 * width or
                not .069 * width <= b - t <= .18 * width or not 3.8 <= (r - l) / (b - t) <= 11 or
                (l + r) / 2 < .38 * width or (l + r) / 2 > .62 * width):
            continue
        boxes.append(box)
    result = []
    for group in _groups(boxes):
        witnesses = []
        outline_count = 0
        for box in group:
            fit = _surface_fill(frame, box) or _outline_owner(frame, box)
            if not fit:
                continue
            owner, plane, kind = fit
            px, py = _center(owner)
            if any(l <= px < r and t <= py < b for l, t, r, b in regions.get("floating_controls", ())):
                continue
            if caption := _visible_label(frame, owner, plane, index):
                witnesses.append((owner, caption, kind))
                outline_count += kind == "outline"
        keys = [_text_key(caption) for _, caption, _ in witnesses]
        if len(witnesses) < 2 or len(set(keys)) != len(keys) or outline_count == len(witnesses):
            continue
        for (box, caption, kind), key in zip(witnesses, keys, strict=True):
            if key in _CONTEXT_DEPENDENT and not (_PERMISSION_WORDS & set(keys)):
                continue
            point = _center(box)
            if not _unclaimed(box, point, (*known, *result)):
                continue
            label = _TERMS[key]
            action = {"role": "labeled_control", "function": label,
                      "box": [int(v) for v in box], "click_point": point,
                      "text": caption, "preview_text": caption}
            if diagnostic:
                action["witness"] = kind
            result.append(action)
    return result


def _raised_sheet_edge(frame):
    """Locate a uniform near-white sheet crossing a neutral dimmed backdrop."""
    h, w = frame.shape[:2]
    if not 1.7 * w <= h <= 2.64 * w:
        return None
    start, end = round(.49 * h), round(.84 * h)
    xs = np.arange(round(.08 * w), round(.92 * w), max(2, w // 130))
    sampled = cv2.cvtColor(frame[start:end, xs], cv2.COLOR_BGR2GRAY).astype(np.int16)
    lead = (sampled[3:] - sampled[:-3] >= 23).mean(axis=1) >= .92
    edges = _bands(lead)
    if len(edges) != 1:
        return None
    top = start + edges[0][1]
    size = max(3, round(.0055 * w))
    prior = frame[top - 2 * size:top - size, xs]
    inside = frame[top + size:top + 2 * size, xs]
    dark = np.median(prior, axis=(0, 1))
    light = np.median(inside, axis=(0, 1))
    if (min(light) < 247 or np.ptp(light) > 6 or not 174 <= min(dark) <= 227 or np.ptp(dark) > 8
            or (np.abs(prior - dark).max(axis=2) <= 6).mean() < .90
            or (np.abs(inside - light).max(axis=2) <= 4).mean() < .94):
        return None
    backdrop = frame[round(.075 * h):top - 3 * size:max(2, h // 200), ::max(2, w // 170)]
    if (np.abs(backdrop - dark).max(axis=2) <= 7).mean() < .46 or (backdrop.max(axis=2) > max(dark) + 13).mean() > .017:
        return None
    # Both rounded corners must expose the same backdrop at the edge, while
    # the same columns are filled by the sheet immediately below the radius.
    for corner in (frame[:, 0], frame[:, -1]):
        if (np.abs(corner[top + 1:top + size] - dark).max(axis=1) <= 7).mean() < .76:
            return None
        if (np.abs(corner[top + 4 * size:top + 7 * size] - light).max(axis=1) <= 6).mean() < .92:
            return None
    side = np.concatenate((frame[top + 7 * size:h - 2 * size:max(3, h // 160), :size],
                           frame[top + 7 * size:h - 2 * size:max(3, h // 160), -size:]), axis=1)
    if (np.abs(side - light).max(axis=2) <= 6).mean() < .90:
        return None
    return (0, top, w, h)


def verified_raised_sheet(frame, regions, index):
    """Scope previously unclassified proposals only after surface and command proof agree."""
    if regions.get("layer") != "page" or regions.get("regions") or index is None:
        return None
    height, width = frame.shape[:2]
    proposals = [row for row in regions.get("candidates", ()) if row.get("score", 0) >= .61
                 and .49 * width <= row["box"][2] - row["box"][0] <= .97 * width
                 and width * .06 <= row["box"][3] - row["box"][1] <= width * .19
                 and row["box"][1] > .49 * height]
    if len(proposals) < 2 or not _groups([tuple(row["box"]) for row in proposals]):
        return None
    panel = _raised_sheet_edge(frame)
    if panel is None:
        return None
    next_regions = {**regions, "layer": "bottom_sheet", "regions": [list(panel)],
                    "candidates": [{**row, "visibility": "foreground"} for row in regions["candidates"]
                                   if row["box"][1] >= panel[1]]}
    controls = scoped_row_actions(frame, next_regions, index, diagnostic=True)
    positively_painted = {row["witness"] for row in controls} == {"fill", "outline"}
    if not positively_painted or not {"取消", "关闭", "不允许", "拒绝"}.intersection(row["function"] for row in controls):
        return None
    return next_regions, [{key: value for key, value in row.items() if key != "witness"} for row in controls]
