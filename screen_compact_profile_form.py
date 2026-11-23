"""Find the name field and avatar chooser in a visibly framed profile form."""

import cv2
import numpy as np

from screen_component_captions import _bands


def _paper(gray, panel):
    l, t, r, b = panel
    w, h = r - l, b - t
    height, width = gray.shape
    if (not 0 < l < r < width or not 0 < t < b < height or min(w, h) < 106
            or not .52 * width <= w <= .94 * width or not .33 * w <= h <= .89 * w):
        return None
    pad = max(3, round(.027 * w))
    if min(l, t, width - r, height - b) <= pad:
        return None
    face = gray[t + pad:b - pad:2, l + pad:r - pad:2]
    tone = float(np.median(face))
    if tone < 183 or (abs(face.astype(np.int16) - tone) <= 13).mean() < .88:
        return None
    x1, x2 = l + round(.19 * w), r - round(.19 * w)
    y1, y2 = t + round(.17 * h), b - round(.15 * h)
    sides = (gray[y1:y2:2, l - pad], gray[y1:y2:2, r + pad],
             gray[t - pad, x1:x2:2], gray[b + pad, x1:x2:2])
    if any((side < tone - 47).mean() < .69 for side in sides):
        return None
    return tone


def _underlines(gray, panel, tone):
    left, top, right, bottom = panel
    w, h = right - left, bottom - top
    start, end = left + round(.18 * w), right - round(.036 * w)
    reach = max(2, round(.015 * h))
    found = []
    for y in range(top + round(.34 * h), top + round(.70 * h)):
        row = gray[y, start:end].astype(np.int16)
        around = np.minimum(gray[y - reach, start:end], gray[y + reach, start:end]).astype(np.int16)
        ink = (around - row >= 18) & (row <= tone - 22) & (row >= tone - 111)
        for a, b in _bands(ink):
            x1, x2 = start + a, start + b
            if (.46 * w <= x2 - x1 <= .77 * w and .18 * w <= x1 - left <= .37 * w
                    and .032 * w <= right - x2 <= .107 * w):
                found.append((x1, y, x2))
    if not found or np.ptp(found, axis=0)[1] > max(2, .026 * h):
        return None
    return max(found, key=lambda edge: int(tone - np.median(gray[edge[1], edge[0]:edge[2]])))


def _field_text(gray, line, tone, panel):
    a, y, b = line
    _, top, _, bottom = panel
    h = bottom - top
    width = b - a
    d = max(1, round(.018 * h))
    start = y - round(.19 * h)
    stop = y - d
    mark = np.uint8(gray[start:stop, a:b] <= tone - 66)
    x, t, w, height = cv2.boundingRect(mark)
    if (not .041 * h <= height <= .13 * h or not .67 <= w / height <= 15
            or x > .062 * width or not .011 * h <= y - (start + t + height) <= .092 * h
            or not .012 * width <= w <= .76 * width):
        return None
    if not 2 <= len(_bands(mark[t:t + height].any(axis=0))) <= 30:
        return None
    return a + x, start + t, a + x + w, start + t + height


def _triangle(gray, icon, line, tone):
    x, y, r, b = icon
    _, base, _ = line
    size = r - x
    reach = max(3, round(.60 * size))
    start, end = y + round(.69 * (b - y)), base + round(.24 * (b - y))
    chip = gray[start:end, r:r + reach]
    mark = np.uint8((chip <= tone - 13) & (chip >= tone - 75))
    amount, ids, stats, _ = cv2.connectedComponentsWithStats(mark, connectivity=8)
    for i in range(1, amount):
        a, b, w, h, area = stats[i]
        if (.19 * size <= min(w, h) <= max(w, h) <= .54 * size and .70 <= w / h <= 1.34
                and .38 <= area / (w * h) <= .73):
            mask = ids[b:b + h, a:a + w] == i
            yy, xx = np.ogrid[:h, :w]
            wedge = ((yy + .5) / h + (xx + .5) / w) >= .89
            if (mask & wedge).sum() >= .77 * (mask | wedge).sum():
                return (r + int(a), start + int(b), r + int(a + w), start + int(b + h))
    return None


def _avatar(frame, gray, line, tone, panel):
    l, t, _, b = panel
    left, base, _ = line
    h = b - t
    x0, x1 = l + round(.033 * (left - l)), left - max(4, round(.043 * h))
    y0, y1 = base - round(.285 * h), base - max(1, round(.013 * h))
    patch = frame[y0:y1, x0:x1]
    ink = np.uint8((np.ptp(patch, axis=2) >= 48)
                   & (abs(patch.astype(np.int16) - tone).max(axis=2) >= 54))
    _, _, stats, centers = cv2.connectedComponentsWithStats(ink, connectivity=8)
    candidates = []
    for i, (x, y, w, k, area) in enumerate(stats[1:], 1):
        if (not .105 * h <= min(w, k) <= max(w, k) <= .213 * h or not .83 <= w / k <= 1.22
                or not .29 <= area / (w * k) <= .58 or not 0 < x < x + w < ink.shape[1]
                or not 0 < y < y + k < ink.shape[0] or not .009 * h <= base - y0 - y - k <= .055 * h):
            continue
        cx = x + w / 2
        heads = [j for j, (a, b, q, s, mass) in enumerate(stats[1:], 1)
                 if j != i and .23 * w <= q <= .51 * w and .24 * k <= s <= .50 * k
                 and .60 <= mass / (q * s) <= .95 and abs(centers[j][0] - cx) <= .11 * w
                 and .14 * k <= b - y <= .33 * k and .41 * k <= b + s - y <= .63 * k]
        shoulders = ink[y + round(.66 * k):y + round(.86 * k), x + round(.27 * w):x + round(.73 * w)]
        if len(heads) != 1 or shoulders.mean() < .59:
            continue
        icon = tuple(map(int, (x + x0, y + y0, x + x0 + w, y + y0 + k)))
        chooser = _triangle(gray, icon, line, tone)
        if chooser:
            candidates.append((icon, chooser))
    return candidates[0] if len(candidates) == 1 else None


def compact_profile_form_actions(frame, panel):
    """Return two actions only when the input and avatar chooser prove one form."""
    if frame.shape[0] < 1.31 * frame.shape[1]:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tone = _paper(gray, panel)
    if tone is None or (line := _underlines(gray, panel, tone)) is None:
        return []
    text = _field_text(gray, line, tone, panel)
    if text is None or (avatar := _avatar(frame, gray, line, tone, panel)) is None:
        return []
    a, b, c, d = text
    x, y, r, z = avatar[0]
    _, _, mark_right, mark_bottom = avatar[1]
    return [
        {"role": "input", "function": "编辑用户名称", "preview_text": "编辑用户名称",
         "box": [line[0], b - 2, line[2], line[1] + 2], "click_point": [(a + c) // 2, (b + d) // 2]},
        {"role": "pick_photo", "function": "选择头像", "preview_text": "选择头像",
         "box": [x - 2, y - 2, max(r + 2, mark_right), max(z + 2, mark_bottom)],
         "click_point": [(x + r) // 2, (y + z) // 2]},
    ]


def complete_compact_profile(frame, panel, index):
    """Verify that a compact profile panel has only its title, fields, and two footer buttons."""
    if index is None:
        return None
    actions = compact_profile_form_actions(frame, panel)
    if not actions:
        return None
    from screen_compact_dialog_footer import compact_dialog_footer_actions

    footer = compact_dialog_footer_actions(frame, panel, index)
    if len(footer) != 2:
        return None
    left, top, right, bottom = panel
    w, h = right - left, bottom - top
    dx, dy = max(2, round(.035 * w)), max(2, round(.038 * h))
    x0, y0, x1, y1 = left + dx, top + dy, right - dx, bottom - dy
    patch = frame[y0:y1, x0:x1].astype(np.int16)
    color = np.median(patch[::2, ::2], axis=(0, 1))
    residual = np.uint8(abs(patch - color).max(axis=2) > 22)
    depth = round(.26 * h)
    reach = round(.67 * w)
    title = residual[:depth, :reach].copy()
    x, y, a, b = cv2.boundingRect(title)
    if (not .09 * w <= a <= .61 * w or not .043 * h <= b <= .133 * h
            or x > .078 * w or not 3 <= len(_bands(title[y:y + b].any(axis=0))) <= 25):
        return None
    residual[y:y + b, x:x + a] = 0
    for item in (*actions, *footer):
        l, t, r, btm = item["box"]
        residual[max(0, t - y0 - 2):btm - y0 + 2, max(0, l - x0 - 2):r - x0 + 2] = 0
    _, _, parts, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    if parts[1:, cv2.CC_STAT_AREA].max(initial=0) >= max(4, round(.00037 * w * h)):
        return None
    return actions + footer
