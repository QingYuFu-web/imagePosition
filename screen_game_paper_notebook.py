"""Read owned controls of a landscape illustrated notebook page."""

import cv2
import numpy as np

from inventory_setting_switches import _stadium_mask
from screen_caption_transcript import _cjk_parts
from screen_component_captions import _bands
from screen_landscape_settings_rail import _literal_word


_TITLES = ("友情手账", "友情手帐", "好友手账", "好友笔记", "社交手册", "心情日记", "冒险手册", "游戏设置")
_TABS = ("友友特报", "邀请好友", "好友列表", "好友动态", "发现好友", "通讯录", "添加好友")
_PURPOSES = {"不感兴趣": "dismiss_friend_suggestion", "好友申请": "friend_request",
             "添加好友": "friend_request", "发送邀请": "friend_request"}
_OTHER_CAPTIONS = ("好友推荐", "申请失败", "已经发送", "已经邀请", "取消申请", "好友请求", "查看好友", "好友已满")


def _center(box):
    l, t, r, b = box
    return (l + r) // 2, (t + b) // 2


def _action(role, purpose, box):
    return {"role": role, "function": purpose, "box": list(box), "click_point": list(_center(box))}


def _color_planes(image):
    b, g, r = (image[..., channel] for channel in range(3))
    return np.minimum(np.minimum(b, g), r), np.maximum(np.maximum(b, g), r)


def _jigsaw_rows(paper, box, span):
    left, top, right, bottom = box
    start, stop = left + round(.075 * span), right - round(.19 * span)
    check = paper[:, start:stop]
    if check.shape[1] < .35 * span:
        return None
    for row in range(top, min(bottom, top + round(.133 * span))):
        gaps = [(x + r) / 2 for x, r in _bands(check[row] == 0)
                if 0 < x < r < check.shape[1] and .012 * span <= r - x <= .064 * span]
        if len(gaps) < 5 or check[row].mean() < .59:
            continue
        tail = check[row + round(.024 * span):row + round(.040 * span)]
        steps = np.diff(gaps)
        for start_at in range(len(steps) - 3):
            run = steps[start_at:start_at + 4]
            pitch = np.median(run)
            a = round(gaps[start_at] - .20 * pitch)
            b = round(gaps[start_at + 4] + .20 * pitch)
            if (.086 * span <= pitch <= .173 * span and np.ptp(run) <= .14 * pitch
                    and len(tail) and tail[:, max(0, a):min(tail.shape[1], b)].mean() >= .93):
                return row
    return None


def _paper_panel(frame):
    height, width = frame.shape[:2]
    factor = min(1., 520 / height)
    reduced = cv2.resize(frame, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)
    small_h, small_w = reduced.shape[:2]
    small = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    low, high = _color_planes(reduced)
    paper = np.uint8((small >= 218) & (high - low <= 41))
    _, _, stats, _ = cv2.connectedComponentsWithStats(paper)
    components = [(x, y, x + w, y + h, n / (w * h)) for x, y, w, h, n in stats[1:]
             if .49 * small_h <= h <= .87 * small_h and .36 * small_h <= w <= 1.56 * small_h
             and .110 * small_h <= y <= .28 * small_h and y + h >= .83 * small_h and n >= .20 * w * h]
    panes = [row[:4] for row in components if row[4] >= .38]
    siblings = list(panes)
    # The neighboring card can be covered by the bezel or a fixture. Its
    # lower paper rim can still establish the second surface and gutter.
    for left, top, right, bottom, density in components:
        if density >= .38:
            continue
        inset = round(.02 * small_h)
        rim = paper[bottom - inset:bottom - max(2, inset // 3), left + inset:right - inset]
        if rim.size and rim.mean() >= .83:
            siblings.append((left, top, right, bottom))
    matches = []
    for p in panes:
        x, y, r, b = p
        if not .29 * small_h <= x <= .69 * small_h or not .78 * small_h <= r - x <= 1.46 * small_h:
            continue
        rim = _jigsaw_rows(paper, p, small_h)
        if rim is None or not any(q[0] > r and .018 * small_h <= q[0] - r <= .13 * small_h
                                  and abs(q[3] - b) <= .037 * small_h and abs(q[1] - rim) <= .077 * small_h
                                  for q in siblings):
            continue
        matches.append((round(x * width / small_w), round(rim * height / small_h),
                        round(r * width / small_w), round(b * height / small_h)))
    return matches[0] if len(matches) == 1 else None


def _pills(mask, minimum, maximum, ratio):
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not minimum <= h <= maximum or not ratio[0] <= w / h <= ratio[1]:
            continue
        area = np.zeros((h, w), np.uint8)
        cv2.drawContours(area, [c], -1, 1, cv2.FILLED, offset=(-x, -y))
        ideal = _stadium_mask(h, w)
        if np.count_nonzero(area & ideal) >= .90 * np.count_nonzero(area | ideal):
            out.append((x, y, x + w, y + h))
    return out


def _caption_box(box, horizontal=.16):
    l, t, r, b = box
    dx, dy = round(horizontal * (r - l)), round(.16 * (b - t))
    return l + dx, t + dy, r - dx, b - dy


def _bright_word(frame, box, words, index):
    left, top, right, bottom = box
    tile = frame[top:bottom, left:right]
    reading = _bright_cuts(tile, words, index)
    if reading is not None or 52 <= len(tile) <= 62:
        return reading
    scale = 58 / len(tile)
    sample = cv2.resize(tile, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return _bright_cuts(sample, words, index)


def _bright_cuts(tile, words, index):
    low, high = _color_planes(tile)
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    votes = {}
    for plane, cuts in ((gray, (.71, .75, .79, .65, .82)), (low, (.41, .51, .73)), (high, (.52, .61, .73, .82))):
        base, peak = np.percentile(plane, (50, 98))
        if peak - base < 37:
            continue
        for cut in cuts:
            ink = np.uint8(plane >= base + cut * (peak - base))
            x, y, w, h = cv2.boundingRect(ink)
            if not .36 * len(tile) <= h <= .79 * len(tile) or not 3.18 * h <= w <= 5.64 * h:
                continue
            parts = _cjk_parts(ink[y:y + h, x:x + w], h, join_radicals=True)
            guesses = {word for word in words if len(word) == len(parts)}
            margins = []
            for column, (_, _, _, _, glyph) in enumerate(parts):
                if not guesses:
                    break
                shapes = index.rank(glyph, alphabet="cjk", limit=2)
                if len(shapes) < 2 or shapes[0]["distance"] > .196:
                    guesses.clear()
                    break
                guesses = {word for word in guesses if word[column] == shapes[0]["text"]}
                margins.append(shapes[1]["distance"] - shapes[0]["distance"])
            if len(guesses) == 1 and np.mean(margins) >= .024:
                word = guesses.pop()
                key = (ink.shape, ink.tobytes())
                votes.setdefault(word, set()).add(key)
                if len(votes[word]) >= 2:
                    return word
    return None


def _selected_and_inactive(frame, gray, panel, index):
    height = gray.shape[0]
    left, top, _, _ = panel
    margin = round(.055 * height)
    y0, y1 = max(0, top - margin), top + round(.30 * height)
    tray = gray[y0:y1, :left - round(.027 * height)]
    colors = frame[y0:y1, :tray.shape[1]]
    low, high = _color_planes(colors)
    bright = _pills((tray >= 222) & (high - low <= 24), .083 * height, .141 * height, (2.15, 3.22))
    bright = [p for p in bright if abs(p[1] + y0 - top) <= .039 * height]
    if len(bright) != 1:
        return None
    x, y, r, b = bright[0]

    def inactive(base):
        return [p for p in _pills(tray <= base - 17, .079 * height, .132 * height, (2.12, 3.22))
                if .011 * height <= p[1] - b <= .061 * height and abs((x + r) - (p[0] + p[2])) <= .041 * height
                and .77 <= (p[2] - p[0]) / (r - x) <= 1.12]

    muted = inactive(np.median(tray[:, :max(2, x - round(.024 * height)):4]))
    if not muted:
        shoulder = tray[:, r + round(.006 * height):]
        if shoulder.shape[1] >= .011 * height:
            base = np.median(shoulder)
            if (np.abs(shoulder.astype(np.int16) - base) <= 13).mean() >= .64:
                muted = inactive(base)
    if len(muted) != 1:
        return None
    selected = x, y0 + y, r, y0 + b
    a, t, end, stop = muted[0]
    inactive = a, y0 + t, end, y0 + stop
    chip = _caption_box(selected, .17)
    seen = _literal_word(255 - gray, chip, _TABS, index)
    if seen is None:
        l, t, r, b = chip
        scale = min(1., 57 / (b - t))
        if scale < 1:
            sample = cv2.resize(255 - gray[t:b, l:r], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            seen = _literal_word(sample, (0, 0, sample.shape[1], sample.shape[0]), _TABS, index)
    if seen is None:
        return None
    read = _bright_word(frame, _caption_box(inactive, .15), _TABS, index)
    if read is None:
        printed = _literal_word(gray, _caption_box(inactive, .11), _TABS + _OTHER_CAPTIONS, index)
        if printed and printed[0] in _TABS:
            read = printed[0]
    if read is None or read == seen[0]:
        return selected, None
    return selected, _action("switch_tab", f"切换到{read}", inactive)


def _footer(frame, panel, index):
    left, _, right, bottom = panel
    height = frame.shape[0]
    x0 = left + round(.405 * (right - left))
    y0 = bottom - round(.162 * height)
    tray = frame[y0:bottom, x0:right - round(.018 * height)]
    low, high = _color_planes(tray)
    pills = _pills((high - low >= 20) & (high >= 115), .057 * height, .106 * height, (2.35, 3.68))
    pairs = [(a, b) for i, a in enumerate(pills) for b in pills[i + 1:]
             if abs(_center(a)[1] - _center(b)[1]) <= .011 * height and .23 * height <= abs(_center(a)[0] - _center(b)[0])
             <= .41 * height and abs((a[2] - a[0]) - (b[2] - b[0])) <= .032 * height
             and min(bottom - (a[3] + y0), bottom - (b[3] + y0)) >= .012 * height]
    if len(pairs) != 1:
        return []
    actions = []
    for x, y, right, b in sorted(pairs[0]):
        pill = x0 + x, y0 + y, x0 + right, y0 + b
        text = _bright_word(frame, _caption_box(pill, .143), _PURPOSES, index)
        if text is None:
            l, t, r, b = _caption_box(pill, .16)
            gray = cv2.cvtColor(frame[t:b, l:r], cv2.COLOR_BGR2GRAY)
            printed = _literal_word(gray, (0, 0, r - l, b - t), tuple(_PURPOSES) + _OTHER_CAPTIONS, index)
            if printed and printed[0] in _PURPOSES:
                text = printed[0]
        if text is not None:
            actions.append(_action(_PURPOSES[text], text, pill))
    return actions


def _back(frame, gray, panel, selected, index):
    height, width = gray.shape
    xstop = min(selected[0] + round(.067 * height), panel[0])
    ystop = selected[1] - round(.028 * height)
    if min(ystop, xstop) < .12 * height:
        return None
    tray = gray[:ystop, :xstop]
    contours, _ = cv2.findContours(np.uint8(tray >= 221), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    matches = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (.070 * height <= min(w, h) <= max(w, h) <= .122 * height and .84 <= w / h <= 1.19
                and .66 * w * h <= cv2.contourArea(c) <= .84 * w * h):
            continue
        d = round(.19 * min(w, h))
        chip = tray[y + d:y + h - d, x + d:x + w - d]
        ink = np.uint8(chip <= np.median(chip) - 53)
        ix, iy, iw, ih = cv2.boundingRect(ink)
        if not .28 * h <= ih <= .66 * h or not .53 <= iw / ih <= 1.24:
            continue
        mark = ink[iy:iy + ih, ix:ix + iw]
        third = max(1, iw // 3)
        first, last = mark[:, :third], mark[:, -third:]
        unfilled = .33 * ih
        if (not .36 <= mark.mean() <= .82 or not .43 <= last.mean() <= .96
                or last.mean() - first.mean() < .06 or not first[round(unfilled):round(2 * unfilled)].any()
                or first[:round(.14 * ih)].any() or first[-round(.14 * ih):].any()):
            continue
        title = gray[max(0, y - round(.03 * h)):min(ystop, y + h + round(.03 * h)),
                     x + w + round(.14 * h):min(width, x + w + round(3.10 * h))]
        if title.size == 0:
            continue
        factor = min(1., 75 / h)
        label = cv2.resize(title, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        if _literal_word(label, (0, 0, label.shape[1], label.shape[0]), _TITLES, index) is not None:
            matches.append((x, y, x + w, y + h))
    return _action("back", "返回上一级", matches[0]) if len(matches) == 1 else None


def game_paper_notebook_actions(frame, regions, font_index):
    """Recognize independently measured controls and abstain on masked scenes."""
    height, width = frame.shape[:2]
    if (font_index is None or regions.get("layer") != "page" or regions.get("regions")
            or height < 280 or not 1.54 * height <= width <= 2.82 * height):
        return []
    panel = _paper_panel(frame)
    if panel is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tabs = _selected_and_inactive(frame, gray, panel, font_index)
    if tabs is None:
        return []
    selected, switch = tabs
    actions = [switch] if switch is not None else []
    actions.extend(_footer(frame, panel, font_index))
    if back := _back(frame, gray, panel, selected, font_index):
        actions.append(back)
    return actions
