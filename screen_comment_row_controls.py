"""Locate independently verified action links and reactions inside comment rows."""

from collections import defaultdict

import cv2
import numpy as np

from inventory_small_reaction_glyphs import _thumb_shape
from screen_caption_transcript import read_caption
from screen_comment_sheet_modal import _phrase
from screen_component_captions import _bands
from screen_landscape_search_field import _available
from screen_social_comment_surface import _active_comment_tab, _close, _heading


_REPLY = ("Reply", "Responder", "Répondre", "Antworten", "Trả lời", "回复", "回覆", "回復")
_EXPAND = ("View replies", "See replies", "View reply", "查看回复", "展开回复", "查看回覆", "展開回覆")


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _panel_tone(gray, panel):
    l, t, r, b = panel
    return float(np.median(gray[t:min(b, t + round(.22 * (r - l))), l:r][::2, ::5]))


def has_comment_sheet_header(frame, panel, font_index, *, working_view=None):
    """Confirm a visible comment title or its selected speech/count tab."""
    height, width = frame.shape[:2]
    l, t, r, b = panel
    if (font_index is None or not 0 <= l <= .07 * width or not .91 * width <= r <= width or
            not 0 <= t < b <= height or b - t < .38 * width):
        return False
    small = working_view[1] if working_view is not None else None
    if small is not None:
        sx, sy = small.shape[1] / width, small.shape[0] / height
        sample = tuple(round(v * k) for v, k in zip(panel, (sx, sy, sx, sy), strict=True))
        tone = _panel_tone(small, sample)
    else:
        stroke = cv2.cvtColor(frame[t:min(b, t + round(.22 * width)), l:r], cv2.COLOR_BGR2GRAY)
        tone = float(np.median(stroke[::2, ::5]))
    if 67 < tone < 212:
        return False
    if _heading(frame, t, font_index, dark=tone <= 67):
        return True
    if tone >= 212:
        gray = cv2.cvtColor(frame[:min(b, t + round(.28 * width))], cv2.COLOR_BGR2GRAY)
        close = _close(gray, t, tone)
        if close is not None:
            return (_active_comment_tab(gray, t, close, font_index)
                    or _heading(frame, t, font_index, dark=False, weak_close=close, faint_ink=True))
    return False


def _avatars(ink, panel):
    l, t, r, b = panel
    span = r - l
    start = t + round(.082 * span)
    end = b - round(.12 * span)
    tiles = ink[start:end, l:l + round(.178 * span)]
    contours, _ = cv2.findContours(np.uint8(tiles >= 18), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    owners = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (not .044 * span <= min(w, h) <= max(w, h) <= .132 * span
                or not .79 <= w / h <= 1.24 or not .014 * span <= x <= .176 * span
                or cv2.contourArea(contour) < .46 * w * h):
            continue
        text = ink[start + y:start + y + h + round(.060 * span), l + x + w + 3:l + round(.67 * span)]
        if text.size and ((text >= 54).sum(axis=1) >= .044 * span).any():
            owners.append((l + x, start + y, l + x + w, start + y + h))
    if len(owners) < 2:
        return []
    groups = [[b for b in owners if abs(_center(a)[0] - _center(b)[0]) <= .023 * span
               and abs(a[2] - a[0] - b[2] + b[0]) <= .030 * span] for a in owners]
    peers = max(groups, key=len)
    return sorted(peers, key=lambda box: box[1]) if len(peers) >= 2 else []


def _heart_shape(mask):
    h, w = mask.shape
    if not .86 <= w / h <= 1.46 or not .55 <= mask.mean() <= .88:
        return False
    symmetry = (mask & mask[:, ::-1]).sum() / (mask | mask[:, ::-1]).sum()
    tip = np.flatnonzero(mask[-1])
    cap = mask[:max(2, round(.27 * h))]
    middle = slice(round(.45 * w), max(round(.45 * w) + 1, round(.55 * w)))
    lobes = ((cap[:, :round(.44 * w)].sum(axis=1) >= .16 * w)
             & (cap[:, round(.56 * w):].sum(axis=1) >= .16 * w) & (~cap[:, middle].any(axis=1)))
    return (symmetry >= .71 and 0 < tip.size <= .40 * w and abs(tip.mean() + .5 - w / 2) <= .15 * w
            and lobes.any() and np.median(mask[round(.26 * h):round(.58 * h)].sum(axis=1)) >= .72 * w)


def _reactions(ink, panel):
    l, t, r, b = panel
    span = r - l
    left, top = l + round(.64 * span), t + round(.091 * span)
    crop = ink[top:b - round(.078 * span), left:r]
    found = defaultdict(list)
    for cut in (16, 23, 35, 51, 73, 103):
        contours, hierarchy = cv2.findContours(np.uint8(crop >= cut), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        links = hierarchy[0]
        for contour, (_, _, child, parent) in zip(contours, links, strict=True):
            x, y, w, h = cv2.boundingRect(contour)
            if (parent >= 0 or not .021 * span <= min(w, h) <= max(w, h) <= .074 * span
                    or not .76 <= w / h <= 1.46):
                continue
            mask = cv2.drawContours(np.zeros((h, w), np.uint8), [contour], 0, 1, cv2.FILLED, offset=(-x, -y)).astype(bool)
            pockets = []
            while child >= 0:
                if cv2.contourArea(contours[child]) >= .09 * w * h:
                    pockets.append(contours[child])
                child = links[child][0]
            if len(pockets) <= 1 and _heart_shape(mask):
                role = "heart"
            elif len(pockets) == 1 and .23 <= cv2.contourArea(pockets[0]) / (w * h) <= .68:
                up, down = _thumb_shape(mask), _thumb_shape(mask[::-1, ::-1])
                if up == down:
                    continue
                role = "thumb_up" if up else "thumb_down"
            else:
                continue
            found[role].append((cut, (left + x, top + y, left + x + w, top + y + h)))
    return {role: _stable_boxes(samples, max(2, round(.004 * span))) for role, samples in found.items()}


def _stable_boxes(samples, tolerance):
    def near(a, b):
        return max(abs(x - y) for x, y in zip(a, b, strict=True)) <= tolerance

    merged = []
    for cut, box in samples:
        if (any(other != cut and near(box, peer) for other, peer in samples)
                and not any(near(box, previous) for previous in merged)):
            merged.append(box)
    return merged


def _button(box, panel, role, function, preview):
    l, t, r, b = panel
    pad = max(2, round(.24 * (box[3] - box[1])))
    bounds = [max(l, box[0] - pad), max(t, box[1] - pad), min(r, box[2] + pad), min(b, box[3] + pad)]
    return {"role": role, "function": function, "box": bounds, "click_point": list(_center(box)), "preview_text": preview}


def _selected(frame, box):
    l, t, r, b = box
    blue, green, red = cv2.split(frame[t:b, l:r].astype(np.int16))
    return ((red >= 144) & (red - green >= 43) & (red - blue >= 25)).mean() >= .24


def _row_lanes(owners, reactions, panel):
    span = panel[2] - panel[0]
    stops = [b[1] for b in owners[1:]] + [panel[3] - round(.10 * span)]
    limits = [(a[1] - round(.019 * span), min(b - round(.013 * span), a[3] + round(.49 * span)))
              for a, b in zip(owners, stops, strict=True)]
    rows = [{} for _ in owners]
    for kind, boxes in reactions.items():
        fits = [[box for box in boxes if start <= _center(box)[1] < stop] for start, stop in limits]
        for i, marks in enumerate(fits):
            peers = [box for j, other in enumerate(fits) if j != i for box in other]
            linked = [box for box in marks if any(abs(_center(box)[0] - _center(peer)[0]) <= .043 * span
                      and abs((box[3] - box[1]) - (peer[3] - peer[1])) <= .020 * span for peer in peers)]
            if linked and kind == "thumb_down":
                rows[i][kind] = linked
            elif linked:
                rows[i][kind] = min(linked, key=lambda box: box[1])
    return limits, rows


def _words(row, em, gap_fraction):
    runs = _bands(row.any(axis=0))
    groups = []
    for start, stop in runs:
        if groups and start - groups[-1][1] <= gap_fraction * em:
            groups[-1] = groups[-1][0], stop
        else:
            groups.append((start, stop))
    return groups


def _expand_end(row, bands, end, em):
    """Include a nearby printed reply count when it ends with a down chevron."""
    following = [box for box in bands if end <= box[0] <= end + 6.5 * em]
    for x, stop in following:
        mark = np.uint8(row[:, x:stop])
        _, _, w, h = cv2.boundingRect(mark)
        if not .29 * em <= h <= .79 * em or not .64 * em <= w <= 1.40 * em or not .36 <= h / w <= .79:
            continue
        depth, width = np.nonzero(mark)
        center = (width >= .33 * w) & (width <= .67 * w)
        outside = (width <= .23 * w) | (width >= .77 * w)
        if center.any() and outside.any() and depth[center].mean() - depth[outside].mean() >= .25 * h:
            chained = [box for box in following if box[1] <= stop]
            if chained[0][0] - end <= .95 * em and all(b[0] - a[1] <= .95 * em
                       for a, b in zip(chained[:-1], chained[1:], strict=True)):
                return stop
    return end


def _word_similarity(first, second):
    if not .81 <= first.shape[1] * second.shape[0] / (first.shape[0] * second.shape[1]) <= 1.24:
        return 0.
    a, b = (cv2.resize(tile.astype(np.float32) / max(1, int(tile.max())), (42, 18),
                      interpolation=cv2.INTER_LINEAR) for tile in (first, second))
    return float(np.minimum(a, b).sum() / max(1., np.maximum(a, b).sum()))


def _read_reply_pair(frame, roi, em, index):
    """Read a compact two-character reply label on fonts with different outlines."""
    left, top, right, bottom = roi
    if not 1.55 * em <= right - left <= 2.66 * em + 2:
        return False
    horizontal, vertical = max(2, round(.56 * em)), max(2, round(.21 * em))
    area = (max(0, left - horizontal), max(0, top - vertical),
            min(frame.shape[1], right + horizontal), min(frame.shape[0], bottom + vertical))
    label = read_caption(frame, area, index, cache={}, choice=True)
    return bool(label and label["text"] in {"回复", "回覆", "回復"} and label["support"] >= 2
                and label["max_distance"] <= .18 and label["min_margin"] >= .028)


def _caption_links(frame, gray, ink, bounds, span, index, reaction_y, verified_words, *, expand=True):
    left, top, right, stop = bounds
    if index is None or stop <= top or right <= left:
        return {}
    tile = ink[top:stop, left:right]
    found = {}
    signal = tile >= 36
    for a, b in _bands(signal.sum(axis=1) >= max(2, round(.006 * span))):
        em = b - a
        if not max(5, .010 * span) <= em <= .052 * span:
            continue
        bands = _words(signal[a:b], em, .28)
        if not bands:
            continue
        for i, (x, end) in enumerate(bands[:6]):
            # A close timestamp or a proven reaction lane distinguishes Reply from body prose.
            reply_lane = ((i > 0 and x - bands[i - 1][1] <= 2.2 * em) or
                          (reaction_y is not None and abs(top + (a + b) / 2 - reaction_y) <= 1.30 * em))
            variants = []
            if reply_lane and "reply" not in found and 1.45 * em <= end - x <= 7.2 * em:
                variants.append(("reply", _REPLY, end))
                if i + 1 < len(bands) and .27 * em <= bands[i + 1][0] - end <= .80 * em:
                    if 2.7 * em <= bands[i + 1][1] - x <= 7.2 * em:
                        variants.append(("reply", ("Trả lời",), bands[i + 1][1]))
            if expand and i == 0 and "expand" not in found and x <= .055 * span:
                if 2.1 * em <= end - x <= 7.0 * em:
                    variants.append(("expand", _EXPAND, end))
                if len(bands) > 1 and bands[1][0] - end <= .95 * em and 3.1 * em <= bands[1][1] - x <= 9.1 * em:
                    variants.append(("expand", _EXPAND, bands[1][1]))
            for role, names, word_end in variants:
                pad = max(2, round(.28 * em))
                roi = (max(left, left + x - 1), max(0, top + a - pad), min(right, left + word_end + 1),
                       min(len(gray), top + b + pad))
                word = tile[a:b, x:word_end]
                if np.median(word[word >= 36]) > 165:
                    continue
                previous = verified_words.get(role)
                similarity = _word_similarity(word, previous) if previous is not None else .72
                aligned_reply = (role == "reply" and reaction_y is not None
                                 and abs(top + (a + b) / 2 - reaction_y) <= 1.5 * em)
                if similarity < .68 and not aligned_reply:
                    continue
                if (similarity < .86 and not _phrase(gray, roi, names, index)
                        and not (aligned_reply and _read_reply_pair(frame, roi, em, index))):
                    continue
                if previous is None:
                    verified_words[role] = word.copy()
                control_end = _expand_end(signal[a:b], bands, word_end, em) if role == "expand" else word_end
                found[role] = (left + x, top + a, left + control_end, top + b)
                break
    return found


def _links(frame, gray, ink, owner, limits, span, index, reaction_y, verified_words):
    left = owner[2] + max(2, round(.018 * span))
    top = owner[1] + round(.73 * (owner[3] - owner[1]))
    if reaction_y is not None:
        top = max(top, round(reaction_y - .045 * span))
    bounds = left, top, min(gray.shape[1], owner[0] + round(.56 * span)), min(limits[1], len(gray))
    return _caption_links(frame, gray, ink, bounds, span, index, reaction_y, verified_words)


def _repeated_heart_controls(frame, gray, ink, panel, hearts, existing, index, verified_words):
    """Recover reactions when thread indentation or redaction obscures avatars."""
    left, top, right, bottom = panel
    span = right - left
    groups = [[other for other in hearts if abs(_center(heart)[0] - _center(other)[0]) <= .044 * span]
              for heart in hearts]
    if not groups or len(peers := max(groups, key=len)) < 3:
        return []
    peers = sorted(peers, key=lambda box: box[1])
    centers = [_center(box) for box in peers]
    if any(b[1] - a[1] < .095 * span for a, b in zip(centers, centers[1:])):
        return []
    actions = []
    for i, (box, (x, y)) in enumerate(zip(peers, centers, strict=True)):
        if not any(item["role"] == "comment_like" and abs(item["click_point"][1] - y) < .028 * span
                   for item in existing):
            purpose = ("取消点赞" if _selected(frame, box) else "点赞") + "此条评论"
            actions.append(_button(box, panel, "comment_like", purpose, "Like"))
        following = centers[i + 1][1] if i + 1 < len(centers) else bottom - .10 * span
        start = max(top, round(y - .041 * span))
        stop = round(min(following - .055 * span, y + .23 * span))
        if any(item["role"] == "comment_reply" and start <= item["click_point"][1] < stop for item in existing):
            continue
        search = (left + round(.125 * span), start, left + round(.61 * span), stop)
        links = _caption_links(frame, gray, ink, search, span, index, y, verified_words, expand=False)
        if reply := links.get("reply"):
            actions.append(_button(reply, panel, "comment_reply", "回复此条评论", "Reply"))
    return actions


def comment_row_controls(frame, regions, font_index, *, confirmed_comment=False, working_view=None):
    """Emit named row controls only inside a positively identified comment sheet."""
    if regions.get("layer") != "bottom_sheet" or not regions.get("regions"):
        return []
    panel = regions["regions"][0]
    height, width = frame.shape[:2]
    l, t, r, b = panel
    if not 0 <= l <= .07 * width < .91 * width <= r <= width or not 0 <= t < b <= height or b - t < .40 * width:
        return []
    if not confirmed_comment and not has_comment_sheet_header(frame, panel, font_index, working_view=working_view):
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tone = _panel_tone(gray, panel)
    if tone < 99:
        gray, tone = 255 - gray, 255 - tone
    ink = cv2.subtract(int(round(tone)), gray)
    reactions = _reactions(ink, panel)
    owners = _avatars(ink, panel)
    limits, rows = _row_lanes(owners, reactions, panel) if owners else ([], [])
    actions = []
    verified_words = {}
    for owner, limit, votes in zip(owners, limits, rows, strict=True):
        target = "此条评论"
        like = votes.get("heart") or votes.get("thumb_up")
        if like is not None:
            label = ("取消点赞" if _selected(frame, like) else "点赞") + target
            actions.append(_button(like, panel, "comment_like", label, "Like"))
        if like is not None and votes.get("thumb_down"):
            x, y = _center(like)
            paired = [box for box in votes["thumb_down"] if .077 * (r - l) <= _center(box)[0] - x <= .32 * (r - l)
                      and abs(_center(box)[1] - y) <= .025 * (r - l)]
            if len(paired) == 1:
                actions.append(_button(paired[0], panel, "comment_dislike", "点踩" + target, "Dislike"))
        links = _links(frame, gray, ink, owner, limit, r - l, font_index, _center(like)[1] if like else None,
                       verified_words)
        if (reply := links.get("reply")) is not None:
            actions.append(_button(reply, panel, "comment_reply", "回复" + target, "Reply"))
        if (expand := links.get("expand")) is not None:
            actions.append(_button(expand, panel, "comment_expand_replies", f"展开{target}的回复", "View replies"))
    actions.extend(_repeated_heart_controls(frame, gray, ink, panel, reactions.get("heart", ()),
                                            actions, font_index, verified_words))
    return [action for action in actions if _available(action["box"], regions)]
