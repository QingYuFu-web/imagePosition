"""Pixel-only recognition for repeated, uncaptioned post reply toolbars."""

from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np

from inventory_list_rows import _Box, _Mark
from inventory_paper_plane_share import _heart_outline
from screen_counted_post_actions import _count_is_printed


@dataclass(frozen=True)
class _Row:
    heart: _Box
    reply: _Box
    plane: _Box

    @property
    def pitch(self):
        return self.reply.center[0] - self.heart.center[0]


def _shell(contour, box):
    return cv2.drawContours(np.zeros((box.height, box.width), np.uint8), [contour], 0, 1,
                            cv2.FILLED, offset=(-box.left, -box.top))


def _bubble_tail(shell, hole, box):
    """Require an ellipse-like closed pocket and an asymmetric lower tail."""
    h, w = shell.shape
    x, y, iw, ih = cv2.boundingRect(hole)
    cx = x - box.left + (iw - 1) / 2
    cy = y - box.top + (ih - 1) / 2
    inner = cv2.contourArea(hole)
    if not (.69 <= iw / ih <= 1.37 and .63 * w <= iw <= w
            and .58 * h <= ih <= .97 * h and .57 <= inner / (iw * ih) <= .88
            and abs(cx - (w - 1) / 2) <= .12 * w and .34 * h <= cy <= .53 * h):
        return False
    edge = shell[round(.73 * h):]
    skew = edge[:, :round(.36 * w)].sum() - float(edge[:, round(.64 * w):].sum())
    if abs(skew) < .025 * w * h:
        return False
    for scanline in shell[-max(1, round(.085 * h)):]:
        spans = np.flatnonzero(np.diff(np.r_[0, scanline, 0])).reshape(-1, 2)
        tips = [cx - (a + b - 1) / 2 for a, b in spans if b - a <= .33 * w]
        if sum(abs(dx) >= .25 * w and np.sign(dx) == np.sign(skew) for dx in tips) == 1:
            return True
    return False


def _folded_plane(shell, holes, box):
    """Require the two separate light pockets and the low point of an outline plane."""
    h, w = shell.shape
    if not .31 <= shell.mean() <= .57:
        return False
    tip = np.flatnonzero(shell[-1])
    if (not 0 < tip.size <= .25 * w or not .33 * w <= tip.mean() + .5 <= .72 * w
            or shell[:round(.28 * h)].sum(axis=1).max() < .73 * w):
        return False
    pockets = sorted((cv2.boundingRect(hole) for hole in holes), key=lambda item: item[1])
    x, y, _, dh = pockets[0]
    a, b, _, bh = pockets[1]
    return .11 * h <= b + bh / 2 - y - dh / 2 <= .56 * h and abs(x - a) >= .12 * box.width


def _marks(gray, cut, *, only_replies=False):
    contours, hierarchy = cv2.findContours(cv2.inRange(gray, 0, cut), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    marks = {"heart": [], "reply": [], "plane": []}
    if hierarchy is None:
        return marks
    height, width = gray.shape
    light = None if only_replies else 255 - gray
    children = hierarchy[0, :, 2]
    owners = hierarchy[0, :, 3]
    for i in np.flatnonzero((owners == -1) & (children != -1)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if not (.021 * width <= h <= .064 * width and .79 <= w / h <= 1.35
                and .13 * height <= y <= .88 * height):
            continue
        box = _Box(x, y, x + w, y + h)
        holes, child = [], children[i]
        while child != -1:
            hole = contours[child]
            if cv2.contourArea(hole) >= .021 * w * h:
                holes.append(hole)
            child = hierarchy[0, child, 0]
        if len(holes) == 2 and not only_replies:
            if _folded_plane(_shell(contours[i], box), holes, box):
                marks["plane"].append(box)
        elif len(holes) == 1 and cv2.contourArea(holes[0]) >= .29 * w * h:
            shell = _shell(contours[i], box)
            if _bubble_tail(shell, holes[0], box):
                marks["reply"].append(box)
            if light is not None and _heart_outline(_Mark(box, float(shell.mean()), shell), light):
                marks["heart"].append(box)
    return marks


def _repost_pair(gray, cx, cy, span):
    reach = round(.65 * span)
    x, y = round(cx), round(cy)
    chip = gray[y - reach:y + reach + 1, x - reach:x + reach + 1]
    if min(chip.shape) < 8:
        return False
    for cut in (93, 133):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.inRange(chip, 0, cut))
        pieces = [(i, row) for i, row in enumerate(stats[1:], 1)
                  if .35 * span <= min(row[2:4]) <= max(row[2:4]) <= .87 * span]
        if len(pieces) != 2:
            continue
        (i, (l, t, w, h, area)), (j, (r, b, ww, hh, other)) = sorted(pieces, key=lambda item: item[1][1])
        if not (.20 * span <= b - t <= .73 * span and .03 * span <= r - l <= .43 * span
                and .72 <= w / ww <= 1.39 and .73 <= h / hh <= 1.36
                and .21 <= area / (w * h) <= .59 and .65 <= area / other <= 1.53):
            continue
        first = cv2.resize(np.uint8(labels[t:t + h, l:l + w] == i), (21, 21), interpolation=cv2.INTER_NEAREST)
        second = cv2.resize(np.uint8(labels[b:b + hh, r:r + ww] == j), (21, 21), interpolation=cv2.INTER_NEAREST)
        second = cv2.rotate(second, cv2.ROTATE_180)
        if (first & second).sum() >= .42 * (first | second).sum():
            return True
    return False


def _post_text(gray, row):
    span = row.reply.height
    y = row.reply.center[1]
    left = row.heart.left
    right = min(gray.shape[1], row.plane.right + 4 * span)
    context = gray[round(y - 2.7 * span):round(y - .62 * span), left:right]
    if not context.size or (context >= 192).mean() < .72:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.inRange(context, 0, 150))
    text = [(x, top + h) for x, top, w, h, area in stats[1:]
            if .23 * span <= h <= .88 * span and 2 <= w <= .87 * span and area >= 5]
    return any(sum(abs(baseline - peer[1]) <= .15 * span for peer in text) >= 6 for _, baseline in text)


def _same_box(first, second):
    return max(abs(a - b) for a, b in zip(first.xyxy(), second.xyxy(), strict=True)) <= max(2, .13 * first.height)


def _rows(gray):
    first = _marks(gray, 88)
    if not all(first.values()):
        return
    second = _marks(gray, 132)
    stable = {role: [box for box in boxes if any(_same_box(box, another) for another in second[role])]
              for role, boxes in first.items()}
    for reply in stable["reply"]:
        x, y = reply.center
        span = reply.height
        for heart in stable["heart"]:
            hx, hy = heart.center
            pitch = x - hx
            if not 1.60 * span <= pitch <= 3.1 * span or abs(hy - y) > .20 * span:
                continue
            for plane in stable["plane"]:
                px, py = plane.center
                if not (abs(px - x - 2 * pitch) <= .18 * pitch and abs(py - y) <= .22 * span
                        and .76 * span <= min(heart.height, plane.height)
                        <= max(heart.height, plane.height) <= 1.24 * span):
                    continue
                row = _Row(heart, reply, plane)
                if _repost_pair(gray, x + pitch, y, span) and _post_text(gray, row):
                    yield row


def _shares_grid(first, peer):
    return (abs(first.reply.center[1] - peer.reply.center[1]) >= 3.6 * first.reply.height
            and all(abs(a.center[0] - b.center[0]) <= .16 * first.pitch
                    for a, b in ((first.heart, peer.heart), (first.reply, peer.reply), (first.plane, peer.plane))))


def _row_parts(gray, reply, l, r, cut):
    y = reply.center[1]
    h = reply.height
    t = round(y - .63 * h)
    b = round(y + .65 * h)
    chip = cv2.inRange(gray[t:b, l:r], 0, cut)
    contours, _ = cv2.findContours(chip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, v, w, depth = cv2.boundingRect(contour)
        if .17 * h <= w <= 1.44 * h and .46 * h <= depth <= 1.27 * h:
            box = _Box(x, v, x + w, v + depth)
            mask = _shell(contour, box)
            yield _Box(l + x, t + v, l + x + w, t + v + depth), mask


def _share_tray(parts, span):
    for tray, shell in parts:
        h, w = shell.shape
        if not (.85 * span <= w <= 1.32 * span and .88 * span <= h <= 1.30 * span
                and shell[-max(2, round(.18 * h)):].sum(axis=1).max() >= .77 * w
                and shell[:, :max(2, round(.17 * w))].any(axis=1).sum() >= .76 * h
                and shell[:round(.62 * h), round(.48 * w):].mean() <= .18):
            continue
        for arrow, mark in parts:
            if not (.75 * span <= arrow.width <= 1.16 * span and .63 * span <= arrow.height <= 1.06 * span
                    and .12 * span <= arrow.left - tray.left <= .40 * span
                    and -.17 * span <= arrow.top - tray.top <= .12 * span
                    and .13 * span <= tray.bottom - arrow.bottom <= .43 * span):
                continue
            leading = np.where(mark[:, :max(1, round(.17 * mark.shape[1]))])[0]
            tip = np.where(mark[:, -max(1, round(.13 * mark.shape[1])):])[0]
            if leading.size and tip.size and np.median(leading) - np.median(tip) >= .16 * arrow.height:
                yield _Box(tray.left, min(tray.top, arrow.top), max(tray.right, arrow.right), tray.bottom)


def _thumb_cuff(parts, span):
    for cuff, mask in parts:
        if not (.15 * span <= cuff.width <= .37 * span and .45 * span <= cuff.height <= .76 * span
                and mask.mean() >= .78):
            continue
        for hand, shell in parts:
            if not (.53 * span <= hand.width <= 1.05 * span and .77 * span <= hand.height <= 1.2 * span
                    and .01 * span <= hand.left - cuff.right <= .23 * span
                    and .20 * span <= cuff.top - hand.top <= .49 * span
                    and abs(cuff.bottom - hand.bottom) <= .19 * span):
                continue
            w = hand.width
            upper = shell[:max(2, round(.17 * hand.height))].sum(axis=1).max()
            base = shell[round(.72 * hand.height):].sum(axis=1).max()
            if upper <= .61 * w and base >= .62 * w:
                yield _Box(cuff.left, hand.top, hand.right, max(cuff.bottom, hand.bottom))


def _numeric_owner(gray, glyph, index, cache):
    span = glyph.height
    y = glyph.center[1]
    x = glyph.right + max(2, round(.16 * span))
    t = round(y - .48 * span)
    r = min(gray.shape[1], x + round(3.1 * span))
    b = round(y + .50 * span)
    chip = gray[t:b, x:r]
    if not chip.size:
        return None
    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.inRange(chip, 0, 174))
    glyphs = sorted((v, u, v + w, u + h) for v, u, w, h, area in stats[1:]
                    if .51 * span <= h <= .86 * span and .13 * span <= w <= .75 * span
                    and area >= .055 * span * span)
    if not 1 <= len(glyphs) <= 4 or glyphs[0][0] > .71 * span:
        return None
    bottoms = [item[3] for item in glyphs]
    if max(bottoms) - min(bottoms) > .13 * span or any(
            not .04 * span <= following[0] - current[2] <= .38 * span
            for current, following in zip(glyphs, glyphs[1:])):
        return None
    number = _Box(x + glyphs[0][0], t + min(item[1] for item in glyphs), x + glyphs[-1][2], t + max(bottoms))
    return number if _count_is_printed(gray, number, index, cache) else None


def _media_above(gray, reply):
    """Check for a broad preceding post image and a separate light toolbar surface."""
    y = round(reply.center[1])
    span = reply.height
    width = gray.shape[1]
    t = max(0, y - round(6.1 * span))
    b = y - round(.88 * span)
    post = gray[t:b, :]
    # Small dark margins can survive display rectification along both sides.
    inset = round(.025 * width)
    bare = gray[y + round(.77 * span):y + round(1.35 * span), inset:width - inset]
    if min(post.shape, default=0) == 0 or min(bare.shape, default=0) == 0 or (bare >= 222).mean() < .97:
        return False
    raster = cv2.inRange(post, 0, 208)
    count = (raster > 0).mean(axis=0) > .46
    runs = np.flatnonzero(np.diff(np.r_[False, count, False])).reshape(-1, 2)
    return any(b - a >= .22 * width for a, b in runs)


def _counted_rows(gray, index):
    width = gray.shape[1]
    low = [reply for reply in _marks(gray, 155, only_replies=True)["reply"] if _media_above(gray, reply)]
    if not low:
        return
    high = _marks(gray, 185, only_replies=True)["reply"]
    cache = {}
    for reply in low:
        if not any(_same_box(reply, peer) for peer in high):
            continue
        x, y = reply.center
        span = reply.height
        left = max(0, round(x - .40 * width - span))
        left_end = round(x - .20 * width + span)
        right = round(x + .20 * width - span)
        right_end = min(width, round(x + .40 * width + span))
        if left_end - left < span or right_end - right < span:
            continue
        shares = list(_share_tray(list(_row_parts(gray, reply, left, left_end, 165)), span))
        likes = list(_thumb_cuff(list(_row_parts(gray, reply, right, right_end, 165)), span))
        for share in shares:
            gap = x - share.center[0]
            if not .22 * width <= gap <= .38 * width or abs(y - share.center[1]) > .18 * span:
                continue
            for like in likes:
                if abs(like.center[0] - x - gap) > .16 * gap or abs(y - like.center[1]) > .18 * span:
                    continue
                reply_count = _numeric_owner(gray, reply, index, cache)
                if reply_count is None:
                    continue
                share_count = _numeric_owner(gray, share, index, cache)
                if share_count is not None and _numeric_owner(gray, like, index, cache) is not None:
                    yield reply, share, reply_count, share_count


def _action(box, role, sx, sy, source_size, *, count=None):
    name, preview = {"comment_reply": ("回复帖子", "Reply"), "comments": ("打开评论", "Comments"),
                     "share": ("打开分享菜单", "Share")}[role]
    x, y = box.center
    bounds = [round(box.left * sx), round(box.top * sy), round(box.right * sx), round(box.bottom * sy)]
    action = {"role": role, "function": name, "preview_text": preview, "box": bounds,
              "click_point": [round(x * sx), round(y * sy)]}
    if count is not None:
        width, height = source_size
        left, top = max(0, floor(count.left * sx) - 1), max(0, floor(count.top * sy) - 1)
        right, bottom = min(width, ceil(count.right * sx) + 1), min(height, ceil(count.bottom * sy) + 1)
        number = [left, top, right, bottom]
        action["engagement_count_box"] = number
        action["context_box"] = [min(bounds[0], left), min(bounds[1], top),
                                 max(bounds[2], right), max(bounds[3], bottom)]
    return action


def compact_comment_row_actions(frame, regions, index=None):
    """Return visible replies in repeated rows, or comments/share in counted rows."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or width < 315 or not 1.53 * width <= height <= 2.8 * width:
        return []
    factor = min(1, 530 / width)
    gray = cv2.cvtColor(cv2.resize(frame, (round(factor * width), round(factor * height)),
                                   interpolation=cv2.INTER_AREA) if factor < 1 else frame, cv2.COLOR_BGR2GRAY)
    if (gray[::4, ::4] >= 191).mean() < .64:
        return []
    sy, sx = height / gray.shape[0], width / gray.shape[1]
    source_size = width, height
    rows = list(_rows(gray))
    output = {}
    for row in rows:
        if not any(_shares_grid(row, peer) for peer in rows):
            continue
        box = row.reply
        output[box.xyxy()] = _action(box, "comment_reply", sx, sy, source_size)
    if index is not None:
        for reply, share, reply_count, share_count in _counted_rows(gray, index):
            output[reply.xyxy()] = _action(reply, "comments", sx, sy, source_size, count=reply_count)
            output[share.xyxy()] = _action(share, "share", sx, sy, source_size, count=share_count)
    return sorted(output.values(), key=lambda item: (item["box"][1], item["box"][0]))
