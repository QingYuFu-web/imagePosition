"""Ground separately typed controls in a measured, counted post footer."""

from collections import Counter

import cv2
import numpy as np

from inventory_unlabeled_engagement_row import _KERNEL, _rows, _similar_ink, _stable_glyphs


_ACTIONS = {
    "reply": ("comment_reply", "回复帖子", "Reply"),
    "repost": ("repost", "转发", "Repost"),
    "share": ("share", "打开分享菜单", "Share"),
}


def _count_is_printed(gray, box, index, cache):
    left, top = max(0, box.left - 2), max(0, box.top - 2)
    chip = gray[top:box.bottom + 2, left:box.right + 2].astype(np.int16)
    contrast = np.median(chip) - chip
    votes = Counter()
    for cut in (41, 59, 77):
        ink = np.uint8(contrast >= cut)
        groups = np.flatnonzero(np.diff(np.r_[False, ink.any(axis=0), False])).reshape(-1, 2)
        if not 1 <= len(groups) <= 5:
            continue
        digits = []
        for a, b in groups:
            glyph = ink[:, a:b]
            _, _, w, h = cv2.boundingRect(glyph)
            if not .62 * box.height <= h <= 1.22 * box.height or not .13 <= w / h <= .91:
                break
            key = glyph.shape, glyph.tobytes()
            if key not in cache:
                rank = index.rank(glyph, alphabet="code", limit=2)
                cache[key] = (rank[0]["text"] if len(rank) == 2 and rank[0]["text"] in "0123456789"
                              and rank[0]["distance"] <= .225 and rank[1]["distance"] - rank[0]["distance"] >= .019
                              else None)
            digit = cache[key]
            if digit is None:
                break
            digits.append(digit)
        if len(digits) == len(groups):
            votes["".join(digits)] += 1
    return sum(count >= 2 for count in votes.values()) == 1


def _publish(icon, caption, frame_shape, working_shape):
    height, width = frame_shape[:2]
    h, w = working_shape[:2]
    l, t, r, b = icon.box.xyxy()
    x, y = icon.box.center
    if caption is not None:
        l, t, r, b = min(l, caption.left), min(t, caption.top), max(r, caption.right), max(b, caption.bottom)
    pad = max(2, round(.31 * icon.box.height))
    box = [max(0, round((l - pad) * width / w)), max(0, round((t - pad) * height / h)),
           min(width, round((r + pad) * width / w)), min(height, round((b + pad) * height / h))]
    role, function, preview = _ACTIONS[icon.role]
    return {"role": role, "function": function, "preview_text": preview, "box": box,
            "click_point": [round(x * width / w), round(y * height / h)]}


def counted_post_actions(frame, regions, index):
    """Return reply, repost and branching share when all four owners agree."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or width < 250 or not 1.57 * width <= height <= 2.81 * width:
        return []
    posts = [entry["box"] for entry in regions["candidates"] if entry["visibility"] == "page"
             and entry["box"][2] - entry["box"][0] >= .63 * width
             and .20 * width <= entry["box"][3] - entry["box"][1] <= .70 * height
             and .11 * height <= entry["box"][1] < entry["box"][3] <= .90 * height]
    if not posts:
        return []
    ratio = min(1., 420 / width)
    working = cv2.resize(frame, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA) if ratio < 1 else frame
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    relief = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _KERNEL)
    glyphs = _stable_glyphs(working, relief)
    cache = {}
    found = {}
    for icons, captions, span, pitch in _rows(working, relief, glyphs):
        if tuple(icon.role for icon in icons[:2]) != ("reply", "repost") or icons[2].role not in {"favorite", "like"}:
            continue
        _, y = icons[0].box.center
        source_y = y * height / working.shape[0]
        if not any(-.125 * width <= source_y - box[3] <= .18 * width for box in posts):
            continue
        last = icons[-1]
        x, y = last.box.center
        share = [item for item in glyphs if item.role == "share" and abs(item.box.center[0] - x - pitch) <= .11 * pitch
                 and abs(item.box.center[1] - y) <= .27 * span and _similar_ink(last.color, item.color)]
        if len(share) != 1 or not all(_count_is_printed(gray, box, index, cache) for box in captions):
            continue
        for icon, count in (*zip(icons[:2], captions[:2], strict=True), (share[0], None)):
            found[icon.role, icon.box.xyxy()] = _publish(icon, count, frame.shape, working.shape)
    return list(found.values())
