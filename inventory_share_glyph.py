"""Check a separated up arrow and open tray inside a verified UI surface."""

import cv2
import numpy as np


def _tray(mask):
    height, width = mask.shape
    if not .72 <= width / height <= 1.55:
        return False
    flank = max(2, round(.18 * width))
    walls = mask[round(.25 * height):round(.77 * height)]
    if min(walls[:, :flank].any(axis=1).mean(), walls[:, -flank:].any(axis=1).mean()) < .81:
        return False
    if walls[:, flank:-flank].mean() > .07:
        return False
    opening = mask[:round(.37 * height), round(.29 * width):round(.71 * width)]
    floor = mask[round(.77 * height):]
    return bool(opening.mean() <= .10 and floor.mean(axis=1).max() >= .76)


def _up_arrow(mask):
    height, width = mask.shape
    if not .40 <= width / height <= 1.06:
        return False
    top = np.flatnonzero(mask[0])
    if top.size > .62 * width or abs(top.mean() - (width - 1) / 2) > .19 * width:
        return False
    # The apex is central. Both head edges extend down and out from it.
    shoulder = max(1, round(.25 * width))
    left = np.nonzero(mask[:, :shoulder])[0]
    right = np.nonzero(mask[:, -shoulder:])[0]
    if not left.size or not right.size:
        return False
    left_y, right_y = np.median(left), np.median(right)
    if (not .10 * height <= min(left_y, right_y) <= max(left_y, right_y) <= .53 * height
            or abs(left_y - right_y) > .15 * height):
        return False
    stem = mask[round(.50 * height):]
    at = np.flatnonzero(stem.any(axis=0))
    return bool(at.size and at[-1] - at[0] + 1 <= max(2, .36 * width)
                and abs(at.mean() - (width - 1) / 2) <= .14 * width
                and stem.any(axis=1).mean() >= .90)


def _share_parts(mask, short):
    if not .028 <= mask.mean() <= .21:
        return False
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask), connectivity=8)
    parts = [i for i, (_, _, _, _, area) in enumerate(stats[1:], 1)
             if area >= max(3, round(.0015 * short * short))]
    if len(parts) != 2:
        return False
    tray_id, arrow_id = sorted(parts, key=lambda i: stats[i, cv2.CC_STAT_WIDTH], reverse=True)
    tx, ty, tw, th, _ = stats[tray_id]
    ax, ay, aw, ah, _ = stats[arrow_id]
    if not (.35 * short <= tw <= .71 * short and .29 * short <= th <= .68 * short
            and .28 * tw <= aw <= .72 * tw and .46 * th <= ah <= 1.02 * th
            and abs(tx + tw / 2 - (ax + aw / 2)) <= .10 * tw
            and abs(tx + tw / 2 - mask.shape[1] / 2) <= .12 * short
            and .11 * th <= ty - ay <= .53 * th
            and ty + .17 * th <= ay + ah <= ty + .74 * th):
        return False
    tray = labels[ty:ty + th, tx:tx + tw] == tray_id
    arrow = labels[ay:ay + ah, ax:ax + aw] == arrow_id
    return _tray(tray) and _up_arrow(arrow)


def is_share_glyph(gray, box):
    """Verify the glyph inside a half-open xyxy surface, independent of clickability."""
    if gray.ndim != 2:
        return False
    left, top, right, bottom = map(round, box.xyxy() if hasattr(box, "xyxy") else box)
    height, width = bottom - top, right - left
    short = min(height, width)
    if (short < 26 or not .79 <= width / height <= 1.26
            or not (0 <= left < right <= gray.shape[1] and 0 <= top < bottom <= gray.shape[0])):
        return False
    patch = gray[top:bottom, left:right].astype(np.int16)
    y, x = np.ogrid[:height, :width]
    radius = np.hypot(y - (height - 1) / 2, x - (width - 1) / 2) / short
    interior = radius <= .435
    ring = patch[(radius >= .35) & interior]
    dark, ground, light = np.percentile(ring, (10, 50, 90))
    if light - dark > 31:
        return False
    mark = patch - ground
    threshold = max(29, round(1.4 * (light - dark) + 17))
    for polarity in (1, -1):
        ink = polarity * mark
        confirmations = 0
        for level in (threshold, threshold + 12, threshold + 24):
            if _share_parts((ink >= level) & interior, short):
                confirmations += 1
                if confirmations >= 2:
                    return True
    return False
