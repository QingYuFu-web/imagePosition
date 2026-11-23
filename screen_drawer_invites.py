"""Read invitation buttons in a visually isolated side drawer."""

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts
from screen_component_captions import _rectangular_button


def _invitation(gray, box, index):
    left, top, right, bottom = box
    dx, dy = round(.19 * (right - left)), round(.17 * (bottom - top))
    tile = gray[top + dy:bottom - dy, left + dx:right - dx]
    paper, highlight = np.percentile(tile, (50, 96))
    if highlight - paper < 72:
        return False
    samples = {}
    for ratio in (.30, .44, .58, .70):
        ink = np.uint8(tile >= paper + ratio * (highlight - paper))
        x, y, w, h = cv2.boundingRect(ink)
        if not (.38 * tile.shape[0] <= h <= .89 * tile.shape[0] and 1.63 <= w / h <= 2.62
                and x > 0 and y > 0 and x + w < tile.shape[1] and y + h < tile.shape[0]):
            continue
        cut = ink[y:y + h, x:x + w]
        fingerprint = (cut.shape, cut.tobytes())
        if fingerprint in samples:
            continue
        parts = _cjk_parts(cut, h)
        if len(parts) != 2:
            continue
        ranks = [index.rank(part[-1], alphabet="cjk", limit=2) for part in parts]
        if any(len(rank) != 2 or rank[0]["distance"] > .18 for rank in ranks):
            continue
        if "".join(rank[0]["text"] for rank in ranks) not in {"邀请", "邀請"}:
            continue
        samples[fingerprint] = tuple(rank[1]["distance"] - rank[0]["distance"] for rank in ranks)
    return len(samples) >= 2 and all(max(values) >= .011 for values in zip(*samples.values(), strict=True))


def drawer_invitation_actions(frame, regions, index):
    """Verify repeated painted carriers, their row text and a literal invite caption."""
    if regions["layer"] != "side_drawer":
        return []
    left, top, right, bottom = regions["regions"][0]
    width = right - left
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    buttons = set()
    for row in regions["candidates"]:
        l, t, r, b = row["box"]
        w, h = r - l, b - t
        if (row["visibility"] != "foreground" or not left <= l < r <= right or not top <= t < b <= bottom
                or not .18 * width <= w <= .43 * width or not 2.1 <= w / h <= 4.3):
            continue
        box = _rectangular_button(gray, row["box"])
        if box is not None:
            buttons.add(box)
    active = []
    for l, t, r, b in sorted(buttons, key=lambda item: item[1]):
        w, h = r - l, b - t
        if sum(abs(l - pl) <= .13 * h and abs(r - pr) <= .13 * h and .76 * h <= pb - pt <= 1.24 * h
               and 1.2 * h <= abs(pt - t) <= 5.2 * h for pl, pt, pr, pb in buttons) < 2:
            continue
        caption = gray[t:b, max(left, l - round(1.80 * w)):max(left, l - round(.78 * w))]
        if not caption.size or (caption >= 160).mean() < .037 or not _invitation(gray, (l, t, r, b), index):
            continue
        active.append({"role": "invite_friend", "function": "邀请好友", "box": [l, t, r, b],
                       "click_point": [(l + r) / 2, (t + b) / 2], "preview_text": "Invite"})
    return active
