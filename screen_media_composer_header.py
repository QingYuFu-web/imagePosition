"""Read the separately lit composer toolbar above a dimmed message body."""

import cv2
import numpy as np

from screen_dark_barrage_input import _round_fields
from screen_reply_draft_page import _read_word


_CANCEL = ("Cancel", "取消")
_SEND = ("Send", "发送", "發送")


def _toolbar_bottom(gray):
    width = gray.shape[1]
    rail = gray[:, round(.65 * width):round(.83 * width):2]
    bright = (rail >= 229).mean(axis=1)
    dim = ((rail >= 131) & (rail <= 212)).mean(axis=1)
    depth = max(3, round(.008 * width))
    edges = []
    for y in range(round(.105 * width), round(.238 * width)):
        if (bright[y - depth:y].mean() >= .77 and dim[y:y + depth].mean() >= .76
                and bright[y - 1] - bright[y] >= .27 and dim[y + depth:y + 2 * depth].mean() >= .84):
            edges.append((bright[y - 1] - bright[y], y))
    if not edges:
        return None
    _, edge = max(edges)
    body = gray[edge + round(.046 * width):edge + round(.101 * width):2, round(.06 * width):round(.94 * width):3]
    if not body.size or ((body >= 129) & (body <= 215)).mean() < .75:
        return None
    return edge


def _colored_pill(frame, gray, edge):
    width = frame.shape[1]
    left, top = round(.745 * width), max(0, edge - round(.118 * width))
    tile = frame[top:edge, left:]
    mask = (tile.max(axis=2) >= 174) & (np.ptp(tile, axis=2) >= 84)
    pills = []
    for a, t, b, z in _round_fields(mask, width, 1.39, 2.45):
        a, t, b, z = left + a, top + t, left + b, top + z
        if (not .81 * width <= a < b <= .99 * width or not .008 * width <= edge - z <= .043 * width
                or z - t > .091 * width):
            continue
        margin = round(.13 * (z - t))
        paint = gray[t + margin:z - margin, a - round(.072 * width):a - round(.017 * width)]
        if paint.size and (paint >= 228).mean() >= .76:
            pills.append((a, t, b, z))
    return pills[0] if len(pills) == 1 else None


def media_composer_header(frame, index):
    """Return only literally verified actions on the separate composer toolbar."""
    height, width = frame.shape[:2]
    if index is None or width < 260 or height < 1.43 * width:
        return []
    gray = cv2.cvtColor(frame[:round(.35 * width)], cv2.COLOR_BGR2GRAY)
    edge = _toolbar_bottom(gray)
    if edge is None or (pill := _colored_pill(frame, gray, edge)) is None:
        return []
    left, top, right, bottom = pill
    tall, wide = bottom - top, right - left
    cancel_window = (round(.016 * width), top + round(.11 * tall), round(.16 * width), bottom - round(.13 * tall))
    send_window = (left + round(.12 * wide), top + round(.14 * tall), right - round(.12 * wide), bottom - round(.13 * tall))
    actions = []
    cancel = _read_word(255 - gray, cancel_window, _CANCEL, index, rivals=("Back", "Close", "Done", *_SEND), threshold=78)
    if cancel is not None:
        printed, (a, t, b, z) = cancel
        pad = max(3, round(.30 * (z - t)))
        actions.append({"role": "cancel", "function": "取消当前编辑", "preview_text": printed,
                        "box": [max(0, a - pad), max(top, t - pad), b + pad, min(edge, z + pad)],
                        "click_point": [(a + b) // 2, (t + z) // 2], "fixed_center": True,
                        "control_scope": "composer"})
    send = _read_word(gray, send_window, _SEND, index, rivals=("Post", "Save", "Share", "Next", *_CANCEL), threshold=220)
    if send is not None:
        printed, (a, t, b, z) = send
        actions.append({"role": "send", "function": "发送当前编辑", "preview_text": printed, "box": list(pill),
                        "click_point": [(a + b) // 2, (t + z) // 2], "fixed_center": True,
                        "control_scope": "composer"})
    return actions
