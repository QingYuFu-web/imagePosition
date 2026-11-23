"""Find a printed dismissal immediately below a blocking dark dialog."""

import math

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_foreground import _working_view, detect_dark_dialog_from_dismissal
from screen_reply_draft_page import _read_word


_DISMISSALS = ("Dismiss", "Close", "关闭", "取消")
_OTHER_WORDS = ("Cancel", "Keep", "Back", "Confirm", "Continue", "OK", "Done", "Retry", "Later",
                "Maybe Later", "Ignore", "Allow", "Disable", "Close All", "Dismiss All", "Close Window",
                "返回", "继续", "关闭全部", "确定", "完成", "忽略")


def _text_windows(gray):
    height, width = gray.shape
    x0, x1 = round(.34 * width), round(.66 * width)
    y0, y1 = round(.28 * height), round(.78 * height)
    ink = gray[y0:y1, x0:x1] >= 76
    for start, stop in _bands(ink.sum(axis=1) >= max(2, round(.005 * width))):
        length = stop - start
        if not .012 * width <= length <= .063 * width:
            continue
        x, y, span, depth = cv2.boundingRect(np.uint8(ink[start:stop]))
        center = x0 + x + span / 2
        if not (.048 * width <= span <= .26 * width and .42 * width <= center <= .58 * width
                and x > 0 and x + span < x1 - x0):
            continue
        top, bottom = y0 + start + y, y0 + start + y + depth
        # A detached label is drawn onto quiet dark space below the panel.
        # Reject rows on cards, illustrations and full-width controls early.
        flanks = (gray[top:bottom, round(.15 * width):x0], gray[top:bottom, x1:round(.85 * width)])
        if any((side < 57).mean() < .89 for side in flanks):
            continue
        pad = max(3, round(.68 * depth))
        yield x0 + x - pad, top - pad, x0 + x + span + pad, bottom + pad


def dark_dialog_literal_dismissal(frame, font_index, *, working_view=None):
    """Return foreground geometry supported by a detached literal and a raised panel."""
    height, width = frame.shape[:2]
    if font_index is None or not 1.38 * width <= height <= 2.85 * width:
        return None
    _, gray = _working_view(frame) if working_view is None else working_view
    if gray is None:
        return None
    windows = list(_text_windows(gray))
    if not 1 <= len(windows) <= 6:
        return None
    small_h, small_w = gray.shape
    sx, sy = width / small_w, height / small_h
    matches = []
    for x0, y0, x1, y1 in windows:
        left, right = math.floor(x0 * sx), math.ceil(x1 * sx)
        top, bottom = math.floor(y0 * sy), math.ceil(y1 * sy)
        chip = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        box = (0, 0, right - left, bottom - top)
        if (reading := _read_word(chip, box, _DISMISSALS, font_index, rivals=_OTHER_WORDS, threshold=76)) is None:
            continue
        name, (a, b, c, d) = reading
        a, b, c, d = a + left, b + top, c + left, d + top
        # The geometry detector uses score as its accepted-text gate. Here
        # acceptance comes from matching the literal at two stroke levels.
        matches.append({"text": name, "score": 1.0, "polygon": ((a, b), (c, b), (c, d), (a, d))})
    return detect_dark_dialog_from_dismissal(frame, matches) if len(matches) == 1 else None
