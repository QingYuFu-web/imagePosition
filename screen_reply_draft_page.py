"""Recognize a quoted-comment editor and its enabled publication action."""

from statistics import median

import cv2
import numpy as np

from screen_colored_settings_panel import _printed
from screen_component_captions import _bands
from screen_visible_draft import has_visible_draft


_HEADERS = ("New Comment", "Write Comment", "Edit Comment", "Reply", "New Reply", "发表回复", "发表评论", "编辑评论")
_OTHER_HEADERS = ("New Post", "New Message", "New Message Draft", "Comment Settings", "Post Preview", "Settings", "编辑帖子")
_CANCEL = ("Cancel", "Discard", "Close", "取消", "放弃")
_COMMAND = ("Post", "Publish", "Send", "Reply", "发送", "发布", "回复", "提交")


def _read_word(gray, window, texts, index, *, rivals=(), threshold):
    left, top, right, bottom = window
    chip = gray[top:bottom, left:right]
    if min(chip.shape) < 8:
        return None
    x, y, w, h = cv2.boundingRect(cv2.compare(chip, threshold, cv2.CMP_GE))
    if not 7 <= h <= .81 * len(chip) or not 1.30 * h <= w <= 13 * h:
        return None
    if min(x, y, chip.shape[1] - x - w, len(chip) - y - h) < 1:
        return None
    pad = max(2, round(.28 * h))
    x0, y0 = left + x, top + y
    area = (max(left, x0 - pad), max(top, y0 - pad), min(right, x0 + w + pad), min(bottom, y0 + h + pad))
    reading = _printed(gray, area, index, vocabulary=(*texts, *rivals), minimum_score=.83, minimum_margin=.032)
    return (reading, (x0, y0, x0 + w, y0 + h)) if reading in texts else None


def _keyboard(gray):
    height, width = gray.shape
    start = round(.65 * height)
    chip = gray[start:]
    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.inRange(chip, 37, 93))
    rows = [(float(x + w / 2), int(y + start), int(w), int(h)) for x, y, w, h, n in stats[1:]
            if .044 * width <= w <= .122 * width and .053 * width <= h <= .158 * width and n >= .64 * w * h]
    for _, y, w, h in sorted(rows, key=lambda row: row[1]):
        group = sorted(item[0] for item in rows if abs(item[1] - y) <= .10 * h and abs(item[2] - w) <= .18 * w
                       and abs(item[3] - h) <= .16 * h)
        if not 7 <= len(group) <= 11 or group[-1] - group[0] <= .73 * width:
            continue
        spacing = np.diff(group)
        if (spacing >= .70 * w).all() and (spacing <= 1.27 * w).mean() >= .69:
            return y
    return None


def _quote(gray, keyboard_top):
    height, width = gray.shape
    samples = gray[:, round(.03 * width):round(.97 * width):4]
    mid = np.median(samples, axis=1)
    surface = (samples >= 23) & (samples <= 74)
    broad = surface.mean(axis=1) >= .84
    gaps = max(3, round(.030 * width) | 1)
    joined = cv2.morphologyEx(np.uint8(broad[:, None]), cv2.MORPH_CLOSE, np.ones((gaps, 1), np.uint8))[:, 0]
    found = []
    for top, bottom in _bands(joined != 0):
        if (not .31 * height <= top <= .78 * height or not .105 * height <= bottom - top <= .33 * height
                or not .105 * width <= keyboard_top - bottom <= .45 * width):
            continue
        margin = max(3, round(.033 * width))
        # Resampling widens the bright strokes inside the quote. Measure the
        # gray surface across the band and at both sides.
        paper = surface[top:bottom]
        edge = max(1, round(.08 * paper.shape[1]))
        if (paper.mean() < .90 or min(paper[:, :edge].mean(), paper[:, -edge:].mean()) < .84
                or np.median(mid[top - margin:top]) >= 19
                or np.median(mid[bottom:bottom + margin]) >= 19):
            continue
        ink = samples[top:bottom] >= float(np.median(mid[top:bottom])) + 66
        text_rows = ink.sum(axis=1) >= max(2, round(.015 * width))
        if sum(.016 * width <= b - a <= .055 * width for a, b in _bands(text_rows)) >= 2:
            found.append((top, bottom))
    return found[0] if len(found) == 1 else None


def _ink(frame, box):
    left, top, right, bottom = box
    body = frame[top:bottom, left:right]
    low, high = body.min(axis=2), body.max(axis=2)
    colored = body[(high >= 123) & (high - low >= 45)]
    if len(colored) < .16 * (right - left) * (bottom - top):
        return None
    color = np.median(colored, axis=0)
    return color if np.ptp(color) >= 55 else None


def _accent_toolbar(frame, start, stop, expected):
    width = frame.shape[1]
    tile = frame[start:stop]
    if tile.size == 0:
        return False
    samples = tile.astype(np.int16)
    hue = samples - samples.min(axis=2, keepdims=True)
    profile = expected - min(expected)
    ink = (abs(hue - profile).max(axis=2) <= 55) & (hue.max(axis=2) >= 55)
    sizes = _bands(ink.sum(axis=1) >= .043 * width)
    for top, bottom in sizes:
        if not .021 * width <= bottom - top <= .102 * width:
            continue
        lane = np.uint8(ink[top:bottom].any(axis=0))[None]
        lane = cv2.morphologyEx(lane, cv2.MORPH_CLOSE, np.ones((1, max(3, round(.019 * width)) | 1), np.uint8))[0]
        groups = [(a + b) / 2 for a, b in _bands(lane != 0) if .015 * width <= b - a <= .115 * width]
        if not 4 <= len(groups) <= 9 or groups[-1] - groups[0] <= .60 * width:
            continue
        spacing = np.diff(groups)
        pitch = median(spacing)
        if len(spacing) >= 3 and (abs(spacing - pitch) <= .19 * pitch).mean() >= .72:
            return True
    return False


def reply_draft_page(frame, overlay, index):
    """Find the real Cancel and Send on a comment editor with a separated quote."""
    height, width = frame.shape[:2]
    if index is None or overlay is not None or width < 300 or not 1.40 * width <= height <= 2.95 * width:
        return None
    overview_w = min(width, 640)
    overview_h = round(height * overview_w / width)
    work = cv2.resize(frame, (overview_w, overview_h), interpolation=cv2.INTER_AREA) if overview_w < width else frame
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    header_limit = round(.103 * h)
    if (gray[:header_limit:2, ::4] <= 30).mean() < .84:
        return None
    title = _read_word(gray, (round(.276 * w), round(.019 * h), round(.726 * w), round(.065 * h)),
                       _HEADERS, index, rivals=_OTHER_HEADERS, threshold=171)
    if title is None:
        return None
    first = max(0, title[1][1] - round(.012 * h))
    last = round(.085 * h)
    button = _read_word(gray, (round(.020 * w), first, round(.253 * w), last), _CANCEL, index,
                        rivals=_COMMAND, threshold=90)
    if button is None or (paint := _ink(work, button[1])) is None:
        return None
    key_y = _keyboard(gray)
    panel = _quote(gray, key_y) if key_y is not None else None
    if panel is None or not _accent_toolbar(work, panel[1], key_y, paint):
        return None
    tw = title[1][3] - title[1][1]
    post = _read_word(gray, (round(.752 * w), first, round(.987 * w), last), _COMMAND, index,
                      rivals=_CANCEL, threshold=90)
    labels = [("cancel", "取消回复", button[1], button[0])]
    if post is not None:
        cy = (post[1][1] + post[1][3] - button[1][1] - button[1][3]) / 2
        color = _ink(work, post[1])
        body = (round(.027 * w), header_limit + max(2, round(.02 * w)), round(.98 * w), panel[0])
        if (abs(cy) <= .45 * tw and color is not None and np.max(abs(paint - color)) <= 36
                and has_visible_draft(work, body, index, trailing_fields=.10, line_span=max(22, 2.50 * tw))):
            labels.append(("comment_send", "发送回复", post[1], post[0]))
    actions = []
    for role, function, (a, b, c, d), printed in labels:
        box = [round(a * width / w), round(b * height / h), round(c * width / w), round(d * height / h)]
        actions.append({"role": role, "function": function, "preview_text": printed, "box": box,
                        "click_point": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]})
    return (0, 0, width, height), actions
