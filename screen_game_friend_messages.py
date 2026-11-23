"""Recognize a chat control on an explicitly available friend row."""

import cv2
import numpy as np

from inventory_comment_evidence import _speech_bubble
from screen_caption_transcript import read_caption
from screen_component_captions import _bands
from screen_landscape_search_field import _available


_AVAILABLE = {"在线", "在線", "可邀请", "可邀請", "online", "available"}


def _diamond(crop, button_height):
    paper = np.median(crop[::3, ::3], axis=(0, 1))
    distance = np.max(crop.astype(np.int16) - paper, axis=2)
    candidates = []
    for level in (8, 12, 18):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(distance >= level))
        for index, (x, y, w, h, _) in enumerate(stats[1:], 1):
            if (not .78 * button_height <= min(w, h) <= max(w, h) <= 1.14 * button_height
                    or not .84 <= w / h <= 1.18 or not 0 < x < x + w < crop.shape[1]
                    or not 0 < y < y + h < crop.shape[0]):
                continue
            component = labels[y:y + h, x:x + w] == index
            yy, xx = np.ogrid[:h, :w]
            shape = abs((2 * xx + 1) / w - 1) + abs((2 * yy + 1) / h - 1) <= 1.03
            if (component & shape).sum() / (component | shape).sum() >= .71:
                candidates.append((level, (int(x), int(y), int(x + w), int(y + h))))
    boxes = [box for _, box in candidates]
    if len({level for level, _ in candidates}) < 2 or np.ptp(boxes, axis=0).max() > .12 * button_height:
        return None
    return boxes[len(boxes) // 2]


def _dotted_speech(mask):
    enlarged = cv2.resize(mask, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    contours, parents = cv2.findContours(enlarged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if parents is None:
        return False
    parents = parents[0, :, 3]
    for index, outline in enumerate(contours):
        if parents[index] < 0:
            holes = [hole for hole, parent in zip(contours, parents, strict=True) if parent == index]
            if _speech_bubble(cv2.boundingRect(outline), cv2.contourArea(outline), outline, holes):
                return True
    return False


def _bubble_on_carrier(gray, carrier):
    height, width = gray.shape
    left, top, right, bottom = carrier
    span = right - left
    delta = gray.astype(np.int16) - np.median(gray)
    bubbles = []
    for level in (55, 74, 95):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(delta >= level))
        for index, (x, y, w, h, _) in enumerate(stats[1:], 1):
            if (not .38 * span <= w <= .70 * span or not .32 * span <= h <= .59 * span
                    or not 0 < x < x + w < width or not 0 < y < y + h < height
                    or abs(x + w / 2 - (left + right) / 2) > .11 * span
                    or abs(y + h / 2 - (top + bottom) / 2) > .11 * span):
                continue
            glyph = np.uint8(labels[y:y + h, x:x + w] == index)
            if _dotted_speech(glyph):
                bubbles.append((level, (int(x), int(y), int(x + w), int(y + h))))
    return len({level for level, _ in bubbles}) >= 2 and np.ptp([box for _, box in bubbles], axis=0).max() <= .10 * span


def _peer(frame, panel, invite):
    left, top, _, bottom = invite
    h = bottom - top
    cy = (top + bottom) / 2
    x0, x1 = max(panel[0], left - round(1.58 * h)), left - round(.20 * h)
    y0, y1 = max(panel[1], round(cy - .64 * h)), min(panel[3], round(cy + .64 * h))
    if min(x1 - x0, y1 - y0) < h:
        return None
    crop = frame[y0:y1, x0:x1]
    carrier = _diamond(crop, h)
    if carrier is None or not _bubble_on_carrier(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), carrier):
        return None
    a, b, c, d = carrier
    if (not .24 * h <= left - x0 - c <= .86 * h or
            abs(cy - y0 - (b + d) / 2) > .14 * h):
        return None
    return x0 + a, y0 + b, x0 + c, y0 + d


def _available_friend(frame, panel, invite, index):
    left, top, _, bottom = invite
    h = bottom - top
    x0, x1 = max(panel[0] + round(.174 * (panel[2] - panel[0])), left - round(5.8 * h)), left - round(2.84 * h)
    y0, y1 = top + round(.60 * h), min(panel[3], bottom + round(.09 * h))
    if min(x1 - x0, y1 - y0) < 9:
        return False
    crop = frame[y0:y1, x0:x1].astype(np.int16)
    blue, green, red = crop.transpose(2, 0, 1)
    visible = (green - np.maximum(blue, red) >= 30) & (green >= 103)
    runs = []
    for a, b in _bands(visible.any(axis=0)):
        if runs and a - runs[-1][1] <= .145 * h:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    options = []
    for a, b in runs:
        x, y, w, height = cv2.boundingRect(np.uint8(visible[:, a:b]))
        if not .19 * h <= height <= .48 * h or not .46 * h <= w <= 2.10 * h:
            continue
        margin = max(2, round(.07 * h))
        area = (max(x0, x0 + a + x - margin), max(y0, y0 + y - margin),
                min(x1, x0 + a + x + w + margin), min(y1, y0 + y + height + margin))
        reading = read_caption(frame, area, index, cache={}, choice=True)
        if reading is not None and reading["text"].casefold().strip() in _AVAILABLE:
            options.append(area)
    return len(options) == 1


def game_friend_message_actions(frame, regions, index, *, invitations):
    """Export only a readable online friend's measured bubble control."""
    height, width = frame.shape[:2]
    scopes = regions.get("regions", ())
    if index is None or width < 1.40 * height or regions.get("layer") != "side_drawer" or not scopes:
        return []
    panel = tuple(scopes[0])
    left, top, right, bottom = panel
    if not (0 <= left <= .03 * width and top <= .06 * height and right <= .72 * width and bottom >= .93 * height):
        return []
    pairs = []
    for entry in invitations:
        if entry["role"] != "invite_friend":
            continue
        l, t, r, b = box = tuple(entry["box"])
        if left <= l < r <= right and top <= t < b <= bottom and (mark := _peer(frame, panel, box)) is not None:
            pairs.append((box, mark))
    if len(pairs) < 3:
        return []
    pairs.sort(key=lambda pair: pair[0][1])
    heights = [b - t for (l, t, r, b), _ in pairs]
    typical = float(np.median(heights))
    centers = [((l + r) / 2, (t + b) / 2) for _, (l, t, r, b) in pairs]
    x, y = zip(*centers, strict=True)
    pitches = np.diff(y)
    if (np.ptp(heights) > .18 * typical or np.ptp(x) > .16 * typical
            or not 1.55 * typical <= min(pitches) <= max(pitches) <= 3.75 * typical
            or np.ptp(pitches) > .30 * float(np.median(pitches))):
        return []
    actions = []
    for invite, box in pairs:
        if _available(box, regions) and _available_friend(frame, panel, invite, index):
            l, t, r, b = box
            actions.append({"role": "open_chat", "function": "与好友聊天", "preview_text": "与好友聊天",
                            "box": [l, t, r, b], "click_point": [(l + r) // 2, (t + b) // 2]})
    return actions
