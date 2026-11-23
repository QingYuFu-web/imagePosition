"""Verify an editable messaging row that belongs to a docked phone keyboard."""

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_caption_transcript import _bands
from screen_dark_reply_editor import _components, _send
from screen_landscape_search_field import _available
from screen_mail_compose_controls import _paperclip, _stable_glyph
from screen_native_tab_bar import _distance


def _send_surface(frame, keyboard_top):
    width = frame.shape[1]
    x0 = round(.824 * width)
    y0 = max(0, keyboard_top - round(.223 * width))
    y1 = keyboard_top - round(.020 * width)
    zone = frame[y0:y1, x0:]
    paint = np.median(zone[::5, ::5], axis=(0, 1))
    if paint.mean() < 196 or np.ptp(paint) > 27:
        return None
    mask = np.uint8(_distance(zone, paint) >= 35)
    outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caps = []
    for outline in outlines:
        x, y, w, h = cv2.boundingRect(outline)
        if (not .071 * width <= h <= .148 * width or not .91 * h <= w <= 1.12 * h
                or not .71 * w * h <= cv2.contourArea(outline) <= .82 * w * h
                or not 0 <= x < x + w < mask.shape[1] or not 0 < y < y + h < mask.shape[0]):
            continue
        corner = max(2, round(.15 * h))
        part = mask[y:y + h, x:x + w]
        if max(part[:corner, :corner].mean(), part[:corner, -corner:].mean(),
               part[-corner:, :corner].mean(), part[-corner:, -corner:].mean()) > .46:
            continue
        inner = zone[y + round(.25 * h):y + round(.74 * h), x + round(.18 * w):x + round(.82 * w)]
        fill = np.median(inner[::3, ::3], axis=(0, 1))
        if (_distance(inner, fill) <= 12).mean() >= .64:
            caps.append(((x0 + x, y0 + y, x0 + x + w, y0 + y + h), fill))
    return caps[0] if len(caps) == 1 else None


def _attachment(gray, cap):
    _, top, _, bottom = cap
    height = bottom - top
    w = gray.shape[1]
    return _stable_glyph(
        gray, (round(.021 * w), top + round(.095 * height), round(.132 * w), bottom - round(.19 * height)),
        _paperclip, lambda x, y: .35 * height <= x <= .65 * height and .17 * height <= y <= .35 * height,
        cuts=(34, 41, 48),
    )


def _draft_carrier(frame, cap, clip):
    left, top, _, bottom = cap
    height, width = bottom - top, frame.shape[1]
    a, r = clip[2] + round(.16 * height), left - round(.10 * height)
    y, b = top - round(.19 * height), bottom + round(.16 * height)
    chip = frame[y:b, a:r]
    fill = np.median(chip[round(.30 * height):round(.64 * height):3,
                         round(.22 * (r - a)):round(.77 * (r - a)):5], axis=(0, 1))
    gutter = frame[y:b:3, clip[2] + 2:a]
    surround = np.median(gutter, axis=(0, 1))
    if (fill.mean() < 226 or np.ptp(fill) > 19 or fill.mean() - surround.mean() < 11
            or abs(fill - surround).max() > 63):
        return None
    mask = np.uint8(_distance(chip, fill) <= 7)
    outlines, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fields = []
    for outline in outlines:
        x, t, w, h = cv2.boundingRect(outline)
        if (.56 * width <= w <= .74 * width and .85 * height <= h <= 1.18 * height
                and .25 * height <= (a + x) - clip[2] <= .93 * height
                and abs(y + t + h / 2 - (top + bottom) / 2) <= .095 * height
                and .89 * w * h <= cv2.contourArea(outline)):
            core = mask[t + round(.33 * h):t + round(.74 * h), x + w // 2:x + w]
            edge = np.flatnonzero(core.mean(axis=0) >= .76)
            if edge.size:
                fields.append((a + x, y + t, a + x + w // 2 + int(edge[-1]) + 1, y + t + h))
    return fields[0] if len(fields) == 1 else None


def _visible_draft(gray, field):
    left, top, right, bottom = field
    depth = bottom - top
    x, r = left + round(.105 * depth), right - round(.15 * depth)
    y, b = top + round(.13 * depth), top + round(.80 * depth)
    body = gray[y:b, x:r]
    ink = body <= np.median(body) - 52
    spans = _bands(ink[round(.24 * depth):round(.51 * depth)].mean(axis=0) >= .93)
    found = []
    for a, end in spans:
        if not 1 <= end - a <= .055 * depth or a < .51 * depth:
            continue
        strokes = _bands(ink[:, (a + end) // 2])
        if len(strokes) != 1:
            continue
        i, j = strokes[0]
        if not .43 * depth <= j - i <= .67 * depth:
            continue
        vx, vy, w, h = cv2.boundingRect(np.uint8(ink[:, :a]))
        if (not .04 * depth <= vx <= .44 * depth or not .20 * depth <= h <= .49 * depth
                or a - vx - w > .15 * depth or vy - i < .027 * depth or j - vy - h < .023 * depth):
            continue
        if (r - x - end >= .27 * depth
                and ink[:, end + round(.09 * depth):].mean() <= .011):
            found.append(a)
    return len(found) == 1


def _toolbar_back(frame):
    height, width = frame.shape[:2]
    header = frame[:round(.178 * height)]
    profile = np.median(header[:, round(.33 * width):round(.70 * width):4], axis=1)
    bar_candidates = []
    for start in range(round(.048 * height), round(.115 * height), max(2, round(.013 * height))):
        paint = profile[start]
        if np.ptp(paint) < 42:
            continue
        for top, bottom in _bands(abs(profile - paint).max(axis=1) <= 10):
            after = bottom + round(.019 * width)
            if (.016 * height <= top <= .071 * height and .077 * width <= bottom - top <= .191 * width
                    and after < len(profile) and np.ptp(profile[after]) <= 19
                    and np.median(abs(profile[:top] - paint)) >= 14):
                bar_candidates.append((top, bottom))
    bars = set(bar_candidates)
    if len(bars) != 1:
        return None
    top, bottom = bars.pop()
    size = bottom - top
    gray = cv2.cvtColor(header[top:bottom], cv2.COLOR_BGR2GRAY)
    lane = (round(.022 * width), round(.14 * size), round(.117 * width), size - round(.14 * size))
    mark = None
    for cut in (42, 56, 72):
        parts = _components(gray, lane, cut)
        hits = [(x, y, r, b) for x, y, r, b, mask in parts
                if .22 * size <= b - y <= .40 * size and is_shaft_back_arrow(mask)]
        if len(hits) != 1 or (mark is not None and max(abs(a - b) for a, b in zip(mark, hits[0])) > 2):
            return None
        mark = hits[0]
    title = gray[lane[1]:lane[3], round(.145 * width):round(.58 * width)]
    _, _, ink, _ = cv2.connectedComponentsWithStats(np.uint8(title >= np.median(title) + 54))
    glyphs = [(x, y + h, w) for x, y, w, h, area in ink[1:]
              if .15 * size <= h <= .48 * size and .044 * size <= w <= .47 * size and area >= .004 * size**2]
    if (not 3 <= len(glyphs) <= 18 or np.ptp([bottom for _, bottom, _ in glyphs]) > .11 * size
            or min(x for x, _, _ in glyphs) > .43 * size):
        return None
    l, t, r, b = mark
    pad = round(.15 * size)
    return l - pad, top + t - pad, r + pad, top + b + pad


def message_composer_actions(frame, regions, keyboard=None, *, get_keyboard=None):
    """Read an entered message, attachment, and explicit paper-plane Send."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or width < 270 or not 1.53 * width <= height <= 2.78 * width:
        return []
    toolbar = None
    if keyboard is None and get_keyboard is not None:
        # Only a native toolbar can request the more expensive ordinary keyboard scan.
        toolbar = _toolbar_back(frame)
        if toolbar is not None:
            keyboard = get_keyboard()
    if keyboard is None:
        return []
    surface = _send_surface(frame, keyboard[1])
    if surface is None:
        return []
    cap, paint = surface
    gray = cv2.cvtColor(frame[:keyboard[1]], cv2.COLOR_BGR2GRAY)
    if (clip := _attachment(gray, cap)) is None:
        return []
    field = _draft_carrier(frame, cap, clip)
    if field is None or not _visible_draft(gray, field):
        return []
    l, t, r, b = cap
    depth = b - t
    pad = round(.15 * depth)
    clip_box = (clip[0] - pad, clip[1] - pad, clip[2] + pad, clip[3] + pad)
    targets = [(field, "input", "输入消息"), (clip_box, "attach", "添加附件")]
    core = (l + pad, t + pad, r - pad, b - pad)
    if np.ptp(paint) >= 42 and _send(_components(gray, core, 52), depth):
        targets.append((cap, "send", "发送消息"))
    if toolbar is None:
        toolbar = _toolbar_back(frame)
    if toolbar is not None:
        targets.append((toolbar, "back", "返回"))
    return [{"role": role, "function": name, "preview_text": name, "box": list(box),
             "click_point": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]}
            for box, role, name in targets if _available(box, regions)]
