"""Own the underlined fields of a visible portrait compose form."""

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_caption_transcript import _bands, read_caption
from screen_keyboard_footer_geometry import keyboard_footer_bounds
from screen_landscape_search_field import _available
from screen_native_tab_bar import _distance


def _paper_rules(ink):
    height, width = ink.shape
    inset = round(.026 * width)
    center = ink[:, inset:width - inset]
    dense = (center >= 9).mean(axis=1) >= .97
    runs = _bands(dense)
    thin = [(a, b, int(np.median(center[a:b]))) for a, b in runs
            if b - a <= .010 * width and a > .16 * height and b < .89 * height]
    families = []
    for a, b, c, d in zip(thin, thin[1:], thin[2:], thin[3:]):
        pitches = (b[0] - a[0], c[0] - b[0])
        if (not .093 * width <= min(pitches) <= max(pitches) <= .167 * width
                or max(pitches) > 1.21 * min(pitches) or not 1.91 * min(pitches) <= d[0] - c[0] <= 3.40 * max(pitches)
                or not 17 <= min(a[2], b[2], c[2]) <= max(a[2], b[2], c[2]) <= 66
                or max(a[2], b[2], c[2]) - min(a[2], b[2], c[2]) > 9 or d[2] - c[2] < 24):
            continue
        upper = [(top, bottom) for top, bottom in runs if bottom - top >= .095 * width
                 and .078 * width <= a[0] - bottom <= .203 * width]
        if len(upper) == 1:
            families.append((upper[0][1], a, b, c, d))
    return families[0] if len(families) == 1 else None


def _toolbar(gray, edge, pitch):
    width = gray.shape[1]
    y0, y1 = max(0, edge - round(1.09 * pitch)), edge - round(.21 * pitch)
    if y1 <= y0:
        return None
    zone = gray[y0:y1, :round(.122 * width)]
    paint = float(np.median(zone))
    _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(zone <= paint - 53), connectivity=8)
    arrows = [(int(x), y0 + int(y), int(w), int(h)) for k, (x, y, w, h, _) in enumerate(stats[1:], 1)
              if .024 * width <= h <= .068 * width and .80 <= w / h <= 1.23
              and .028 * width <= x <= .088 * width
              and is_shaft_back_arrow(ids[y:y + h, x:x + w] == k)]
    if len(arrows) != 1:
        return None
    x, y, w, h = arrows[0]
    y0, y1 = max(0, y - round(.62 * h)), min(edge, y + round(1.62 * h))
    tile = gray[y0:y1, x + w + round(.71 * h):round(.55 * width)]
    title = cv2.boundingRect(np.uint8(tile <= paint - 64))
    if .69 * h <= title[3] <= 1.72 * h and 2.06 * h <= title[2] <= 8.4 * h:
        return 0, y0, width, y1
    return None


def _placeholder(ink, top, bottom, reach, *, tall=False):
    width = ink.shape[1]
    depth = bottom - top
    x0, x1 = round(.010 * width), round(reach * width)
    y0 = top + round(.143 * depth if not tall else .056 * depth)
    y1 = top + round(.86 * depth if not tall else .33 * depth)
    tile = ink[y0:y1, x0:x1]
    mask = np.uint8(tile >= 44)
    x, y, w, h = cv2.boundingRect(mask)
    if (not .018 * width <= h <= .068 * width or not 1.10 <= w / h <= 11.3
            or not .018 * width <= x + x0 <= .055 * width or x + w >= x1 - x0 - 2
            or not 57 <= np.percentile(tile[mask != 0], 80) <= 132):
        return None
    if not tall and not .34 * depth <= y0 + y + h / 2 - top <= .76 * depth:
        return None
    pad = max(2, round(.0030 * width))
    return x0 + x - pad, y0 + y - pad, x0 + x + w + pad, y0 + y + h + pad


def _sender_text(ink, top, bottom, after):
    width = ink.shape[1]
    x0, x1 = after + round(.016 * width), round(.88 * width)
    depth = bottom - top
    y0, y1 = top + round(.21 * depth), bottom - round(.20 * depth)
    mask = ink[y0:y1, x0:x1] >= 130
    x, y, w, h = cv2.boundingRect(np.uint8(mask))
    if not (.094 * width <= w <= .72 * width and .020 * width <= h <= .073 * width
            and .04 * depth <= y and len(_bands(mask.any(axis=0))) >= 7):
        return None
    return x0 + x


def _visible_caret(frame, top, bottom, sender):
    width = frame.shape[1]
    before, after = sender - round(.027 * width), sender + round(.053 * width)
    depth = bottom - top
    pixel = frame[top:bottom, before:after]
    chroma = np.ptp(pixel, axis=2)
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(chroma >= 60), connectivity=8)
    marks = [(before + int(x), top + int(y), before + int(x + w), top + int(y + h))
             for x, y, w, h, area in stats[1:]
             if 2 <= w <= .012 * width and .31 * depth <= h <= .69 * depth
             and area >= .81 * w * h and abs(y + h / 2 - depth / 2) <= .13 * depth]
    return marks[0] if len(marks) == 1 else None


def mail_compose_layout(frame):
    """Return measured form and toolbar bounds for a focused full-page composer."""
    height, width = frame.shape[:2]
    if width < 270 or not 1.53 * width <= height <= 2.78 * width:
        return None
    scan = round(.65 * height)
    paint = np.median(frame[round(.238 * height):round(.483 * height):5,
                            round(.17 * width):round(.79 * width):5], axis=(0, 1))
    if np.mean(paint) < 225 or np.ptp(paint) > 19:
        return None
    ink = _distance(frame[:scan], paint)
    rules = _paper_rules(ink)
    if rules is None:
        return None
    top, first, second, third, base = rules
    pitch = second[0] - first[0]
    sender = _placeholder(ink, top, first[0], .14)
    recipient = _placeholder(ink, first[1], second[0], .12)
    subject = _placeholder(ink, second[1], third[0], .49)
    body = _placeholder(ink, third[1], base[0], .67, tall=True)
    if not (sender and recipient and subject and body):
        return None
    left_edges = [caption[0] for caption in (sender, recipient, subject, body)]
    if max(left_edges) - min(left_edges) > .019 * width:
        return None
    sender_x = _sender_text(ink, top, first[0], sender[2])
    if sender_x is None or (caret := _visible_caret(frame, first[1], second[0], sender_x)) is None:
        return None
    quiet = (ink[first[1]:second[0], caret[2] + round(.024 * width):round(.89 * width)],
             ink[second[1]:third[0], subject[2] + round(.016 * width):round(.96 * width)],
             ink[body[3] + round(.021 * width):base[0], round(.025 * width):round(.976 * width)])
    if any(tile.size == 0 or (tile <= 15).mean() < .986 for tile in quiet):
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    header = _toolbar(gray, top, pitch)
    if header is None:
        return None
    keyboard = keyboard_footer_bounds(frame)
    if keyboard is None or not .024 * width <= keyboard[1] - base[1] <= .121 * width:
        return None
    margin = round(.017 * width)
    return {
        "toolbar": header, "keyboard_top": keyboard[1], "caret": caret,
        "recipient_row": (0, first[1], width, second[0]),
        "recipient": (max(recipient[2] + margin, caret[0] - margin), first[1], round(.889 * width), second[0]),
        "subject": (margin, second[1], width - margin, third[0]),
        "body": (margin, third[1], width - margin, base[0]),
        "captions": (recipient, subject, body),
    }


def mail_compose_fields(frame, regions, index, *, layout=None):
    """Read the editable placeholder associated with each supported compose field."""
    if index is None or regions.get("layer") != "page":
        return []
    if layout is None:
        layout = mail_compose_layout(frame)
    if layout is None:
        return []
    cache, actions = {}, []
    for name, caption in zip(("recipient", "subject", "body"), layout["captions"], strict=True):
        box = layout[name]
        if not _available(box, regions) or not _available(caption, regions):
            continue
        reading = read_caption(frame, caption, index, cache=cache)
        if reading is None:
            reading = read_caption(frame, caption, index, cache=cache, choice=True)
        if reading is None:
            continue
        label = reading["text"]
        x, y, r, b = box
        actions.append({"role": "input", "function": f"输入{label}", "preview_text": label,
                        "box": list(box), "click_point": [round(x + .39 * (r - x)), round(y + .43 * (b - y))]})
    return actions
