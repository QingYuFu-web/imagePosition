"""Read grouped icon-and-headline drawer rows inside an already verified panel."""

from collections import Counter
import re

import cv2
import numpy as np

from screen_caption_transcript import _bands
from screen_native_tab_bar import _distance
from screen_portrait_drawer import _fallback_caption, _glyph_box, _split_lane


def _bands_with_ink(ink, minimum, bridge):
    marks = np.uint8((ink.sum(axis=1) >= minimum)[:, None])
    return _bands(cv2.morphologyEx(marks, cv2.MORPH_CLOSE, np.ones((bridge, 1), np.uint8)).ravel() != 0)


def _shapes(layer, shift, low, high, min_ink, bridge):
    a, y0 = shift
    output = []
    for top, bottom in _bands_with_ink(layer, min_ink, bridge):
        if not low <= bottom - top <= high:
            continue
        start = max(0, top - 2)
        x, y, w, h = cv2.boundingRect(np.uint8(layer[start:bottom + 2]))
        if w and h:
            output.append((a + x, y0 + start + y, a + x + w, y0 + start + y + h))
    return output


def _text_groups(icons, text, width):
    rows = []
    for x, y, right, bottom in icons:
        w, h = right - x, bottom - y
        center = (y + bottom) / 2
        if not (.037 * width <= h <= .085 * width and .031 * width <= w <= .093 * width
                and .48 <= w / h <= 1.87):
            continue
        matching = [head for head in text if abs((head[1] + head[3]) / 2 - center) <= .72 * h]
        if not 1 <= len(matching) <= 2:
            continue
        first = matching[0]
        a, t, b, end = first
        if not .025 * width <= end - t <= .072 * width or not .079 * width <= b - a <= .63 * width:
            continue
        detail = None
        if len(matching) == 2:
            left, top, _, last = matching[1]
            if not (.45 * (end - t) <= last - top <= .94 * (end - t)
                    and abs(a - left) <= .027 * width and .12 * h <= top - end <= .55 * h
                    and (t + end) / 2 < center < (top + last) / 2):
                continue
            detail = matching[1]
        elif abs((t + end) / 2 - center) > .23 * h:
            continue
        rows.append((first, detail, x, center))
    if not 5 <= len(rows) <= 20:
        return []
    expected = float(np.median([item[0][0] for item in rows]))
    lead = float(np.median([item[2] for item in rows]))
    return [item for item in rows if abs(item[0][0] - expected) <= .024 * width and abs(item[2] - lead) <= .032 * width]


def _flow(frame, box, body, paint):
    left, start, right, bottom = box
    width, height = right - left, bottom - start
    top = start + round(.037 * height)
    if not top < body + .20 * height < bottom:
        return None
    tile = frame[top:bottom, left:right]
    ink = _distance(tile, paint)
    paper_samples = tile[:, round(.46 * width):round(.92 * width):4]
    line_color = np.median(paper_samples, axis=1)
    uniformity = (abs(paper_samples - line_color[:, None, :]).max(axis=2) <= 10).mean(axis=1)
    quiet = (abs(line_color - paint).max(axis=1) < 37) & (uniformity >= .70)
    binary = (ink >= 43) & quiet[:, None]
    if not (split := _split_lane(binary)):
        return None
    text = _shapes(binary[:, split:round(.944 * width)], (left + split, top), .016 * width, .092 * width,
                   max(4, round(.006 * width)), max(3, round(.0064 * width) | 1))
    icons = _shapes(binary[:, round(.025 * width):split], (left + round(.025 * width), top),
                    .028 * width, .109 * width, max(3, round(.0045 * width)), max(3, round(.0078 * width) | 1))
    rows = _text_groups(icons, text, width)
    if len(rows) < 5:
        return None
    gaps = np.diff([row[3] for row in rows])
    core = gaps[(gaps >= .096 * width) & (gaps <= .224 * width)]
    if len(core) < 4:
        return None
    pitch = float(np.median(core))
    repeated = abs(gaps - pitch) <= .087 * pitch
    if repeated.sum() < max(4, round(.56 * len(gaps))) or max(gaps) > 2.2 * pitch:
        return None
    separators = [(top + a, top + b) for a, b in _bands((ink[:, round(.07 * width):round(.92 * width)] >= 13).mean(axis=1) >= .92)
                  if 1 <= b - a <= .083 * pitch]
    inside = [edge for edge in separators if any(old[3] + .25 * pitch < edge[0] < new[3] - .25 * pitch
              for old, new in zip(rows[:-1], rows[1:], strict=True))]
    if len(inside) < 2 and sum(row[1] is not None for row in rows) < 2:
        return None
    return rows, pitch, separators


def _selected(frame, box, center, pitch, paint):
    left, _, right, _ = box
    width = right - left
    y0, y1 = round(center - .208 * pitch), round(center + .208 * pitch)
    sample = frame[y0:y1:2, left + round(.54 * width):left + round(.88 * width):3]
    ink = np.median(sample, axis=(0, 1))
    return abs(ink - paint).max() >= 13 and (abs(sample - ink).max(axis=2) <= 9).mean() >= .77


def _ellipsis(mask, text_height):
    spans = _bands(mask.any(axis=0))
    if len(spans) < 5:
        return None
    last = spans[-3:]
    dims = []
    for a, b in last:
        _, y, w, h = cv2.boundingRect(mask[:, a:b])
        if not .072 * text_height <= min(w, h) <= max(w, h) <= .26 * text_height or y < .63 * text_height:
            return None
        dims.append((y, w, h))
    if (np.ptp([a for a, _, _ in dims]) > .08 * text_height
            or not .041 * text_height <= min(b[0] - a[1] for a, b in zip(last[:-1], last[1:], strict=True))):
        return None
    return last[0][0]


def _latin_case(frame, caption, paint, index):
    left, top, right, bottom = caption
    ink = _distance(frame[top:bottom, left:right], paint)
    peak = float(np.percentile(ink, 97))
    if peak < 70:
        return None
    votes = {}
    for k in (.30, .43, .56, .65):
        mask = np.uint8(ink >= k * peak)
        spans = _bands(mask.any(axis=0))
        if not 3 <= len(spans) <= 23:
            continue
        pieces = []
        for a, b in spans:
            _, y, _, h = cv2.boundingRect(mask[:, a:b])
            rank = index.rank(mask[:, a:b], alphabet="choice_latin", limit=3)
            if len(rank) >= 2:
                pieces.append((a, b, y, y + h, rank))
        if len(pieces) != len(spans):
            continue
        cap = [(y, end) for a, b, y, end, ranks in pieces if ranks[0]["text"].isupper()
               and ranks[0]["distance"] < .145 and ranks[1]["distance"] - ranks[0]["distance"] >= .043
               and b - a > .38 * (end - y)]
        if not cap:
            continue
        upper, base = np.median(cap, axis=0)
        height = base - upper
        text = []
        if height < 12:
            continue
        for a, b, y, end, ranks in pieces:
            target, alt = ranks[:2]
            c, other = target["text"], alt["text"]
            near_cap = abs(y - upper) <= .06 * height and abs(end - base) <= .074 * height
            low = .16 * height <= y - upper <= .40 * height and abs(end - base) <= .11 * height
            if {c, other} == {"l", "I"} and target["distance"] < .037 and b - a <= .245 * height:
                if .65 <= upper - y <= .15 * height and abs(end - base) <= .086 * height:
                    c = "l"
                elif near_cap and len(cap) >= 2:
                    c = "I"
                else:
                    break
            elif c.isalpha() and other.isalpha() and c.lower() == other.lower():
                if not (near_cap or low):
                    break
                c = c.upper() if near_cap else c.lower()
            elif target["distance"] > .175 or alt["distance"] - target["distance"] < .024 or not c.isalnum():
                break
            text.append(c)
        if len(text) != len(spans):
            continue
        literal = text[0] + "".join((" " if a - before > .30 * height else "") + c
                                    for (_, before), (a, _), c in zip(spans[:-1], spans[1:], text[1:], strict=True))
        votes.setdefault(literal, set()).add(mask.tobytes())
    consensus = [label for label, masks in votes.items() if len(masks) >= 2]
    return consensus[0] if len(consensus) == 1 else None


def _headline(frame, caption, paint, index, cache):
    if label := _glyph_box(frame, caption, index, cache):
        return label
    x, y, right, bottom = caption
    cut = _ellipsis(np.uint8(_distance(frame[y:bottom, x:right], paint) >= 62), bottom - y)
    if cut:
        # Give the transcript its normal padding without including the punctuation.
        pad = max(3, round(.13 * (bottom - y)))
        trimmed = (x, y, x + cut - pad - 1, bottom)
        if trimmed[2] > x and (base := _glyph_box(frame, trimmed, index, cache)):
            return base + "…"
    return _latin_case(frame, caption, paint, index) or _fallback_caption(frame, caption, paint, index, cache)


def grouped_drawer_rows(frame, box, body, paint, index):
    """Read complete owned rows and exclude selected, subtitle, and status text."""
    frame_height, frame_width = frame.shape[:2]
    left, top, right, bottom = box
    width = right - left
    if (index is None or not 0 <= left < right <= frame_width or not 0 <= top < bottom <= frame_height
            or not .45 * frame_width <= width <= .91 * frame_width or not top <= body < bottom):
        return []
    base = np.asarray(paint).reshape(3)
    result = _flow(frame, box, body, base)
    if not result:
        return []
    rows, pitch, dividers = result
    actions, cache = [], {}
    for i, (title, _, _, center) in enumerate(rows):
        if _selected(frame, box, center, pitch, base):
            continue
        name = _headline(frame, title, base, index, cache)
        if not name or re.search(r"\d+(?:\.\d+){1,4}(?:[-+][A-Za-z0-9.]+)?", name):
            continue
        y0, y1 = round(center - pitch / 2), round(center + pitch / 2)
        previous = max((b for a, b in dividers if b < center), default=top)
        following = min((a for a, b in dividers if a > center), default=bottom)
        if i:
            previous = max(previous, round((rows[i - 1][3] + center) / 2))
        if i + 1 < len(rows):
            following = min(following, round((rows[i + 1][3] + center) / 2))
        actions.append({"role": "app_drawer_row", "function": f"打开{name}", "preview_text": name,
                        "box": [left, max(previous, y0), right - round(.013 * width), min(following, y1)],
                        "click_point": [(title[0] + title[2]) // 2, round(center)]})
    counts = Counter(row["function"] for row in actions)
    result = [row for row in actions if counts[row["function"]] == 1]
    return result if len(result) >= 3 else []
