"""Read paired text buttons that form an owned compact dialog footer."""

import cv2
import numpy as np

from screen_caption_transcript import _bands
from screen_native_tab_bar import _distance


def _footer_words(frame, panel, paint):
    left, top, right, bottom = panel
    w, h = right - left, bottom - top
    x0, x1 = left + round(.038 * w), right - round(.032 * w)
    y0, y1 = top + round(.626 * h), bottom - round(.052 * h)
    band = frame[y0:y1, x0:x1]
    ink = _distance(band, paint)
    color = np.ptp(band, axis=2)
    visible = (ink >= 30) & (color >= max(20., np.ptp(paint) + 16))
    matches = [(a, b) for a, b in _bands(visible.sum(axis=1) >= max(3, round(.017 * w)))
               if .028 * h <= b - a <= .092 * h]
    if len(matches) != 1:
        return None
    a, b = matches[0]
    if not .111 * h <= bottom - (a + b) / 2 - y0 <= .235 * h:
        return None
    letters = visible[max(0, a - 1):b + 1]
    y0 += max(0, a - 1)
    groups = []
    for start, end in _bands(letters.any(axis=0)):
        if groups and start - groups[-1][1] <= max(.040 * w, 1.30 * (b - a)):
            groups[-1][1] = end
        else:
            groups.append([start, end])
    if len(groups) != 2:
        return None
    boxes = []
    for start, end in groups:
        u, v, width, height = cv2.boundingRect(np.uint8(letters[:, start:end]))
        if not .85 * (b - a) <= height <= 1.25 * (b - a) or not 1.16 <= width / height <= 9.7:
            return None
        boxes.append((x0 + start + u, y0 + v, x0 + start + u + width, y0 + v + height))
    first, second = boxes
    if (first[0] < left + .465 * w or not .057 * w <= right - second[2] <= .187 * w
            or not .055 * w <= second[0] - first[2] <= .22 * w
            or not .198 * w <= second[2] - first[0] <= .52 * w
            or abs(first[1] - second[1]) > max(1, .006 * h)
            or abs(first[3] - second[3]) > max(1, .006 * h)):
        return None
    return boxes


def _footer_surface(frame, panel, paint, words):
    left, top, right, bottom = panel
    w, h = right - left, bottom - top
    a, b = left + round(.074 * w), right - round(.074 * w)
    first = min(word[1] for word in words)
    last = max(word[3] for word in words)
    leading = frame[first - round(.155 * h):first - round(.028 * h), a:b]
    trailing = frame[last + round(.021 * h):bottom - round(.048 * h), a:b]
    if any(tile.size == 0 or (_distance(tile, paint) <= 13).mean() < .968 for tile in (leading, trailing)):
        return False
    # There is visible content before the quiet gap. Plain colored letters in an
    # empty drawing or a panel-wide indicator are not sufficient.
    content = frame[top + round(.063 * h):first - round(.174 * h), a:b]
    ink = _distance(content, paint) >= 35
    rows = _bands(ink.sum(axis=1) >= max(5, round(.032 * w)))
    if sum(end - start >= .022 * h for start, end in rows) < 2:
        return False
    pigments = []
    for left, top, right, bottom in words:
        module = frame[top:bottom, left:right]
        color = np.ptp(module, axis=2)
        crisp = module[(color >= 27) & (_distance(module, paint) >= 71)]
        if len(crisp) < 8:
            return False
        pigment = np.median(crisp, axis=0) - paint
        if np.ptp(pigment) < 40:
            return False
        pigments.append(pigment)
    a, b = pigments
    return ((a @ b) >= .962 * np.linalg.norm(a) * np.linalg.norm(b)
            and min(np.linalg.norm(a), np.linalg.norm(b)) >= .71 * max(np.linalg.norm(a), np.linalg.norm(b)))


def _capital(ranks, width, height, allow_stem):
    nearest = ranks[0]
    letter = next((entry for entry in ranks if entry["text"].isascii() and entry["text"].isupper()), None)
    ceiling = .224 if height <= 12 else .183
    if not letter or letter["distance"] > ceiling or letter["distance"] - nearest["distance"] > .055:
        return None
    character, winner = letter["text"], nearest["text"]
    if character == winner and ranks[1]["distance"] - nearest["distance"] >= .023:
        return character, True
    if winner.upper() == character:
        return character, False
    if character == "O" and winner == "0" and .683 <= width / height <= 1.01:
        return character, False
    if allow_stem and character == "I" and winner == "l" and width <= .24 * height:
        return character, False
    return None


def _read_caps(frame, box, paint, index):
    x, y, right, bottom = box
    height = bottom - y
    ink = _distance(frame[y:bottom, x:right], paint)
    peak = float(np.percentile(ink, 97))
    if peak < 73:
        return None
    votes = {}
    for level in (.194, .233, .277, .326, .379):
        cut = np.uint8(ink >= level * peak)
        spans = _bands(cut.any(axis=0))
        if not 2 <= len(spans) <= 14:
            continue
        glyphs = []
        for a, b in spans:
            x0, y0, w, h = cv2.boundingRect(cut[:, a:b])
            if not .82 * height <= h <= height or not .13 <= w / h <= 1.11:
                break
            ranks = index.rank(cut[y0:y0 + h, a + x0:a + x0 + w], alphabet="choice_latin", limit=8)
            if len(ranks) < 2 or not (letter := _capital(ranks, w, h, len(spans) >= 4)):
                break
            glyphs.append((*letter, y0, y0 + h))
        if len(glyphs) != len(spans):
            continue
        _, anchors, tops, bottoms = zip(*glyphs, strict=True)
        if (max(np.ptp(tops), np.ptp(bottoms)) > max(1, .081 * height)
                or sum(anchors) < max(1, round(.41 * len(glyphs)))):
            continue
        label = glyphs[0][0]
        for (_, previous_end), (start, _), glyph in zip(spans[:-1], spans[1:], glyphs[1:], strict=True):
            label += (" " if start - previous_end >= .46 * height else "") + glyph[0]
        votes.setdefault(label, set()).add(cut.tobytes())
    confirmed = [name for name, cuts in votes.items() if len(cuts) >= 2]
    return confirmed[0] if len(confirmed) == 1 else None


def compact_dialog_footer_actions(frame, panel, index):
    """Read button captions on the shared footer of an owned dialog."""
    height, width = frame.shape[:2]
    if index is None or panel is None or len(panel) != 4:
        return []
    left, top, right, bottom = map(int, panel)
    w, h = right - left, bottom - top
    if not (0 <= left < right <= width and 0 <= top < bottom <= height and w >= 135
            and .56 * width <= w <= .945 * width and .39 * w <= h <= .87 * w):
        return []
    panel = (left, top, right, bottom)
    sample = frame[top + round(.13 * h):bottom - round(.071 * h):2,
                   left + round(.054 * w):right - round(.056 * w):2]
    paint = np.median(sample, axis=(0, 1))
    if (_distance(sample, paint) <= 12).mean() < .859:
        return []
    words = _footer_words(frame, panel, paint)
    if not words or not _footer_surface(frame, panel, paint, words):
        return []
    dx, dy = round(.043 * w), round(.070 * h)
    actions = []
    for word in words:
        label = _read_caps(frame, word, paint, index)
        if label is None:
            continue
        x0, y0, x1, y1 = word
        actions.append({
            "role": "modal_footer_action",
            "function": label,
            "preview_text": label,
            "box": [max(left, x0 - dx), max(top, y0 - dy), min(right, x1 + dx), min(bottom, y1 + dy)],
            "click_point": [(x0 + x1) // 2, (y0 + y1) // 2],
        })
    return actions if len({row["function"] for row in actions}) == len(actions) else []
