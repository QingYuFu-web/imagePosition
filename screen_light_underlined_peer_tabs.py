"""Recognize unselected peers in a light tab rail with a marked selection."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_component_captions import _bands, _tight


def _chroma(frame):
    b, g, r = cv2.split(frame)
    return cv2.subtract(cv2.max(cv2.max(b, g), r), cv2.min(cv2.min(b, g), r))


def _marks(frame):
    height, width = frame.shape[:2]
    start, stop = round(.070 * height), round(.46 * height)
    crop = frame[start:stop]
    full = cv2.inRange(crop, (219, 219, 219), (255, 255, 255))
    if (full != 0).mean() < .37:
        return []
    _, _, rows, _ = cv2.connectedComponentsWithStats(cv2.compare(_chroma(crop), 72, cv2.CMP_GE))
    return [(int(x), int(start + y), int(x + w), int(start + y + h)) for x, y, w, h, area in rows[1:]
            if .025 * width <= w <= .13 * width and max(2, .003 * width) <= h <= .017 * width
            and 4.45 <= w / h <= 14.5 and area >= .78 * w * h
            and .028 * width <= x and x + w <= .978 * width]


def _ink_profile(pixels):
    lower = pixels.min(axis=2, keepdims=True)
    delta = np.float32(pixels - lower)
    return delta / np.maximum(1, delta.max(axis=2, keepdims=True))


def _selected(frame, mark):
    width = frame.shape[1]
    l, t, r, b = mark
    mid = (l + r) / 2
    reach = round(1.35 * (r - l))
    left, right = max(0, round(mid) - reach), min(width, round(mid) + reach)
    start = max(0, t - round(.102 * width))
    stop = t - max(2, round(.006 * width))
    mask = frame[start:stop, left:right]
    if min(mask.shape[:2]) < 4:
        return None
    profile = np.median(_ink_profile(frame[t:b, l:r]), axis=(0, 1))
    ink = (_chroma(mask) >= 61) & (np.max(np.abs(_ink_profile(mask) - profile), axis=2) <= .22)
    band = [span for span in _bands(ink.sum(axis=1) >= max(3, round(.10 * (r - l)))) if span[1] - span[0] >= .020 * width]
    if len(band) != 1:
        return None
    a, z = band[0]
    box = _tight(ink[a:z], left, start + a)
    if box is None:
        return None
    x, y, xx, yy = box
    size = yy - y
    if (not .021 * width <= size <= .059 * width or not 1.0 <= (xx - x) / size <= 4.5
            or not -.02 * size <= t - yy <= .94 * size or abs((x + xx) / 2 - mid) > .24 * size
            or x <= left + 1 or xx >= right - 1):
        return None
    pixels = ink[a:z, x - left:xx - left]
    if not .20 <= pixels.mean() <= .72:
        return None
    return box


def _paper(frame, selected, mark):
    width = frame.shape[1]
    _, top, _, bottom = selected
    _, _, _, line_bottom = mark
    size = bottom - top
    inset = round(.025 * width)
    margins = ((top - round(.61 * size), top - round(.13 * size)),
               (line_bottom + 1, line_bottom + round(.48 * size)))
    for l, r in margins:
        if r - l < 3 or l < 0 or r > len(frame):
            return False
        stripe = frame[l:r, inset:width - inset:2]
        if ((stripe.min(axis=2) >= 224).mean() < .82 or np.median(stripe) < 242
                or (_chroma(stripe) >= 48).mean() > .12):
            return False
    return True


def _badge_mask(chroma, size):
    mask = cv2.inRange(chroma, 52, 255)
    close = max(2, round(.14 * size))
    drawn = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    contours, _ = cv2.findContours(drawn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    covered = np.zeros_like(mask, dtype=bool)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        chip = drawn[y:y + h, x:x + w]
        if .7 * size <= w <= 3.4 * size and .48 * size <= h <= 1.3 * size and (chip != 0).mean() >= .60:
            covered[y:y + h, x:x + w] = True
    return covered


def _line_peers(frame, selected):
    l, t, r, b = selected
    height, width = frame.shape[:2]
    size = b - t
    pad = max(3, round(.22 * size))
    top, bottom = max(0, t - pad), min(height, b + pad)
    tile = frame[top:bottom]
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    chroma = _chroma(tile)
    badges = _badge_mask(chroma, size)
    # Exclude the measured selected word and neighboring badge paint. Dark
    # artwork lettering cannot supply several centered same-size peers here.
    badges[:, max(0, l - pad):min(width, r + pad)] = True
    traces = []
    for cut in (143, 179):
        ink = (gray <= cut) & (chroma <= 38) & ~badges
        cols = _bands(ink.sum(axis=0) >= max(2, round(.065 * size)))
        groups = []
        for first, last in cols:
            if groups and first - groups[-1][1] <= .73 * size:
                groups[-1] = groups[-1][0], last
            else:
                groups.append((first, last))
        boxes = []
        for first, last in groups:
            box = _tight(ink[:, first:last], first, top)
            if box is None:
                continue
            left, first_row, right, last_row = box
            extent = last_row - first_row
            if (not .77 * size <= extent <= 1.22 * size
                    or not .68 * size <= right - left <= 5.9 * size
                    or abs((first_row + last_row - t - b) / 2) > .21 * size):
                continue
            boxes.append(box)
        traces.append(boxes)
    first, second = traces
    limit = max(2, round(.11 * size))
    stable = [box for box in first if any(max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= limit
                                          for peer in second)]
    if len(stable) < 3:
        return None
    captions = sorted([selected, *stable])
    centers = [(a + z) / 2 for a, _, z, _ in captions]
    gaps = np.diff(centers)
    pitch = np.median(gaps)
    if (centers[-1] - centers[0] < .44 * width or not np.all((gaps >= .63 * pitch) & (gaps <= 1.72 * pitch))
            or gaps.min() < 1.85 * size or gaps.max() > .34 * width):
        return None
    return captions


def _read_name(frame, line, index, cache):
    l, t, r, b = line
    inset = max(3, round(.14 * (b - t)))
    tile = frame[max(0, t - inset):b + inset, max(0, l - inset):r + inset]
    work = tile.copy()
    work[_chroma(work) >= 47] = 255
    # Replicate a tall blank margin for the shared reader's viewport-relative
    # bounds, leaving device/font data local and surrounding controls masked.
    extra = max(0, round(16 * (b - t)) - work.shape[1])
    if extra:
        work = cv2.copyMakeBorder(work, 0, 0, 0, extra, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    area = 0, 0, tile.shape[1], tile.shape[0]
    result = read_caption(work, area, index, cache=cache, choice=True, join_radicals=True)
    return result["text"] if result else None


def light_underlined_peer_actions(frame, regions, index):
    """Return unselected tab click targets after locating their shared rail."""
    height, width = frame.shape[:2]
    if index is None or regions.get("layer") != "page" or regions.get("regions") or width < 248 or not 1.42 * width <= height <= 2.64 * width:
        return []
    matches = []
    for mark in _marks(frame):
        selected = _selected(frame, mark)
        if selected is None or not _paper(frame, selected, mark):
            continue
        if (labels := _line_peers(frame, selected)) is not None:
            matches.append((selected, mark, labels))
    if len(matches) != 1:
        return []
    selected, mark, labels = matches[0]
    candidates = []
    cache = {}
    for line in labels:
        l, t, r, b = line
        if line == selected or l <= .013 * width or r >= .977 * width:
            continue
        text = _read_name(frame, line, index, cache)
        if text and 2 <= len(text) <= 10 and all(char.isalpha() or char == " " for char in text):
            off = round(.32 * (b - t))
            box = [max(0, l - off), t - off, min(width, r + off), mark[3]]
            candidates.append({"role": "switch_tab", "function": f"切换到{text}", "preview_text": text,
                               "box": box, "click_point": [round((l + r) / 2), round((t + b) / 2)]})
    duplicates = {row["preview_text"] for row in candidates if sum(
        other["preview_text"] == row["preview_text"] for other in candidates) > 1}
    verified = [row for row in candidates if row["preview_text"] not in duplicates]
    return verified if len(verified) >= 2 else []
