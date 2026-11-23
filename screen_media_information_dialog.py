"""Identify the foreground close on a video diagnostics overlay."""

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_dialog_close import _cross_footprint


VIDEO = {"video", "视频", "視訊", "ビデオ"}
AUDIO = {"audio", "音频", "音訊", "オーディオ"}
INFORMATION = {"video information", "视频信息", "視訊資訊", "ビデオ情報"}
METRICS = {"codec", "编解码器", "編解碼器", "コーデック"}


def _crosses(gray):
    height, width = gray.shape
    left, top = round(.85 * width), round(.029 * height)
    right, bottom = round(.994 * width), round(.205 * height)
    tile = gray[top:bottom, left:right]
    ink = tile >= 192
    contours, _ = cv2.findContours(np.uint8(ink), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not .022 * width <= min(w, h) <= max(w, h) <= .063 * width or not .83 <= w / h <= 1.20:
            continue
        glyph = tile[y:y + h, x:x + w]
        if all(.19 <= mark.mean() <= .53 and _cross_footprint(mark)
               for mark in (glyph >= 192, glyph >= 213)):
            yield (x + left, y + top, x + left + w, y + top + h)


def _panel_roof(gray, cross):
    height, width = gray.shape
    _, top, _, bottom = cross
    size = bottom - top
    active = gray[:, round(.15 * width):round(.86 * width)]
    ends = gray[:, np.r_[round(.06 * width):round(.17 * width), round(.85 * width):round(.95 * width)]]
    advance = max(3, round(.40 * size))
    witness = []
    for y in range(max(advance, top - round(3.2 * size)), top - max(5, round(.30 * size))):
        edge = ends[y].astype(np.int16) - ends[y - 1]
        before = np.median(ends[y - advance:y:2], axis=0)
        after = np.median(ends[y:y + advance:2], axis=0)
        if (((after >= 8) & (after <= 87) & (after - before >= 6)).mean() >= .88
                and (active[y + advance:min(height, y + round(3.1 * size)):3] <= 88).mean() >= .89
                and (edge >= 3).mean() >= .80):
            witness.append((float(np.median(edge)), y))
    return max(witness)[1] if witness else None


def _heading_boxes(frame, gray, cross, roof):
    height, width = gray.shape
    size = cross[3] - cross[1]
    start, limit = roof + max(2, round(.10 * size)), min(height, cross[3] + round(36 * size))
    stop = round(.37 * width)
    tile = frame[start:limit, :stop]
    magnitude = tile.max(axis=2) - tile.min(axis=2)
    ink = (magnitude >= 59) & (gray[start:limit, :stop] >= 78)
    for first, last in _bands(ink.sum(axis=1) >= max(4, round(.24 * size))):
        if not .47 * size <= last - first <= 1.36 * size:
            continue
        x, y, w, h = cv2.boundingRect(np.uint8(ink[first:last]))
        if x > .073 * width or not .073 * width <= w <= .331 * width:
            continue
        pad = max(3, round(.35 * size))
        yield (max(0, x - pad), first + y + start - pad,
               min(width, x + w + pad), first + y + start + h + pad)


def _word(frame, box, index, cache):
    record = read_caption(frame, box, index, cache=cache, choice=True, sentence_case=True)
    return record["text"].casefold() if record else None


def _codec_below(frame, gray, heading, size, index, cache):
    _, _, _, last = heading
    first = max(0, last + round(.11 * size))
    end = min(gray.shape[0], first + round(2.5 * size))
    x = round(.119 * gray.shape[1])
    patch = gray[first:end, :x] >= 139
    for start, stop in _bands(patch.sum(axis=1) >= 3):
        if not .58 * size <= stop - start <= 1.25 * size:
            continue
        pad = max(3, round(.16 * size))
        if _word(frame, (max(0, heading[0]), first + start - pad, x, first + stop + pad), index, cache) in METRICS:
            return True
    return False


def _belongs(frame, gray, cross, roof, index):
    size = cross[3] - cross[1]
    center_y = (cross[1] + cross[3]) / 2
    titles = list(_heading_boxes(frame, gray, cross, roof))
    primary = [box for box in titles if abs((box[1] + box[3]) / 2 - center_y) <= .83 * size
               and box[2] - box[0] <= 5.15 * size]
    following = [box for box in titles if 3.2 * size <= box[1] - center_y <= 18.5 * size
                 and box[2] - box[0] <= 5.15 * size]
    charts = [box for box in titles if 11.0 * size <= box[1] - center_y <= 34 * size
              and 3.45 * size <= box[2] - box[0] <= 11.1 * size]
    if len(primary) != 1 or not 1 <= len(following) <= 4 or not 1 <= len(charts) <= 4:
        return False
    cache = {}
    return (_word(frame, primary[0], index, cache) in VIDEO
            and any(_word(frame, box, index, cache) in AUDIO for box in following)
            and any(_word(frame, box, index, cache) in INFORMATION for box in charts)
            and _codec_below(frame, gray, primary[0], size, index, cache))


def media_information_dialog(frame, overlay, index):
    """Scope an identified information overlay without proposing actions on telemetry."""
    height, width = frame.shape[:2]
    if index is None or width < 340 or not 1.14 * width <= height <= 2.81 * width:
        return None
    if overlay is not None and overlay.kind != "dialog":
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.percentile(gray[:round(.19 * height):4, ::5], 72) > 88:
        return None
    matches = []
    for shape in _crosses(gray):
        roof = _panel_roof(gray, shape)
        if roof is None or not _belongs(frame, gray, shape, roof, index):
            continue
        left, top, right, bottom = shape
        pad = max(4, round(.23 * (bottom - top)))
        center = [(left + right) // 2, (top + bottom) // 2]
        if overlay is not None:
            if not overlay.contains(center):
                continue
            panel = tuple(map(int, overlay.panel))
        else:
            panel = (0, roof, width, height)
        box = [left - pad, max(panel[1], top - pad), min(width, right + pad), min(height, bottom + pad)]
        matches.append((panel, [{"role": "close", "function": "关闭视频信息", "box": box, "click_point": center}]))
    return matches[0] if len(matches) == 1 else None
