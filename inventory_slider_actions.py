"""Locate labeled horizontal sliders from a visible rail and a distinct thumb."""

from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import median

import cv2
import numpy as np

from inventory_list_rows import _Box


_RANGE_LABEL = re.compile(
    r"(?:\b(?:(?:screen|display|input|output|playback|recording|sound)\s+)?"
    r"(?:brightness|contrast|saturation|hue|exposure|gamma|opacity|volume|gain|balance|speed|strength|"
    r"threshold|intensity|sensitivity|temperature|tint|blur|radius|sharpness|quality|minimum|maximum|"
    r"similarity|zoom|slider)\b|\blevel\s*\(\s*db\s*\)|"
    r"亮度|对比度|對比度|饱和度|飽和度|色调|色調|曝光|透明度|不透明度|音量|增益|速度|灵敏度|靈敏度|"
    r"阈值|閾值|强度|強度|缩放|縮放|滑块|滑塊)(?:\s*[:：])?",
    re.IGNORECASE,
)
_UNIT_LABEL = re.compile(r"(?<=\))\s*[:：]?\s*[-−]?\d+$")


@dataclass(frozen=True)
class _Rail:
    left: int
    inner_left: int
    inner_right: int
    right: int
    y: int


@dataclass(frozen=True)
class _Slider:
    thumb: _Box
    track: _Box


def _spans(line):
    limits = np.flatnonzero(np.diff(np.r_[False, line, False])).tolist()
    return zip(limits[::2], limits[1::2])


def _inside(box, region):
    return (region[0] <= box.left < box.right <= region[2]
            and region[1] <= box.top < box.bottom <= region[3])


def _label(text):
    candidate = _UNIT_LABEL.sub("", text).strip().rstrip(":：")
    return candidate if _RANGE_LABEL.fullmatch(candidate) else None


def _range_labels(nodes, regions):
    for node in nodes:
        text = str(node.get("content") or "").strip()
        if (node.get("kind") != "text" or (node.get("score") or 0) < 84
                or not 1 <= len(text) <= 72 or not any(letter.isalpha() for letter in text)):
            continue
        box = _Box(*map(round, node["box"]))
        for region in regions:
            limit = min(region[2] - region[0], region[3] - region[1])
            if 8 <= box.height <= .145 * limit and _inside(box, region):
                yield box, text, region
                break


def _local_bands(box, region):
    x0, y0, x1, y1 = region
    h = box.height
    cy = box.center[1]
    yield "side", (max(x0, box.right), max(y0, round(cy - 1.30 * h)), x1,
                   min(y1, round(cy + 1.30 * h)))
    yield "below", (max(x0, round(box.left - 1.15 * h)), max(y0, box.bottom), x1,
                    min(y1, round(box.bottom + 3.85 * h)))


def _aligned(box, rail, mode):
    h = box.height
    if mode == "side":
        return abs(rail.y - box.center[1]) <= .62 * h and .07 * h <= rail.left - box.right <= 4.0 * h
    return .55 * h <= rail.y - box.bottom <= 3.12 * h and -.8 * h <= rail.left - box.left <= 3.35 * h


def _rails(frame, band, label, mode):
    x0, y0, x1, y1 = band
    h = label.height
    chip = frame[y0:y1, x0:x1]
    if chip.shape[0] < h or chip.shape[1] < 4.8 * h:
        return
    spans = set()
    for offset in sorted({max(2, round(h * n)) for n in (.21, .32, .43)}):
        row_count = chip.shape[0] - 2 * offset
        if row_count <= 0:
            continue
        body = chip[offset:offset + row_count]
        above = cv2.absdiff(body, chip[:row_count]).max(axis=2)
        below = cv2.absdiff(body, chip[2 * offset:]).max(axis=2)
        contrast = np.minimum(above, below)
        kernel = np.ones((1, max(3, round(.79 * h))), np.uint8)
        for threshold in (13, 20):
            mask = cv2.morphologyEx(np.uint8(contrast >= threshold), cv2.MORPH_OPEN, kernel)
            for y, row in enumerate(mask):
                bands = list(_spans(row))
                for (a, b), (c, d) in zip(bands, bands[1:]):
                    if (not .20 * h <= c - b <= 2.35 * h or min(b - a, d - c) < .81 * h
                            or d - a < max(4.95 * h, 3.3 * (c - b)) or a <= 0 or d >= chip.shape[1]):
                        continue
                    rail = _Rail(x0 + a, x0 + b, x0 + c, x0 + d, y0 + offset + y)
                    if rail not in spans and _aligned(label, rail, mode):
                        spans.add(rail)
                        yield rail


def _rail_box(frame, rail, height):
    widths = (rail.inner_left - rail.left, rail.right - rail.inner_right)
    left, right = ((rail.left, rail.inner_left) if widths[0] >= widths[1]
                   else (rail.inner_right, rail.right))
    left += round((right - left) * .18)
    right -= round((right - left) * .20)
    pad = max(3, round(.66 * height))
    pixels = frame[rail.y - pad:rail.y + pad + 1, left:right]
    if pixels.shape[0] < 2 * pad + 1 or pixels.shape[1] < 4:
        return None
    trace = np.median(pixels, axis=1)
    background = np.median(np.vstack((trace[:2], trace[-2:])), axis=0)
    occupied = np.flatnonzero(np.max(np.abs(trace - background), axis=1) >= 12)
    occupied = occupied[np.abs(occupied - pad) <= max(3, round(.31 * height))]
    if not occupied.size:
        return None
    return _Box(rail.left, rail.y - pad + int(occupied.min()), rail.right, rail.y - pad + int(occupied.max()) + 1)


def _colored_thumb(frame, rail, track, height, region):
    pad = max(2, round(.22 * height))
    x0, x1 = max(region[0], rail.inner_left - pad), min(region[2], rail.inner_right + pad)
    y0, y1 = max(region[1], round(rail.y - 1.11 * height)), min(region[3], round(rail.y + 1.11 * height))
    pixels = frame[y0:y1, x0:x1]
    if min(pixels.shape[:2]) < 7:
        return None
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(chroma >= 26), connectivity=8)
    results = []
    for x, y, w, h, area in stats[1:]:
        box = _Box(int(x0 + x), int(y0 + y), int(x0 + x + w), int(y0 + y + h))
        if (not .36 * height <= w <= 2.3 * height or not .55 * height <= h <= 1.82 * height
                or not .40 <= w / h <= 2.08 or area < .54 * w * h
                or box.top > track.top - 2 or box.bottom < track.bottom + 2
                or abs(box.center[1] - rail.y) > .31 * h
                or abs(box.center[0] - (rail.inner_left + rail.inner_right) / 2) > .42 * w
                or w < .64 * (rail.inner_right - rail.inner_left)):
            continue
        results.append(box)
    return results[0] if len(results) == 1 else None


def _light_thumb(frame, rail, track, height, region):
    gap = rail.inner_right - rail.inner_left
    if gap < 1.03 * height:
        return None
    arms = (frame[rail.y, rail.left:rail.inner_left], frame[rail.y, rail.inner_right:rail.right])
    if not any(np.median(arm.max(axis=1) - arm.min(axis=1)) >= 22 for arm in arms):
        return None
    cx = (rail.inner_left + rail.inner_right) / 2
    left, right = round(cx - .20 * gap), round(cx + .20 * gap)
    y0 = max(region[1], round(rail.y - max(1.03 * height, .72 * gap)))
    y1 = min(region[3], round(rail.y + max(1.03 * height, .72 * gap)))
    lightness = np.median(frame[y0:y1, left:right].min(axis=2), axis=1)
    middle = rail.y - y0
    if not 2 <= middle < len(lightness) - 2 or lightness[middle] < 232:
        return None
    solid = lightness >= lightness[middle] - 2
    for top, bottom in _spans(solid):
        if not top <= middle < bottom or top < 2 or bottom + 2 > len(lightness):
            continue
        body_h = bottom - top
        if (not .59 * height <= body_h <= 1.9 * height or not 1.04 <= gap / body_h <= 2.42
                or y0 + top > track.top - 2 or y0 + bottom < track.bottom + 2
                or abs(middle - (top + bottom) / 2) > .23 * body_h
                or min(lightness[middle] - lightness[top - 2:top].min(),
                       lightness[middle] - lightness[bottom:bottom + 2].min()) < 7):
            continue
        return _Box(rail.inner_left, y0 + int(top), rail.inner_right, y0 + int(bottom))
    return None


def _find_slider(frame, label, region):
    candidates = []
    for mode, band in _local_bands(label, region):
        for rail in _rails(frame, band, label, mode):
            track = _rail_box(frame, rail, label.height)
            if track is None or track.height > .58 * label.height:
                continue
            thumb = _colored_thumb(frame, rail, track, label.height, region)
            if thumb is None:
                thumb = _light_thumb(frame, rail, track, label.height, region)
            if thumb is None or not _inside(track, region) or not _inside(thumb, region):
                continue
            slider = _Slider(thumb, track)
            if not any(abs(thumb.center[0] - old.thumb.center[0]) <= .33 * thumb.width
                       and abs(thumb.center[1] - old.thumb.center[1]) <= .32 * thumb.height for old in candidates):
                candidates.append(slider)
    return candidates[0] if len(candidates) == 1 else None


def _stepper_boundary(gray, thumb, rail_left, end):
    """Confirm the pair of small glyphs beyond a classic beveled rail."""
    h = thumb.height
    start = thumb.right + max(1, round(.18 * h))
    if end - start < 1.28 * h:
        return None
    mid = round(thumb.center[1])
    top, bottom = mid - round(.36 * h), mid + round(.36 * h) + 1
    pixels = gray[top:bottom, start:end]
    mask = np.uint8(pixels < np.median(pixels) - 73)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    buttons = []
    for x, y, w, height, area in stats[1:]:
        if (.21 * h <= min(w, height) <= .70 * h and max(w, height) <= .82 * h
                and .27 * w * height <= area <= .89 * w * height and y > 0 and y + height < mask.shape[0]):
            buttons.append((int(start + x), int(w), int(y), int(height)))
    buttons.sort()
    for (first, w, top, height), (second, _, other_top, other_h) in zip(buttons, buttons[1:]):
        if (.65 * h <= second - first <= 1.95 * h and end - second <= .97 * h
                and abs(top + height / 2 - other_top - other_h / 2) <= .19 * h):
            cut = first - max(1, round(.28 * h))
            if cut >= thumb.right + .15 * h and cut - rail_left >= 3.1 * h and w < second - first:
                return cut
    return None


def _beveled_rail(gray, thumb, region):
    h = thumb.height
    x0 = max(region[0], round(thumb.center[0] - 8.5 * h))
    x1 = min(region[2], round(thumb.center[0] + 8.5 * h))
    best = None
    for y in range(max(region[1] + 3, thumb.top - 1), thumb.top + max(2, round(.25 * h))):
        foreground = gray[y, x0:x1].astype(np.float32)
        reference = np.median(gray[y - 3:y - 1, x0:x1], axis=0)
        mask = cv2.morphologyEx(np.uint8([reference - foreground >= 17]), cv2.MORPH_CLOSE,
                                np.ones((1, 3), dtype=np.uint8))[0]
        for start, end in _spans(mask):
            left, right = x0 + int(start), x0 + int(end)
            if (left > thumb.left - .17 * h or right < thumb.right + 1.20 * h
                    or not 4.50 * h <= right - left <= 11.0 * h or left == x0 or right == x1):
                continue
            cut = _stepper_boundary(gray, thumb, left, right)
            if cut is not None and (best is None or right - left > best[0]):
                best = (right - left, _Box(left, min(y, thumb.top), cut, thumb.bottom))
    return best[1] if best else None


def _labels_share_axis(first, second):
    smallest, largest = sorted((first.height, second.height))
    return (abs(first.center[0] - second.center[0]) <= .82 * smallest
            and largest <= 1.54 * smallest
            and 1.22 * smallest <= abs(first.center[1] - second.center[1]) <= 5.5 * largest)


def _compact_thumbs(frame, label, region):
    h = label.height
    mid = label.center[0]
    x0, x1 = max(region[0], round(mid - 6 * h)), min(region[2], round(mid + 6 * h))
    y0, y1 = max(region[1], round(label.top - 1.44 * h)), min(region[3], round(label.top + .20 * h))
    pixels = frame[y0:y1, x0:x1]
    if min(pixels.shape[:2]) <= 8:
        return
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(chroma >= 24), connectivity=8)
    for x, y, w, height, area in stats[1:]:
        thumb = _Box(int(x0 + x), int(y0 + y), int(x0 + x + w), int(y0 + y + height))
        if (.93 * h <= w <= 1.85 * h and .54 * h <= height <= 1.02 * h and 1.12 <= w / height <= 2.1
                and area >= .68 * w * height and -.10 * h <= label.top - thumb.bottom <= .38 * h
                and x > 0 and x + w < pixels.shape[1] and y > 0 and y + height < pixels.shape[0]):
            yield thumb


def _tracks_share_axis(first, second):
    return (abs(first.left - second.left) <= .52 * first.height
            and abs(first.right - second.right) <= .52 * first.height
            and 1.35 * first.height <= abs(first.center[1] - second.center[1]) <= 6.9 * first.height)


def _recaptured_rail(gray, thumb, region, peers):
    """Check a thin, broken bevel against independently measured nearby axes."""
    if not peers:
        return None
    height = thumb.height
    expected_left = median(track.left for track in peers)
    expected_right = median(track.right for track in peers)
    start = max(region[0], round(expected_left - .35 * height))
    end = min(region[2], round(expected_right + 1.96 * height))
    right = _stepper_boundary(gray, thumb, round(expected_left), end)
    if (right is None or abs(right - expected_right) > .36 * height
            or not start < thumb.left < thumb.right < right):
        return None
    best = None
    for y in range(max(region[1] + 3, thumb.top - 1), thumb.top + max(2, round(.30 * height))):
        reference = np.median(gray[y - 3:y - 1, start:right], axis=0)
        occupied = reference - gray[y, start:right] >= 12
        edge = next((start + first for first, last in _spans(occupied)
                     if last - first >= 2 and abs(start + first - expected_left) <= .35 * height), None)
        if edge is None:
            continue
        rail = occupied[edge - start:]
        third = len(rail) // 3
        density = float(rail.mean())
        longest_gap = max((last - first for first, last in _spans(~rail)), default=0)
        if (density >= .86 and min(rail[:third].mean(), rail[-third:].mean()) >= .74
                and longest_gap <= .32 * height and (best is None or density > best[0])):
            best = density, _Box(edge, min(y, thumb.top), right, thumb.bottom)
    return best[1] if best else None


def _find_compact_slider(frame, gray, label, region, *, peers=()):
    x = label.center[0]
    candidates = []
    for thumb in _compact_thumbs(frame, label, region):
        track = _recaptured_rail(gray, thumb, region, peers) if peers else _beveled_rail(gray, thumb, region)
        if track is not None and track.left <= x <= track.right + .40 * label.height:
            candidates.append(_Slider(thumb, track))
    return candidates[0] if len(candidates) == 1 else None


def _compact_name(gray, box, text, region, reader):
    label = _label(text)
    read = getattr(reader, "recognize_prepared_lines", None)
    if label is not None or read is None or not (text.isascii() and text.isalpha()):
        return label or text
    # A tiny title may be misread through its JPEG halo. Confirm any
    # one-letter correction from two views of the detached title pixels.
    patch = gray[max(region[1], box.top - 1):min(region[3], box.bottom + 1),
                 max(region[0], box.left - 2):min(region[2], box.right + 2)]
    _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inverse = cv2.resize(255 - binary, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    try:
        readings = read([cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in (binary, inverse)])
    except RuntimeError:
        return text
    names = {entry["text"] for entry in readings if entry["score"] >= 92}
    if len(readings) != 2 or len(names) != 1 or any(entry["score"] < 92 for entry in readings):
        return text
    name = next(iter(names))
    if (len(name) == len(text) and sum(a != b for a, b in zip(name.casefold(), text.casefold())) == 1
            and (corrected := _label(name)) is not None):
        return corrected
    return text


def _compact_family(frame, labels, line_reader):
    supported = [entry for entry in labels if _label(entry[1])]
    if len(supported) < 2:
        return
    gray = None
    candidates, unresolved = [], []
    for bounds, text, region in labels:
        if sum(other_region == region and _labels_share_axis(bounds, other)
               for other, _, other_region in supported) < 2:
            continue
        if not 1.6 * bounds.height <= bounds.width <= 6.2 * bounds.height:
            continue
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        slider = _find_compact_slider(frame, gray, bounds, region)
        if slider is not None:
            candidates.append((bounds, text, slider, region))
        else:
            unresolved.append((bounds, text, region))
    verified = tuple(candidates)
    for bounds, text, region in unresolved:
        peers = [slider.track for other, _, slider, area in verified
                 if area == region and _labels_share_axis(bounds, other)]
        first = next((track for track in peers if any(_tracks_share_axis(track, other) for other in peers)), None)
        if first is not None:
            lane = [first, *(track for track in peers if _tracks_share_axis(first, track))]
            slider = _find_compact_slider(frame, gray, bounds, region, peers=lane)
            if slider is not None:
                candidates.append((bounds, text, slider, region))
    for bounds, label, slider, region in candidates:
        peer_count = sum(other_region == region and other != label and _tracks_share_axis(slider.track, peer.track)
                         for _, other, peer, other_region in candidates)
        if peer_count >= 2:
            yield _compact_name(gray, bounds, label, region, line_reader), slider


def suggest_slider_actions(frame_bgr, nodes, *, overlay=None, line_reader=None):
    """Return supported drag starts and measured track boxes without a value scale."""
    height, width = frame_bgr.shape[:2]
    regions = ((0, 0, width, height),) if overlay is None else overlay.regions
    labels = list(_range_labels(nodes, regions))
    sliders = [(label, slider) for bounds, text, region in labels if (label := _label(text)) is not None
               and (slider := _find_slider(frame_bgr, bounds, region)) is not None]
    sliders.extend(_compact_family(frame_bgr, labels, line_reader))
    actions = []
    for label, slider in sliders:
        thumb, track = slider.thumb, slider.track
        box = (min(thumb.left, track.left), min(thumb.top, track.top),
               max(thumb.right, track.right), max(thumb.bottom, track.bottom))
        actions.append({"role": "adjust_slider", "function": f"拖动{label}滑块", "setting_label": label,
                        "center": thumb.center, "box": box, "slider_thumb_box": thumb.xyxy(),
                        "slider_track_box": track.xyxy(), "slider_axis": "horizontal"})
    return [action for action in actions if sum(
        left <= action["center"][0] < right and top <= action["center"][1] < bottom
        for left, top, right, bottom in (other["slider_thumb_box"] for other in actions)
    ) == 1]
