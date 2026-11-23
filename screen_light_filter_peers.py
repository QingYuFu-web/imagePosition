"""Read unselected filters in a light horizontal row of painted chips."""

import cv2
import numpy as np

from inventory_list_rows import _Box
from inventory_top_filter_chips import _separate_groups, _Chip
from screen_caption_transcript import read_caption


def _enclosure(contour, box):
    h = box.height
    if not .79 * box.width * h <= cv2.contourArea(contour) <= .987 * box.width * h:
        return False
    inset = max(1, round(.026 * h))
    corners = [(x, y) for x in (box.left + inset, box.right - 1 - inset)
               for y in (box.top + inset, box.bottom - 1 - inset)]
    return all(cv2.pointPolygonTest(contour, corner, False) <= 0 for corner in corners)


def _chip_boxes(gray):
    _, width = gray.shape
    paper = int(np.median(gray[::3, round(.025 * width):round(.975 * width):3]))
    if paper < 236:
        return []
    measurements = []
    for cut in (paper - 4, paper - 6, paper - 8, paper - 11, paper - 17):
        image = cv2.inRange(gray, 0, cut)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for shape in contours:
            x, y, w, h = cv2.boundingRect(shape)
            if (not .043 * width <= h <= .091 * width or not 1.26 * h <= w <= min(5.4 * h, .38 * width)
                    or not 1 < x < x + w < width - 2 or not 1 < y < y + h < len(gray) - 2):
                continue
            if _enclosure(shape, _Box(x, y, x + w, y + h)):
                measurements.append((cut, _Box(x, y, x + w, y + h)))
    groups = []
    for cut, box in measurements:
        linked = [group for group in groups if max(abs(a - b) for a, b in zip(box.xyxy(), group[0][1].xyxy(), strict=True))
                  <= max(2, round(.08 * box.height))]
        if not linked:
            groups.append([(cut, box)])
        elif len(linked) == 1:
            linked[0].append((cut, box))
    return [_Box(*map(int, np.rint(np.median([box.xyxy() for _, box in group], axis=0))))
            for group in groups if len({cut for cut, _ in group}) >= 2]


def _paper_edges(gray, box, color):
    x, y, r, b = box.xyxy()
    size = box.height
    inset = round(.28 * size)
    pad = max(2, round(.11 * size))
    above = gray[y - pad:y, x + inset:r - inset]
    below = gray[b:b + pad, x + inset:r - inset]
    brightness = round(.114 * color[0] + .587 * color[1] + .299 * color[2])
    return all(edge.size and np.median(edge) >= 234 and (edge >= brightness + 4).mean() >= .80
               for edge in (above, below))


def _face(frame, box):
    x, y, r, b = box.xyxy()
    size = box.height
    horizontal, vertical = round(.22 * size), round(.11 * size)
    tile = frame[y + vertical:b - vertical, x + horizontal:r - horizontal]
    paint = np.median(tile.reshape(-1, 3), axis=0)
    close = (np.max(np.abs(tile - paint), axis=2) <= 10).mean()
    if close < (.48 if np.ptp(paint) >= 15 else .57):
        return None
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).astype(np.int16)
    ink = gray - round(.114 * paint[0] + .587 * paint[1] + .299 * paint[2])
    high, low = (ink >= 56).mean(), (ink <= -56).mean()
    if not .035 <= max(high, low) <= .39 or min(high, low) > .025:
        return None
    return tuple(map(int, np.rint(paint))), 1 if high > low else -1


def _colored_selection(frame, chip):
    fill = np.asarray(chip.fill, dtype=np.int16)
    if np.ptp(fill) < 15:
        return False
    left, top, right, bottom = chip.box.xyxy()
    tile = frame[top:bottom, left:right]
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    mid = round(.114 * fill[0] + .587 * fill[1] + .299 * fill[2])
    strong = tile[gray <= mid - 57].astype(np.int16)
    if len(strong) < .039 * gray.size:
        return False
    direction = fill - fill.mean()
    dominant = max(np.abs(direction))
    color = np.median(strong, axis=0)
    return np.ptp(color) >= 53 and (direction @ (color - color.mean())) >= dominant * 40


def _unselected(frame, group):
    verified = []
    for active in group:
        peers = [chip for chip in group if chip is not active]
        anchor = peers[0]
        if not all(chip.text_polarity < 0 and np.ptp(chip.fill) <= 12 and min(chip.fill) >= 210
                   and max(abs(a - b) for a, b in zip(chip.fill, anchor.fill, strict=True)) <= 7 for chip in peers):
            continue
        if ((active.text_polarity > 0 and max(active.fill) <= min(anchor.fill) - 35)
                or (active.text_polarity < 0 and _colored_selection(frame, active))):
            verified.append(peers)
    return verified[0] if len(verified) == 1 else []


def _caption(frame, chip, index, cache):
    x, y, right, bottom = chip.box.xyxy()
    h = chip.box.height
    dx, dy = round(.19 * h), round(.13 * h)
    crop = (x + dx, y + dy, right - dx, bottom - dy)
    reading = read_caption(frame, crop, index, cache=cache, join_radicals=True)
    if reading is None:
        reading = read_caption(frame, crop, index, cache=cache, choice=True, join_radicals=True,
                               sentence_case=True, join_letters=True)
    if reading is None:
        return None
    left, top, end, stop = reading["line_box"]
    text = reading["text"]
    if (not .25 * h <= stop - top <= .67 * h or not .39 * h <= end - left <= chip.box.width - .33 * h
            or abs(left + end - x - right) > .27 * h or abs(top + stop - y - bottom) > .23 * h
            or not 2 <= len(text) <= 24 or not all(char.isalpha() or char == " " for char in text)):
        return None
    return text


def _filter_row(frame, scan_start, scan_end):
    width = frame.shape[1]
    # Shape measurement uses a bounded raster; reading retains native pixels.
    short = min(width, 640)
    scale = short / width
    if scan_end - scan_start < .06 * width:
        return None
    header = frame[scan_start:scan_end]
    complete = header if scale == 1 else cv2.resize(header, (short, round(len(header) * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(complete, cv2.COLOR_BGR2GRAY)
    chips = []
    for box in _chip_boxes(gray):
        face = _face(complete, box)
        if face is not None and _paper_edges(gray, box, face[0]):
            chips.append(_Chip(box, *face))
    rows = {tuple(sorted((peer for peer in chips if abs(peer.box.center[1] - chip.box.center[1]) <= .11 * chip.box.height),
                        key=lambda peer: peer.box.left)) for chip in chips}
    groups = {tuple(group) for row in rows for group in _separate_groups(row) if len(group) >= 4}
    matched = [(group, peers) for group in groups if (peers := _unselected(complete, group))]
    if len(matched) != 1:
        return None
    y_scale = len(complete) / len(header)
    group, peers = matched[0]
    native = {}
    for chip in group:
        l, t, r, b = chip.box.xyxy()
        box = _Box(round(l / scale), scan_start + round(t / y_scale),
                   round(r / scale), scan_start + round(b / y_scale))
        native[chip] = _Chip(box, chip.fill, chip.text_polarity)
    selected = next(native[chip] for chip in group if chip not in peers)
    return selected, [native[chip] for chip in peers]


def selected_light_filter_at(frame, box, point):
    """Check the selected chip in the local header row without reading text."""
    height, width = frame.shape[:2]
    _, top, _, bottom = box
    x, y = point
    if width < 248 or not 1.42 * width <= height <= 2.72 * width or not .025 * height <= y <= .225 * height:
        return False
    reach = round(max(12, .44 * (bottom - top)))
    result = _filter_row(frame, max(0, top - reach), min(height, bottom + reach))
    if result is None:
        return False
    l, t, r, b = result[0].box.xyxy()
    return l <= x < r and t <= y < b


def light_filter_peer_actions(frame, regions, index, *, below=None):
    """Return one point per fully visible neutral filter with a selected peer."""
    height, width = frame.shape[:2]
    if index is None or regions.get("layer") != "page" or regions.get("regions") or width < 248 or not 1.42 * width <= height <= 2.72 * width:
        return []
    result = _filter_row(frame, max(round(.045 * height), below or 0), round(.225 * height))
    if result is None:
        return []
    _, peers = result
    cache = {}
    results = []
    for chip in peers:
        if text := _caption(frame, chip, index, cache):
            results.append({"role": "switch_tab", "function": f"切换到{text}", "preview_text": text,
                            "box": list(chip.box.xyxy()), "click_point": list(chip.box.center)})
    if len(results) < 2 or len({row["preview_text"].casefold() for row in results}) != len(results):
        return []
    return results
