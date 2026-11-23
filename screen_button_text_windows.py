"""Split isolated short labels into glyph masks using only pixel geometry."""

from math import ceil

import cv2
import numpy as np


_THRESHOLDS = (62, 82, 102, 122, 142, 162, 182, 202)
_BRIGHTNESS_GAP = 64


def _overlap(first, second):
    l0, t0, r0, b0 = first
    l1, t1, r1, b1 = second
    shared = max(0, min(r0, r1) - max(l0, l1)) * max(0, min(b0, b1) - max(t0, t1))
    area = (r0 - l0) * (b0 - t0) + (r1 - l1) * (b1 - t1)
    return shared / (area - shared) if shared else 0.


def _spans(columns, gap):
    limits = np.flatnonzero(np.diff(np.r_[False, columns, False]))
    merged = []
    for left, right in limits.reshape(-1, 2):
        left, right = int(left), int(right)
        if merged and left - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], right)
        else:
            merged.append((left, right))
    return merged


def _split_line(mask, left, top):
    height, width = mask.shape
    if width > 6.7 * height or width < .27 * height or not .13 <= mask.mean() <= .72:
        return None
    spans = _spans(mask.any(axis=0), max(1, round(.032 * height)))
    if not 1 <= len(spans) <= 7:
        return None
    masks = []
    boxes = []
    for index, (start, stop) in enumerate(spans):
        x, y, w, h = cv2.boundingRect(mask[:, start:stop])
        punct = (0 < index < len(spans) - 1 and .13 * height <= w <= .67 * height and w >= 2.1 * h
                 and h <= .29 * height and .27 * height <= y + h / 2 <= .69 * height)
        glyph = mask[y:y + h, start + x:start + x + w]
        if not punct and (not .68 * height <= h <= height or not .20 <= w / h <= 1.32
                          or not .12 <= glyph.mean() <= .80):
            return None
        masks.append(glyph.copy())
        boxes.append((left + start + x, top + y, left + start + x + w, top + y + h))
    return tuple(boxes), tuple(masks)


def _contrast(gray, mask, box):
    left, top, right, bottom = box
    pad = max(3, round(.10 * (bottom - top)))
    x0, y0 = max(0, left - pad), max(0, top - pad)
    x1, y1 = min(gray.shape[1], right + pad), min(gray.shape[0], bottom + pad)
    surround = gray[y0:y1, x0:x1]
    ink = np.zeros(surround.shape, np.uint8)
    ink[top - y0:bottom - y0, left - x0:right - x0] = mask
    clear = cv2.dilate(ink, np.ones((3, 3), np.uint8)) == 0
    if clear.sum() < .20 * surround.size:
        return False
    background = surround[clear]
    dark = float(np.median(surround[ink != 0]))
    if (background >= dark + 39).mean() < .78:
        return False
    cutoff = dark + _BRIGHTNESS_GAP
    if background.dtype == np.uint8:
        # Computing the 72nd percentile is unnecessary when both of
        # its bracketing order statistics are on the same side of the cutoff.
        below = np.count_nonzero(background < cutoff)
        position = .72 * (background.size - 1)
        if below <= int(position):
            return True
        if below > ceil(position):
            return False
    return bool(np.percentile(background, 72) >= cutoff)


def _groups(parts, height):
    current = []
    edge = 0
    for part in parts:
        _, left, _, right, _ = part
        if current and left - edge > .40 * height:
            yield current
            current = []
        edge = max(edge, right)
        current.append(part)
    if current:
        yield current


def _threshold_lines(gray, level):
    crop_height, crop_width = gray.shape
    ink = cv2.compare(gray, level, cv2.CMP_LE) if gray.dtype == np.uint8 else np.uint8(gray <= level)
    minimum = max(12, round(.16 * crop_height))
    x, y, w, h = cv2.boundingRect(ink)
    if h < minimum or w < 2:
        return
    offset_x, offset_y = x & -2, y & -2
    if cv2.getNumThreads() <= 1 and (x + w - offset_x) * (y + h - offset_y) < .75 * ink.size:
        # The 8-connected scan uses 2×2 blocks. Retain their original alignment
        # when removing an empty border, including the order of interior labels.
        ink = ink[offset_y:y + h, offset_x:x + w]
    else:
        offset_x = offset_y = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    membership = np.zeros(len(stats), dtype=np.uint8)
    parts = []
    for index, (x, y, w, h, area) in enumerate(stats[1:].tolist(), start=1):
        x, y = x + offset_x, y + offset_y
        if (0 < x < x + w < crop_width and 0 < y < y + h < crop_height and w >= 2 and h >= 2 and area >= 5
                and h <= .81 * crop_height and w <= max(1.62 * h, .26 * crop_height)):
            parts.append((index, x, y, x + w, y + h))
    # A tall fragment determines the line height. The complete label can have
    # smaller disconnected radicals or punctuation inside the same band.
    by_left = sorted(parts, key=lambda item: item[1])
    seen = set()
    grouped = {}
    for seed, _, y0, _, y1 in parts:
        height = y1 - y0
        if height < minimum:
            continue
        # The tighter padded band avoids absorbing a small second line beneath
        # a larger label. The wider band retains disconnected high/low strokes.
        for allowance in (.22, .44):
            key = y0, y1, allowance
            if key not in grouped:
                margin = allowance * height
                peers = [part for part in by_left if y0 - margin <= part[2] and part[4] <= y1 + margin]
                by_seed = {}
                for group in _groups(peers, height):
                    ids = tuple(part[0] for part in group)
                    by_seed.update((number, (ids, group)) for number in ids)
                grouped[key] = by_seed
            found = grouped[key].get(seed)
            if found is None or found[0] in seen:
                continue
            ids, group = found
            seen.add(ids)
            left = min(part[1] for part in group)
            top = min(part[2] for part in group)
            right = max(part[3] for part in group)
            bottom = max(part[4] for part in group)
            span = bottom - top
            if span < minimum or not .27 * span <= right - left <= 6.7 * span:
                continue
            if span > .86 * crop_height or right - left > .93 * crop_width:
                continue
            components = list(ids)
            membership[components] = 1
            mask = membership[labels[top - offset_y:bottom - offset_y, left - offset_x:right - offset_x]]
            membership[components] = 0
            split = _split_line(mask, left, top)
            if split is not None and _contrast(gray, mask, (left, top, right, bottom)):
                yield (left, top, right, bottom), split


def _threshold_observations(gray):
    if gray.dtype != np.uint8:
        for level in _THRESHOLDS:
            yield level, _threshold_lines(gray, level)
        return
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    counts = np.cumsum(histogram, dtype=np.int64)
    previous = -1
    for level in _THRESHOLDS:
        # The foreground masks are nested, so equal counts imply identical pixels.
        count = counts[level]
        if count != previous:
            # A full mask only has a border-touching component. OpenCV's
            # float32 histogram counts are exact up to 2**24 pixels.
            full = gray.size <= 2**24 and count == gray.size
            samples = [] if count == 0 or full else list(_threshold_lines(gray, level))
            previous = count
        yield level, samples


def _stable_lines(gray, polarity):
    measurements = []
    for level, samples in _threshold_observations(gray):
        for line, (boxes, masks) in samples:
            matches = [records for records in measurements if len(records[0][2]) == len(boxes)
                       and _overlap(records[0][1], line) >= .79 and all(
                           _overlap(a, b) >= .67 for a, b in zip(records[0][2], boxes, strict=True))]
            if matches:
                for samples in matches:
                    if all(other[0] != level for other in samples):
                        samples.append((level, line, boxes, masks))
            else:
                measurements.append([(level, line, boxes, masks)])
    for records in measurements:
        if len(records) < 3:
            continue
        level, line, boxes, masks = records[len(records) // 2]
        yield {"line_box": line, "glyph_boxes": boxes, "masks": masks,
               "variants": (records[0][3], records[-1][3]),
               "polarity": polarity, "support": len(records), "threshold": level}


def _duplicates(line, prior):
    left, top, right, bottom = line
    x0, y0, x1, y1 = prior
    return _overlap(line, prior) >= .40 or (x0 <= left < right <= x1 and y0 <= top < bottom <= y1)


def text_windows(frame, regions):
    """Return short lines with stable thresholds and intact original glyph masks.

    Each result has an original candidate ID and box, a line box and ordered
    glyph boxes in half-open original pixels. Masks are tightly cropped uint8
    arrays of zeros and ones. Candidate ownership is inherited from the input.
    """
    height, width = frame.shape[:2]
    short = min(height, width)
    output = []
    for item in regions["candidates"]:
        left, top, right, bottom = item["box"]
        h, w = bottom - top, right - left
        if (not 0 <= left < right <= width or not 0 <= top < bottom <= height
                or not 21 <= h <= .39 * short or not 25 <= w <= .64 * width):
            continue
        gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        if gray.dtype == np.uint8:
            low, high, _, _ = cv2.minMaxLoc(gray)
            # Even the padded contrast sample stays inside this crop. Both
            # polarities have the same intensity range after inversion.
            if high - low < _BRIGHTNESS_GAP:
                continue
        plausible = [*_stable_lines(gray, "dark"), *_stable_lines(255 - gray, "light")]
        plausible.sort(key=lambda row: (-(row["line_box"][3] - row["line_box"][1]), -row["support"]))
        accepted = []
        for row in plausible:
            if any(_duplicates(row["line_box"], prior["line_box"]) for prior in accepted):
                continue
            accepted.append(row)
            if len(accepted) == 2:
                break
        def absolute(box):
            return [box[0] + left, box[1] + top, box[2] + left, box[3] + top]

        for row in accepted:
            output.append({**row, "id": item["id"], "box": list(item["box"]),
                           "visibility": item.get("visibility", "page"),
                           "line_box": absolute(row["line_box"]),
                           "glyph_boxes": [absolute(box) for box in row["glyph_boxes"]]})
    return output
