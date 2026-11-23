"""Locate labeled collection cards with repeated edges and progress tracks."""

import cv2
import numpy as np

from screen_component_captions import _bands


def _midpoint(box):
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def _grid(candidates):
    if len(candidates) < 6:
        return []
    sizes = np.array([(r - l, b - t) for l, t, r, b in candidates])
    for width, height in sizes:
        aligned = [box for box, (w, h) in zip(candidates, sizes, strict=True)
                   if .89 * width <= w <= 1.12 * width and .88 * height <= h <= 1.14 * height]
        rows = []
        for box in sorted(aligned, key=lambda item: (_midpoint(item)[1], item[0])):
            if rows and abs(_midpoint(rows[-1][0])[1] - _midpoint(box)[1]) <= .10 * height:
                rows[-1].append(box)
            else:
                rows.append([box])
        full = []
        for row in rows:
            if len(row) < 3:
                continue
            ordered = sorted(row)
            gaps = np.diff([_midpoint(box)[0] for box in ordered])
            if .91 * width <= gaps.min() <= gaps.max() <= 1.31 * width and np.ptp(gaps) <= .13 * width:
                full.append(ordered)
        for a, b in zip(full, full[1:]):
            separation = _midpoint(b[0])[1] - _midpoint(a[0])[1]
            columns = sum(any(abs(_midpoint(first)[0] - _midpoint(second)[0]) <= .12 * width for second in b)
                          for first in a)
            if .90 * height <= separation <= 1.47 * height and columns >= 3:
                return [a, b]
    return []


def _edge_support(gray, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    start, end = top + round(.35 * height), bottom - round(.19 * height)
    edge = round(.15 * width)
    for x in (left, right - edge):
        side = gray[start:end:2, x:x + edge].astype(np.int16)
        contrast = abs(np.diff(side, axis=1))
        support = (contrast >= 8).mean(axis=0)
        if support.max(initial=0.) < .67:
            return False
    return True


def _progress_track(gray, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    a, b = left + round(.10 * w), right - round(.10 * w)
    start, end = top + round(.71 * h), top + round(.865 * h)
    crop = gray[start:end, a:b]
    paper = float(np.median(crop))
    ink = np.uint8(crop >= paper + 48)
    _, _, parts, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    return any(.24 * w <= width <= .79 * w and 2 <= height <= .034 * h and width >= 11 * height
               and area >= .61 * width * height and abs(a + x + width / 2 - (left + right) / 2) <= .19 * w
               for x, _, width, height, area in parts[1:])


def _letter_parts(ink):
    x, y, w, h = cv2.boundingRect(ink)
    if h < 10 or w < 1.60 * h:
        return []
    text = ink[y:y + h, x:x + w]
    spans = []
    projection = text.sum(axis=0)
    for left, right in _bands(projection > 0):
        width = right - left
        if 1.52 * h <= width <= 2.45 * h:
            start, end = left + round(.39 * width), left + round(.61 * width)
            cut = min(range(start, end), key=lambda at: (int(projection[at]), abs(at - (left + right) / 2)))
            if projection[cut] <= .30 * h:
                spans.extend(((left, cut), (cut, right)))
                continue
        spans.append((left, right))
    parts = []
    for left, right in spans:
        _, top, width, height = cv2.boundingRect(text[:, left:right])
        if (.11 * h <= width <= .33 * h and .11 * h <= height <= .32 * h
                and .30 * h <= top + height / 2 <= .69 * h):
            parts.append("·")
        elif (.70 * h <= width <= 1.33 * h and 1 <= height <= .18 * h
              and .32 * h <= top + height / 2 <= .68 * h):
            parts.append("一")
        elif .58 * h <= width <= 1.35 * h and .70 * h <= height <= h:
            parts.append(text[top:top + height, left:right])
        else:
            return []
    return parts if 2 <= len(parts) <= 11 and sum(type(part) is str for part in parts) <= 2 else []


def _line_text(roi, index, cache):
    paper = float(np.median(roi))
    observations = [_letter_parts(np.uint8(roi >= paper + bias)) for bias in (39, 45, 51, 63, 76, 90)]
    counts = sorted({len(parts) for parts in observations if parts
                     and sum(len(other) == len(parts) for other in observations) >= 2})
    if len(counts) != 1:
        return None
    votes = [parts for parts in observations if len(parts) == counts[0]]
    result = []
    for samples in zip(*votes, strict=True):
        ratings = {}
        for sample in samples:
            if isinstance(sample, str):
                ratings.setdefault(sample, {})[sample] = (.0, .10)
                continue
            key = sample.shape, sample.tobytes()
            if key not in cache:
                cache[key] = index.rank(sample, alphabet="cjk", limit=2)
            candidates = cache[key]
            if len(candidates) == 2 and candidates[0]["distance"] <= .208:
                best, other = candidates
                ratings.setdefault(best["text"], {})[key] = (best["distance"], other["distance"] - best["distance"])
        agreed = []
        for label, evidence in ratings.items():
            distances, margins = zip(*evidence.values(), strict=True)
            if label in {"·", "一"}:
                if sum(isinstance(sample, str) and sample == label for sample in samples) == len(samples):
                    agreed.append(label)
            elif min(distances) <= .172:
                strong = sum(margin >= .002 for margin in margins)
                if ((strong >= 2 and max(margins) >= .016)
                        or (strong >= 3 and max(margins) >= .008)):
                    agreed.append(label)
        if len(agreed) != 1:
            return None
        result.append(agreed[0])
    return "".join(result)


def _title(gray, box, index, cache):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    a, b = left + round(.11 * w), right - round(.11 * w)
    start, end = top + round(.52 * h), top + round(.711 * h)
    roi = gray[start:end, a:b]
    paper = float(np.median(roi))
    ink = roi >= paper + 56
    bands = [(y0, y1) for y0, y1 in _bands(ink.sum(axis=1) >= max(4, round(.054 * w)))
             if .041 * h <= y1 - y0 <= .11 * h and .545 * h <= start + (y0 + y1) / 2 - top <= .681 * h]
    if not bands or len(bands) > 2:
        return None
    results = []
    for y0, y1 in bands:
        x, _, width, _ = cv2.boundingRect(np.uint8(ink[y0:y1]))
        if not .025 * w <= width <= .78 * w or abs(a + x + width / 2 - (left + right) / 2) > .16 * w:
            return None
        pad = max(3, round(.17 * (y1 - y0)))
        area = (max(a, a + x - pad), start + y0 - pad, min(b, a + x + width + pad), start + y1 + pad)
        left, top, right, bottom = area
        result = _line_text(gray[top:bottom, left:right], index, cache)
        if result is None:
            return None
        results.append(result)
    return "".join(results)


def game_collection_card_actions(frame, regions, index):
    """Return exactly one target per visually verified and legible collection card."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or width < 1.4 * height or height < 265:
        return []
    candidates = []
    for row in regions["candidates"]:
        left, top, right, bottom = row["box"]
        w, h = right - left, bottom - top
        if (row.get("visibility") == "page" and .025 * height <= top < bottom <= .98 * height
                and 0 <= left < right <= width and .20 * height <= h <= .52 * height and .48 <= w / h <= .91):
            candidates.append((left, top, right, bottom))
    rows = _grid(candidates)
    if not rows:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    owned = [[box for box in row if _edge_support(gray, box)] for row in rows]
    if any(len(row) < 3 or sum(_progress_track(gray, box) for box in row) < max(2, len(row) // 2) for row in owned):
        return []
    actions, cache = [], {}
    for row in owned:
        for box in row:
            label = _title(gray, box, index, cache)
            if label and label[0].isalpha() and len(label.replace("·", "")) >= 2:
                x, y = _midpoint(box)
                actions.append({"role": "collection_card", "function": f"打开{label}", "preview_text": label,
                                "box": list(box), "click_point": [round(x), round(y)]})
    return [item for item in actions if sum(item["function"] == other["function"] for other in actions) == 1]
