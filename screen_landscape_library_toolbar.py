"""Verify compact Search and Sort icons on a native landscape library header."""

import cv2
import numpy as np

from screen_candidate_glyphs import _overlap
from screen_landscape_header import _magnifier


def _tile_rows(boxes, width, height):
    for columns in range(3, 8):
        pitch = width / columns
        potential = [box for box in boxes if .10 * height <= box[1] <= .61 * height
                     and .093 * height <= box[3] - box[1] <= .33 * height
                     and .72 * pitch <= box[2] - box[0] <= 1.02 * pitch
                     and 1.22 <= (box[2] - box[0]) / (box[3] - box[1]) <= 2.46]
        known = set()
        for anchor in sorted(potential, key=lambda box: box[1]):
            depth = anchor[3] - anchor[1]
            peers = [peer for peer in potential if abs(anchor[1] - peer[1]) <= .081 * depth
                     and .89 * depth <= peer[3] - peer[1] <= 1.11 * depth]
            slots = [[peer for peer in peers if abs((peer[0] + peer[2]) / 2 - (i + .5) * pitch) <= .13 * pitch]
                     for i in range(columns)]
            if any(len(slot) != 1 for slot in slots):
                continue
            row = tuple(slot[0] for slot in slots)
            if row not in known:
                known.add(row)
                yield row


def _media_context(gray, boxes):
    height, width = gray.shape
    rows = list(_tile_rows(boxes, width, height))
    pairs = []
    for upper in rows:
        y = float(np.median([box[1] for box in upper]))
        depth = float(np.median([b - t for _, t, _, b in upper]))
        if not .125 * height <= y <= .275 * height:
            continue
        for lower in rows:
            if len(lower) != len(upper):
                continue
            pitch = width / len(upper)
            if (not .96 * depth <= np.median([box[1] for box in lower]) - y <= 1.32 * depth
                    or any(abs((a[0] + a[2]) - (b[0] + b[2])) > .15 * pitch
                           or abs((a[3] - a[1]) - (b[3] - b[1])) > .13 * depth
                           for a, b in zip(upper, lower, strict=True))):
                continue
            pairs.append((upper, lower))
    if len(pairs) != 1:
        return None
    upper, lower = pairs[0]
    textured = 0
    for left, top, right, bottom in (*upper, *lower):
        px, py = round(.12 * (right - left)), round(.17 * (bottom - top))
        patch = gray[top + py:bottom - py, left + px:right - px].astype(np.int16)
        if patch.std() >= 26 and (abs(np.diff(patch, axis=0)) >= 10).mean() >= .087:
            textured += 1
    if textured < .80 * (len(upper) + len(lower)):
        return None
    return round(np.median([box[1] for box in upper]))


def _toolbar_groups(gray, boxes, seam, paint):
    height, width = gray.shape
    found = []
    for left, top, right, bottom in boxes:
        w, h = right - left, bottom - top
        if (left < .48 * width or top < .028 * height or bottom > seam + .020 * height
                or not .047 * height <= h <= .128 * height or not .65 <= w / h <= 1.49):
            continue
        pad = gray[top:bottom, left:right].astype(np.int16)
        ink = np.uint8(abs(pad - paint) >= 47)
        x, y, iw, ih = cv2.boundingRect(ink)
        if (not .29 * h <= ih <= .68 * h or not .064 * w <= iw <= .70 * w
                or abs(y + ih / 2 - h / 2) > .16 * h or abs(x + iw / 2 - w / 2) > .16 * w
                or (abs(pad - paint) <= 15).mean() < .78):
            continue
        found.append((left, top, right, bottom))
    if not 4 <= len(found) <= 7:
        return None
    found.sort()
    centers = [(left + right) / 2 for left, _, right, _ in found]
    vertical = [(top + bottom) / 2 for _, top, _, bottom in found]
    gaps = np.diff(centers)
    typical = np.median(gaps)
    skipped = gaps > 1.60 * typical
    spacing = gaps / (1 + skipped)
    if (not .030 * width <= typical <= .088 * width or skipped.sum() > 1
            or np.ptp(spacing) > .32 * typical or np.ptp(vertical) > .023 * height
            or found[-1][2] < .90 * width):
        return None
    return found


def _status_and_surface(gray, stripe, seam, paint):
    height, width = gray.shape
    cap = min(box[1] for box in stripe)
    top = gray[:cap].astype(np.int16)
    status = abs(top - paint) >= 30
    if not cap or status[:, round(.105 * width):round(.835 * width)].mean() > .021:
        return False
    left, right = status[:, :round(.10 * width)], status[:, round(.86 * width):]
    spans = [cv2.boundingRect(np.uint8(part)) for part in (left, right)]
    if any(not .028 * width <= w <= .165 * width or not .013 * height <= h <= .047 * height
           for _, _, w, h in spans):
        return False
    if abs((spans[0][1] + spans[0][3] / 2) - (spans[1][1] + spans[1][3] / 2)) > .017 * height:
        return False
    post_tools = max(box[3] for box in stripe)
    buffer = gray[post_tools:seam].astype(np.int16)
    if len(buffer) < .010 * height or (abs(buffer - paint) <= 15).mean() < .95:
        return False
    before = gray[seam - round(.012 * height):seam].astype(np.int16)
    after = gray[seam:seam + round(.028 * height)].astype(np.int16)
    return (abs(before - paint) <= 15).mean() >= .92 and (abs(after - paint) >= 30).mean() >= .61


def _sort_strokes(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Ignore isolated antialiasing specks while requiring all three bars.
    bars = sorted((row.tolist() for row in stats[1:count] if row[4] >= 4), key=lambda item: item[1])
    if len(bars) != 3:
        return None
    left, top, width, height, _ = bars[0]
    bottoms = [y + h for _, y, _, h, _ in bars]
    steps = np.diff([y + h / 2 for _, y, _, h, _ in bars])
    if (not .42 * width <= bars[1][2] <= .76 * width or not .16 * width <= bars[2][2] <= .42 * width
            or bars[2][2] >= .76 * bars[1][2] or max(abs(x - left) for x, _, _, _, _ in bars) > .084 * width
            or not .21 * width <= min(steps) <= max(steps) <= .48 * width or np.ptp(steps) > .095 * width
            or any(not 1 <= h <= .22 * width or area < .68 * w * h for _, _, w, h, area in bars)
            or sum(row[4] for row in bars) < .89 * mask.sum()):
        return None
    return left, top, left + width, max(bottoms)


def _sort_mark(gray, box, paint):
    left, top, right, bottom = box
    tile = abs(gray[top:bottom, left:right].astype(np.int16) - paint)
    peak = float(np.percentile(tile, 99.4))
    if peak < 54:
        return None
    marks = []
    for cut in (.38, .52, .65):
        result = _sort_strokes(np.uint8(tile >= cut * peak))
        if result:
            x, y, X, Y = result
            if (.24 * (right - left) <= X - x <= .69 * (right - left)
                    and .24 * (bottom - top) <= Y - y <= .62 * (bottom - top)
                    and min(x, y, right - left - X, bottom - top - Y) >= 2):
                marks.append((left + x, top + y, left + X, top + Y))
    if len(marks) >= 2 and all(_overlap(marks[0], box) >= .76 for box in marks):
        return marks[len(marks) // 2]
    return None


def landscape_library_toolbar_actions(frame, regions, index):
    """Name unique header glyphs only when the library owns their row."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or height < 224 or width < 1.45 * height:
        return []
    boxes = [tuple(x["box"]) for x in regions["candidates"] if x.get("visibility") == "page"
             and 0 <= x["box"][0] < x["box"][2] <= width and 0 <= x["box"][1] < x["box"][3] <= height]
    if len(boxes) < 10:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    seam = _media_context(gray, boxes)
    if seam is None:
        return []
    paint = float(np.median(gray[:seam, round(.28 * width):round(.52 * width)]))
    stripe = _toolbar_groups(gray, boxes, seam, paint)
    if not stripe or not _status_and_surface(gray, stripe, seam, paint):
        return []
    ink = 255 - gray if paint >= 128 else gray
    searches = [mark for box in stripe if (mark := _magnifier(ink, box))]
    sorts = [mark for box in stripe if (mark := _sort_mark(gray, box, paint))]
    if len(searches) == 1 and len(sorts) == 1 and _overlap(searches[0], sorts[0]) != 0:
        return []
    return [{"role": role, "function": label, "preview_text": label, "box": list(box),
             "click_point": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]}
            for role, label, marks in (("search", "搜索", searches), ("sort", "排序", sorts))
            if len(marks) == 1 for box in marks]
