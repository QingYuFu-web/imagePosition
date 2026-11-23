"""Verify leading navigation glyphs against a physically independent toolbar title."""

import cv2
import numpy as np

from inventory_back_evidence import _chevrons, is_shaft_back_arrow
from screen_back_control import _adjacent_native_title, _center, _heading, _status_band, _title_run, _unique
from screen_candidate_glyphs import _overlap


def _joined_shaft(gray, box, width):
    left, top, right, bottom = box
    chip = gray[top:bottom, left:right].astype(np.int16)
    delta = chip - np.median(chip)
    if (np.abs(delta) <= 12).mean() < .86:
        return []
    closed_size = max(3, round(.004 * width) | 1)
    kernel = np.ones((closed_size, closed_size), np.uint8)
    votes = []
    for sign in (-1, 1):
        for cut in (31, 49, 72):
            mask = np.uint8(sign * delta >= cut)
            _, _, fragments, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if (fragments[1:, cv2.CC_STAT_AREA] >= .004 * mask.size).sum() != 2:
                continue
            stitched = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            _, ids, parts, _ = cv2.connectedComponentsWithStats(stitched, connectivity=8)
            for i, (x, y, w, h, area) in enumerate(parts[1:], 1):
                if (not .028 * width <= h <= .072 * width or not 0 < x < x + w < chip.shape[1]
                        or not 0 < y < y + h < chip.shape[0] or abs(x + w / 2 - chip.shape[1] / 2) > .17 * chip.shape[1]
                        or abs(y + h / 2 - chip.shape[0] / 2) > .15 * chip.shape[0]):
                    continue
                body = np.uint8(ids[y:y + h, x:x + w] == i)
                ink = mask[y:y + h, x:x + w]
                if .81 * area <= int((body & ink).sum()) < area and is_shaft_back_arrow(body):
                    mark = (left + x, top + y, left + x + w, top + y + h)
                    votes.append((sign, cut, mark))
    return [mark for sign, cut, mark in votes if any(
        polarity == sign and other_cut != cut and _overlap(mark, other) >= .84
        for polarity, other_cut, other in votes
    )]


def _breadcrumb(frame, gray, box):
    left, top, right, bottom = box
    height = bottom - top
    seam = left + round(.72 * height)
    icon = gray[top:bottom, left:seam].astype(np.int16)
    paper = float(np.median(icon))
    if paper < 184 or (np.abs(icon - paper) <= 10).mean() < .82:
        return []
    eroder = np.ones((3, 3), np.uint8)
    votes = []
    for cut in (37, 55, 77):
        ink = np.uint8(icon <= paper - cut)
        x, y, w, h = cv2.boundingRect(ink)
        if not (0 < x < x + w < icon.shape[1] - 3 and 0 < y < y + h < height
                and .39 * height <= h <= .76 * height and .40 <= w / h <= .73):
            continue
        thinner = cv2.erode(ink, eroder, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        if not list(_chevrons(thinner, y + h / 2, .78 * h, h)):
            continue
        mark = (left + x, top + y, left + x + w, top + y + h)
        votes.append((cut, mark))
    stable = _unique([mark for cut, mark in votes if any(
        cut != other and _overlap(mark, alt) >= .85 for other, alt in votes)])
    if len(stable) != 1:
        return []

    a, t, b, d = mark = stable[0]
    size = d - t
    if not _status_band(gray, mark) or not _heading(gray, mark, below=False):
        return []
    style = frame[t:d, a:b][gray[t:d, a:b] <= paper - 68]
    label = gray[top:bottom, seam:right]
    extent = _title_run(label, paper, size, minimum=4, high=1.04)
    if not style.size or extent is None:
        return []
    paint = np.median(style, axis=0)
    first, last = extent
    if (np.ptp(paint) < 63 or not .31 * size <= seam + first - b <= .97 * size
            or not 1.72 * size <= last - first <= 6.1 * size):
        return []
    printed = frame[top:bottom, seam + first:seam + last][label[:, first:last] <= paper - 73]
    if not printed.size or np.max(np.abs(np.median(printed, axis=0) - paint)) > 24:
        return []

    # There are two different text roles on this row: the accented previous
    # destination, and a centered neutral current title beyond that control.
    width = gray.shape[1]
    end = round(.73 * width)
    title = gray[top:bottom, right:end]
    found = _title_run(title, paper, size, minimum=4, high=1.12)
    if found is None:
        return []
    start, stop = found
    center = frame[top:bottom, right + start:right + stop][title[:, start:stop] <= paper - 73]
    if (center.size and np.ptp(np.median(center, axis=0)) <= 20
            and .40 * width <= right + (start + stop) / 2 <= .59 * width
            and right + start - (seam + last) >= .60 * size):
        return [mark]
    return []


def titled_back_action(frame, regions, *, gray=None):
    """Return one back mark with an aligned title or colored breadcrumb."""
    height, width = frame.shape[:2]
    if (regions["layer"] != "page" or regions.get("regions") or width < 260
            or not 1.40 * width <= height <= 2.85 * width):
        return None
    owners = []
    for row in regions["candidates"]:
        a, t, b, d = box = tuple(row["box"])
        if (row["visibility"] == "page" and 0 <= a < b <= width and 0 <= t < d <= height
                and a <= .063 * width and .039 * height <= (t + d) / 2 <= .107 * height
                and .048 * width <= d - t <= .127 * width):
            owners.append(box)
    if not owners:
        return None
    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = []
    for a, t, b, d in owners:
        w, h = b - a, d - t
        if .57 <= w / h <= 1.45 and .055 * width <= w <= .146 * width:
            for mark in _joined_shaft(gray, (a, t, b, d), width):
                if (.024 * width <= _center(mark)[0] <= .105 * width and _status_band(gray, mark)
                        and _adjacent_native_title(gray, mark, True)):
                    found.append(mark)
        elif a <= .032 * width and 2.11 <= w / h <= 5.9 and .164 * width <= w <= .38 * width:
            found.extend(_breadcrumb(frame, gray, (a, t, b, d)))
    unique = _unique(found)
    if len(unique) != 1:
        return None
    box = [int(value) for value in unique[0]]
    x, y = _center(box)
    if any(a <= x < b and t <= y < d for a, t, b, d in regions.get("floating_controls", ())):
        return None
    return {"role": "back", "function": "返回上一级", "box": box, "click_point": [x, y]}
