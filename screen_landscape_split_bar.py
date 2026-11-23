"""Verify a leading arrow and capped pencil in a native split-view toolbar."""

import cv2
import numpy as np

from inventory_back_evidence import is_shaft_back_arrow
from screen_back_control import _title_run
from screen_candidate_glyphs import _overlap
from screen_landscape_app_bar import _clock_glyphs


def _surface(frame):
    height, width = frame.shape[:2]
    # Read down the columns so the clock, titles and icons do not define pane paint.
    strip = frame[round(.042 * height):round(.168 * height):4]
    length = round(.22 * width)
    left = np.median(strip[:, :length], axis=(0, 1)).astype(np.int16)
    right = np.median(strip[:, -length:], axis=(0, 1)).astype(np.int16)
    columns = np.median(strip, axis=0).astype(np.int16)
    match_left = np.max(np.abs(columns - left), axis=1) < 12
    if np.max(np.abs(left - right)) < 28:
        if np.max(np.abs(left - right)) >= 12 or match_left.mean() < .97:
            return None
        tone = cv2.cvtColor(np.uint8([[left]]), cv2.COLOR_BGR2GRAY).item()
        return width, tone, tone
    match_right = np.max(np.abs(columns - right), axis=1) < 12
    edges = np.flatnonzero(match_left[:-1] & ~match_left[1:]) + 1
    gaps = max(2, round(.003 * width))
    cuts = [int(edge) for edge in edges if .28 * width <= edge <= .67 * width
            and match_right[edge:edge + gaps].any()]
    if len(cuts) != 1:
        return None
    seam = int(cuts[0])
    if match_left[:seam].mean() < .96 or match_right[seam:].mean() < .96:
        return None
    tones = cv2.cvtColor(np.uint8([[left, right]]), cv2.COLOR_BGR2GRAY)[0]
    return seam, int(tones[0]), int(tones[1])


def _status(left, right, height):
    width = left.shape[1] if left is right else left.shape[1] + right.shape[1]
    edge = right[:round(.056 * height), -round(.15 * width):]
    _, _, parts, _ = cv2.connectedComponentsWithStats(np.uint8(edge >= 65), connectivity=8)
    for x, y, w, h, area in parts[1:]:
        if not (.012 * height <= h <= .040 * height and 1.46 <= w / h <= 3.8
                and area >= .81 * w * h and x >= .43 * edge.shape[1] and 0 < y):
            continue
        # Require independently drawn radio signal beside the filled battery.
        neighbors = sum(.20 * w <= x - (a + c) <= 2.6 * w and d >= .26 * h
                        and abs(b + d / 2 - y - h / 2) <= .62 * h for a, b, c, d, _ in parts[1:])
        clock = left[:min(round(.067 * height), int(y + 1.76 * h)), :round(.09 * width)]
        if neighbors >= 2 and _clock_glyphs(clock):
            return True
    return False


def _pencil(numbers, parts, index):
    x, y, w, h, area = parts[index]
    if not (.83 <= w / h <= 1.20 and .27 * w * h <= area <= .51 * w * h):
        return None
    yy, xx = np.nonzero(numbers[y:y + h, x:x + w] == index)
    along, across = xx - yy, xx + yy
    if np.corrcoef(xx, yy)[0, 1] > -.77:
        return None
    section = (along - along.min()) / np.ptp(along)
    tip = np.ptp(across[section <= .065])
    middle = across[(section >= .31) & (section <= .68)]
    if not middle.size:
        return None
    shaft = np.ptp(middle)
    if not (.19 * w <= shaft <= .61 * w and tip <= .82 * shaft):
        return None
    for a, b, c, d, cap_area in parts[1:]:
        if not (.25 * w <= c <= .50 * w and .25 * h <= d <= .50 * h and
                .34 * c * d <= cap_area <= .84 * c * d):
            continue
        cap_x, cap_y = a + c / 2, b + d / 2
        if (.97 * w <= cap_x - x <= 1.16 * w and -.31 * h <= cap_y - y <= .17 * h
                and abs(cap_x - x + cap_y - y - np.median(across)) <= .19 * w):
            return int(x), int(b), int(a + c), int(y + h)
    return None


def _marks(gray, ink, tone, seam, y0, height, cut, single):
    _, numbers, parts, _ = cv2.connectedComponentsWithStats(np.uint8(ink[y0:] >= cut), connectivity=8)
    for i, (x, y, w, h, _) in enumerate(parts[1:], 1):
        if not (.017 * height <= h <= .085 * height and .79 <= w / h <= 1.25
                and .26 * h <= x <= min(2.75 * h, .17 * seam)
                and is_shaft_back_arrow(numbers[y:y + h, x:x + w] == i)):
            continue
        x0 = x + w + round(.37 * h)
        x1 = min(seam, x0 + round(17 * h))
        title_end = None
        for rim in (.53, .81):
            top, bottom = max(0, y + y0 - round(rim * h)), min(len(gray), y + y0 + h + round(rim * h))
            run = _title_run(gray[top:bottom, x0:x1], tone, h, minimum=3, high=1.18)
            if run and .54 * h <= x0 + run[0] - x - w <= 2.9 * h and run[1] - run[0] >= 1.9 * h:
                title_end = x0 + run[1]
                break
        if title_end is None:
            continue
        peers = [j for j, (a, b, c, d, _) in enumerate(parts[1:], 1)
                 if a >= title_end + .55 * h and abs(b + d / 2 - y - h / 2) <= .33 * h
                 and .68 * h <= d <= 1.47 * h and .72 <= c / d <= 1.36]
        back = tuple(map(int, (x, y + y0, x + w, y + y0 + h)))
        edits = []
        for peer in peers:
            pencil = _pencil(numbers, parts, peer)
            if pencil is None:
                continue
            a, b, c, d = pencil
            if any(.44 * h <= a - parts[j][0] - parts[j][2] <= 5.0 * h for j in peers if j != peer):
                edits.append((a, b + y0, c, d + y0))
        # Across a continuous surface, require the right-hand command group
        # as well as the title before claiming this is a toolbar.
        if not single or (len(peers) >= 2 and len(edits) == 1 and edits[0][0] >= .69 * seam):
            return back, edits
    return None, []


def _action(box, role, label):
    a, b, c, d = box
    return {"role": role, "function": label, "preview_text": label, "box": list(box),
            "click_point": [(a + c) // 2, (b + d) // 2]}


def landscape_split_bar_actions(frame, regions):
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or width < 1.32 * height or height < 260:
        return []
    surface = _surface(frame)
    if surface is None:
        return []
    seam, left_tone, right_tone = surface
    single = seam == width
    gray = cv2.cvtColor(frame[:round(.204 * height)], cv2.COLOR_BGR2GRAY)
    left, right = gray[:, :seam], gray[:, seam:]
    left_ink = np.abs(left.astype(np.int16) - left_tone)
    right_ink = left_ink if single else np.abs(right.astype(np.int16) - right_tone)
    if not _status(left_ink, right_ink, height):
        return []
    start = round(.054 * height)
    detections = [_marks(left, left_ink, left_tone, seam, start, height, cut, single) for cut in (52, 71)]
    (first, edits), (second, peers) = detections
    if first is None or second is None or _overlap(first, second) < .84:
        return []
    actions = [_action(first, "back", "返回")]
    if len(edits) == len(peers) == 1 and _overlap(edits[0], peers[0]) >= .81:
        actions.append(_action(edits[0], "edit", "编辑"))
    return actions
