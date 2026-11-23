"""Recover controls directly from a confirmed dialog surface."""

import cv2
import numpy as np

from screen_caption_transcript import _bands, _cjk_parts
from screen_dialog_close import _cross_footprint


def _face_edge(gray, edge, first, last, travel, direction):
    """Measure where the bright face rises out of its independently dimmed surroundings."""
    outside_start = edge - direction * travel
    positions = np.arange(outside_start, edge + direction * (travel + 1), direction)
    if len(positions) < 14 or positions.min() < 0 or positions.max() >= len(gray):
        return None
    profile = gray[positions, first:last:3].astype(np.int16)
    samples = max(4, round(.28 * travel))
    backdrop = np.median(profile[:samples], axis=0)
    if (backdrop <= 113).mean() < .83:
        return None
    support = ((profile >= 108) & (profile - backdrop >= 42)).mean(axis=1)
    for i in range(samples, len(positions) - samples + 1):
        if support[i:i + samples].min() >= .79:
            return int(positions[i])
    return None


def verified_dialog_face(frame, panel):
    """Measure all four edges of a painted dialog face from a provisional outer box."""
    height, width = frame.shape[:2]
    left, top, right, bottom = map(int, panel)
    w, h = right - left, bottom - top
    if (not 0 < left < right < width or not 0 < top < bottom < height or min(w, h) < 135
            or not .35 * width <= w <= .92 * width or not .245 * height <= h <= .73 * height):
        return None
    travel = max(9, round(.057 * min(w, h)))
    if min(left, top, width - right, height - bottom) <= travel:
        return None
    x, y = left - travel, top - travel
    gray = cv2.cvtColor(frame[y:bottom + travel + 1, x:right + travel + 1], cv2.COLOR_BGR2GRAY)
    a, b = top - y + round(.18 * h), bottom - y - round(.098 * h)
    p, q = left - x + round(.22 * w), right - x - round(.22 * w)
    x0 = _face_edge(gray.T, left - x, a, b, travel, 1)
    x1 = _face_edge(gray.T, right - x, a, b, travel, -1)
    y0 = _face_edge(gray, top - y, p, q, travel, 1)
    y1 = _face_edge(gray, bottom - y, p, q, travel, -1)
    if None in (x0, x1, y0, y1):
        return None
    if not .87 * w <= x1 - x0 <= 1.06 * w or not .88 * h <= y1 - y0 <= 1.05 * h:
        return None
    return x + x0, y + y0, x + x1 + 1, y + y1 + 1


def _quiet_corner_close(frame, panel, face):
    left, top, right, _ = panel
    _, upper, inner_right, _ = face
    span = right - left
    x0, x1 = right - round(.183 * span), inner_right - round(.011 * span)
    y0, y1 = upper + round(.023 * span), top + round(.145 * span)
    if min(x1 - x0, y1 - y0) < 18:
        return None
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    paint = float(np.median(gray))
    if paint < 160 or (abs(gray.astype(np.int16) - paint) <= 19).mean() < .87:
        return None
    candidates = []
    for cut in (56, 76, 92):
        mask = np.uint8(gray <= paint - cut)
        _, ids, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i, (a, b, w, h, area) in enumerate(stats[1:], 1):
            if (not .024 * span <= min(w, h) <= max(w, h) <= .076 * span or not .77 <= w / h <= 1.29
                    or not .15 <= area / (w * h) <= .51 or not 0 < a < a + w < mask.shape[1]
                    or not 0 < b < b + h < mask.shape[0]
                    or not .018 * span <= right - x0 - a - w / 2 <= .116 * span
                    or not .022 * span <= y0 + b + h / 2 - top <= .109 * span):
                continue
            if _cross_footprint(ids[b:b + h, a:a + w] == i):
                candidates.append(tuple(map(int, (x0 + a, y0 + b, x0 + a + w, y0 + b + h))))
    if len(candidates) < 2 or np.ptp(candidates, axis=0).max() > max(2, .007 * span):
        return None
    a, b, c, d = candidates[-1]
    margin = max(3, round(.15 * min(c - a, d - b)))
    return {"role": "close", "function": "关闭", "preview_text": "关闭",
            "box": [a - margin, b - margin, c + margin, d + margin],
            "click_point": [(a + c) // 2, (b + d) // 2]}


def _envelope_title(frame, panel, index):
    left, top, right, bottom = panel
    w, h = right - left, bottom - top
    pad = round(.12 * w)
    tile = cv2.cvtColor(frame[top + round(.318 * h):top + round(.575 * h), left + pad:right - pad],
                        cv2.COLOR_BGR2GRAY)
    paint = np.median(tile)
    kernel = np.ones((max(3, round(.009 * h)), 1), np.uint8)
    readings = {}
    for distance in (31, 39, 48, 59):
        mask = np.uint8(tile >= paint + distance)
        present = np.uint8(mask.sum(axis=1) >= max(3, round(.052 * w)))
        bands = _bands(cv2.morphologyEx(present[:, None], cv2.MORPH_CLOSE, kernel).ravel())
        for a, b in bands:
            if not .041 * h <= b - a <= .078 * h:
                continue
            x, y, width, height = cv2.boundingRect(mask[a:b])
            if not .195 * w <= width <= .75 * w or abs(x + width / 2 - tile.shape[1] / 2) > .054 * w:
                continue
            crop = mask[a + y:a + y + height, x:x + width]
            glyphs = _cjk_parts(crop, height, join_radicals=True)
            if len(glyphs) < 2:
                continue
            ranks = [index.rank(part[-1], alphabet="cjk", limit=2) for part in glyphs[-2:]]
            if not all(len(rank) == 2 for rank in ranks):
                continue
            word = "".join(rank[0]["text"] for rank in ranks)
            gaps = [rank[1]["distance"] - rank[0]["distance"] for rank in ranks]
            if word in {"红包", "紅包"} and max(rank[0]["distance"] for rank in ranks) <= .207 and min(gaps) >= .005:
                key = crop.tobytes()
                readings[key] = max(readings.get(key, 0), min(gaps))
                if len(readings) >= 2 and max(readings.values()) >= .022:
                    return True
    return False


def _envelope_disk(frame, panel):
    left, top, right, bottom = panel
    w, h = right - left, bottom - top
    if w < 172 or not .54 * frame.shape[1] <= w <= .88 * frame.shape[1] or not 1.16 * w <= h <= 1.87 * w:
        return None
    step = max(2, round(.013 * w))
    for x0, x1 in ((.067, .20), (.80, .93)):
        side = frame[top + round(.21 * h):bottom - round(.09 * h):step,
                     left + round(x0 * w):left + round(x1 * w):step]
        blue, green, red = (side[..., i].astype(np.int16) for i in range(3))
        if ((red >= 159) & (red - green >= 56) & (red - blue >= 65)).mean() < .83:
            return None
    x0, y0 = left + round(.13 * w), top + round(.54 * h)
    tile = frame[y0:bottom - round(.07 * h), x0:right - round(.13 * w)]
    blue, green, red = (channel.astype(np.int16) for channel in cv2.split(tile))
    gold = np.uint8((red >= 162) & (green >= 127) & (green - blue >= 22)
                    & (red - green <= 74) & (red - blue >= 39))
    contours, _ = cv2.findContours(gold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    disks = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if (not .27 * w <= min(width, height) <= max(width, height) <= .47 * w or not .91 <= width / height <= 1.10
                or abs(x0 + x + width / 2 - (left + right) / 2) > .049 * w
                or not .09 * h <= bottom - y0 - y - height <= .225 * h):
            continue
        fill = np.zeros((height, width), np.uint8)
        cv2.drawContours(fill, [contour], -1, 1, cv2.FILLED, offset=(-x, -y))
        yy, xx = np.ogrid[:height, :width]
        circle = ((xx + .5) / width - .5)**2 + ((yy + .5) / height - .5)**2 <= .25
        if np.count_nonzero(fill & circle) >= .94 * np.count_nonzero(fill | circle):
            disks.append((x0 + x, y0 + y, x0 + x + width, y0 + y + height))
    return disks[0] if len(disks) == 1 else None


def _envelope_action(frame, panel, index, existing):
    left, top, right, bottom = panel
    if any(item["function"] in {"打开", "打开红包", "拆开红包", "开启红包"}
           and left <= item["click_point"][0] < right and top <= item["click_point"][1] < bottom
           for item in existing):
        return None
    box = _envelope_disk(frame, panel)
    if box is None:
        return None
    l, t, r, b = box
    size = min(r - l, b - t)
    inset = round(.175 * size)
    tile = cv2.cvtColor(frame[t + inset:b - inset, l + inset:r - inset], cv2.COLOR_BGR2GRAY)
    paint = np.median(tile)
    cuts = {}
    for distance in (34, 45, 65, 85):
        mask = np.uint8(tile <= paint - distance)
        x, y, w, h = cv2.boundingRect(mask)
        if (not .32 * size <= min(w, h) <= max(w, h) <= .58 * size or not .68 <= w / h <= 1.35
                or abs(x + w / 2 - tile.shape[1] / 2) > .052 * size
                or abs(y + h / 2 - tile.shape[0] / 2) > .052 * size):
            continue
        ranks = index.rank(mask, alphabet="cjk", limit=2)
        if (len(ranks) == 2 and ranks[0]["text"] in {"开", "開"} and ranks[0]["distance"] <= .212
                and ranks[1]["distance"] - ranks[0]["distance"] >= .020):
            evidence = cuts.setdefault(ranks[0]["text"], set())
            evidence.add(mask.tobytes())
            if len(evidence) == 2:
                break
    else:
        return None
    if not _envelope_title(frame, panel, index):
        return None
    return {"role": "open_gift", "function": "打开红包", "preview_text": "打开红包",
            "box": list(box), "click_point": [(l + r) // 2, (t + b) // 2]}


def pixel_dialog_controls(frame, regions, index, existing):
    """Add physically verified controls that were missed by proposal-based checks."""
    if regions["layer"] != "dialog":
        return []
    panels = regions["regions"]
    if len(panels) == 2 and index is not None:
        close = regions.get("detected_close")
        outer = tuple(panels[1])
        if (close is not None and outer == tuple(close) and outer[3] <= panels[0][1]
                and any(item["role"] == "close" and outer[0] <= item["click_point"][0] < outer[2]
                        and outer[1] <= item["click_point"][1] < outer[3] for item in existing)):
            action = _envelope_action(frame, panels[0], index, existing)
            return [action] if action is not None else []
    if len(panels) != 1:
        return []
    panel = panels[0]
    face = verified_dialog_face(frame, panel)
    actions = []
    if face is not None:
        from screen_modal_primary_plate import modal_primary_plate_actions
        from screen_rim_attached_dismiss import rim_attached_dismiss

        if not any(item["role"] == "close" for item in existing):
            close = rim_attached_dismiss(frame, panel) or _quiet_corner_close(frame, panel, face)
            if close:
                actions.append(close)
        if index is not None:
            actions.extend(modal_primary_plate_actions(frame, panel, index))
    if index is not None:
        from screen_compact_dialog_footer import compact_dialog_footer_actions
        from screen_compact_profile_form import compact_profile_form_actions

        actions.extend(compact_profile_form_actions(frame, panel))
        actions.extend(compact_dialog_footer_actions(frame, panel, index))
    reach = .023 * min(frame.shape[:2])
    left, top, right, bottom = panel
    output = []
    for item in actions:
        x, y = item["click_point"]
        if left <= x < right and top <= y < bottom and all(
                max(abs(x - prior["click_point"][0]), abs(y - prior["click_point"][1])) > reach
                for rows in (existing, output) for prior in rows):
            output.append(item)
    return output
