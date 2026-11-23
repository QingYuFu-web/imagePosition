"""Verify a filled envelope glyph in an independently supported toolbar."""

import cv2
import numpy as np

from screen_candidate_glyphs import _overlap


def _folds(gray):
    raster = cv2.resize(gray, (64, 42), interpolation=cv2.INTER_LINEAR)
    relief = cv2.morphologyEx(raster, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
    support = 0
    for level in (27, 38, 50):
        ink = cv2.compare(relief, level, cv2.CMP_GE)
        segments = cv2.HoughLinesP(ink, 1, np.pi / 180, 9, minLineLength=12, maxLineGap=4)
        if segments is None:
            continue
        rising, falling = [], []
        for x, y, r, b in segments.reshape(-1, 4).tolist():
            if x > r:
                x, y, r, b = r, b, x, y
            if r - x < 11:
                continue
            slope = (b - y) / (r - x)
            offset = y - x * slope
            if x <= 21 and 22 <= r <= 41 and .35 <= slope <= 1.16 and 2 <= y <= 15:
                rising.append((slope, offset, r))
            if 21 <= x <= 43 and r >= 43 and -1.16 <= slope <= -.35 and 2 <= b <= 15:
                falling.append((slope, offset, x))
        for up, first, end in rising:
            for down, last, start in falling:
                join = (last - first) / (up - down)
                bend = up * join + first
                if not (26 <= join <= 37 and 15 <= bend <= 28 and abs(join - end) <= 9
                        and abs(join - start) <= 9 and .58 <= up / -down <= 1.65):
                    continue
                x = int(round(join))
                y = int(round(bend))
                if np.count_nonzero(ink[y - 3:y + 4, x - 4:x + 5]) >= 10:
                    break
            else:
                continue
            support += 1
            break
    return support >= 2


def _glyph(frame, box):
    left, top, right, bottom = box
    crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    tone, peak = np.percentile(crop, (50, 94))
    if peak - tone < 72:
        return None
    kernel = np.ones((3, 3), np.uint8)
    marks = []
    for portion in (.40, .52, .64, .76, .86):
        level = round(tone + portion * (peak - tone))
        ink = cv2.compare(crop, level, cv2.CMP_GE)
        continuous = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(continuous, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (not 1.16 <= w / h <= 1.83 or not .29 * crop.shape[0] <= h <= .82 * crop.shape[0]
                    or not .33 * crop.shape[1] <= w <= .93 * crop.shape[1]
                    or not 0 < x < x + w < crop.shape[1] or not 0 < y < y + h < crop.shape[0]
                    or abs(x + w / 2 - crop.shape[1] / 2) > .17 * crop.shape[1]
                    or abs(y + h / 2 - crop.shape[0] / 2) > .19 * crop.shape[0]
                    or cv2.contourArea(contour) < .77 * w * h):
                continue
            cap = max(2, round(.08 * h))
            trim = round(.11 * w)
            upper = continuous[y:y + cap, x + trim:x + w - trim]
            lower = continuous[y + h - cap:y + h, x + trim:x + w - trim]
            if min(np.max((upper != 0).mean(axis=1)), np.max((lower != 0).mean(axis=1))) < .82:
                continue
            marks.append((level, (left + x, top + y, left + x + w, top + y + h)))
    stable = [box for level, box in marks if any(level != other and _overlap(box, peer) >= .83
                                               for other, peer in marks)]
    if not stable or not all(_overlap(box, stable[0]) >= .79 for box in stable):
        return None
    for left, top, right, bottom in dict.fromkeys(stable):
        if _folds(cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)):
            return left, top, right, bottom
    return None


def toolbar_mail_actions(frame, regions, known):
    """Return an isolated mail control with a nearby verified settings gear."""
    height, width = frame.shape[:2]
    if regions.get("layer") != "page" or regions.get("regions") or width < 1.46 * height or height < 280:
        return []
    settings = [action for action in known if action["role"] == "settings"]
    if len(settings) != 1:
        return []
    gear = settings[0]
    gx, gy = gear["click_point"]
    gh = gear["box"][3] - gear["box"][1]
    if not (.034 * height <= gh <= .092 * height and .023 * height <= gy <= .124 * height
            and min(gx, width - gx) <= .235 * width):
        return []
    found = []
    peers = [row for row in regions["candidates"] if row.get("visibility") == "page"
             and abs(row["center"][1] - gy) <= .39 * gh and row["box"][3] <= .176 * height
             and .70 * gh <= row["box"][3] - row["box"][1] <= 2.45 * gh]
    for row in peers:
        x, _ = row["center"]
        if not 1.15 * gh <= abs(x - gx) <= 4.7 * gh:
            continue
        if not any(.90 * gh <= abs(x - peer["center"][0]) <= 5.5 * gh
                   and abs(gx - peer["center"][0]) >= .78 * gh for peer in peers if peer is not row):
            continue
        box = _glyph(frame, row["box"])
        if box is not None and all(_overlap(box, other) < .65 for other in found):
            found.append(box)
    if len(found) != 1:
        return []
    left, top, right, bottom = found[0]
    a, b, c, d = gear["box"]
    gear_ink = cv2.cvtColor(frame[b:d, a:c], cv2.COLOR_BGR2GRAY)
    mail_ink = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    if np.percentile(mail_ink, 82) < .79 * np.percentile(gear_ink, 82):
        return []
    x, y = (left + right) / 2, (top + bottom) / 2
    if any(l <= x < r and t <= y < b for l, t, r, b in (item["box"] for item in known)):
        return []
    return [{"role": "mail", "function": "打开邮件", "preview_text": "邮件", "box": list(found[0]), "click_point": [x, y]}]
