"""Verify a dark native reply editor and its two distinct header actions."""

import cv2
import numpy as np

from screen_component_captions import _bands


def _toolbar(frame):
    height, width = frame.shape[:2]
    patch = frame[:round(.211 * height), ::4].astype(np.int16)
    profile = np.median(patch, axis=1)
    rails = set()
    for i in range(round(.045 * height), round(.128 * height), max(2, round(.011 * height))):
        paint = profile[i]
        if np.ptp(paint) < 63:
            continue
        for top, bottom in _bands(np.max(abs(profile - paint), axis=1) <= 11):
            if (.035 * height <= top <= .075 * height and .064 * height <= bottom - top <= .116 * height
                    and (np.max(abs(patch[top:bottom] - paint), axis=2) <= 13).mean() >= .77
                    and np.max(profile[bottom + max(2, round(.008 * height)):bottom + round(.029 * height)], axis=1).mean() <= 25):
                rails.add((top, bottom, tuple(map(int, paint))))
    if not rails:
        return None
    unique = {(t, b) for t, b, _ in rails}
    tops, bottoms = zip(*unique, strict=True)
    if max(np.ptp(tops), np.ptp(bottoms)) > max(2, .003 * height):
        return None
    return round(float(np.median(tops))), round(float(np.median(bottoms)))


def _system_navigation(gray):
    height, width = gray.shape
    top = round(.940 * height)
    patch = gray[top:]
    if np.median(patch) > 27:
        return False
    outlines, hierarchy = cv2.findContours(np.uint8(patch >= 118), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    marks = []
    for i, outline in enumerate(outlines):
        if hierarchy[0, i, 3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(outline)
        hole = hierarchy[0, i, 2]
        if (.032 * width <= h <= .075 * width and .68 <= w / h <= 1.33 and hole >= 0
                and .39 <= cv2.contourArea(outlines[hole]) / cv2.contourArea(outline) <= .83):
            marks.append((x + w / 2, y + h / 2))
    if len(marks) != 3:
        return False
    marks.sort()
    xs, ys = np.array(marks).T
    return abs(xs[1] - width / 2) <= .045 * width and np.ptp(np.diff(xs)) <= .038 * width and np.ptp(ys) <= .015 * width


def _field(frame, gray, candidates, toolbar):
    height, width = gray.shape
    _, after = toolbar
    fields = []
    for left, top, right, bottom in candidates:
        deep = bottom - top
        if not (right - left >= .89 * width and .069 * height <= deep <= .173 * height
                and .034 * height <= top - after <= .104 * height and bottom <= .39 * height):
            continue
        x, r = max(left, round(.017 * width)), min(right, width - round(.017 * width))
        y, end = bottom - round(.21 * deep), min(height, bottom + round(.064 * deep))
        patch = frame[y:end, x:r].astype(np.int16)
        tint = (patch[:, :, 1] >= 129) & (patch[:, :, 1] - patch[:, :, 2] >= 34)
        rails = _bands(tint.mean(axis=1) >= .93)
        if len(rails) != 1 or not 1 <= rails[0][1] - rails[0][0] <= .066 * deep:
            continue
        start, stop = y + rails[0][0], y + rails[0][1]
        blank = gray[top - round(.13 * deep):top + 1, x:r]
        draft = gray[top:start, x:r]
        text_rows = _bands((draft >= 126).mean(axis=1) >= .087)
        if (np.median(draft) > 35 or (blank <= 35).mean() < .95 or not 2 <= len(text_rows) <= 6):
            continue
        fields.append((left, top, right, stop))
    return fields[0] if len(fields) == 1 else None


def _quote(gray, candidates, field):
    height, width = gray.shape
    _, _, _, edge = field
    thumbs = [box for box in candidates if .014 * width <= box[1] - edge <= .054 * width
              and .09 * width <= box[0] <= .25 * width and .14 * width <= box[2] - box[0] <= .35 * width
              and .71 * (box[2] - box[0]) <= box[3] - box[1] <= 1.45 * (box[2] - box[0])]
    if len(thumbs) != 1:
        return False
    a, t, b, d = thumbs[0]
    inset = round(.09 * (b - a))
    thumbnail = gray[t + inset:d - inset, a + inset:b - inset]
    between = gray[edge + max(2, round(.008 * width)):t - 2]
    quote = gray[d + round(.035 * width):round(.912 * height)]
    if not thumbnail.size or not between.size or not quote.size or thumbnail.std() < 24 or (between < 35).mean() < .92:
        return False
    return np.median(quote) <= 28 and sum(.013 * width <= end - begin <= .071 * width
                                            for begin, end in _bands((quote >= 127).mean(axis=1) >= .088)) >= 5


def _components(gray, box, level):
    a, t, b, d = box
    patch = gray[t:d, a:b]
    mask = np.uint8(patch.astype(np.int16) >= np.median(patch) + level)
    _, ids, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(int(x + a), int(y + t), int(x + w + a), int(y + h + t), ids[y:y + h, x:x + w] == i)
            for i, (x, y, w, h, area) in enumerate(stats[1:], 1) if area >= .008 * (b - a) * (d - t)]


def _clip_shape(mask):
    y, x = np.nonzero(mask)
    if not .25 <= mask.mean() <= .52 or np.corrcoef(x, y)[0, 1] >= -.53:
        return False
    along = x - y
    middle = (along >= np.quantile(along, .30)) & (along <= np.quantile(along, .70))
    density = np.bincount((x + y)[middle])
    return len(_bands(density >= max(2, .24 * density.max()))) >= 3


def _plus_shape(mask):
    h, w = mask.shape
    yy, xx = np.nonzero(mask)
    x, y = (xx + .5) / w - .5, (yy + .5) / h - .5
    if not .16 <= mask.mean() <= .40 or (np.minimum(abs(x), abs(y)) <= .16).mean() < .94:
        return False
    arms = [((z * side >= .28) & (abs(other) <= .18)).mean() for z, other in ((x, y), (y, x)) for side in (-1, 1)]
    return min(arms) >= .073


def _attach(parts, size):
    for x, y, r, d, mask in parts:
        h, w = d - y, r - x
        if not (.29 * size <= h <= .48 * size and .79 <= w / h <= 1.22 and _clip_shape(mask)):
            continue
        for a, t, b, e, plus in parts:
            if (.36 * h <= e - t <= .66 * h and .81 <= (b - a) / (e - t) <= 1.23
                    and .32 * h <= (a + b - x - r) / 2 <= .69 * h
                    and .26 * h <= (t + e - y - d) / 2 <= .61 * h and _plus_shape(plus)):
                return x, y, b, e
    return None


def _send(parts, size):
    for x, y, r, d, mask in parts:
        h, w = d - y, r - x
        if not (.25 * size <= h <= .44 * size and .86 <= w / h <= 1.25 and .33 <= mask.mean() <= .56):
            continue
        yy, xx = np.nonzero(mask)
        tip = yy[xx >= .84 * w]
        arms = (xx[yy <= .16 * h], xx[yy >= .84 * h])
        middle = [int(np.flatnonzero(mask[row])[0]) for row in range(round(.38 * h), round(.62 * h)) if mask[row].any()]
        if (tip.size and .34 * h <= tip.mean() <= .65 * h and all(a.size and a.mean() <= .31 * w for a in arms)
                and middle and max(middle) > .24 * w):
            return x, y, r, d
    return None


def dark_reply_editor_actions(frame, regions, index):
    """Name the draft field and its paired attach/send controls on a proven editor."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or not 1.49 * width <= height <= 2.85 * width or width < 240:
        return []
    candidates = [tuple(item["box"]) for item in regions["candidates"] if item.get("visibility") == "page"]
    toolbar = _toolbar(frame)
    if toolbar is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    field = _field(frame, gray, candidates, toolbar)
    if field is None or not _system_navigation(gray) or not _quote(gray, candidates, field):
        return []
    top, bottom = toolbar
    size = bottom - top
    lanes = [(x, max(top, y), r, min(bottom, d)) for x, y, r, d in candidates
             if .48 * width <= x < r <= width and .58 * size <= r - x <= 1.14 * size
             and .58 * size <= min(bottom, d) - max(top, y) <= size]
    attach, send = [], []
    for lane in lanes:
        parts = _components(gray, lane, 52)
        if box := _attach(parts, size):
            attach.append(box)
        if box := _send(parts, size):
            send.append(box)
    if len(attach) != 1 or len(send) != 1 or not .16 * size <= send[0][0] - attach[0][2] <= .62 * size:
        return []
    return [{"role": role, "function": name, "preview_text": preview, "box": list(box),
             "click_point": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]}
            for role, name, preview, box in (("input", "编辑回复", "Edit reply", field),
                                           ("attach", "添加附件", "Attach", attach[0]),
                                           ("send", "发送回复", "Send", send[0]))]
