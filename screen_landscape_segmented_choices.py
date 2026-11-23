"""Read spaced landscape choice cells with a light selected surface."""

from itertools import combinations

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_wide_panel_checks import _read_glyph


def _selected_faces(gray):
    height, width = gray.shape
    contours, _ = cv2.findContours(cv2.inRange(gray, 181, 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (.068 * width <= w <= .37 * width and .072 * height <= h <= .217 * height
                and 1.68 <= w / h <= 5.6 and cv2.contourArea(contour) >= .89 * w * h
                and 1 < x < x + w < width - 1 and 1 < y < y + h < height - 1):
            faces.append((x, y, x + w, y + h))
    return faces


def _centered_ink(gray, box, *, selected=False):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    x, y = left + round(.13 * width), top + round(.12 * height)
    crop = gray[y:bottom - round(.12 * height), x:right - round(.13 * width)].astype(np.float32)
    ground = np.median(crop)
    if selected and (ground < 181 or (crop >= 179).mean() < .61):
        return None
    ink = np.maximum(0, ground - crop if selected else crop - ground)
    peak = float(np.percentile(ink, 98))
    if peak < 57:
        return None
    signal = np.uint8(ink >= .30 * peak)
    u, v, w, h = cv2.boundingRect(signal)
    if (not .25 * height <= h <= .71 * height or not .34 * h <= w <= .75 * width
            or abs(x + u + w / 2 - (left + right) / 2) > .092 * width
            or abs(y + v + h / 2 - (top + bottom) / 2) > .14 * height
            or not .14 <= signal[v:v + h, u:u + w].mean() <= .77):
        return None
    spans = _bands(signal.any(axis=0))
    if not 1 <= len(spans) <= 9:
        return None
    return ink, peak, spans, h


def _lattice(faces):
    groups = set()
    for l, t, r, b in faces:
        w, h = r - l, b - t
        peers = tuple(sorted(((x, y, v, u) for x, y, v, u in faces
                              if abs(v - x - w) <= .075 * w and abs(u - y - h) <= .12 * h),
                             key=lambda box: box[1]))
        if len(peers) < 2:
            continue
        ys = np.sort([(p[1] + p[3]) / 2 for p in peers])
        if (np.diff(ys) < 1.30 * h).any():
            continue
        lefts = sorted({p[0] for p in peers})
        guesses = []
        for x, v in combinations(lefts, 2):
            count = round((v - x) / w)
            if not 1 <= count <= 7:
                continue
            pitch = (v - x) / count
            if not w + .035 * h <= pitch <= w + .37 * h:
                continue
            slots = np.float64(lefts) - lefts[0]
            if np.max(np.abs(slots - pitch * np.rint(slots / pitch))) <= .11 * h:
                guesses.append(pitch)
        if guesses and np.ptp(guesses) <= .15 * h:
            groups.add((peers, float(np.median(guesses))))
    return next(iter(groups)) if len(groups) == 1 else None


def _ridge(profile, target, tolerance, minimum):
    start, stop = max(0, target - tolerance), min(len(profile), target + tolerance + 1)
    if stop <= start:
        return None
    distance = np.abs(np.arange(start, stop) - target)
    response = np.where(profile[start:stop] < 73, profile[start:stop], 0)
    offset = int(np.argmax(response - .37 * distance))
    return start + offset if response[offset] >= minimum else None


def _neighbors(gray, face, anchor, pitch):
    left, top, right, bottom = face
    width, height = right - left, bottom - top
    span = max(2, round(.063 * height))
    gap = max(1, round(.039 * height))
    mid = gray[top + round(.20 * height):bottom - round(.20 * height)]
    side = np.pad(np.median(mid[:, 2 * gap:] - mid[:, :-2 * gap], axis=0), (gap, gap))
    first = int(np.ceil((gap - anchor) / pitch))
    last = int(np.floor((gray.shape[1] - gap - width - anchor) / pitch))
    for j in range(first, last + 1):
        x = round(anchor + j * pitch)
        if abs(x - left) <= .2 * height:
            continue
        l = _ridge(-side, x, span, 2.9)
        r = _ridge(side, x + width, span, 3.8)
        if l is None or r is None or not .94 * width <= r - l <= 1.06 * width:
            continue
        center = gray[:, l + round(.21 * width):r - round(.21 * width)]
        ends = np.pad(np.median(center[2 * gap:] - center[:-2 * gap], axis=1), (gap, gap))
        t = _ridge(-ends, top, span, 3.8)
        b = _ridge(ends, bottom, span, 3.8)
        if t is not None and b is not None and .90 * height <= b - t <= 1.10 * height:
            yield j, (l, t, r, b)


def _digit(ink, peak, index):
    analysis = cv2.resize(ink, None, fx=1.76, fy=1, interpolation=cv2.INTER_LINEAR)
    ranks = [[result for result in index.rank(analysis >= level * peak, alphabet="code", limit=16)
              if result["text"].isdigit()] for level in (.255, .33)]
    if (all(len(row) >= 2 and row[0]["distance"] <= .257
            and row[1]["distance"] - row[0]["distance"] >= .031 for row in ranks)
            and ranks[0][0]["text"] == ranks[1][0]["text"]):
        return ranks[0][0]["text"]
    return None


def _ideograph(ink, peak, index):
    ranks = [index.rank(ink >= level * peak, alphabet="cjk", limit=3) for level in (.27, .39)]
    if (all(len(row) >= 2 and row[0]["distance"] <= .194
            and row[1]["distance"] - row[0]["distance"] >= .014 for row in ranks)
            and ranks[0][0]["text"] == ranks[1][0]["text"]):
        strong = index.rank(ink >= .55 * peak, alphabet="cjk", limit=3)
        if (len(strong) < 2 or strong[0]["distance"] >= .21
                or strong[1]["distance"] - strong[0]["distance"] < .016
                or strong[0]["text"] == ranks[0][0]["text"]):
            return ranks[0][0]["text"]
    return _read_glyph(ink, peak, index)


def _transcribe(details, index):
    ink, peak, spans, height = details
    # The mixed form has a coherent narrow run followed by square glyphs.
    narrow = [stop - start < .56 * height for start, stop in spans]
    if any(narrow) and (sum(narrow) < 2 or narrow != sorted(narrow, reverse=True) or all(narrow)):
        return None
    letters = []
    for (start, stop), digit in zip(spans, narrow, strict=True):
        tile = ink[:, start:stop]
        if not digit and (height < 26 or height > 31):
            scale = 27 / height
            tile = cv2.resize(tile, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        letter = _digit(tile, peak, index) if digit else _ideograph(tile, peak, index)
        if letter is None:
            return None
        letters.append(letter)
    return "".join(letters)


def landscape_segmented_choices(frame, overlay, index):
    """Return measured inactive choices on a verified landscape page."""
    height, width = frame.shape[:2]
    if index is None or overlay is not None or height < 220 or not 1.53 * height <= width <= 3.18 * height:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = [box for box in _selected_faces(gray) if _centered_ink(gray, box, selected=True)]
    geometry = _lattice(faces)
    if geometry is None:
        return []
    gray = gray.astype(np.int16)
    faces, pitch = geometry
    left = min(box[0] for box in faces)
    rows = []
    for face in faces:
        candidates = [(col, box, ink) for col, box in _neighbors(gray, face, left, pitch)
                      if (ink := _centered_ink(gray, box)) is not None]
        if len(candidates) < 2:
            continue
        strongest = max(ink[1] for _, _, ink in candidates)
        readable = [(col, box, ink) for col, box, ink in candidates if ink[1] >= max(76, .66 * strongest)]
        columns = sorted([round((face[0] - left) / pitch), *(col for col, _, _ in candidates)])
        if len(readable) >= 2 and max(np.diff(columns)) <= 1:
            rows.append((columns, readable))
    if len(rows) < 2 or len(set.intersection(*(set(columns) for columns, _ in rows))) < 2:
        return []
    actions = []
    for _, row in rows:
        names = [(box, label) for _, box, ink in row if (label := _transcribe(ink, index)) is not None]
        if not names or len({label for _, label in names}) != len(names):
            continue
        for (l, t, r, b), name in names:
            actions.append({"role": "select_option", "function": f"选择{name}", "preview_text": name,
                            "box": [l, t, r, b], "click_point": [(l + r) // 2, (t + b) // 2]})
    return actions
