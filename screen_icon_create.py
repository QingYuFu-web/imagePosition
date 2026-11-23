"""Recognize a compose mark only on a confirmed mobile social navigation rail."""

from collections import defaultdict

import cv2
import numpy as np

from inventory_compact_video_rail import _curved_arrow, _footer_top, _marks, _neighbor, _speech, _stable
from inventory_labeled_bottom_navigation import _create_bubble
from inventory_small_reaction_glyphs import _heart_candidates
from screen_component_captions import _bands, _tight
from screen_native_tab_bar import _distance
from screen_social_footer import _social_rail
from screen_video_controls import video_control_actions


def _cross(mask):
    height, width = mask.shape
    if not .80 <= width / height <= 1.23 or not .16 <= mask.mean() <= .59:
        return False
    x, y = width // 2, height // 2
    if min(mask[:, x - 1:x + 2].mean(axis=0).max(), mask[y - 1:y + 2].mean(axis=1).max()) < .77:
        return False
    side = max(2, round(.22 * min(height, width)))
    corners = (mask[:side, :side], mask[:side, -side:], mask[-side:, :side], mask[-side:, -side:])
    return max(corner.mean() for corner in corners) <= .16


def _rings(ink, width):
    contours, tree = cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if tree is None:
        return
    links = tree[0]
    for ring, (_, _, hole_id, parent) in zip(contours, links, strict=True):
        if parent >= 0 or hole_id < 0:
            continue
        x, y, w, h = cv2.boundingRect(ring)
        if not (.052 * width <= w <= .128 * width and .050 * width <= h <= .122 * width and
                .85 <= w / h <= 1.26 and .67 <= cv2.contourArea(ring) / (w * h) <= .91):
            continue
        hole = contours[hole_id]
        a, b, hw, hh = cv2.boundingRect(hole)
        if (not .73 * w <= hw <= 1.00 * w or not .73 * h <= hh <= 1.00 * h or
                abs(a + hw / 2 - x - w / 2) > .06 * w or abs(b + hh / 2 - y - h / 2) > .06 * h):
            continue
        inner = links[hole_id][2]
        if inner < 0:
            continue
        l, t, pw, ph = cv2.boundingRect(contours[inner])
        if (not .27 * w <= pw <= .64 * w or not .28 * h <= ph <= .65 * h or
                abs(l + pw / 2 - x - w / 2) > .085 * w or abs(t + ph / 2 - y - h / 2) > .085 * h):
            continue
        if _cross(ink[t:t + ph, l:l + pw]):
            yield x, y, x + w, y + h


def _segment(mask, y, end, x, right, axis):
    h, w = mask.shape
    part = mask[round(y * h):round(end * h), round(x * w):round(right * w)]
    return float(part.mean(axis=axis).max()) if part.size else 0.


def _pencil_segment(mask):
    y, x = np.nonzero(mask)
    if min(len(x), len(y)) < 7:
        return False
    height, width = mask.shape
    eigenvalues = np.linalg.eigvalsh(np.cov(np.vstack((x, y))))
    return (float(np.corrcoef(x, y)[0, 1]) <= -.84 and
            eigenvalues[0] <= .18 * eigenvalues[1] and .13 <= mask.mean() <= .47 and
            .81 <= width / height <= 1.22)


def _pencils(ink, width):
    _, objects, stats, _ = cv2.connectedComponentsWithStats(ink)
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if not (.040 * width <= w <= .105 * width and .040 * width <= h <= .105 * width and
                .89 <= w / h <= 1.17 and .17 <= area / (w * h) <= .42):
            continue
        x, y, w, h = map(int, (x, y, w, h))
        square = objects[y:y + h, x:x + w] == i
        sides = (_segment(square, .01, .22, .23, .50, axis=1),
                 _segment(square, .27, .72, .01, .22, axis=0),
                 _segment(square, .78, .99, .25, .74, axis=1),
                 _segment(square, .62, .79, .80, .99, axis=0))
        if min(sides) < .76 or square[:round(.40 * h), round(.67 * w):].mean() > .11:
            continue
        for j, (l, t, ww, hh, _) in enumerate(stats[1:], 1):
            if (j != i and .41 * w <= l - x <= .59 * w and -.04 * h <= t - y <= .16 * h and
                    .35 * w <= ww <= .58 * w and .35 * h <= hh <= .61 * h):
                if _pencil_segment(objects[t:t + hh, l:l + ww] == j):
                    yield x, y, x + w, y + h


def _central_symbol(distance):
    width = distance.shape[1]
    left, right = round(.34 * width), round(.66 * width)
    central = distance[:, left:right]
    matches = defaultdict(list)
    for level in (30, 49, 66):
        ink = np.uint8(central >= level)
        for label, detector in (("pencil", _pencils), ("plus", _rings)):
            for x, y, r, b in detector(ink, width):
                if abs(left + (x + r) / 2 - width / 2) <= .043 * width:
                    matches[label].append((level, (left + x, y, left + r, b)))
    verified = []
    for label, observed in matches.items():
        agreeing = [box for cut, box in observed if any(peer_cut != cut and
                    max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= max(1, .006 * width)
                    for peer_cut, peer in observed)]
        if agreeing and all(max(abs(a - b) for a, b in zip(agreeing[0], box, strict=True)) <= max(2, .008 * width)
                             for box in agreeing):
            verified.append((label, agreeing[len(agreeing) // 2]))
    return verified[0] if len(verified) == 1 else None


def _painted_plus(frame, offset, paper):
    height, width = frame.shape[:2]
    if np.median(paper) > 25:
        return None
    # Look through a little of the content to locate the start of the dark rail.
    start = max(0, offset - round(.06 * width))
    tile = frame[start:]
    flat = (_distance(tile, paper)[:, round(.07 * width):round(.93 * width)] <= 15).mean(axis=1)
    seam = []
    for y in range(2, len(tile) - round(.117 * width)):
        if flat[y] >= .84 and flat[y - 2:y].max() <= .62 and flat[y:y + max(3, round(.010 * width))].mean() >= .87:
            seam.append(y)
    if len(seam) != 1:
        return None
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    box = _create_bubble(tile, gray, seam[0], float(np.median(paper)), width / 2, compact=True)
    if box is None:
        return None
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    dx, dy = round(.19 * w), round(.15 * h)
    interior = gray[top + dy:bottom - dy, left + dx:right - dx]
    confirmed = 0
    for level in (53, 81, 119):
        ink = np.uint8(interior <= level)
        x, y, gw, gh = cv2.boundingRect(ink)
        if (.27 * w <= gw <= .54 * w and .34 * h <= gh <= .64 * h and
                abs(x + gw / 2 - interior.shape[1] / 2) <= .06 * w and
                abs(y + gh / 2 - interior.shape[0] / 2) <= .057 * h and _cross(ink[y:y + gh, x:x + gw])):
            confirmed += 1
    if confirmed < 2:
        return None
    # Use the outer color-carrier boundary, including both visible side accents.
    padding = round(.12 * w)
    low, high = left - padding, right + padding
    patch = tile[top:bottom, low:high]
    carrier = (np.ptp(patch, axis=2) >= 60) | (patch.min(axis=2) >= 177)
    measured = _tight(carrier, low, top + start - offset)
    if measured is None or not (0 <= measured[1] < measured[3] <= height - offset):
        return None
    l, t, r, b = measured
    if not (l < left < right < r and .10 * width <= r - l <= .145 * width):
        return None
    return "painted_plus", (l, t, r, b)


def _peer_marks(distance, center, mark):
    width = distance.shape[1]
    h = center[3] - center[1]
    if mark == "pencil":
        first, last = center[1] - round(.14 * h), center[3] + round(.14 * h)
        limits = (.67, 1.21)
    elif mark == "plus":
        first, last = center[1] - round(.17 * h), center[1] + round(.72 * h)
        limits = (.35, .76)
    elif mark == "mini_plus":
        first, last = max(1, center[1] - round(.21 * h)), center[1] + round(.85 * h)
        limits = (.43, .86)
    else:
        first, last = center[1] - round(.20 * h), center[1] + round(.72 * h)
        limits = (.61, .86)
    if not 0 <= first < last <= len(distance):
        return None
    ink = distance[first:last] >= 34
    # Merge nearby components of one icon, such as the separate head and shoulders.
    marks = []
    for a, b in _bands(ink.sum(axis=0) >= max(1, round(.045 * h))):
        if marks and a - marks[-1][1] <= .13 * h:
            marks[-1] = marks[-1][0], b
        else:
            marks.append((a, b))
    boxes = []
    for x, right in marks:
        if x < center[2] and right > center[0] or not .025 * width <= right - x <= .087 * width:
            continue
        spot = _tight(ink[:, x:right], x, first)
        if spot is None:
            continue
        _, t, _, b = spot
        if limits[0] * h <= b - t <= limits[1] * h and first < t < b < last + (1 if mark == "plus" else 0):
            boxes.append(spot)
    left = [box for box in boxes if box[2] < center[0]]
    right = [box for box in boxes if box[0] > center[2]]
    if len(left) != 2 or len(right) != 2:
        return None
    peers = left + right
    centers = [(a + b) / 2 for a, _, b, _ in left + [center] + right]
    pitch = np.diff(centers)
    if (min(pitch) < .14 * width or max(pitch) > .239 * width or np.ptp(pitch) > .028 * width or
            np.ptp([(t + b) / 2 for _, t, _, b in peers]) > .22 * h):
        return None
    return peers


def _selected_peer(distance, peers, mark):
    patches, strength, filled = [], [], []
    for left, top, right, bottom in peers:
        span = distance[top:bottom, left:right]
        patches.append(span)
        contrast = span[span >= 34]
        strength.append(float(np.percentile(contrast, 79)))
        filled.append(float((span >= 37).mean()))
    winner = int(np.argmax(strength))
    remaining = [value for i, value in enumerate(strength) if i != winner]
    if mark == "pencil":
        return strength[winner] >= max(remaining) + 47 and filled[winner] >= max(filled) - .025
    cut = max(55, round(.47 * strength[winner]))
    filling = [float((patch >= cut).mean()) for patch in patches]
    return (strength[winner] >= max(remaining) + 32 and
            filling[winner] >= max(value for i, value in enumerate(filling) if i != winner) + .12)


def _compact_nav_captions(distance, peers, center, *, edge=0):
    width = distance.shape[1]
    first = max(box[3] for box in peers) + max(1, round(.005 * width))
    last = min(len(distance), center[3] + round(.027 * width))
    if last <= first:
        return False
    matches = []
    for left, _, right, _ in peers:
        x = (left + right) / 2
        start, stop = max(edge, round(x - .095 * width)), min(width - edge, round(x + .095 * width))
        box = _tight(distance[first:last, start:stop] >= 58, start, first)
        if box is not None:
            a, t, b, z = box
            if (.012 * width <= z - t <= .035 * width and .033 * width <= b - a <= .184 * width
                    and abs((a + b) / 2 - x) <= .026 * width):
                matches.append((t, z))
    return len(matches) >= 3 and np.ptp([t for t, _ in matches]) <= .011 * width


def _two_post_rows(frame, footer):
    height, width = frame.shape[:2]
    top = round(.18 * height)
    scale = min(1., 660 / width)
    field = cv2.cvtColor(frame[top:footer, :round(.65 * width)], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(field, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else field
    hits = defaultdict(set)
    for heart, contrast in _heart_candidates(cv2.bitwise_not(gray), .047 * width * scale):
        hits[heart.xyxy()].add(contrast)
    rows = []
    for (x, y, r, bottom), levels in hits.items():
        h = bottom - y
        if len(levels) < 2:
            continue
        dy = max(2, round(.18 * h))
        tail = gray[max(0, y - dy):bottom + dy, r:min(gray.shape[1], r + round(6.3 * h))]
        if not tail.size or (tail >= 235).mean() < .71:
            continue
        peers = _bands((tail <= 148).sum(axis=0) >= 1)
        if len(peers) != 3 or any(not .68 * h <= right - left <= 1.25 * h for left, right in peers):
            continue
        # A reaction row repeats three equally sized marks after the heart.
        centers = [(x + r) / 2] + [r + (a + b) / 2 for a, b in peers]
        gaps = np.diff(centers)
        box = _tight(tail[:, peers[0][0]:peers[0][1]] <= 148, 0, 0)
        if (box is None or not .71 * h <= box[3] - box[1] <= 1.22 * h or
                not 1.48 * h <= min(gaps) <= max(gaps) <= 2.63 * h or np.ptp(gaps) > .35 * h):
            continue
        rows.append((x, y, h))
    return any(abs(x - xx) <= .37 * h and 2.5 * h <= abs(y - yy) and .76 <= h / hh <= 1.30
               for x, y, h in rows for xx, yy, hh in rows)


def _compact_rail(frame, approximate):
    footer = _footer_top(frame)
    if footer is None or not 0 < approximate - footer <= .13 * frame.shape[1]:
        return False
    marks = list(_marks(frame, footer))
    for bubble in _stable(marks, _speech):
        for arrow in _stable(marks, _curved_arrow):
            cx, cy = bubble.box.center
            sx, sy = arrow.box.center
            h = (bubble.box.height + arrow.box.height) / 2
            pitch = sy - cy
            if (not 1.72 * h <= pitch <= 4.05 * h or abs(cx - sx) > .25 * h or
                    not .61 <= arrow.box.height / bubble.box.height <= 1.38):
                continue
            axis = (cx + sx) / 2
            if all(any(_neighbor(mark, bubble, h, pitch, axis, at) for mark in marks) for at in (-2, -1)):
                return True
    return False


def _body_video_rail(frame, regions, footer):
    if not regions.get("candidates"):
        return False
    width = frame.shape[1]
    active, _ = video_control_actions(frame, regions)
    marks = {row["role"]: row["click_point"] for row in active if row["role"] in {"comments", "share"}}
    if "comments" not in marks or "share" not in marks:
        return False
    cx, cy = marks["comments"]
    sx, sy = marks["share"]
    return (abs(cx - sx) <= .029 * width and abs(cy - sy) >= .11 * width and
            max(cy, sy) <= footer - .07 * width and
            (max(cx, sx) <= .20 * width or min(cx, sx) >= .80 * width))


def _mini_carrier(gray, paint):
    width = gray.shape[1]
    lo, hi = round(.35 * width), round(.65 * width)
    search = gray[1:round(.145 * width), lo:hi]
    matches = []
    for cut in (14, 17, 20):
        ink = np.uint8(search >= paint + cut)
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (not .062 * width <= min(w, h) <= max(w, h) <= .116 * width or
                    not .83 <= w / h <= 1.20 or not .56 <= cv2.contourArea(contour) / (w * h) <= .81 or
                    abs(x + lo + w / 2 - width / 2) > .026 * width):
                continue
            patch = search[y:y + h, x:x + w]
            good = 0
            for contrast in (73, 92, 121):
                mark = np.uint8(patch >= paint + contrast)
                xx, yy, a, b = cv2.boundingRect(mark)
                if (max(5, .36 * w) <= a <= .65 * w and max(5, .36 * h) <= b <= .68 * h and
                        abs(xx + a / 2 - w / 2) <= .13 * w and abs(yy + b / 2 - h / 2) <= .15 * h and
                        _cross(mark[yy:yy + b, xx:xx + a])):
                    good += 1
            if good >= 2:
                matches.append((cut, (x + lo, y + 1, x + lo + w, y + 1 + h)))
    stable = [box for cut, box in matches if any(
        peer_cut != cut and max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= 2
        for peer_cut, peer in matches)]
    if stable and all(max(abs(a - b) for a, b in zip(stable[0], box, strict=True)) <= 2 for box in stable):
        return stable[len(stable) // 2]
    return None


def _mini_plus(frame):
    width = frame.shape[1]
    origin = _footer_top(frame)
    if origin is None:
        return []
    tile = frame[origin:]
    # The trailing pixels can contain the gesture bar and curved display corners.
    sample = tile[1:len(tile) - round(.045 * width), round(.15 * width):round(.85 * width)]
    paper = np.median(sample[::2, ::3], axis=(0, 1))
    if np.ptp(paper) > 13 or max(paper) > 26:
        return []
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    box = _mini_carrier(gray, round(float(np.median(paper))))
    if box is None:
        return []
    distance = _distance(tile, paper)
    peers = _peer_marks(distance, box, "mini_plus")
    if not peers or not _compact_nav_captions(distance, peers, box, edge=round(.034 * width)):
        return []
    neutral = []
    for left, top, right, bottom in peers:
        source = tile[top:bottom, left:right]
        ink = distance[top:bottom, left:right] >= 34
        if (np.ptp(source, axis=2)[ink] <= 49).mean() >= .80:
            neutral.append((left, top, right, bottom))
    if len(neutral) < 3 or not _selected_peer(distance, neutral, "plus") or not _compact_rail(frame, origin + box[1]):
        return []
    return _create_result(box, origin)


def _create_result(box, offset):
    left, top, right, bottom = box
    owner = [left, offset + top, right, offset + bottom]
    return [{"role": "open_post_composer", "function": "打开创作", "box": owner,
             "click_point": [(left + right) // 2, offset + (top + bottom) // 2], "preview_text": "Create"}]


def icon_create_actions(frame, regions):
    """Publish one center compose target with matching tab and body evidence."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or width < 128 or not 1.53 * width <= height <= 2.68 * width:
        return []
    if width < 190:
        return _mini_plus(frame)
    offset = height - round(.205 * width)
    tile = frame[offset:]
    strip = tile[-max(3, round(.023 * width)):, round(.20 * width):round(.80 * width)]
    paper = np.median(strip[::2, ::3], axis=(0, 1))
    if np.ptp(paper) > 17:
        return []
    distance = _distance(tile, paper)
    symbol = _central_symbol(distance) or _painted_plus(frame, offset, paper)
    if symbol is None:
        return []
    mark, box = symbol
    left, top, right, bottom = box
    inset = round(.083 * width)
    peers = _peer_marks(distance, box, mark)
    minimum = {"pencil": .83, "plus": .77, "painted_plus": .72}[mark]
    if (not peers or not _selected_peer(distance, peers, mark) or
            (distance[top:bottom, inset:width - inset] <= 15).mean() < minimum):
        return []
    footer = offset + top
    if mark == "pencil":
        proved = np.median(paper) >= 209 and _two_post_rows(frame, footer)
    elif mark == "painted_plus":
        below = distance[bottom + max(2, round(.006 * width)):min(len(distance), bottom + round(.029 * width)),
                         left:right]
        proved = (below.size and (below <= 16).mean() >= .90 and _compact_nav_captions(distance, peers, box) and
                  _body_video_rail(frame, regions, footer))
    else:
        proved = np.median(paper) <= 76 and _compact_nav_captions(distance, peers, box) and (
            _compact_rail(frame, footer) if width <= 420 else _social_rail(frame, regions, footer))
    if not proved:
        return []
    return _create_result(box, offset)
