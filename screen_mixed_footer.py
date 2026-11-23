"""Find mobile footer tabs whose selected or central slot uses a painted mark."""

import cv2
import numpy as np

from screen_component_captions import _bands, _outlier, _tight
from screen_native_tab_bar import _action, _distance, _ink_color, _pair, _transcript


def _marks(tile, distance, width):
    b, g, r = cv2.split(tile)
    chroma = cv2.subtract(cv2.max(cv2.max(b, g), r), cv2.min(cv2.min(b, g), r))
    colored = np.uint8((chroma >= 66) & (distance >= 59))
    _, _, records, _ = cv2.connectedComponentsWithStats(colored)
    for x, y, w, h, area in records[1:]:
        if (.054 * width <= w <= .175 * width and .048 * width <= h <= .128 * width and
                area >= .55 * w * h and y + h >= len(tile) - .13 * width):
            yield int(x), int(y), int(x + w), int(y + h)


def _without_bezel(distance):
    height, width = distance.shape
    _, objects, stats, _ = cv2.connectedComponentsWithStats(np.uint8(distance >= 24))
    touching = set(np.unique(objects[:, 0])) | set(np.unique(objects[:, width - 1]))
    large = [i for i in touching if i and stats[i, cv2.CC_STAT_AREA] >= .006 * height * width]
    if not large:
        return distance, np.zeros(distance.shape, bool)
    bezel = np.isin(objects, large)
    return np.where(bezel, 0, distance).astype(np.uint8), bezel


def _plus(tile, box):
    x, y, right, bottom = box
    w, h = right - x, bottom - y
    if not 1.01 <= w / h <= 1.72:
        return False
    dx, dy = round(.15 * w), round(.14 * h)
    interior = tile[y + dy:bottom - dy, x + dx:right - dx]
    neutral = np.uint8((np.ptp(interior, axis=2) <= 58) & (interior.min(axis=2) >= 189))
    l, t, mw, mh = cv2.boundingRect(neutral)
    if (not .29 * w <= mw <= .55 * w or not .30 * h <= mh <= .64 * h or
            not .81 <= mw / mh <= 1.26 or abs(l + mw / 2 - interior.shape[1] / 2) > .065 * w or
            abs(t + mh / 2 - interior.shape[0] / 2) > .077 * h):
        return False
    mask = neutral[t:t + mh, l:l + mw]
    center_x, center_y = slice(round(.40 * mw), round(.60 * mw)), slice(round(.40 * mh), round(.60 * mh))
    if mask[:, center_x].mean() < .78 or mask[center_y, :].mean() < .78:
        return False
    corners = [mask[ys, xs] for ys in (slice(0, round(.28 * mh)), slice(round(.72 * mh), mh))
               for xs in (slice(0, round(.28 * mw)), slice(round(.72 * mw), mw))]
    return max(corner.mean() for corner in corners) <= .16


def _round_brand(tile, box):
    x, y, right, bottom = box
    w, h = right - x, bottom - y
    if not .86 <= w / h <= 1.17:
        return False
    color = np.ptp(tile[y:bottom, x:right], axis=2) >= 66
    side = max(2, round(.14 * min(w, h)))
    corners = (color[:side, :side], color[:side, -side:], color[-side:, :side], color[-side:, -side:])
    return (max(corner.mean() for corner in corners) <= .35 and
            min(color[h // 2, :].mean(), color[:, w // 2].mean()) >= .54)


def _spans(width, count):
    edges = [round(i * width / count) for i in range(count + 1)]
    pad = round(.014 * width)
    return edges, [(a + pad, b - pad) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def _centers(owners):
    return [(owner[0] + owner[2]) / 2 for owner in owners]


def _cuts(centers, width):
    left = max(0, round(1.5 * centers[0] - .5 * centers[1]))
    right = min(width, round(1.5 * centers[-1] - .5 * centers[-2]))
    return [left, *(round((a + b) / 2) for a, b in zip(centers[:-1], centers[1:], strict=True)), right]


def _text_only(tile, distance, box):
    width = tile.shape[1]
    x, top, right, bottom = box
    if abs((x + right) / 2 - width / 2) > .052 * width:
        return None
    for count in (5, 7):
        edges, spans = _spans(width, count)
        center = count // 2
        if not edges[center] + .01 * width <= x < right <= edges[center + 1] - .01 * width:
            continue
        shift = round(.17 * (bottom - top))
        start, end = top + shift, bottom - shift
        boxes = [box if i == center else _tight(distance[start:end, l:r] >= 27, l, start)
                 for i, (l, r) in enumerate(spans)]
        if any(item is None for item in boxes):
            continue
        labels = [item for i, item in enumerate(boxes) if i != center]
        if any(not .024 * width <= b - t <= .057 * width or not .61 * (b - t) <= r - l <= .132 * width
               or abs((t + b - top - bottom) / 2) > .020 * width or t == start or b == end
               for l, t, r, b in labels):
            continue
        centers = _centers(boxes)
        if (any(abs(cx - (a + z) / 2) > .048 * width for cx, (a, z) in zip(centers, spans, strict=True))
                or np.ptp(np.diff(centers)) > .043 * width or np.ptp([item[1] for item in labels]) > .016 * width):
            continue
        ink = [_ink_color(tile[t:b, l:r], distance[t:b, l:r]) for l, t, r, b in labels]
        selected = _outlier(ink, same_max=36, gap=43)
        if selected is None:
            continue
        edges = _cuts(centers, width)
        owners = []
        for i, ink_box in enumerate(boxes):
            if i == center or ink_box == labels[selected]:
                continue
            l, t, r, b = ink_box
            owners.append((ink_box, [edges[i], top, edges[i + 1], bottom],
                           [round(centers[i]), (t + b) // 2]))
        return owners
    return None


def _branded(tile, distance, bezel, box):
    width = tile.shape[1]
    x, top, right, bottom = box
    h = bottom - top
    start, end = top + round(.56 * h), min(len(tile), bottom + round(.018 * width))
    ink = distance >= 24
    for count in (5, 6, 4):
        edges, spans = _spans(width, count)
        picked = [i for i, (l, r) in enumerate(spans)
                  if l <= x < right <= r and abs((l + r - x - right) / 2) <= .042 * width]
        if len(picked) != 1:
            continue
        selected = picked[0]
        other = [(i, span) for i, span in enumerate(spans) if i != selected]
        alignment = np.sum([ink[start:end, l:r].sum(axis=1) >= 3 for _, (l, r) in other], axis=0)
        bands = [(start + a, start + b) for a, b in _bands(alignment >= max(3, count - 2))
                 if .014 * width <= b - a <= .042 * width]
        if len(bands) != 1:
            continue
        a, b = bands[0]
        row = max(start, a - 3), min(end, b + 3)
        peers, bezel_slots = [], []
        for i, (l, r) in other:
            # A curved crop can remove final letters while leaving a valid-looking prefix.
            if i in {0, count - 1} and bezel[row[0]:row[1], l:r].mean() >= .052:
                bezel_slots.append(i)
                continue
            pair = _pair(tile, distance, (l, r), row)
            if pair is not None:
                peers.append((i, pair))
        if len(peers) < 3 or len(peers) + len(bezel_slots) != len(other):
            continue
        icons = [pair[1] for _, pair in peers]
        colors = np.asarray([pair[2] for _, pair in peers])
        if (np.ptp([icon[1] for icon in icons]) > .019 * width or
                np.ptp([icon[3] for icon in icons]) > .019 * width or
                not top - .017 * width <= min(icon[1] for icon in icons) <= top + .027 * width or
                np.ptp(colors, axis=0).max() > 43 or
                any(not .12 <= ink[t:b, l:r].mean() <= .49 for l, t, r, b in icons)):
            continue
        centers = [(l + r) / 2 for l, r in spans]
        centers[selected] = (x + right) / 2
        for i, (_, icon, _, _) in peers:
            centers[i] = (icon[0] + icon[2]) / 2
        if np.ptp(np.diff(centers)) > .041 * width:
            continue
        edges = _cuts(centers, width)
        result = []
        for i, (text, icon, _, _) in peers:
            owner = [edges[i], max(0, min(top, icon[1]) - round(.013 * width)), edges[i + 1],
                     min(len(tile), max(bottom, text[3]))]
            result.append((text, owner, [(icon[0] + icon[2]) // 2, (icon[1] + icon[3]) // 2]))
        return result
    return None


def _caption_cuts(distance):
    contrast = np.percentile(distance, 99)
    if contrast < 57:
        return
    for ratio in (.19, .25, .31, .365, .445):
        mask = np.uint8(distance >= ratio * contrast)
        x, y, w, h = cv2.boundingRect(mask)
        count = round(w / max(1, h))
        if not 1 <= count <= 6 or not .77 * count <= w / max(1, h) <= 1.28 * count or h < 11:
            continue
        glyphs = mask[y:y + h, x:x + w]
        ink_count = glyphs.sum(axis=0)
        edges = [0]
        for i in range(1, count):
            center = w * i / count
            reach = .16 * h
            left, right = max(1, round(center - reach)), min(w, round(center + reach) + 1)
            edge = min(range(left, right), key=lambda j: (int(ink_count[j]), abs(j - center)))
            if ink_count[edge] > .08 * h:
                break
            edges.append(edge)
        if len(edges) == count:
            yield [glyphs[:, a:b] for a, b in zip(edges, [*edges[1:], w], strict=True)]


def _angled_dot(glyph):
    # 市 has an angled dot above the crossbar; the similar 巿 has an upright stroke.
    x, y, w, h = cv2.boundingRect(glyph)
    if h < 25:
        return False
    cropped = glyph[y:y + h, x:x + w]
    bars = np.flatnonzero(cropped.sum(axis=1) >= .65 * w)
    if not len(bars) or not 4 <= bars[0] <= .21 * h:
        return False
    dot = cropped[:bars[0]]
    if not all(0 < line.sum() <= .25 * w for line in dot):
        return False
    centers = [np.flatnonzero(line).mean() for line in dot]
    return np.mean(centers[-2:]) - np.mean(centers[:2]) >= max(.65, .019 * h)


def _read_caption(frame, box, index, cache):
    x, y, end, bottom = box
    height = bottom - y
    pad = max(3, round(.13 * height))
    patch = frame[max(0, y - pad):min(frame.shape[0], bottom + pad),
                  max(0, x - pad):min(frame.shape[1], end + pad)]
    paint = np.median(patch.reshape(-1, 3), axis=0)
    seen, readings = set(), {}
    ambiguous_stroke = False
    for glyphs in _caption_cuts(_distance(patch, paint)):
        fingerprint = tuple((glyph.shape, glyph.tobytes()) for glyph in glyphs)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        chars, margins = [], []
        for glyph in glyphs:
            ranked = index.rank(glyph, alphabet="cjk", limit=3)
            if len(ranked) < 3 or ranked[0]["distance"] > .190:
                break
            best, second, third = ranked
            selected, margin = best["text"], second["distance"] - best["distance"]
            if selected == "巿" and second["text"] == "市":
                ambiguous_stroke = True
                if second["distance"] > .17 or not _angled_dot(glyph):
                    break
                selected, margin = "市", third["distance"] - second["distance"]
            if margin < .006:
                break
            chars.append(selected)
            margins.append(margin)
        if len(chars) == len(glyphs):
            readings.setdefault("".join(chars), []).append(margins)
    verified = [text for text, scores in readings.items() if len(scores) >= 3
                and min(np.max(scores, axis=0)) >= .011]
    if len(verified) == 1:
        return verified[0]
    if ambiguous_stroke:
        return None
    return _transcript(frame, box, index, cache)


def mixed_footer_actions(frame, regions, index):
    """Return literal peer tabs while withholding cropped, central and selected slots."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or index is None or width < 200 or not 1.4 * width <= height <= 2.65 * width:
        return []
    offset = height - round(.265 * width)
    tile = frame[offset:]
    paint = np.median(tile[-round(.025 * width):, round(.18 * width):round(.82 * width)][::2, ::4], axis=(0, 1))
    if np.ptp(paint) > 22:
        return []
    distance = _distance(tile, paint)
    found = []
    for box in _marks(tile, distance, width):
        x, y, r, b = box
        inset = round(.079 * width)
        if (distance[y:b, inset:width - inset] <= 13).mean() < .77:
            continue
        if _plus(tile, box):
            owners = _text_only(tile, distance, box)
        elif _round_brand(tile, box):
            clean, bezel = _without_bezel(distance)
            owners = _branded(tile, clean, bezel, box)
        else:
            continue
        if owners:
            found.append(owners)
    if len(found) != 1:
        return []
    cache, result = {}, []
    for (left, top, right, bottom), (x, y, r, b), (cx, cy) in found[0]:
        text = _read_caption(frame, (left, top + offset, right, bottom + offset), index, cache)
        if text:
            result.append(_action(text, [x, y + offset, r, b + offset], [cx, cy + offset]))
    return result
