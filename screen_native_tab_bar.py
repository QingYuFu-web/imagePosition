"""Own visible mobile navigation tabs and transcribe their printed captions."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_component_captions import _bands, _outlier, _tight
from screen_dark_accented_tabs import accented_tab_pair
from screen_light_underlined_peer_tabs import light_underlined_peer_actions


def _distance(patch, paint):
    b, g, r = (cv2.absdiff(plane, int(color)) for plane, color in zip(cv2.split(patch), paint, strict=True))
    return cv2.max(cv2.max(b, g), r)


def _ink_color(patch, distance):
    peak = np.percentile(distance[distance >= 24], 76)
    return np.median(patch[distance >= peak], axis=0)


def _pair(frame, distance, span, band):
    width = frame.shape[1]
    left, right = span
    first, last = band
    text = _tight(distance[first:last, left:right] >= 24, left, first)
    if text is None:
        return None
    l, t, r, b = text
    text_width, text_height = r - l, b - t
    center = (l + r) / 2
    if (not .013 * width <= text_height <= .043 * width or
            not .56 * text_height <= text_width <= min(.158 * width, 9 * text_height) or
            abs(center - (left + right) / 2) > .053 * width):
        return None
    icon_start = max(0, t - round(.098 * width))
    icon_end = t - max(2, round(.008 * width))
    reach = round(.065 * width)
    il, ir = max(left, round(center) - reach), min(right, round(center) + reach)
    icon_distances = distance[icon_start:icon_end, il:ir]
    # A boundary shared across the slot is the top of the rail, not an icon.
    if not icon_distances.size:
        return None
    icon_rows = icon_distances >= 24
    boundaries = np.flatnonzero(icon_rows.mean(axis=1) > .67)
    if len(boundaries):
        shift = int(boundaries[-1]) + 1
        icon_start += shift
        icon_rows = icon_rows[shift:]
    icon = _tight(icon_rows, il, icon_start) if icon_rows.size else None
    if icon is None:
        return None
    x, y, xx, yy = icon
    iw, ih = xx - x, yy - y
    if (not .024 * width <= ih <= .079 * width or not .024 * width <= iw <= .107 * width or
            not .69 <= iw / ih <= 1.68 or not .007 * width <= t - yy <= .033 * width or
            abs(center - (x + xx) / 2) > .026 * width or x == il or xx == ir):
        return None
    return (text, icon, _ink_color(frame[t:b, l:r], distance[t:b, l:r]),
            _ink_color(frame[y:yy, x:xx], distance[y:yy, x:xx]))


def _bottom_pairs(frame):
    height, width = frame.shape[:2]
    offset = height - round(.244 * width)
    tile = frame[offset:]
    paint = np.median(tile[-max(4, round(.024 * width)):, round(.035 * width):round(.965 * width)][::2, ::4],
                      axis=(0, 1))
    distance = _distance(tile, paint)
    minimum = max(3, round(.003 * width))
    present = distance >= 24
    if not present[-round(.126 * width):].any():
        return None
    for count in (6, 5, 4, 3):
        edges = [round(part * width / count) for part in range(count + 1)]
        inset = round(.018 * width)
        spans = [(left + inset, right - inset) for left, right in zip(edges[:-1], edges[1:], strict=True)]
        ink_rows = [present[:, left:right].sum(axis=1) >= minimum for left, right in spans]
        aligned = np.sum(ink_rows, axis=0) >= count - 1
        for first, last in reversed(_bands(aligned)):
            if not .010 * width <= last - first <= .044 * width or len(tile) - last > .126 * width:
                continue
            pad = max(2, round(.004 * width))
            band = (max(0, first - pad), min(len(tile), last + pad))
            pairs = [_pair(tile, distance, span, band) for span in spans]
            if any(pair is None for pair in pairs):
                continue
            icons = [icon for _, icon, _, _ in pairs]
            if np.ptp([box[1] for box in icons]) > .018 * width or np.ptp([box[3] for box in icons]) > .018 * width:
                continue
            centers = [(left + right) / 2 for left, _, right, _ in icons]
            if np.ptp(np.diff(centers)) > .039 * width:
                continue
            selected = _outlier([color for _, _, color, _ in pairs], same_max=32, gap=40)
            if selected is None or selected != _outlier([color for _, _, _, color in pairs], same_max=42, gap=44):
                continue
            top = max(0, min(box[1] for box in icons) - round(.019 * width))
            bar = distance[top:min(len(tile), last + pad), inset:width - inset]
            if (bar <= 13).mean() < .78:
                continue
            return offset, pairs, selected
    return None


def _single_caption(frame, area, index):
    left, top, right, bottom = area
    patch = frame[top:bottom, left:right]
    paper = np.median(patch[::2, ::2], axis=(0, 1))
    distance = _distance(patch, paper)
    contrast = np.percentile(distance, 99)
    if contrast < 56:
        return None
    accepted = {}
    for ratio in (.26, .34, .43, .51):
        ink = np.uint8(distance >= contrast * ratio)
        x, y, w, h = cv2.boundingRect(ink)
        if not .74 <= w / max(h, 1) <= 1.23:
            continue
        cut = ink[y:y + h, x:x + w]
        key = (cut.shape, cut.tobytes())
        if key in accepted:
            continue
        reading = index.recognize(cut, alphabet="all")
        if reading is not None:
            accepted[key] = reading["text"]
    texts = set(accepted.values())
    return next(iter(texts)) if len(accepted) >= 2 and len(texts) == 1 else None


def _transcript(frame, ink_box, index, cache):
    x, y, right, bottom = ink_box
    h = bottom - y
    margin = max(4, round(.28 * h))
    area = (max(0, x - margin), max(0, y - margin), min(frame.shape[1], right + margin),
            min(frame.shape[0], bottom + margin))
    reading = read_caption(frame, area, index, cache=cache)
    if reading is not None:
        return reading["text"]
    if .74 * h <= right - x <= 1.23 * h:
        return _single_caption(frame, area, index)
    return None


def _serif_letter(glyph, peak, index):
    def votes(masks, gap):
        observed = {}
        for mask in masks:
            identity = mask.tobytes()
            if identity not in observed:
                options = index.rank(mask, alphabet="cjk", limit=2)
                if (len(options) >= 2 and options[0]["distance"] <= .215
                        and options[1]["distance"] - options[0]["distance"] >= gap):
                    observed[identity] = options[0]["text"]
        return list(observed.values())

    wins = votes((np.uint8(glyph >= cut * peak) for cut in (.17, .28, .39, .50, .61, .73, .82)), .014)
    if len(wins) < 2 or len(set(wins)) != 1:
        return None
    if len(wins) >= 3:
        return wins[0]
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thinned = votes((cv2.erode(np.uint8(glyph >= cut * peak), cross) for cut in (.17, .28, .39)), .013)
    return wins[0] if len(thinned) >= 2 and set(thinned) == {wins[0]} else None


def _serif_caption(frame, ink_box, index):
    """Read a thick CJK caption only inside a measured navigation slot."""
    x, y, right, bottom = ink_box
    size = bottom - y
    if not 22 <= size <= 66 or not 1.78 * size <= right - x <= 4.60 * size:
        return None
    pad = max(4, round(.235 * size))
    left, top = max(0, x - pad), max(0, y - pad)
    tile = frame[top:bottom + pad, left:right + pad]
    paper = np.median(tile[::2, ::2], axis=(0, 1))
    signal = _distance(tile, paper)
    peak = float(np.percentile(signal, 99))
    if peak < 66:
        return None
    ink = signal >= .30 * peak
    spans = _bands(ink.any(axis=0))
    if not 2 <= len(spans) <= 4:
        return None
    for a, b in spans:
        h = cv2.boundingRect(np.uint8(ink[:, a:b]))[3]
        if not .76 * size <= h <= 1.13 * size or not .83 * h <= b - a <= 1.21 * h:
            return None
    names = [_serif_letter(signal[:, a:b], peak, index) for a, b in spans]
    return "".join(names) if all(names) else None


def _action(text, box, point):
    return {"role": "switch_tab", "function": f"切换到{text}", "box": box, "click_point": point,
            "preview_text": text}


def _bottom_actions(frame, index, cache):
    height, width = frame.shape[:2]
    result = _bottom_pairs(frame)
    if result is None:
        return []
    offset, pairs, selected = result
    centers = [(icon[0] + icon[2]) / 2 for _, icon, _, _ in pairs]
    edges = [0, *(round((first + second) / 2) for first, second in zip(centers[:-1], centers[1:], strict=True)), width]
    actions = []
    for i, (ink, icon, _, _) in enumerate(pairs):
        if i == selected:
            continue
        x, y, r, b = ink
        caption = x, y + offset, r, b + offset
        text = _transcript(frame, caption, index, cache) or _serif_caption(frame, caption, index)
        if text:
            box = [edges[i], offset + max(0, icon[1] - round(.016 * width)), edges[i + 1],
                   min(height, offset + b + round(.019 * width))]
            actions.append(_action(text, box, [round(centers[i]), offset + round((icon[1] + icon[3]) / 2)]))
    return actions


def _panel_seam(frame, mark, bottom):
    width = frame.shape[1]
    gap = max(2, round(.0035 * width))
    inset = round(.036 * width)
    before = frame[mark - gap:bottom + gap, inset:width - inset].astype(np.int16)
    shift = 2 * gap
    trailing = frame[mark + gap:bottom + gap + shift, inset:width - inset].astype(np.int16)
    contrasts = np.abs(trailing - before).max(axis=2)
    support = np.mean(contrasts >= 11, axis=1)
    strongest = int(np.argmax(support))
    return mark + strongest if support[strongest] >= .79 else None


def _strip_titles(frame, mark):
    left, y, right, last = mark
    width = frame.shape[1]
    seam = _panel_seam(frame, y, last)
    if seam is None:
        return None
    gap = max(3, round(.007 * width))
    offset, end = y - round(.081 * width), y - gap
    tile = frame[offset:end]
    paint = np.median(frame[end:y, round(.08 * width):round(.92 * width)][::2, ::4], axis=(0, 1))
    distance = _distance(tile, paint)
    if (distance <= 13).mean() < .70:
        return None
    mask = distance >= 24
    rows = _bands(mask.sum(axis=1) >= .045 * width)
    suitable = [(t, b) for t, b in rows if .015 * width <= b - t <= .045 * width]
    if len(suitable) != 1:
        return None
    t, b = suitable[0]
    h = b - t
    top, bottom = max(0, t - round(.19 * h)), min(len(tile), b + round(.19 * h))
    runs = _bands(mask[top:bottom].sum(axis=0) >= max(2, round(.08 * h)))
    titles = []
    for a, z in runs:
        if titles and a - titles[-1][1] < .70 * h:
            titles[-1] = titles[-1][0], z
        else:
            titles.append((a, z))
    if not 3 <= len(titles) <= 8:
        return None
    spaces = np.asarray([r[0] - l[1] for l, r in zip(titles[:-1], titles[1:], strict=True)])
    if spaces.min() < .91 * h or spaces.max() > 4.1 * h or np.ptp(spaces) > 1.10 * h:
        return None
    tight = [_tight(mask[top:bottom, a:z], a, offset + top) for a, z in titles]
    if any(box is None for box in tight):
        return None
    if any(not .78 * h <= box[3] - box[1] <= 1.42 * h or box[2] - box[0] < .72 * h for box in tight):
        return None
    picked = [i for i, (a, _, z, _) in enumerate(tight)
              if max(0, min(z, right) - max(a, left)) >= .84 * (z - a)
              and abs((a + z - left - right) / 2) <= .42 * h]
    if len(picked) != 1:
        return None
    colors = [_ink_color(frame[t:b, a:z], distance[t - offset:b - offset, a:z]) for a, t, z, b in tight]
    if _outlier(colors, same_max=45, gap=43) != picked[0]:
        return None
    mark_color = np.median(frame[y:last, left + 3:right - 3].reshape(-1, 3), axis=0)
    if np.linalg.norm(mark_color - colors[picked[0]]) > 58:
        return None
    return tight, picked[0], seam


def _underlined_actions(frame, index, cache):
    width = frame.shape[1]
    top, bottom = round(.118 * width), round(.342 * width)
    part = frame[top:bottom]
    b, g, r = cv2.split(part)
    chroma = cv2.subtract(cv2.max(cv2.max(b, g), r), cv2.min(cv2.min(b, g), r))
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(chroma >= 49), connectivity=8)
    found = []
    for x, y, w, h, area in stats[1:]:
        if (2 <= h <= .015 * width and max(.038 * width, 6.2 * h) <= w <= .245 * width
                and area >= .76 * w * h):
            bounds = int(x), top + int(y), int(x + w), top + int(y + h)
            strip = _strip_titles(frame, bounds)
            if strip:
                found.append(strip)
    if len(found) != 1:
        return []
    titles, selected, seam = found[0]
    pad = round(.023 * width)
    result = []
    for i, title in enumerate(titles):
        left, top, right, bottom = title
        # A tab cut off by the edge can lose its final glyph.
        if i == selected or left < .018 * width or right >= .982 * width:
            continue
        text = _transcript(frame, title, index, cache)
        if text:
            box = [max(0, left - pad), top - pad // 2, min(width, right + pad), seam]
            result.append(_action(text, box, [round((left + right) / 2), round((top + bottom) / 2)]))
    return result


def _accented_caption(frame, caption, index, cache):
    text = _transcript(frame, caption, index, cache)
    left, top, right, bottom = caption
    height = bottom - top
    if text or not 14 <= height <= 22:
        return text
    pad = max(4, round(.43 * height))
    start = max(0, top - pad)
    enlarged = cv2.resize(frame[start:bottom + pad], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return _transcript(enlarged, (2 * left, 2 * (top - start), 2 * right, 2 * (bottom - start)), index, {})


def native_tab_actions(frame, regions, index):
    """Publish legible nonselected bottom tabs and underlined header tabs."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or index is None or width < 200 or not 1.40 * width <= height <= 2.65 * width:
        return []
    cache = {}
    bottom = _bottom_actions(frame, index, cache)
    header = _underlined_actions(frame, index, cache)
    results = bottom + (header or light_underlined_peer_actions(frame, regions, index))
    if not regions.get("regions") and not regions.get("floating_controls"):
        if (pair := accented_tab_pair(frame)) is not None:
            caption, context = pair
            if text := _accented_caption(frame, caption, index, cache):
                left, top, right, last = caption
                point = [round((left + right) / 2), round((top + last) / 2)]
                results.append({**_action(text, list(caption), point), "context_box": context})
    return results


_ORDER_SEARCH = {
    "搜索订单": "订单", "搜索我的订单": "订单", "查找订单": "订单", "查找我的订单": "订单",
    "搜尋訂單": "訂單", "搜尋我的訂單": "訂單", "查找訂單": "訂單", "查找我的訂單": "訂單",
}


def apply_header_tab_context(tabs, header_controls):
    """Name a tab for the collection identified in its adjacent search plate."""
    fields = [item for item in header_controls if item["role"] == "search"]
    if len(fields) != 1:
        return None
    field = fields[0]
    noun = _ORDER_SEARCH.get("".join(field["preview_text"].split()))
    if noun is None:
        return None
    left, top, right, bottom = field["box"]
    depth = bottom - top
    rail = [item for item in tabs if .18 * depth <= item["box"][1] - bottom <= 1.04 * depth
            and .44 * depth <= item["box"][3] - item["box"][1] <= 1.20 * depth
            and left <= item["click_point"][0] <= right]
    if len(rail) < 2:
        return None
    tops = [item["box"][1] for item in rail]
    ends = [item["box"][3] for item in rail]
    if max(tops) - min(tops) > .23 * depth or max(ends) - min(ends) > .12 * depth:
        return None
    for item in rail:
        label = item["preview_text"]
        if not label.isalpha() or label.isascii():
            continue
        title = label if label.endswith(noun) else label + noun
        item.update(function=f"切换到{title}", setting_label=label, setting_display_label=title, setting_state="off")
    return noun, max(ends)
