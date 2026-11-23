"""Read portrait native settings using a title, grouped rows and visible switches."""

from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np

from inventory_setting_switches import find_setting_switches
from inventory_settings_navigation import _HEADINGS, _normalized, settings_back
from screen_caption_transcript import _contrast, _ranked_glyph, _read_cut, _shape_parts, read_caption
from screen_component_captions import _bands
from screen_landscape_search_field import _available


_SENTENCE_CASE = re.compile(r"[A-Z][a-z]{2,}(?: [a-z]{2,}){1,6}")


@dataclass(frozen=True)
class _Line:
    box: tuple
    contrast: float
    accent: tuple | None

    @property
    def height(self):
        return self.box[3] - self.box[1]

    @property
    def y(self):
        return (self.box[1] + self.box[3]) / 2


def _center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def _heading(gray):
    h, w = gray.shape
    x, end = round(.137 * w), round(.59 * w)
    y, bottom = round(.045 * h), round(.111 * h)
    tile = gray[y:bottom, x:end]
    ink = np.uint8(tile <= 140)
    a, t, width, depth = cv2.boundingRect(ink)
    if not (.025 * w <= depth <= .063 * w and 1.25 * depth <= width <= 7.0 * depth
            and t >= 2 and t + depth < len(tile) and x + a <= .36 * w
            and .014 <= ink.mean() <= .19):
        return None
    return x + a, y + t, x + a + width, y + t + depth


def _body_lines(frame, gray, top, paper):
    height, width = gray.shape
    x, right = round(.026 * width), round(.80 * width)
    top = top + round(.068 * width)
    signal = cv2.subtract(int(paper), gray[top:height - round(.033 * width), x:right])
    mask = signal >= 77
    result = []
    for a, b in _bands(mask.sum(axis=1) >= max(3, round(.004 * width))):
        if not .018 * width <= b - a <= .057 * width:
            continue
        dx, dy, w, h = cv2.boundingRect(np.uint8(mask[a:b]))
        left, y = x + dx, top + a + dy
        if not (.028 * width <= left <= .17 * width and 1.44 * h <= w <= .76 * width and y + h < height):
            continue
        cut = mask[a + dy:a + dy + h, dx:dx + w]
        pixels = frame[y:y + h, left:left + w][cut]
        saturation = np.ptp(pixels, axis=1)
        accent = tuple(np.median(pixels, axis=0)) if (saturation >= 40).mean() >= .76 else None
        if accent is None and (saturation <= 19).mean() < .91:
            continue
        contrast = float(np.percentile(signal[a + dy:a + dy + h, dx:dx + w][cut], 80))
        result.append(_Line((left, y, left + w, y + h), contrast, accent))
    return result


def _hierarchy(gray, lines, header):
    height, width = gray.shape
    accent = [line for line in lines if line.accent is not None and line.box[2] < .41 * width]
    if (len(accent) < 2 or not .038 * height <= accent[0].box[1] - header[3] <= .12 * height
            or np.ptp([line.box[0] for line in accent]) > .026 * width
            or np.max(np.ptp([line.accent for line in accent], axis=0)) > 22):
        return []
    second = accent[1].box[1]
    strip = gray[max(accent[0].box[3], second - round(.12 * width)):second - round(.014 * width):1, ::4]
    if not strip.size or not np.any((np.abs(strip.astype(np.int16) - np.median(strip)) >= 11).mean(axis=1) >= .80):
        return []
    rows = [line for line in lines if line.accent is None and line.contrast >= 161
            and line.box[1] >= accent[0].box[3] + .030 * width
            and abs(line.box[0] - accent[0].box[0]) <= .023 * width]
    if len(rows) < 5 or np.median([line.height for line in rows]) < .021 * width:
        return []
    grouped = []
    for line in rows:
        summaries = [item for item in lines if item.accent is None
                     and .46 * line.contrast <= item.contrast <= .77 * line.contrast
                     and .006 * width <= item.box[1] - line.box[3] <= .032 * width
                     and abs(item.box[0] - line.box[0]) <= .019 * width
                     and .52 * line.height <= item.height <= 1.32 * line.height]
        grouped.append((line, summaries[0] if len(summaries) == 1 else None))
    return grouped if sum(item is not None for _, item in grouped) >= 3 else []


def _upper_edge(signal, dx, dy, glyph):
    left, top, right, bottom, _ = glyph
    start = max(0, dy + top - 3)
    values = signal[start:dy + bottom, dx + left:dx + right].max(axis=1)
    level = .60 * np.percentile(values, 83)
    positions = np.flatnonzero(values >= level)
    if positions.size == 0 or not positions[0]:
        return None
    i = int(positions[0])
    return start + i - (values[i] - level) / (values[i] - values[i - 1])


def _ascender_overrides(parts, signal, dx, dy, index, cache):
    bank = "choice_latin"
    cap = parts[0]
    top = _upper_edge(signal, dx, dy, cap)
    size = cap[3] - cap[1]
    if top is None or cap[2] - cap[0] < .38 * size:
        return {}
    shape = _ranked_glyph(index, cache, bank, cap[4])
    if (len(shape) < 2 or not shape[0]["text"].isupper() or shape[0]["distance"] > .16
            or shape[1]["distance"] - shape[0]["distance"] < .029):
        return {}
    confirmed, unresolved = {}, set()
    for left, first, right, bottom, ink in parts[1:]:
        if right - left > .19 * size or abs(bottom - cap[3]) > .095 * size or bottom - first < .88 * size:
            continue
        options = _ranked_glyph(index, cache, bank, ink)
        if (len(options) < 3 or {options[0]["text"], options[1]["text"]} != {"I", "l"}
                or options[1]["distance"] > .102 or options[2]["distance"] - options[1]["distance"] < .031):
            continue
        key = ("caption_rank", bank, ink.shape, ink.tobytes())
        edge = _upper_edge(signal, dx, dy, (left, first, right, bottom, ink))
        if edge is None or not max(.36, .022 * size) <= top - edge <= .20 * size:
            unresolved.add(key)
            continue
        confirmed[key] = [option for option in options if option["text"] != "I"]
    return {key: options for key, options in confirmed.items() if key not in unresolved}


def _read_ascenders(frame, area, index, cache, *, enlarge=False):
    """Resolve a straight lowercase stem only when its ink rises past a proven capital."""
    left, top, right, bottom = area
    tile = frame[top:bottom, left:right]
    if enlarge:
        tile = cv2.resize(tile, (round(1.5 * tile.shape[1]), round(1.5 * tile.shape[0])),
                          interpolation=cv2.INTER_CUBIC)
    sy, sx = (bottom - top) / tile.shape[0], (right - left) / tile.shape[1]
    signal, _ = _contrast(tile)
    peak = float(np.percentile(signal, 98))
    if peak < 96:
        return None
    fonts = index.fonts if index.fonts is not None else index.choice_latin_sources
    paths = {Path(path).name: path for path in fonts}
    reports, seen = {}, set()
    for level in (.25, .30, .35, .41, .48, .53, .64):
        mask = np.uint8(signal >= level * peak)
        x, y, w, h = cv2.boundingRect(mask)
        if h < 13 or not 4.3 * h <= w <= 24 * h:
            continue
        cut = mask[y:y + h, x:x + w]
        identity = cut.shape, cut.tobytes()
        if identity in seen:
            continue
        seen.add(identity)
        parts = _shape_parts(cut, h, choice=True, join_letters=True)
        if len(parts) < 9 or not (ranks := _ascender_overrides(parts, signal, x, y, index, cache)):
            continue
        reading = _read_cut(parts, h, index, paths, ChainMap(ranks, cache), "latin",
                            choice=True, sentence_case=True, join_letters=True)
        if reading is None or not _SENTENCE_CASE.fullmatch(reading[0]) or reading[2] < .021:
            continue
        text, error, margin = reading
        box = [left + round(x * sx), top + round(y * sy), left + round((x + w) * sx), top + round((y + h) * sy)]
        reports.setdefault(text, []).append((error, margin, box))
    if len(reports) == 1:
        text, observations = next(iter(reports.items()))
        if len(observations) >= 2:
            error, margin, box = min(observations)
            return {"text": text, "line_box": box, "max_distance": error, "min_margin": margin}
    return None


def _read(frame, box, index, cache, *, expanded=False):
    x, y, right, bottom = box
    pad = max(3, round((.24 if expanded else .13) * (bottom - y)))
    area = (max(0, x - pad), max(0, y - pad), min(frame.shape[1], right + pad), min(frame.shape[0], bottom + pad))
    result = read_caption(frame, area, index, cache=cache, choice=True, join_radicals=True,
                          sentence_case=True, join_letters=True)
    if result is None and expanded:
        result = _read_ascenders(frame, area, index, cache)
        if result is None and bottom - y < 28:
            result = _read_ascenders(frame, area, index, cache, enlarge=True)
    if result:
        line = result["line_box"]
        size = bottom - y
        if max(abs(a - b) for a, b in zip(line[:3], box[:3], strict=True)) <= max(3, .16 * size) and (
                abs(line[3] - bottom) <= max(3, .29 * size)):
            return result
    return None


def _owned_tracks(line, switches):
    return [switch for switch in switches if abs(switch["center"][1] - line.y) <= .62 * line.height]


def _raw_colored_thumb(frame, box, paper):
    x, y, right, bottom = box
    tile = frame[y:bottom, x:right].astype(np.int16)
    h, w = tile.shape[:2]
    v, u = np.ogrid[:h, :w]
    cy = (h - 1) / 2
    distance = (v - cy) ** 2 + (u - (w - 1 - cy)) ** 2
    thumb = np.median(tile[distance <= (.27 * h) ** 2], axis=0)
    track = np.median(tile[round(.35 * h):round(.64 * h), round(.13 * w):round(.27 * w)], axis=(0, 1))
    if (np.ptp(thumb) < 49 or np.ptp(track) < 19 or np.max(paper - track) < 31
            or np.mean(track - thumb) < 30):
        return False
    hues = (thumb - thumb.mean(), track - track.mean())
    if np.dot(*hues) < .87 * np.linalg.norm(hues[0]) * np.linalg.norm(hues[1]):
        return False
    round_thumb = distance <= (.475 * h) ** 2
    ink = (np.max(np.abs(tile - thumb), axis=2) <= 33) & (np.ptp(tile, axis=2) >= 44)
    return ((ink & round_thumb).sum() >= .80 * (ink | round_thumb).sum()
            and ink[distance <= (.33 * h) ** 2].mean() >= .94)


def _recover_colored_tracks(frame, rows, switches, paper, regions):
    """Confirm a denoised track against the original colored circular thumb."""
    height, width = frame.shape[:2]
    peers = [s for s in switches if s["state"] in {"on", "off"}]
    if len(peers) != 1:
        return []
    peer = peers[0]
    x, _, right, _ = peer["box"]
    reach = round(.105 * width)
    recovered = []
    for line, summary in rows:
        known = _owned_tracks(line, switches)
        if summary is not None or any(track["state"] != "unknown" for track in known):
            continue
        top, bottom = max(0, round(line.y) - reach), min(height, round(line.y) + reach)
        left = round(.68 * width)
        rail = frame[top:bottom].copy()
        rail[:, left:] = cv2.medianBlur(rail[:, left:], 5 if width >= 500 else 3)
        margin = max(0, 140 - len(rail))
        padded = cv2.copyMakeBorder(rail, margin, margin, 0, 0, cv2.BORDER_REPLICATE) if margin else rail
        possible = []
        for mark in find_setting_switches(padded, top=margin, bottom=margin + len(rail)):
            a, t, b, end = mark["box"]
            box = a, t + top - margin, b, end + top - margin
            center = _center(box)
            if (mark["state"] == "on" and .74 * (right - x) <= b - a <= 1.25 * (right - x)
                    and abs(center[0] - peer["center"][0]) <= .045 * width
                    and abs(center[1] - line.y) <= .62 * line.height
                    and _available(box, regions) and _raw_colored_thumb(frame, box, paper)):
                possible.append({**mark, "box": box, "center": tuple(center)})
        if len(possible) == 1:
            recovered.append(possible[0])
    return recovered


def _color_swatch(frame, rows, switches, regions):
    """Find a solid painted chooser mark in the trailing settings lane."""
    width = frame.shape[1]
    span = round(.079 * width)
    left = round(.82 * width)
    found = []
    for line, summary in rows:
        if summary is None or _owned_tracks(line, switches):
            continue
        y = round((line.y + summary.y) / 2)
        tile = frame[y - span:y + span, left:round(.978 * width)]
        if not tile.size:
            continue
        mask = np.uint8(np.ptp(tile, axis=2) >= 54)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i, (x, t, w, h, count) in enumerate(stats[1:], 1):
            if (not .046 * width <= h <= .106 * width or not .88 <= w / h <= 1.14
                    or not .69 <= count / (w * h) <= .86 or not 0 < t < t + h < len(tile)
                    or abs(t + h / 2 - span) > .36 * line.height or not 0 < x < x + w < tile.shape[1]):
                continue
            component = labels[t:t + h, x:x + w] == i
            v, u = np.ogrid[:h, :w]
            circle = ((u - (w - 1) / 2) / (.48 * w)) ** 2 + ((v - (h - 1) / 2) / (.48 * h)) ** 2 <= 1
            paint = tile[t:t + h, x:x + w][component].astype(np.int16)
            box = (left + x, y - span + t, left + x + w, y - span + t + h)
            if ((component & circle).sum() >= .87 * (component | circle).sum()
                    and (np.abs(paint - np.median(paint, axis=0)).max(axis=1) <= 24).mean() >= .85
                    and _available(box, regions)):
                found.append(box)
    return len(found) == 1


def native_switch_list_actions(frame, regions, index):
    """Return visible choices and switches on a sectioned native settings screen."""
    height, width = frame.shape[:2]
    if index is None or regions["layer"] != "page" or width < 360 or not 1.46 * width <= height <= 2.62 * width:
        return []
    sample = frame[::12, ::12]
    paper = np.median(sample, axis=(0, 1))
    if min(paper) < 231 or np.ptp(paper) > 14 or (np.abs(sample.astype(np.int16) - paper).max(axis=2) <= 12).mean() < .88:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (header := _heading(gray)) is None:
        return []
    cache = {}
    rows = _hierarchy(gray, _body_lines(frame, gray, header[3], np.median(paper)), header)
    if not rows:
        return []
    heading = _read(frame, header, index, cache)
    back = settings_back(frame, (), header)
    if not heading or _normalized(heading["text"]) not in _HEADINGS or back is None or not _available(header, regions):
        return []
    all_switches = find_setting_switches(frame, top=rows[0][0].box[1] - round(.04 * width))
    all_switches = [s for s in all_switches if _available(s["box"], regions)]
    switches = [switch for switch in all_switches if switch["state"] in {"on", "off"}]
    if not switches:
        return []
    recovered = _recover_colored_tracks(frame, rows, all_switches, paper, regions)
    all_switches = [s for s in all_switches if not any(abs(s["center"][1] - added["center"][1]) <= .021 * width
                                                      for added in recovered)] + recovered
    switches = [s for s in all_switches if s["state"] in {"on", "off"}]
    if np.ptp([switch["center"][0] for switch in switches]) > .05 * width:
        return []
    owned = [(line, summary, _owned_tracks(line, all_switches)) for line, summary in rows]
    owned_count = sum(len(tracks) == 1 and tracks[0] in switches for _, _, tracks in owned)
    if owned_count < 1 or (owned_count == 1 and not _color_swatch(frame, rows, all_switches, regions)):
        return []
    actions = []
    if _available(back["box"], regions):
        actions.append({"role": "back", "function": back["function"], "box": list(back["box"]),
                        "click_point": list(back["center"])})
    for line, summary, tracks in owned:
        if len(tracks) > 1 or (not tracks and summary is None) or (tracks and tracks[0] not in switches):
            continue
        area = tracks[0]["box"] if tracks else line.box
        paired = (line.box[0], min(line.box[1], area[1]), max(line.box[2], area[2]),
                  max(line.box[3], area[3], summary.box[3] if summary else 0))
        if not _available(paired, regions):
            continue
        reading = _read(frame, line.box, index, cache, expanded=True)
        if not reading or reading["min_margin"] < .021 or not _available(reading["line_box"], regions):
            continue
        text = reading["text"]
        verb, role = "打开", "open_setting"
        if tracks:
            verb = "关闭" if tracks[0]["state"] == "on" else "开启"
            role = "toggle_switch"
        action = {"role": role, "function": f"{verb}{text}", "preview_text": text,
                  "box": list(area), "click_point": _center(area)}
        caption = reading["line_box"]
        if tracks and caption[2] < area[0]:
            context = [caption[0], min(caption[1], area[1]), area[2], max(caption[3], area[3])]
            if _available(context, regions):
                action["context_box"] = context
        actions.append(action)
    return actions
