"""Locate independently labeled gain handles on a multi-band equalizer surface."""

from pathlib import Path
from statistics import median

import cv2
import numpy as np

from inventory_list_rows import _Box
from inventory_slider_actions import _spans
from screen_colored_settings_panel import _normalize, _printed, _word_shapes
from screen_range_geometry import ranged_controls
from screen_setting_ranges import _caption, _caption_windows, range_action


_TITLE = ("Equalizer", "均衡器", "イコライザー")
_OTHER_HEADINGS = ("Visualizer", "Playback", "Settings", "Controls", "Effects", "Audio")
_FREQ = ("20Hz", "25Hz", "31Hz", "32Hz", "40Hz", "50Hz", "60Hz", "63Hz", "80Hz", "100Hz", "125Hz",
         "150Hz", "160Hz", "200Hz", "250Hz", "300Hz", "315Hz", "400Hz", "500Hz", "630Hz", "800Hz",
         "1kHz", "1.25kHz", "1.6kHz", "2kHz", "2.5kHz", "3.2kHz", "4kHz", "5kHz", "6.3kHz", "8kHz",
         "10kHz", "12kHz", "12.5kHz", "16kHz", "20kHz")
_DERAIL = ("-20dB", "+20dB", "-15dB", "+15dB", "-12dB", "+12dB", "-10dB", "+10dB", "0dB")
_PREAMP = {"Preamp", "Gain", "前置增益"}


def _frequency(label):
    return float(label[:-3]) * 1000 if label.endswith("kHz") else float(label[:-2])


def _band_windows(gray, track, spacing):
    height, width = gray.shape
    x = round(track.center[0])
    glyph = max(7., .28 * spacing)
    side = round(.48 * spacing)
    bounds = (max(0, x - side), min(height, track.bottom + round(.25 * glyph)),
              min(width, x + side), min(height, track.bottom + round(5.13 * glyph)))
    return list(_caption_windows(gray, bounds, glyph, side=False))


def _split_frequency(gray, box, light, index):
    left, top, right, bottom = box
    chip = gray[top:bottom, left:right]
    mode = cv2.MORPH_TOPHAT if light else cv2.MORPH_BLACKHAT
    ink = cv2.morphologyEx(chip, mode, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    peak = float(np.percentile(ink, 98))
    if peak < 38:
        return None
    fonts = tuple(index.fonts or index.choice_latin_sources)
    if fonts and index.fonts is None:
        sibling = Path(fonts[0]).parent
        fonts += tuple(path for name in ("bahnschrift.ttf", "arialn.ttf") if (path := sibling / name).is_file())
    glyphs = tuple("0123456789kHzdB")
    words, ratios, bank = _word_shapes(fonts, glyphs)
    if not words:
        return None
    candidates = []
    for level in (.30, .41, .52):
        cut = ink >= peak * level
        spans = list(_spans(cut.any(axis=0)))
        if not 3 <= len(spans) <= 6:
            continue
        chars = []
        for start, end in spans:
            _, y, _, h = cv2.boundingRect(np.uint8(cut[:, start:end]))
            if not (.35 * chip.shape[0] <= h <= .85 * chip.shape[0]):
                break
            sample = _normalize(np.minimum(1, ink[y:y + h, start:end] / peak))
            aspect = ((end - start) / h) / ratios
            selected = np.flatnonzero((aspect >= .58) & (aspect <= 1.56))
            if not selected.size:
                break
            scores = np.einsum("ij,j->i", bank[selected], sample) - .15 * abs(np.log(aspect[selected]))
            position = int(scores.argmax())
            char = words[selected[position]]
            rival = max((score for i, score in zip(selected, scores, strict=True) if words[i] != char), default=0.)
            if scores[position] < .84 or scores[position] - rival < .017:
                break
            chars.append(char)
        if len(chars) == len(spans) and (name := "".join(chars)) in _FREQ:
            candidates.append(name)
    return candidates[0] if len(candidates) >= 2 and len(set(candidates)) == 1 else None


def _read_bands(gray, common, spacing, index):
    rows, uncertain, caption_rise = [], [], []
    for thumb, track in common:
        windows = _band_windows(gray, track, spacing)
        words = {word: box[1] - track.bottom for box, light in windows if (word := _printed(
             gray, box, index, vocabulary=_FREQ + _DERAIL, light=light,
             minimum_score=.87, minimum_margin=.018)) in _FREQ}
        if len(words) == 1:
            name, rise = next(iter(words.items()))
            rows.append((thumb, track, name))
            caption_rise.append(rise)
        else:
            uncertain.append((thumb, track, windows))
    if len(rows) < .60 * len(common):
        return rows
    if uncertain:
        # Narrow this second read to the same caption line proven by the other bands.
        baseline = median(caption_rise)
        for thumb, track, windows in uncertain:
            for box, light in windows:
                if abs(box[1] - track.bottom - baseline) >= .12 * spacing:
                    continue
                if name := _split_frequency(gray, box, light, index):
                    rows.append((thumb, track, name))
                    break
    return sorted(rows, key=lambda row: row[0].center[0])


def _portrait_box(box, height):
    return _Box(box.top, height - box.right, box.bottom, height - box.left)


def _trim_rail(frame, thumb, track):
    cx, cy = round(track.center[0]), round(thumb.center[1])
    gap = max(3, round(1.35 * track.width))
    if not gap <= cx < frame.shape[1] - gap:
        return None
    first = max(0, track.top - 2 * gap)
    last = min(frame.shape[0], track.bottom + 2 * gap)
    chip = frame[first:last, cx - gap:cx + gap + 1].astype(np.int16)
    center = chip[:, gap]
    opposed = np.minimum(np.abs(center - chip[:, 0]).max(axis=1), np.abs(center - chip[:, -1]).max(axis=1))
    ink = np.uint8(opposed >= 17)
    ink[max(0, thumb.top - first):thumb.bottom - first] = 1
    valid = cv2.morphologyEx(ink[None], cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))[0]
    return next((_Box(track.left, first + lo, track.right, first + hi) for lo, hi in _spans(valid)
                 if lo <= cy - first < hi and hi - lo >= .62 * track.height), None)


def equalizer_band_actions(frame, overlay, index):
    """Return a scoped foreground when a title and a bank of real gain tracks agree."""
    height, width = frame.shape[:2]
    if index is None or min(width, height) < 340:
        return None
    if overlay is None:
        panel = 0, 0, width, height
    elif overlay.kind == "dialog":
        panel = overlay.panel
    else:
        return None
    left, top, right, bottom = panel
    pixels = frame[top:bottom, left:right]
    h, w = pixels.shape[:2]
    if min(w, h) < 320:
        return None
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    heading = (round(.011 * w), round(.010 * h), round(.44 * w), round(.094 * h))
    title_gray = gray[heading[1]:heading[3], heading[0]:heading[2]]
    light = float(np.median(title_gray[::5, ::7])) < 140
    if _printed(gray, heading, index, vocabulary=_TITLE + _OTHER_HEADINGS, light=light) not in _TITLE:
        return None
    vertical = [(_portrait_box(row.thumb, h), _portrait_box(row.track, h)) for row in
                ranged_controls(cv2.rotate(pixels, cv2.ROTATE_90_CLOCKWISE))]
    vertical = [(thumb, trim) for thumb, track in vertical if .165 * h <= track.height <= .50 * h
                and (trim := _trim_rail(pixels, thumb, track)) is not None
                and .12 * h <= trim.top <= .64 * h and trim.bottom <= .83 * h]
    if not 5 <= len(vertical) <= 19:
        return None
    recent = sorted(vertical, key=lambda row: row[1].center[0])
    rail_top, rail_end = (median(getattr(track, key) for _, track in recent) for key in ("top", "bottom"))
    common = [(thumb, track) for thumb, track in recent if abs(track.top - rail_top) <= .03 * h
              and abs(track.bottom - rail_end) <= .062 * h]
    if len(common) < 5:
        return None
    gaps = [b.center[0] - a.center[0] for (_, a), (_, b) in zip(common, common[1:])]
    spacing = median(gaps)
    if spacing < 21 or sum(abs(gap - spacing) <= .12 * spacing for gap in gaps) < .74 * len(gaps):
        return None
    band_rows = _read_bands(gray, common, spacing, index)
    if len(band_rows) < max(4, .55 * len(common)) or any(
            _frequency(first[2]) >= _frequency(second[2]) for first, second in zip(band_rows, band_rows[1:])):
        return None
    actions = [{**range_action(thumb, rail, name, name, (left, top), axis="vertical"), "preview_text": name}
               for thumb, rail, name in band_rows]
    for item in ranged_controls(pixels):
        if item.track.bottom < rail_top and (name := _caption(gray, item, index)) in _PREAMP:
            label = "前置增益"
            actions.append(range_action(item.thumb, item.track, label, name, (left, top)))
    return panel, actions
