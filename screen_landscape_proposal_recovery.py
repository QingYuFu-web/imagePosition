"""Recover small, physically bounded layout witnesses from existing raster evidence."""

from itertools import combinations

import cv2
import numpy as np

from screen_landscape_stacked_tabs import (
    _CUSTOMIZE, _pane_peers, _ranked_views, _read_heading, _read_outlined, _supported_phases, _winner,
    stacked_setting_tabs,
)


def title_from_pixels(gray, paths):
    height, width = gray.shape
    left, top = round(.029*width), round(.004*height)
    section = gray[top:round(.093*height), left:round(.255*width)].astype(np.float32)
    signal = section - np.median(section, axis=1)[:,None]
    peak = float(np.percentile(signal, 99.2))
    if peak < 42:
        return False
    mask = np.uint8(signal >= max(34, .40*peak))
    # Compact neighboring glyph strokes while avoiding the separate back icon.
    bridge = max(2, round(.007*height))
    columns = cv2.morphologyEx(np.max(mask, axis=0)[None,:], cv2.MORPH_CLOSE,
                               np.ones((1, bridge), np.uint8))[0]
    switches = np.flatnonzero(np.diff(np.pad(columns, 1).astype(np.int16)))
    ranges = list(zip(switches[::2], switches[1::2]))
    for i, (a, r) in enumerate(ranges):
        ends = [r]
        if i+1 < len(ranges) and ranges[i+1][0]-r <= .015*height:
            ends.append(ranges[i+1][1])
        for end in ends:
            if not .040*height <= end-a <= .096*height:
                continue
            if _read_heading(signal[:, max(0, a-1):end+1], paths):
                return True
    return False


def recentered_rim(gray, box):
    a, t, r, b = box
    w, h = r-a, b-t
    radius = max(2, round(.16*h))
    delta = max(2, round(.072*h))
    planes = gray[max(0,t-radius-delta):b+radius+delta, max(0,a-radius-delta):r+radius+delta].astype(np.int16)
    x0, y0 = max(0,a-radius-delta), max(0,t-radius-delta)
    a, r, t, b = a-x0, r-x0, t-y0, b-y0

    def find_edge(field, center):
        alternatives = []
        for offset in {2, delta}:
            start, stop = max(offset, center-radius), min(field.shape[0]-offset, center+radius+1)
            alternatives.extend(((field[p]-np.maximum(field[p-offset], field[p+offset]) >= 6).mean(), -abs(center-p), p)
                                for p in range(start, stop))
        if not alternatives:
            return None
        score, _, position = max(alternatives)
        return position if score >= .72 else None

    left, right = (find_edge(planes[t+round(.20*h):b-round(.20*h)].T, x) for x in (a, r))
    top, bottom = (find_edge(planes[:,a+round(.16*w):r-round(.16*w)], y) for y in (t, b))
    if None in (left, right, top, bottom):
        return None
    return left+x0, top+y0, right+x0+1, bottom+y0+1


def read_recentered_label(signal, paths):
    native = _read_outlined(signal, paths)
    if native:
        return native
    reads = _ranked_views(signal, paths)
    likely = _winner(reads, .75, .035)
    if len(reads) < 3 or reads[0][0][0] not in _CUSTOMIZE or not all(view[0][0] == reads[0][0][0] for view in reads):
        return None
    if likely:
        for scale in (.86, 1.17):
            adjusted = cv2.resize(signal, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            if _winner(_ranked_views(adjusted, paths), .80, .045) == likely:
                return likely
    likely = reads[0][0][0]
    if all(view[0][1] >= .70 and view[0][1]-view[1][1] >= .033 for view in reads):
        finalists = tuple(sorted({word for rank in reads for word, score in rank}))
        if _supported_phases(signal, paths, finalists, likely) and _supported_phases(
                cv2.GaussianBlur(signal, (0,0), .65), paths, finalists, likely) >= 2:
            return likely
    return None


def recovered_outlined(frame, regions, index):
    height, width = frame.shape[:2]
    if (index is None or regions.get("layer") != "page" or regions.get("regions")
            or height < 220 or width < 1.55*height):
        return []
    boxes = [tuple(p["box"]) for p in regions["candidates"] if p.get("visibility") == "page"
             and 0 <= p["box"][0] < p["box"][2] <= width and 0 <= p["box"][1] < p["box"][3] <= height]
    options = [p for p in boxes if .089*width <= p[2]-p[0] <= .21*width
               and .048*height <= p[3]-p[1] <= .112*height and p[1] >= .072*height and p[3] <= .34*height]
    paths = tuple(index.fonts if index.fonts is not None else index.cjk_sources)
    if not options or not paths:
        return []
    gray = None
    outcome = []
    for owner in options:
        a, t, r, b = owner
        w, h = r-a, b-t
        parallels = [p for p in boxes if p[2] <= a and abs((p[1]+p[3]-t-b)/2) <= .14*h
                     and .72*h <= p[3]-p[1] <= 1.20*h and p[2]-p[0] >= 1.24*w]
        if not any(abs((left[2]-left[0])-(right[2]-right[0])) <= .14*w
                   and -.05*w <= right[0]-left[2] <= .14*w for left, right in combinations(sorted(parallels), 2)):
            continue
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rim = recentered_rim(gray, owner)
        if rim is None:
            continue
        if not title_from_pixels(gray, paths):
            continue
        reads = set()
        for a, t, r, b in (rim, owner):
            w, h = r-a, b-t
            roi = gray[t+round(.16*h):b-round(.18*h), a+round(.092*w):r-round(.092*w)].astype(np.float32)
            word = read_recentered_label(roi - np.median(roi, axis=1)[:, None], paths)
            if word in _CUSTOMIZE:
                reads.add(word)
        if len(reads) == 1:
            text = reads.pop()
            a, t, r, b = rim
            outcome.append({"role": "customize_game_controls", "function": f"打开{text}", "preview_text": text,
                            "box": list(rim), "click_point": [(a+r)//2, (t+b)//2]})
    return outcome if len(outcome) == 1 else []


def _runs(mask):
    switches = np.flatnonzero(np.diff(np.pad(mask, 1).astype(np.int16)))
    return list(zip(switches[::2].tolist(), switches[1::2].tolist()))


def rail_pixel_owners(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame[:, :round(.25*w)], cv2.COLOR_BGR2GRAY).astype(np.int16)
    watch = gray[round(.19*h):round(.74*h)]
    changes = watch[:,2:] - watch[:,:-2]
    rising = (changes >= 4).mean(axis=0)

    def horizontal_edge(lo, hi, quorum):
        lo, hi = round(lo*w), round(hi*w)
        winner = lo + int(rising[lo:hi].argmax())
        return winner if rising[winner] >= quorum else None

    start = horizontal_edge(.012, .062, .47)
    separator = horizontal_edge(.14, .233, .80)
    if None in (start, separator):
        return []
    gap = max(2, round(.012*w))
    band = (changes[:,separator-gap:separator] <= -4).mean(axis=0)
    if band.max() < .35:
        return []
    a, r = start+1, separator-gap+int(band.argmax())+2
    width = r-a
    if not .108*w <= width <= .215*w:
        return []
    watch = gray[:,a+round(.14*width):a+round(.39*width)]
    vertical = np.pad(np.median(watch[2:]-watch[:-2], axis=1), 1)
    start, stop = round(.085*h), round(.80*h)
    upper = [start+a+int(vertical[start+a:start+b].argmax()) for a,b in _runs(vertical[start:stop] >= 6)]
    lower = [start+a+int(vertical[start+a:start+b].argmin()) for a,b in _runs(vertical[start:stop] <= -6)]
    pairs = []
    for top in upper:
        bottom = next((y for y in lower if y > top), None)
        if bottom is not None and .052*h <= bottom-top <= .093*h and not any(top < y < bottom for y in upper):
            # A colored selection can produce a positive derivative but lacks a raised body.
            paint = np.median(watch[top+2:bottom-1])
            outside = np.median(watch[bottom+2:bottom+max(4, round(.025*h))])
            if paint - outside >= 7:
                pairs.append((top, bottom+1))
    if not 4 <= len(pairs) <= 7:
        return []
    typical = int(np.median([b-t for t,b in pairs]))
    chroma_pixels = frame[start:stop, a+round(.16*width):r-round(.16*width)]
    chroma = np.ptp(np.median(chroma_pixels, axis=1), axis=1)
    strong = _runs(chroma >= max(20, .61*chroma.max()))
    if len(strong) != 1:
        return []
    y0, y1 = strong[0]
    if y1-y0 > max(2, round(.19*typical)):
        return []
    bottom = start + y1
    active = (bottom-typical, bottom)
    pairs = sorted([*pairs, active])
    if any(b > nxt for (_, b),(nxt, _) in zip(pairs,pairs[1:])):
        return []
    return [(a,t,r,b) for t,b in pairs]


def recovered_tabs(frame, regions, index):
    height, width = frame.shape[:2]
    if (index is None or regions.get("layer") != "page" or regions.get("regions")
            or height < 210 or width < 1.55*height):
        return []
    rows = rail_pixel_owners(frame)
    if not rows:
        return []
    lane = rows[0][2]
    proposals = [p for p in regions['candidates'] if p.get('visibility') == 'page' and p['box'][0] >= lane]
    if not _pane_peers([p['box'] for p in proposals], lane, width, height):
        return []
    recovered = [{"box": list(row), "visibility": "page"} for row in rows]
    return stacked_setting_tabs(frame, {**regions, "candidates": [*recovered, *proposals]}, index)
