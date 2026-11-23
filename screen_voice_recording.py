"""Confirm a held voice gesture from the waveform and release surfaces."""

import cv2
import numpy as np


def _release_arc(gray, veil, *, subdued=False):
    height, width = gray.shape
    start = round(.75 * height)
    inset = round(.025 * width) if subdued else 0
    tail = gray[start:, inset:width - inset]
    if subdued:
        ink = np.uint8((tail >= veil + 10) & (tail <= veil + 34))
    else:
        ink = np.uint8(tail >= max(112, veil + 43))
    kernel = np.ones((5 if subdued else max(5, round(.075 * width) | 1), 1), np.uint8)
    joined = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if not (.91 * width <= w and y + h >= ink.shape[0] - 1 and .79 <= area / (w * h) <= 1.):
            continue
        body = labels[y:y + h, x:x + w] == i
        x += inset
        left, right = round(.055 * width) - x, round(.945 * width) - x
        if left < 0 or right > w:
            continue
        profile = (y + start + np.argmax(body[:, left:right], axis=0)).astype(np.int16)
        core = profile[round(.41 * len(profile)):round(.59 * len(profile))]
        side = max(3, round(.16 * width))
        middle = float(np.median(core))
        outer = [float(np.median(part)) for part in (profile[:side], profile[-side:])]
        if (not .80 * height <= middle <= .94 * height
                or not all(.029 * width <= end - middle <= .16 * width for end in outer)
                or abs(outer[0] - outer[1]) > .032 * width
                or (np.abs(np.diff(profile)) <= max(2, .008 * width)).mean() < .94):
            continue
        return round(middle)
    return None


def _gesture_guides(gray, middle, veil):
    height, width = gray.shape
    top = max(round(.70 * height), middle - round(.285 * width))
    end = middle - round(.017 * width)
    ink = np.uint8(gray[top:end] >= veil + 10)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    guides = []
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if not (.30 * width <= w <= .49 * width and .123 * width <= h <= .27 * width and area >= .65 * w * h):
            continue
        body = labels[y:y + h, x:x + w] == i
        present = np.flatnonzero(body.sum(axis=1) >= .30 * w)
        if present.size:
            shift = int(present[0])
            guides.append((x, y + shift, w, h - shift))
    guides.sort()
    if len(guides) != 2:
        return None
    (l, t, w, h), (r, y, rw, rh) = guides
    if (l > .047 * width or r + rw < .953 * width or not .055 * width <= r - l - w <= .17 * width
            or abs(t - y) > .026 * width or abs(h - rh) > .041 * width or abs(w - rw) > .055 * width):
        return None
    tones = [np.median(gray[top + a:top + a + hh, x + round(.20 * ww):x + round(.81 * ww)])
             for x, a, ww, hh in guides]
    return top + min(t, y), abs(tones[0] - tones[1]) >= 57


def _wave_lane(tile, *, short=False):
    height, width = tile.shape
    surface = float(np.median(tile))
    for lift in (32, 46, 60):
        ink = np.uint8(tile <= surface - lift)
        _, _, stats, centers = cv2.connectedComponentsWithStats(ink, connectivity=8)
        marks = sorted((x, y, h) for (a, b, w, h, area), (x, y) in zip(stats[1:], centers[1:], strict=True)
                       if 1 <= w <= .038 * width and .085 * height <= h <= .73 * height and area >= .61 * w * h
                       and 0 < a < a + w < width and 0 < b < b + h < height)
        if not (8 if short else 13) <= len(marks) <= (17 if short else 42):
            continue
        xs, ys, heights = np.array(marks).T
        steps = np.diff(xs)
        pitch = float(np.median(steps))
        variable = (heights.max() >= 1.53 * np.median(heights) and
                    (np.median(heights[-4:]) <= .63 * heights.max() if short else heights.min() <= .63 * np.median(heights)))
        if (.35 * width <= xs[-1] - xs[0] <= .85 * width and .019 * width <= pitch <= .066 * width
                and (abs(steps - pitch) <= .35 * pitch).mean() >= .82 and np.ptp(ys) <= .081 * height
                and variable):
            return True
    return False


def _transcript_indicator(gray, source, box):
    x, y, w, h = box
    title = gray[y + round(.14 * h):y + round(.49 * h), x + round(.042 * w):x + round(.88 * w)]
    ink = np.uint8(title <= float(np.median(title)) - 46)
    _, _, letters, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    text_parts = [a for a, _, span, k, area in letters[1:] if .25 * title.shape[0] <= k and span >= 2 and area >= 5]
    if len(text_parts) < 5 or max(text_parts) - min(text_parts) < .35 * w:
        return False
    sx, sy = source.shape[1] / gray.shape[1], source.shape[0] / gray.shape[0]
    pulse = source[round(sy * (y + .53 * h)):round(sy * (y + .89 * h)),
                   round(sx * (x + .80 * w)):round(sx * (x + .96 * w))]
    return pulse.size > 0 and _wave_lane(cv2.cvtColor(pulse, cv2.COLOR_BGR2GRAY), short=True)


def _indicator(gray, veil, guides, highlighted, source):
    height, width = gray.shape
    start, stop = round(.34 * height), min(guides - round(.17 * width), round(.66 * height))
    contours, _ = cv2.findContours(np.uint8(gray[start:stop] >= veil + 51), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for item in contours:
        x, y, w, h = cv2.boundingRect(item)
        compact = .28 * width <= w <= .56 * width and .145 * width <= h <= .285 * width
        transcript = highlighted and source is not None and .77 * width <= w <= .966 * width and .17 * width <= h <= .34 * width
        if (not (compact or transcript) or abs(x + w / 2 - .5 * width) > .038 * width
                or not .77 <= cv2.contourArea(item) / (w * h) <= .976):
            continue
        y += start
        gap = gray[y + h + round(.053 * width):guides - round(.03 * width), round(.10 * width):round(.90 * width)]
        if gap.size == 0 or (abs(gap.astype(np.int16) - veil) <= 11).mean() < .91:
            continue
        if transcript:
            if _transcript_indicator(gray, source, (x, y, w, h)):
                return True
        elif _wave_lane(gray[y + round(.12 * h):y + round(.80 * h), x + round(.13 * w):x + round(.87 * w)]):
            return True
    return False


def voice_recording_bounds(gray, *, source=None):
    """Return the gesture area in working pixels when the host is visibly veiled."""
    height, width = gray.shape
    if width < 150 or not 1.56 * width <= height <= 2.85 * width:
        return None
    center = slice(round(.11 * width), round(.89 * width))
    underlay = gray[round(.065 * height):round(.37 * height), center]
    veil = round(float(np.percentile(underlay, 78)))
    if not 34 <= veil <= 122 or (underlay >= veil + 39).mean() > .014:
        return None
    for subdued in (False, True):
        arc = _release_arc(gray, veil, subdued=subdued)
        if arc is None or (guides := _gesture_guides(gray, arc, veil)) is None:
            continue
        top, highlighted = guides
        if (highlighted or not subdued) and _indicator(gray, veil, top, highlighted, source):
            return 0, top, width, height
    return None
