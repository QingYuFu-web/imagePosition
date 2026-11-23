"""Locate labeled setting boxes below a measured preview on a wide overlay."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np
from PIL import ImageFont

from inventory_choice_glyphs import _outline
from screen_component_captions import _bands, _tight
from screen_font_glyphs import _mask_pixels


@dataclass(frozen=True)
class _Check:
    box: tuple[int, int, int, int]
    state: str
    label: tuple[int, int, int, int]
    cue: tuple[int, int, int, int]

    @property
    def height(self):
        return self.box[3] - self.box[1]

    @property
    def center(self):
        a, t, b, z = self.box
        return (a + b) / 2, (t + z) / 2


def _boundary_steps(image, gap, axis):
    shifted = np.moveaxis(image.astype(np.int16), axis, 0)
    diff = np.abs(shifted[2 * gap:] - shifted[:-2 * gap]).max(axis=2)
    return np.pad((diff >= 25).mean(axis=1), (gap, gap))


def _preview_floor(panel):
    height, width = panel.shape[:2]
    gap = max(2, round(.0015 * width))
    field = panel[:, round(.178 * width):round(.863 * width):3]
    steps = _boundary_steps(field, gap, axis=0)
    tops = np.flatnonzero((steps >= .48) & (np.arange(height) >= .078 * height) &
                          (np.arange(height) <= .30 * height))
    bottoms = np.flatnonzero((steps >= .65) & (np.arange(height) >= .36 * height) &
                             (np.arange(height) <= .70 * height))
    separated = []
    for a, b in _bands(np.isin(np.arange(height), bottoms)):
        if b - a > 3 * gap:
            continue
        floor = b
        for u, v in _bands(np.isin(np.arange(height), tops)):
            if not .25 * height <= floor - v <= .59 * height or v - u > 3 * gap:
                continue
            vertical = _boundary_steps(panel[v + gap:floor - gap:3], gap, axis=1)
            x = np.arange(width)
            lhs = np.flatnonzero((vertical >= .55) & (x >= .09 * width) & (x <= .31 * width))
            rhs = np.flatnonzero((vertical >= .35) & (x >= .70 * width) & (x <= .94 * width))
            if lhs.size and rhs.size and rhs[0] - lhs[-1] >= .44 * width:
                separated.append(floor)
    if not separated or max(separated) - min(separated) > 2 * gap:
        return None
    return max(separated)


def _square_boxes(gray, blue, height):
    observed = []
    for low, high in ((7, 16), (12, 25), (19, 39)):
        ink = cv2.bitwise_or(cv2.Canny(gray, low, high), cv2.Canny(blue, low, high))
        contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            bounds = cv2.boundingRect(contour)
            x, y, w, h = bounds
            if (not .046 * height <= min(w, h) <= max(w, h) <= .083 * height or not .88 <= w / h <= 1.13
                    or cv2.contourArea(contour) < .83 * w * h):
                continue
            shape = _outline(contour, bounds)
            if shape is not None and shape[0] == "checkbox":
                observed.append((low, (x, y, x + w, y + h)))
    accepted = []
    for low, box in sorted(observed, key=lambda item: -((item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))):
        a, t, b, z = box
        size = z - t
        if any(abs((a + b - x - r) / 2) <= .20 * size and abs((t + z - y - q) / 2) <= .20 * size
               for x, y, r, q in accepted):
            continue
        if any(l != low and max(abs(v - o) for v, o in zip(box, other, strict=True)) <= .13 * size
               for l, other in observed):
            accepted.append(box)
    if len(accepted) == 2 and all(_state(gray, box) == "off" for box in accepted):
        accepted.extend(_checked_surface_boxes(gray, height))
    return accepted


def _state(gray, box):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    margin = max(3, round(.15 * h))
    chip = gray[top + margin:bottom - margin, left + margin:right - margin]
    fill = float(np.median(chip))
    if fill >= 66:
        mark = np.uint8(chip >= fill + 46)
        x, y, span, depth = cv2.boundingRect(mark)
        if (.20 * w <= span <= .41 * w and .20 * h <= depth <= .41 * h and
                .83 <= span / depth <= 1.21 and abs(x + span / 2 - chip.shape[1] / 2) <= .13 * w and
                abs(y + depth / 2 - chip.shape[0] / 2) <= .13 * h and mark.sum() >= .80 * span * depth):
            return "on"
    if fill <= 56 and (np.abs(chip.astype(np.int16) - fill) <= 11).mean() >= .88:
        return "off"
    return None


def _checked_surface_boxes(gray, height):
    """Recover a filled square when the outer outline is fragmented."""
    observations = []
    for level in (55, 67, 79):
        ink = np.uint8(gray >= level)
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = bounds = cv2.boundingRect(contour)
            if (not .042 * height <= min(w, h) <= max(w, h) <= .072 * height
                    or not .88 <= w / h <= 1.13 or cv2.contourArea(contour) < .84 * w * h):
                continue
            shape = _outline(contour, bounds)
            if shape is None or shape[0] != "checkbox":
                continue
            pad = max(2, round(.09 * min(w, h)))
            left, top, right, bottom = x - pad, y - pad, x + w + pad, y + h + pad
            if min(left, top, gray.shape[1] - right, len(gray) - bottom) < pad:
                continue
            box = left, top, right, bottom
            if _state(gray, box) != "on":
                continue
            first, last = x + w // 4, x + w - w // 4
            start, end = y + h // 4, y + h - h // 4
            border = (gray[top - pad:top, first:last], gray[bottom:bottom + pad, first:last],
                      gray[start:end, left - pad:left], gray[start:end, right:right + pad])
            surface = float(np.median(gray[start:end, first:last]))
            if min(surface - np.median(side) for side in border) >= 24:
                observations.append((level, box))
    accepted = []
    for level, box in observations:
        spread = max(2, round(.13 * (box[3] - box[1])))
        if any(max(abs(a - b) for a, b in zip(box, prior, strict=True)) <= spread for prior in accepted):
            continue
        if any(cut != level and max(abs(a - b) for a, b in zip(box, other, strict=True)) <= spread
               for cut, other in observations):
            accepted.append(box)
    return accepted


def _line(gray, lo, top, hi, bottom, size):
    if hi - lo < size or bottom <= top:
        return None
    field = gray[top:bottom, lo:hi].astype(np.int16)
    light = field - np.median(field)
    peak = float(np.percentile(light, 99))
    if peak < 46:
        return None
    mask = light >= max(34, .32 * peak)
    groups = []
    for a, b in _bands(mask.sum(axis=0) >= 2):
        if groups and a - groups[-1][1] <= .14 * size:
            groups[-1] = groups[-1][0], b
        else:
            groups.append((a, b))
    for a, b in groups:
        box = _tight(mask[:, a:b], lo + a, top)
        if box is None:
            continue
        x, y, r, z = box
        h = z - y
        if .36 * size <= h <= .70 * size and 1.50 * h <= r - x <= 8.7 * h and abs(y + z - top - bottom) <= .45 * size:
            return box
    return None


def _checkboxes(gray, blue, panel_height, floor):
    width = gray.shape[1]
    start = floor + round(.018 * panel_height)
    clipped = gray[start:-round(.04 * panel_height)]
    if not clipped.size:
        return []
    result = []
    for x, y, right, bottom in _square_boxes(clipped, blue[start:start + len(clipped)], panel_height):
        y, bottom = y + start, bottom + start
        size = bottom - y
        box = (x, y, right, bottom)
        state = _state(gray, box)
        if state is None:
            continue
        center = (y + bottom) / 2
        top, end = round(center - .43 * size), round(center + .43 * size)
        label = _line(gray, max(0, x - round(6.8 * size)), top, x - round(.55 * size), end, size)
        cue = _line(gray, right + round(.21 * size), top, min(width, right + round(2.28 * size)), end, size)
        if label and cue and .43 * size <= cue[0] - right <= 1.1 * size:
            result.append(_Check(box, state, label, cue))
    return result


def _cue_shape(gray, box):
    l, t, r, b = box
    inset = 3
    field = gray[t - inset:b + inset, l - inset:r + inset].astype(np.float32)
    rect = field[inset:-inset, inset:-inset]
    paper = np.median(field)
    peak = max(paper + 1., float(np.percentile(rect, 96)))
    return cv2.resize(np.clip((rect - paper) / (peak - paper), 0, 1), (42, 20), interpolation=cv2.INTER_LINEAR)


def _lattice(gray, records):
    viable = []
    for chosen in combinations(records, 4):
        ordered = sorted(chosen, key=lambda check: check.center[1])
        (tl, tr), (bl, br) = [sorted(ordered[a:b], key=lambda check: check.center[0])
                              for a, b in ((0, 2), (2, 4))]
        size = float(np.median([check.height for check in chosen]))
        x, y = tl.center
        x2, y2 = br.center
        if (max(check.height for check in chosen) - min(check.height for check in chosen) > .20 * size or
                not 7.2 * size <= x2 - x <= 17.7 * size or not 1.60 * size <= y2 - y <= 4.55 * size or
                max(abs(tr.center[0] - x2), abs(tr.center[1] - y), abs(bl.center[0] - x),
                    abs(bl.center[1] - y2)) > .21 * size or len({check.state for check in chosen}) < 2):
            continue
        if max(abs(tl.label[0] - bl.label[0]), abs(tr.label[0] - br.label[0])) > .34 * size:
            continue
        cues = [_cue_shape(gray, check.cue) for check in chosen]
        if all(float(np.abs(a - b).mean()) <= .21 for a, b in combinations(cues, 2)):
            viable.append((tl, tr, bl, br))
    return viable[0] if len(viable) == 1 else ()


@lru_cache(maxsize=96)
def _font(path, size):
    return ImageFont.truetype(str(path), size)


@lru_cache(maxsize=8192)
def _clipped_bitmap(path, size, char):
    source = _font(path, size).getmask(char, mode="L")
    if not min(source.size):
        return None
    bitmap = _mask_pixels(source)
    x, y, w, h = cv2.boundingRect(np.uint8(bitmap >= 64))
    return bitmap[y:y + h, x:x + w].copy() if min(w, h) else None


_SHIFT_AXES = ((2, 1), (2, -1), (1, 1), (1, -1))
_SHIFT_WEIGHT = np.float32(round(.3 * cv2.INTER_TAB_SIZE) / cv2.INTER_TAB_SIZE)


@lru_cache(maxsize=64)
def _letter_rasters(paths, char, width, height):
    layouts = []
    for path in paths:
        for size in (22, 25, 29, 37, 52):
            bitmap = _clipped_bitmap(path, size, char)
            if bitmap is None:
                continue
            native = cv2.resize(bitmap, (width, height), interpolation=cv2.INTER_AREA)
            native = np.float32(native) / 255
            # Single-channel blur preserves the original floating-point pixels.
            layouts.extend((native, cv2.GaussianBlur(native, (3, 3), .40)))
    output = np.empty((len(layouts), 1 + len(_SHIFT_AXES), height + 4, width + 4), np.float32)
    if layouts:
        padded = np.pad(np.stack(layouts), ((0, 0), (2, 2), (2, 2)))
        output[:, 0] = padded
        retained = np.float32(1 - _SHIFT_WEIGHT) * padded
        # These translations move along one axis. The zero padding makes
        # each rolled edge equivalent to OpenCV's constant-border sample.
        for position, (axis, step) in enumerate(_SHIFT_AXES, start=1):
            output[:, position] = retained + _SHIFT_WEIGHT * np.roll(padded, step, axis=axis)
    return output.reshape(-1, height + 4, width + 4)


@lru_cache(maxsize=320)
def _soft_letter(paths, options, value, width, height):
    target = np.frombuffer(value, np.float32).reshape(height + 4, width + 4)
    mass = np.square(target).sum()
    scores = []
    for char in options:
        bank = _letter_rasters(paths, char, width, height)
        if not len(bank):
            continue
        energy = np.square(target - bank).sum(axis=(1, 2))
        scores.append((float((energy / (.5 * (mass + np.square(bank).sum(axis=(1, 2))))).min()), char))
    scores.sort()
    if len(scores) < 2 or scores[0][0] > .20 or scores[1][0] - scores[0][0] < .013:
        return None
    # These printed variants can be nearly identical at 20 pixels. Use
    # simplified spelling only when its outline is independently supported.
    if scores[0][1] == "內" and any(char == "内" and score - scores[0][0] <= .030 for score, char in scores[1:]):
        return "内"
    return scores[0][1]


def _read_glyph(glyph, peak, index):
    first, second = (glyph >= cut * peak for cut in (.29, .40))
    middle = [index.rank(mask, alphabet="cjk", limit=6) for mask in (first, second)]
    # A short read needs two distinct masks with a stronger match and margin.
    if (not np.array_equal(first, second) and all(len(row) >= 2 for row in middle)
            and middle[0][0]["text"] == middle[1][0]["text"] != "內"
            and max(row[0]["distance"] for row in middle) <= .148
            and min(row[1]["distance"] - row[0]["distance"] for row in middle) >= .031):
        return middle[0][0]["text"]
    rankings = middle + [index.rank(glyph >= cut * peak, alphabet="cjk", limit=6) for cut in (.19, .53)]
    if all(len(row) >= 2 for row in rankings):
        winners = {row[0]["text"] for row in rankings}
        if (len(winners) == 1 and "內" not in winners and max(row[0]["distance"] for row in rankings) <= .16
                and min(row[1]["distance"] - row[0]["distance"] for row in rankings) >= .022):
            return rankings[0][0]["text"]
    options = tuple(sorted({record["text"] for row in rankings for record in row}))
    x, y, w, h = cv2.boundingRect(np.uint8(glyph >= .25 * peak))
    if not .52 <= w / max(h, 1) <= 1.37:
        return None
    target = np.pad(np.minimum(1, glyph[y:y + h, x:x + w] / peak), 2)
    paths = tuple(index.fonts if index.fonts is not None else index.cjk_sources)
    return _soft_letter(paths, options, target.tobytes(), w, h)


def _read_label(gray, box, index):
    x, y, r, b = box
    pad = 4
    raw = gray[y - pad:b + pad, x - pad:r + pad].astype(np.float32)
    source = np.maximum(raw - np.median(raw), 0)
    peak = float(np.percentile(source, 99))
    if peak < 58:
        return None
    x, y, w, h = cv2.boundingRect(np.uint8(source >= .30 * peak))
    count = round(w / max(1, 1.075 * h))
    if not 2 <= count <= 7 or not 15 <= h <= 47:
        return None
    source = source[y:y + h, x:x + w]
    col = (source >= .30 * peak).sum(axis=0)
    stops = [0]
    for i in range(1, count):
        center = i * w / count
        distance = .14 * h
        start, end = round(center - distance), round(center + distance) + 1
        edge = min(range(start, end), key=lambda j: (int(col[j]), abs(center - j)))
        if col[edge] > .13 * h:
            return None
        stops.append(edge)
    stops.append(w)
    letters = []
    for left, right in zip(stops[:-1], stops[1:], strict=True):
        char = _read_glyph(source[:, left:right], peak, index)
        if not char:
            return None
        letters.append(char)
    return "".join(letters)


def _magnified_label(gray, box, index):
    """Confirm a small owned caption under two contrast treatments."""
    left, top, right, bottom = box
    pad = 4
    if not 12 <= bottom - top <= 26 or min(left, top, gray.shape[1] - right, len(gray) - bottom) < pad:
        return None
    tile = np.uint8(gray[top - pad:bottom + pad, left - pad:right + pad])
    zoom = cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    crop = (2 * pad, 2 * pad, zoom.shape[1] - 2 * pad, len(zoom) - 2 * pad)
    text = _read_label(zoom, crop, index)
    if text is None:
        return None
    sharpened = cv2.addWeighted(zoom, 1.42, cv2.GaussianBlur(zoom, (0, 0), .85), -.42, 0.)
    return text if _read_label(sharpened, crop, index) == text else None


def setting_check_layout(panel, *, denoise=False):
    """Measure a repeated set of captions, checkbox outlines, and state labels."""
    height, width = panel.shape[:2]
    if width < 1.42 * height or height < 360:
        return None, ()
    floor = _preview_floor(panel)
    if floor is None:
        return None, ()
    if denoise:
        panel = cv2.GaussianBlur(panel, (3, 3), .5)
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    blue = cv2.subtract(panel[..., 0], panel[..., 2])
    checks = _checkboxes(gray, blue, height, floor)
    return gray, _lattice(gray, checks) if 4 <= len(checks) <= 20 else ()


def wide_panel_check_actions(frame, regions, index):
    """Return grounded state changes for complete setting controls."""
    if regions["layer"] != "wide_panel" or len(regions["regions"]) < 2 or index is None:
        return []
    left, top, right, bottom = regions["regions"][0]
    gray, checks = setting_check_layout(frame[top:bottom, left:right])
    actions = []
    for check in checks:
        label = _read_label(gray, check.label, index) or _magnified_label(gray, check.label, index)
        if not label:
            continue
        x, y, r, b = check.box
        a, t, _, bz = check.label
        _, ct, cr, cb = check.cue
        box = [left + a, top + min(t, y, ct), left + max(r, cr), top + max(b, bz, cb)]
        verb = "关闭" if check.state == "on" else "开启"
        actions.append({"role": "checkbox_setting", "function": f"{verb}{label}", "preview_text": label,
                        "box": box, "click_point": [left + (x + r) // 2, top + (y + b) // 2]})
    return actions
