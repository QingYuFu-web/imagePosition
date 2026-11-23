"""Recognize live producer controls and a visibly selected picture widget."""

from dataclasses import dataclass
from itertools import combinations
import re

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_component_captions import _bands
from screen_dialog_close import _cross_footprint
from screen_native_numeric_dialog import _symbol


_CLOCK = re.compile(r"\d{1,2}:\d{2}:\d{2}\Z")
_COMPOSER = {"comment here", "write a comment", "leave a comment", "写评论", "輸入留言", "输入评论"}
_DIGITS = frozenset("0123456789")


@dataclass(frozen=True)
class LiveProducerSurface:
    widget: tuple | None
    mode: str
    actions: tuple

    def owns_widget_layer(self, panel):
        if self.widget is None or not panel or len(panel) != 4:
            return False
        left, top, right, bottom = self.widget
        span = right - left
        return max(abs(a - b) for a, b in zip(self.widget, panel, strict=True)) <= .035 * span


def _center(box):
    left, top, right, bottom = box
    return (left + right) // 2, (top + bottom) // 2


def _action(box, function, role):
    box = tuple(map(int, box))
    return {"role": role, "function": function, "box": list(box), "click_point": list(_center(box))}


def _camera_active(frame):
    width = frame.shape[1]
    band = frame[:round(.066 * width), round(.52 * width):round(.87 * width)].astype(np.int16)
    b, g, r = cv2.split(band)
    witnesses = []
    for contrast in (38, 47, 58):
        mask = np.uint8((g >= 87) & (g - r >= contrast) & (g - b >= contrast - 8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        marks = [(x + w / 2, y + h / 2) for x, y, w, h, n in stats[1:]
                 if .0055 * width <= min(w, h) <= max(w, h) <= .020 * width
                 and .67 <= w / h <= 1.46 and .59 * w * h <= n <= w * h]
        if len(marks) == 1:
            witnesses.append(marks[0])
    return len(witnesses) >= 2 and np.ptp(witnesses, axis=0).max() <= max(1.5, .0025 * width)


def _handle_ink(gray, box, cutoff=165):
    left, top, right, bottom = box
    span = min(right - left, bottom - top)
    inset = round(.17 * span)
    chip = gray[top + inset:bottom - inset, left + inset:right - inset]
    if (chip >= 226).mean() < .63:
        return None
    return np.uint8(chip < cutoff)


def _cross_ink(mask):
    if mask is None:
        return False
    x, y, w, h = cv2.boundingRect(mask)
    return (.41 * len(mask) <= h <= .61 * len(mask) and .78 <= w / h <= 1.23
            and abs(x + w / 2 - mask.shape[1] / 2) <= .10 * w and abs(y + h / 2 - len(mask) / 2) <= .10 * h
            and _cross_footprint(mask[y:y + h, x:x + w].astype(bool)))


def _cross_handle(gray, box):
    if _cross_ink(_handle_ink(gray, box)):
        return True
    return sum(_cross_ink(_handle_ink(gray, box, cut)) for cut in (135, 145, 156, 176)) >= 2


def _corner(mask, *, inverted, quantized=False):
    if inverted:
        mask = mask[::-1, ::-1]
    span = min(mask.shape)
    reach = max(1, round(.32 * span))
    if mask[-reach:, -reach:].mean() > .17 or mask.mean() > .63:
        return False
    if mask[:reach].mean() >= .79 and mask[:, :reach].mean() >= .79:
        return True
    # At small raster sizes a crisp two-pixel arm can occupy two of the
    # three sampled rows. Keep the original empty-corner check unchanged.
    if not quantized or not 8 <= span <= 11:
        return False
    arm = int(.32 * span)
    return mask[:arm].mean() >= .79 and mask[:, :arm].mean() >= .79


def _resize_corners(mask, *, quantized=False):
    if mask is None:
        return False
    _, ids, stats, _ = cv2.connectedComponentsWithStats(mask)
    shapes = [(i, x, y, w, h) for i, (x, y, w, h, n) in enumerate(stats[1:], 1)
              if .22 * len(mask) <= min(w, h) <= max(w, h) <= .39 * len(mask) and n >= .32 * w * h]
    if len(shapes) != 2:
        return False
    first, second = sorted(shapes, key=lambda shape: shape[1])
    if not .18 * len(mask) <= second[1] - first[1] <= .39 * len(mask):
        return False
    if not .18 * len(mask) <= second[2] - first[2] <= .39 * len(mask):
        return False
    return all(_corner(ids[y:y + h, x:x + w] == i, inverted=pos == 1, quantized=quantized)
               for pos, (i, x, y, w, h) in enumerate((first, second)))


def _resize_handle(gray, box):
    if _resize_corners(_handle_ink(gray, box)):
        return True
    # The narrow corner strokes can have one partially covered row of pixels.
    return sum(_resize_corners(_handle_ink(gray, box, cut), quantized=True) for cut in (135, 145, 156, 176, 190)) >= 2


def _dash_line(gray, y, end_x, chip_size):
    expected = .335 * chip_size
    radius = max(3, round(.15 * chip_size))
    candidates = []
    for row in range(y - radius, y + radius + 1):
        flags = gray[row, :end_x] >= 194
        spans = [(a, b) for a, b in _bands(flags) if .32 * expected <= b - a <= .69 * expected]
        sections = []
        for a, b in spans:
            if not sections or not .79 * expected <= a - sections[-1][-1][0] <= 1.21 * expected:
                sections.append([])
            sections[-1].append((a, b))
        seq = max(sections, key=len, default=[])
        if len(seq) >= 13 and seq[-1][1] >= end_x - 2.3 * expected:
            candidates.append((len(seq), seq[-1][1] - seq[0][0], -abs(y - row), seq[0][0], row))
    return max(candidates)[3:] if candidates else None


def _selected_picture(gray):
    height, width = gray.shape
    x0, y0 = round(.66 * width), round(.26 * height)
    crop = gray[y0:round(.75 * height), x0:]
    reduced = cv2.resize(crop, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(reduced >= 220))
    discs = [(x0 + 2 * x, y0 + 2 * y, x0 + 2 * (x + w), y0 + 2 * (y + h)) for x, y, w, h, n in stats[1:]
             if .023 * width <= min(w, h) <= max(w, h) <= .045 * width and .87 <= w / h <= 1.15
             and .64 * w * h <= n <= .80 * w * h]
    if not 2 <= len(discs) <= 7:
        return None
    matches = []
    for a, b in combinations(sorted(discs, key=lambda box: box[1]), 2):
        ax, ay = _center(a)
        bx, by = _center(b)
        size = a[2] - a[0]
        if (abs(ax - bx) > .15 * size or abs((b[2] - b[0]) - size) > .15 * size
                or not .125 * height <= by - ay <= .365 * height or not _cross_handle(gray, a)
                or not _resize_handle(gray, b)):
            continue
        end_x = ax - round(.53 * size)
        upper, lower = _dash_line(gray, ay, end_x, size), _dash_line(gray, by, end_x, size)
        if upper is None or lower is None or abs(upper[0] - lower[0]) > .27 * size:
            continue
        left = min(upper[0], lower[0])
        if not .024 * width <= left <= .24 * width or not .47 * width <= end_x - left <= .90 * width:
            continue
        widget = tuple(map(int, (left, a[1], max(a[2], b[2]), b[3])))
        matches.append((widget, tuple(map(int, a))))
    return matches[0] if len(matches) == 1 else None


def _words(frame, box, index, *, numeric=False):
    reading = read_caption(frame, box, index, cache={}, choice=True, numeric=numeric, join_letters=True,
                           sentence_case=True, join_radicals=True)
    return reading["text"].casefold() if reading else ""


def _plain_box(mask, offset_x, offset_y, pad, width, height):
    x, y, w, h = cv2.boundingRect(np.uint8(mask))
    if not w or not h:
        return None
    return (max(0, offset_x + x - pad), max(0, offset_y + y - pad),
            min(width, offset_x + x + w + pad), min(height, offset_y + y + h + pad))


def _dark_footer(gray, *, width, height):
    left, right = round(.058 * width), round(.35 * width)
    top, bottom = height - round(.212 * width), height - round(.112 * width)
    area = gray[top:bottom, left:right]
    box = _plain_box(area >= 158, left, top, max(4, round(.006 * width)), width, height)
    if box is None:
        return None
    x, y, r, b = box
    if not .015 * width <= b - y <= .052 * width or not .15 * width <= r - x <= .32 * width:
        return None
    shoulder = round(.025 * width)
    inside = gray[y:b, x:r][::3, ::4]
    first = gray[y - shoulder:y - shoulder // 2, x:r][::2, ::3]
    last = gray[b + shoulder // 2:b + shoulder, x:r][::2, ::3]
    sections = [np.median(row) for row in (inside, first, last)]
    # On a black camera preview there may be no difference between the filled
    # capsule and the canvas. The other footer peers remain independently visible.
    if max(sections) > 70 or max(sections[1:]) - sections[0] < -5:
        return None
    yy = (y + b) // 2
    return box, yy


def _menu_bars(gray, y, width):
    left, right = round(.859 * width), round(.969 * width)
    pad = round(.032 * width)
    roi = np.uint8(gray[y - pad:y + pad, left:right] >= 173)
    _, _, stats, _ = cv2.connectedComponentsWithStats(roi)
    bars = sorted((yy, x, w, h) for x, yy, w, h, n in stats[1:]
                  if .026 * width <= w <= .066 * width and .003 * width <= h <= .012 * width
                  and w / h >= 4 and n >= .70 * w * h)
    if len(bars) != 3:
        return False
    yy, xs, widths, depths = np.asarray(bars).T
    return np.ptp(xs) <= .007 * width and np.ptp(widths) <= .011 * width and np.ptp(np.diff(yy)) <= .008 * width


def _printed_comment_here(frame, gray, box, index):
    x, y, right, bottom = box
    chip = gray[y:bottom, x:right]
    padding = max(3, round(.17 * (bottom - y)))
    votes = 0
    verified_first = None
    for cutoff in (115, 137, 158):
        mask = np.uint8(chip >= cutoff)
        spans = _bands(mask.any(axis=0))
        if len(spans) != 11:
            continue
        stop, begin = spans[6][1], spans[7][0]
        if begin - stop < .24 * (bottom - y):
            continue
        if verified_first is None:
            first = (x, y, x + min(begin, stop + padding), bottom)
            verified_first = _words(frame, first, index) == "comment"
        if not verified_first:
            return False
        distances = []
        for (left, end), expected in zip(spans[7:], "here", strict=True):
            ranks = index.rank(mask[:, left:end], alphabet="choice_latin", limit=2)
            if (len(ranks) < 2 or ranks[0]["text"] != expected or ranks[0]["distance"] > .20
                    or ranks[1]["distance"] - ranks[0]["distance"] < .032):
                break
            distances.append(ranks[0]["distance"])
        else:
            if sum(distances) / len(distances) <= .15:
                votes += 1
    return votes >= 2


def _composer(frame, gray, index):
    height, width = gray.shape
    found = _dark_footer(gray, width=width, height=height)
    if found is None:
        return None
    label, y = found
    if not _menu_bars(gray, y, width):
        return None
    if _words(frame, label, index) not in _COMPOSER and not _printed_comment_here(frame, gray, label, index):
        return None
    left, top, right, bottom = label
    box = (max(0, left - round(.014 * width)), top - round(.020 * width),
           right + round(.041 * width), bottom + round(.021 * width))
    return _action(box, "输入评论", "comment_input")


def _power(gray):
    width = gray.shape[1]
    left, top = round(.86 * width), round(.12 * width)
    chip = gray[top:round(.243 * width), left:round(.977 * width)]
    _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(chip >= 166))
    loops = [(i, x, y, w, h) for i, (x, y, w, h, n) in enumerate(stats[1:], 1)
             if .033 * width <= min(w, h) <= max(w, h) <= .057 * width and .88 <= w / h <= 1.29
             and .16 * w * h <= n <= .39 * w * h]
    if len(loops) != 1:
        return None
    i, x, y, w, h = loops[0]
    bars = [(a, b, c, d) for a, b, c, d, n in stats[1:]
            if .045 * w <= c <= .19 * w and .39 * h <= d <= .68 * h and .65 * c * d <= n
            and abs((a + c / 2) - (x + w / 2)) <= .10 * w and -.25 * h <= b - y <= .035 * h]
    if len(bars) != 1:
        return None
    loop = ids[y:y + h, x:x + w] == i
    side = round(.24 * w)
    if (loop[:round(.64 * h), side:-side].mean() >= .12 or not loop[round(.32 * h):round(.61 * h), :side].any()
            or not loop[round(.32 * h):round(.61 * h), -side:].any()
            or not loop[round(.80 * h):, side:-side].any()):
        return None
    _, up, _, _ = bars[0]
    pad = round(.21 * w)
    return left + x - pad, top + min(up, y) - pad, left + x + w + pad, top + y + h + pad


def _clock_digit(chip, index):
    if exact := _symbol(chip, index, _DIGITS, {"O": "0"}):
        return exact
    peak = float(np.percentile(chip, 98))
    votes = []
    for portion in (.29, .44, .57, .68):
        mask = np.uint8(chip >= peak * portion)
        ranks = index.rank(mask, alphabet="code", limit=8)
        if not ranks:
            continue
        first = ranks[0]
        numeral = first["text"]
        if numeral in _DIGITS:
            rival = next((r["distance"] for r in ranks[1:] if r["text"] not in _DIGITS | {"O"}), 1.)
            if first["distance"] <= .236 and rival - first["distance"] >= .021:
                votes.append(numeral)
        elif numeral in {"D", "O"} and any(r["text"] == "0" and r["distance"] <= .20
                                            and r["distance"] - first["distance"] <= .035 for r in ranks[1:]):
            x, y, w, h = cv2.boundingRect(mask)
            cut = mask[y:y + h, x:x + w]
            contours, tree = cv2.findContours(cut, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if tree is None or not .48 <= w / h <= .85:
                continue
            voids = [cv2.contourArea(c) for c, parent in zip(contours, tree[0, :, 3], strict=True) if parent >= 0]
            # A clock zero may rank beside O or D at low resolution. Check the
            # enclosed counter and a left edge shorter than a letter D stem.
            if len(voids) == 1 and voids[0] >= .36 * w * h and cut[:, 0].mean() <= .66:
                votes.append("0")
    return votes[0] if len(votes) >= 2 and len(set(votes)) == 1 else None


def _clock(gray, index):
    level = int(np.median(gray))
    if level > 83:
        return None
    ink = cv2.subtract(gray, level)
    x, y, w, h = cv2.boundingRect(np.uint8(ink >= 111))
    if not w or not h or not 4.2 <= w / h <= 6.35:
        return None
    crop = ink[y:y + h, x:x + w]
    spans = _bands((crop >= 111).any(axis=0))
    if len(spans) != 8:
        return None
    text = []
    for i, (a, b) in enumerate(spans):
        chip = crop[:, a:b]
        if i in (2, 5):
            dots = _bands((chip >= 111).any(axis=1))
            if not .06 * h <= b - a <= .24 * h or len(dots) != 2 or not .24 * h <= dots[1][0] - dots[0][0] <= .62 * h:
                return None
            text.append(":")
        else:
            digit = _clock_digit(chip, index)
            if digit is None:
                return None
            text.append(digit)
    result = "".join(text)
    return result if _CLOCK.fullmatch(result) else None


def _live_end(frame, gray, index, power=None):
    height, width = gray.shape
    power = power or _power(gray)
    if power is None:
        return None
    _, cy = _center(power)
    start, stop = cy + round(.062 * width), cy + round(.132 * width)
    before, after = round(.036 * width), round(.47 * width)
    along = gray[start:stop, before:after]
    spec = np.uint8(along >= 158)
    _, ty, _, hh = cv2.boundingRect(spec)
    if not .015 * width <= hh <= .039 * width:
        return None
    glyph_ranges = _bands(spec.any(axis=0))
    if len(glyph_ranges) < 15:
        return None
    word_ranges = [glyph_ranges[0][0]] + [b for (_, a), (b, _) in zip(glyph_ranges, glyph_ranges[1:])
                                         if b - a >= .37 * hh]
    if len(word_ranges) < 4:
        return None
    last_start, last_end = word_ranges[-1], glyph_ranges[-1][1]
    reach = max(3, round(.17 * hh))
    area = (before + last_start - reach, start + ty - reach,
            before + last_end + reach, start + ty + hh + reach)
    if _words(frame, area, index) not in {"ends", "end"}:
        return None
    left_finish = glyph_ranges[8][0]
    left_clock = _clock(along[:, :left_finish - 2], index)
    right = min(round(.974 * width), power[2] + round(.010 * width))
    right_clock = _clock(gray[start:stop, round(.77 * width):right], index)
    if not left_clock or not right_clock:
        return None
    return _action(power, "结束直播", "end_live")


def _curved_arrow(mask):
    h, w = mask.shape
    if not .17 <= mask.mean() <= .38:
        return False
    solid_rows = mask[round(.19 * h):round(.39 * h)].sum(axis=1)
    tip_rows = np.flatnonzero(mask[:, :max(2, round(.10 * w))].any(axis=1))
    if (max(solid_rows, default=0) < .62 * w or not len(tip_rows)
            or min(tip_rows) < .13 * h or max(tip_rows) > .47 * h):
        return False
    diagonal = mask[:, :round(.29 * w)]
    top, middle, low = (diagonal[round(a * h):round(b * h)] for a, b in ((0, .16), (.19, .36), (.40, .59)))
    ends = mask[round(.38 * h):round(.82 * h), -round(.13 * w):]
    tail = mask[round(.89 * h):, round(.31 * w):round(.88 * w)]
    return (top.any() and middle.any() and low.any() and ends.any(axis=1).mean() >= .57
            and max(tail.mean(axis=1), default=0) >= .49)


def _restore(gray, widget):
    height, width = gray.shape
    _, wt, _, wb = widget
    left, top = round(.81 * width), height - round(.317 * width)
    right, bottom = round(.978 * width), height - round(.161 * width)
    if top <= wb:
        return None
    upper = gray[round(.105 * width):wt - 4:5, 6:width - 6:5]
    lower = gray[wb + 7:top:5, 6:width - 6:5]
    near = gray[top:bottom:4, :left:4]
    if any((section <= 22).mean() < .992 or (section >= 94).mean() > .00055 for section in (upper, lower, near)):
        return None
    crop = gray[top:bottom, left:right]
    mask = np.uint8(crop >= 177)
    _, ids, stats, _ = cv2.connectedComponentsWithStats(mask)
    arrows = [(x, y, w, h) for i, (x, y, w, h, n) in enumerate(stats[1:], 1)
              if .038 * width <= min(w, h) <= max(w, h) <= .077 * width and .88 <= w / h <= 1.30
              and _curved_arrow(ids[y:y + h, x:x + w] == i)]
    if len(arrows) != 1:
        return None
    x, y, w, h = arrows[0]
    pad = round(.12 * min(w, h))
    box = (left + x - pad, top + y - pad, left + x + w + pad, top + y + h + pad)
    return _action(box, "恢复直播控件", "restore_controls")


def live_producer_surface(frame, font_index):
    """Verify on-camera live controls or a clear camera stage with its controls hidden."""
    height, width = frame.shape[:2]
    if font_index is None or width < 380 or not 1.80 * width <= height <= 2.95 * width:
        return None
    camera = _camera_active(frame)
    header = cv2.cvtColor(frame[:round(.255 * width)], cv2.COLOR_BGR2GRAY)
    power = _power(header)
    if not camera and power is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = _selected_picture(gray)
    widget, close = found if found is not None else (None, None)
    composer = _composer(frame, gray, font_index)
    end = _live_end(frame, gray, font_index, power)
    remove = _action(close, "移除画面贴片", "remove_overlay") if close else None
    if composer and end:
        return LiveProducerSurface(widget, "live", tuple(action for action in (composer, end, remove) if action))
    if camera and widget is not None and composer is None and end is None:
        restore = _restore(gray, widget)
        if restore:
            return LiveProducerSurface(widget, "clear", (restore, remove))
    return None


def live_producer_actions(frame, regions, font_index, *, surface=None):
    """Apply the live witness exclusively to its page or matching picture layer."""
    witness = surface if surface is not None else live_producer_surface(frame, font_index)
    if witness is None:
        return []
    layer, panels = regions.get("layer"), regions.get("regions", ())
    if layer == "page" and not panels:
        return list(witness.actions)
    if layer == "dialog" and len(panels) == 1 and witness.owns_widget_layer(panels[0]):
        return list(witness.actions)
    return []
