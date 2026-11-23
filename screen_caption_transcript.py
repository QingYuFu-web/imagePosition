"""Read an isolated component caption with local font outlines."""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

from screen_component_captions import _bands, _contrast


_MAX_GLYPH_DISTANCE = .225


@lru_cache(maxsize=16)
def _font(path):
    return ImageFont.truetype(str(path), 52)


@lru_cache(maxsize=700)
def _metrics(path, char):
    font = _font(path)
    _, top, _, bottom = font.getbbox(char)
    _, cap_top, _, baseline = font.getbbox("H")
    cap = baseline - cap_top
    return (top - baseline) / cap, (bottom - baseline) / cap


def _layout(parts, ranks, paths, height):
    estimates = []
    for (_, top, _, bottom, _), options in zip(parts, ranks, strict=True):
        best = options[0]
        char, dist = best["text"], best["distance"]
        if dist > .195 or not char.isalpha() or options[1]["distance"] - dist < .017:
            continue
        low, high = _metrics(paths[best["font"]], char)
        if high - low >= .56:
            cap = (bottom - top) / (high - low)
            estimates.append((cap, bottom - cap * high))
    if len(estimates) < 2:
        return None
    scale, baseline = np.median(estimates, axis=0)
    if not .63 * height <= scale <= 1.29 * height:
        return None
    return scale, baseline


def _decimal_dot(left, top, right, bottom, height):
    return (.07 * height <= right - left <= .26 * height and
            .075 * height <= bottom - top <= .25 * height and top >= .72 * height)


def _inline_dash(left, top, right, bottom, height):
    return (.15 * height <= right - left <= .65 * height and
            .045 * height <= bottom - top <= .24 * height and right - left > 2.35 * (bottom - top) and
            .30 * height <= (top + bottom) / 2 <= .67 * height)


def _inline_angle(left, top, right, bottom, height):
    return (.28 * height <= right - left <= .65 * height and
            .32 * height <= bottom - top <= .67 * height and .30 * height <= (top + bottom) / 2 <= .67 * height)


def _shape_parts(mask, height, *, choice=False, join_letters=False):
    bands = _bands(mask.any(axis=0))
    if not (1 if choice else 2) <= len(bands) <= (46 if choice else 20):
        return []
    parts = []
    for i, (left, right) in enumerate(bands):
        _, top, _, h = cv2.boundingRect(mask[:, left:right])
        punctuation = choice and 0 < i < len(bands) - 1 and (
            _decimal_dot(left, top, right, top + h, height) or _inline_dash(left, top, right, top + h, height) or
            _inline_angle(left, top, right, top + h, height))
        minimum = .46 if choice else .52
        narrow = .064 if join_letters and (h >= .69 * height or i == len(bands) - 1) else .092
        if not punctuation and (not max(6, minimum * height) <= h <= height or
                                not narrow * height <= right - left <= 1.22 * height):
            return []
        parts.append((left, top, right, top + h, mask[top:top + h, left:right]))
    return parts


def _cjk_parts(mask, height, *, join_radicals=False):
    spans = _bands(mask.any(axis=0))
    if not 2 <= len(spans) <= 15 or not 1.52 * height <= mask.shape[1] <= 9.7 * height:
        return []
    # Full ideographs retain comparable widths and advances even where separate
    # radicals leave blank columns within a character.
    broad = [right - left for left, right in spans if
             right - left >= .64 * height and cv2.boundingRect(mask[:, left:right])[3] >= .83 * height]
    fragments = join_radicals and len(spans) >= 3 and bool(broad)
    if (len(broad) < 2 or sum(broad) < .48 * mask.shape[1]) and not fragments:
        return []

    @lru_cache(maxsize=None)
    def partitions(start):
        if start == len(spans):
            return ((),)
        left = spans[start][0]
        result = []
        for i in range(start, len(spans)):
            right = spans[i][1]
            if right - left > 1.16 * height or (i > start and spans[i][0] - spans[i - 1][1] > .17 * height):
                break
            _, top, _, h = cv2.boundingRect(mask[:, left:right])
            if right - left >= .59 * height and h >= .75 * height:
                part = (left, top, right, top + h, mask[top:top + h, left:right])
                result.extend((part, *tail) for tail in partitions(i + 1))
        return result

    choices = []
    for parts in partitions(0):
        centers = [(part[0] + part[2]) / 2 for part in parts]
        advances = np.diff(centers) / height
        if len(advances) and .76 <= min(advances) <= max(advances) <= 1.45:
            sizes = [(part[2] - part[0]) / height for part in parts]
            choices.append((np.std(advances) + np.std(sizes), parts))
    return min(choices, key=lambda option: option[0])[1] if choices else []


def _numeric_tail(rankings, parts, space):
    """Check an observed run of digits before resolving a look-alike letter."""
    splits = [i for i, (a, b) in enumerate(zip(parts[:-1], parts[1:], strict=True), 1)
              if b[0] - a[2] >= space and rankings[i][0]["text"] != "." and rankings[i - 1][0]["text"] != "."]
    if len(splits) > 1:
        return {}
    start = splits[0] if splits else 0
    if start and (start > 5 or not all(row[0]["text"].isupper() for row in rankings[:start])):
        return {}
    tail = rankings[start:]
    if not 2 <= len(tail) <= 7 or not any(chunk[0]["text"].isdigit() for chunk in tail):
        return {}
    values = {}
    for i, ranking in enumerate(tail, start):
        best = ranking[0]
        if best["text"] == ".":
            continue
        digits = [option for option in ranking if option["text"].isdigit()]
        if (not digits or digits[0]["distance"] > .20 or digits[0]["distance"] - best["distance"] > .058 or
                (len(digits) > 1 and digits[1]["distance"] - digits[0]["distance"] < .024)):
            return {}
        values[i] = digits[0]
    return values if len(values) >= 2 else {}


def _has_t_crossbar(glyph):
    h, w = glyph.shape
    if h < 12 or not .26 * h <= w <= .76 * h:
        return False
    rows = glyph.mean(axis=1)
    top = rows[:round(.18 * h)]
    stem = rows[round(.46 * h):round(.64 * h)]
    cross = rows[round(.20 * h):round(.43 * h)]
    return (max(np.median(top), np.median(stem)) <= .63 and
            (cross >= .81).sum() >= max(2, round(.088 * h)))


def _ranked_glyph(index, cache, bank, glyph):
    key = ("caption_rank", bank, glyph.shape, glyph.tobytes())
    if key not in cache:
        cache[key] = index.rank(glyph, alphabet=bank, limit=5)
    return cache[key]


def _separate_touching_letters(part, index, cache, bank):
    left, top, _, _, glyph = part
    height, width = glyph.shape
    if height < 12 or not .49 * height <= width <= 1.15 * height:
        return None
    whole = _ranked_glyph(index, cache, bank, glyph)
    if not whole or whole[0]["distance"] <= .225:
        return None
    choices = {}
    columns = glyph.sum(axis=0)
    for split in range(max(3, round(.23 * width)), min(width - 2, round(.87 * width))):
        if max(columns[split - 1], columns[split]) > .36 * height:
            continue
        halves, letters = [], []
        for start, stop in ((0, split), (split, width)):
            x, y, w, h = cv2.boundingRect(glyph[:, start:stop])
            if not .61 * height <= h <= height or w < max(2, .067 * height):
                break
            mask = glyph[y:y + h, start + x:start + x + w]
            options = _ranked_glyph(index, cache, bank, mask)
            if (len(options) < 2 or not "a" <= options[0]["text"] <= "z" or options[0]["distance"] > .175
                    or options[1]["distance"] - options[0]["distance"] < .027):
                break
            letters.append(options[0])
            halves.append((left + start + x, top + y, left + start + x + w, top + y + h, mask))
        if len(letters) != 2:
            continue
        name = "".join(letter["text"] for letter in letters)
        score = sum(letter["distance"] for letter in letters)
        if name not in choices or score < choices[name][0]:
            choices[name] = score, halves
    return next(iter(choices.values()))[1] if len(choices) == 1 else None


def _read_cut(parts, height, index, paths, cache, alphabet, *, choice=False, numeric=False, sentence_case=False,
              join_letters=False):
    rankings = []
    bank = "choice_latin" if choice and alphabet == "latin" and not numeric else alphabet
    if join_letters and alphabet == "latin":
        parts = [glyph for part in parts for glyph in (_separate_touching_letters(part, index, cache, bank) or (part,))]
    for left, top, right, bottom, glyph in parts:
        if choice and alphabet == "latin" and _decimal_dot(left, top, right, bottom, height):
            rankings.append([{"text": ".", "distance": 0}, {"text": "·", "distance": .05}])
            continue
        ranks = _ranked_glyph(index, cache, bank, glyph)
        # Layout changes scores, and a numeric override uses one of these same distances.
        if len(ranks) < 2 or all(option["distance"] > _MAX_GLYPH_DISTANCE for option in ranks):
            return None
        rankings.append(ranks)
    layout = _layout(parts, rankings, paths, height) if alphabet == "latin" else None
    gaps = [b[0] - a[2] for a, b in zip(parts[:-1], parts[1:], strict=True)]
    space = max(.224 * height, (np.median(gaps) if gaps else 0) + .128 * height)
    values = _numeric_tail(rankings, parts, space) if numeric and alphabet == "latin" else {}
    letters, shape_errors, margins = [], [], []
    for i, ((left, top, right, bottom, glyph), options) in enumerate(zip(parts, rankings, strict=True)):
        adjusted = []
        crossbar = choice and {"L", "t"} <= {candidate["text"] for candidate in options} and _has_t_crossbar(glyph)
        for candidate in options:
            char, distance = candidate["text"], candidate["distance"]
            penalty = 0
            if layout is not None and "font" in candidate:
                cap, baseline = layout
                a, b = _metrics(paths[candidate["font"]], char)
                penalty = .35 * (abs(top - baseline - a * cap) + abs(bottom - baseline - b * cap)) / height
            if choice and char in "PR":
                foot = glyph[round(.63 * glyph.shape[0]):, round(.48 * glyph.shape[1]):].mean()
                if (char == "P" and foot >= .28) or (char == "R" and foot <= .092):
                    penalty += .075
            if crossbar and char == "L":
                penalty += .10
            elif crossbar and char == "t":
                penalty = min(.019, penalty)
            adjusted.append((distance + penalty, distance, char))
        adjusted.sort()
        score, distance, char = adjusted[0]
        margin = adjusted[1][0] - score
        if i in values:
            char, distance = values[i]["text"], values[i]["distance"]
            margin = min((option["distance"] - distance for option in options
                          if option["text"].isdigit() and option["text"] != char), default=.024)
            score = distance
        required_margin = .027 if {char, adjusted[1][2]} == {"R", "P"} else .010
        stem = (sentence_case and alphabet == "latin" and {char, adjusted[1][2]} == {"I", "l"}
                and len(adjusted) >= 3 and adjusted[2][0] - adjusted[1][0] >= .034
                and right - left <= .19 * height and max(distance, adjusted[1][1]) <= .105 and margin < .026)
        if (not (char.isalnum() or char in ("!?-&.()/<>:" if choice else "!?-&"))
                or distance > _MAX_GLYPH_DISTANCE or score > .255
                or (margin < required_margin and not stem)):
            return None
        if (choice and char in {"I", "l"} and not stem and
                any({char, option["text"]} == {"I", "l"} and abs(option["distance"] - distance) < .022
                    for option in options)):
            return None
        shape_errors.append(distance)
        margins.append(adjusted[2][0] - adjusted[1][0] if stem else margin)
        if letters and alphabet == "latin" and left - previous >= space and (not choice or char not in ")>:"):
            letters.append(" ")
        letters.append("|" if stem else char)
        previous = right
    text = "".join(letters)
    if "|" in text:
        words = (text.rstrip(":") if join_letters else text).split(" ")
        if len(words) < 2 or any(
                not word[0].isascii() or not word[0].isalpha() or word.endswith("|")
                or any(not ("a" <= c <= "z") for c in word[1:] if c != "|")
                for word in words if "|" in word):
            return None
        text = text.replace("|", "l")
    if np.median(shape_errors) > .175 or not any(char.isalnum() if choice else char.isalpha() for char in text):
        return None
    if choice and any(not (0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit())
                      for i, char in enumerate(text) if char == "."):
        return None
    if alphabet == "all" and not all("\u3400" <= char <= "\u9fff" for char in text):
        return None
    return text, max(shape_errors), min(margins)


def _separate_overlapping_ink(parts, height, *, choice=False):
    """Separate two glyphs sharing columns when their visible strokes are disjoint."""
    for i, (left, top, right, _, ink) in enumerate(parts):
        if right - left < (.54 if choice else 1.) * height:
            continue
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        if len(stats) != 3:
            continue
        duo = [(left + x, top + y, left + x + w, top + y + h, np.uint8(labels[y:y + h, x:x + w] == part))
               for part, (x, y, w, h, _) in enumerate(stats[1:], 1)]
        duo.sort(key=lambda item: item[0])
        dot = choice and _decimal_dot(*duo[1][:4], height) and .53 * height <= duo[0][3] - duo[0][1]
        if ((dot or min(stats[1:, 3]) >= .54 * height)
                and (.24 if choice else .29) * height <= duo[1][0] - duo[0][0] <= .87 * height
                and abs(duo[0][3] - duo[1][3]) <= .24 * height):
            yield parts[:i] + duo + parts[i + 1:]


def read_caption(frame, area, index, *, cache, choice=False, numeric=False, join_radicals=False, sentence_case=False,
                 join_letters=False):
    """Check that one full transcript agrees across two different contrast cuts."""
    left, top, right, bottom = area
    distances, _ = _contrast(frame[top:bottom, left:right])
    peak = float(np.percentile(distances, 98))
    if peak < 40:
        return None
    fonts = index.fonts if index.fonts is not None else (index.choice_latin_sources if choice else index.latin_sources)
    paths = {Path(path).name: path for path in fonts}
    reports = defaultdict(dict)
    levels = (.30, .41, .53, .64, .75, .83) if choice else (.30, .41, .53, .64)
    for level in levels:
        ink = np.uint8(distances >= level * peak)
        dx, dy, w, h = cv2.boundingRect(ink)
        if (not 11 <= h <= .074 * frame.shape[1] or
                not (.14 if choice else .83) * h <= w <= (27 if choice else 13.8) * h):
            continue
        cut = ink[dy:dy + h, dx:dx + w]
        cjk = _cjk_parts(cut, h, join_radicals=join_radicals)
        result = _read_cut(cjk, h, index, paths, cache, "all", choice=choice) if cjk else None
        if cjk and result is None and not (numeric or choice):
            continue
        parts = _shape_parts(cut, h, choice=choice, join_letters=join_letters) if result is None else []
        if parts:
            splits = list(_separate_overlapping_ink(parts, h, choice=choice))
            has_dot = lambda glyphs: any(_decimal_dot(*part[:4], h) for part in glyphs)
            decimal_splits = [split for split in splits if numeric and not has_dot(parts) and has_dot(split)]
            if not decimal_splits:
                result = _read_cut(parts, h, index, paths, cache, "latin", choice=choice, numeric=numeric,
                                   sentence_case=sentence_case, join_letters=join_letters)
            if result is None:
                alternatives = [reading for split in decimal_splits or splits
                                if (reading := _read_cut(split, h, index, paths, cache, "latin",
                                                        choice=choice, numeric=numeric, sentence_case=sentence_case,
                                                        join_letters=join_letters)) is not None]
                if len(alternatives) == 1:
                    result = alternatives[0]
        if result is None:
            continue
        text, error, margin = result
        box = [left + dx, top + dy, left + dx + w, top + dy + h]
        reports[text][(cut.shape, cut.tobytes())] = (error, margin, box)
    confirmed = [(text, samples) for text, samples in reports.items() if len(samples) >= 2]
    if len(confirmed) != 1:
        return None
    text, samples = confirmed[0]
    error, margin, box = min(samples.values())
    return {"text": text, "line_box": box, "recognition": "component", "support": len(samples),
            "max_distance": error, "min_margin": round(margin, 5)}
