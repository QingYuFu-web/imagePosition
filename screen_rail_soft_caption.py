"""Read isolated failed captions using the font shared by the visible rail."""

from collections import Counter, defaultdict
from ctypes import create_unicode_buffer
from functools import lru_cache
import os

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts
from screen_component_captions import _contrast
from screen_wide_panel_checks import _letter_rasters


_SIMPLIFIED = 0x02000000
_TRADITIONAL = 0x04000000

if os.name == "nt":
    from ctypes import windll

    _map_string = windll.kernel32.LCMapStringEx
else:
    _map_string = None


@lru_cache(maxsize=2048)
def _variant(char, flag=_SIMPLIFIED):
    if _map_string is None:
        return char
    value = create_unicode_buffer(1)
    count = _map_string("zh-CN", flag, char, 1, value, 1, None, None, 0)
    return value[0] if count == 1 else char


def _parts(signal):
    peak = np.percentile(signal, 98)
    if peak < 44:
        return None
    ink = np.uint8(signal >= .26 * peak)
    x, y, width, height = cv2.boundingRect(ink)
    if not 14 <= height <= 32:
        return None
    letters = _cjk_parts(ink[y:y + height, x:x + width], height, join_radicals=True)
    if not 2 <= len(letters) <= 4:
        return None
    return x, width, letters


def shared_caption_font(peers, index):
    """Start from glyphs already read reliably inside several distinct cells."""
    witnesses, cells = Counter(), defaultdict(set)
    for row, (tile, label) in enumerate(peers):
        signal, _ = _contrast(tile)
        measured = _parts(signal)
        if measured is None or len(measured[2]) != len(label):
            continue
        for (_, _, _, _, mask), char in zip(measured[2], label, strict=True):
            choices = index.rank(mask, alphabet="cjk", limit=2)
            if len(choices) < 2:
                continue
            best, other = choices
            score = best["distance"]
            if best["text"] == char and score <= .17 and other["distance"] - score >= .007:
                witnesses[best["font"]] += 1
                cells[best["font"]].add(row)
    fonts = [name for name, count in witnesses.items()
             if count >= 4 and count >= .75 * witnesses.total() and len(cells[name]) >= 3]
    paths = index.fonts if index.fonts is not None else index.cjk_sources
    return tuple(path for path in paths if path.name in fonts)


def _blur(stack):
    return np.moveaxis(cv2.GaussianBlur(np.moveaxis(stack, 0, -1), (0, 0), .66), -1, 0)


def _error(target, bank):
    mass = np.square(target).sum()
    err = np.square(target - bank).sum(axis=(1, 2))
    return err / (.5 * (mass + np.square(bank).sum(axis=(1, 2))))


def _scores(signal, index, paths, simplified, cache):
    peak = np.percentile(signal[signal >= .1 * signal.max()], 96)
    options = {candidate["text"] for cut in (.19, .23, .30, .39, .51)
               for candidate in index.rank(signal >= cut * peak, alphabet="cjk", limit=5)}
    x, y, width, height = cv2.boundingRect(np.uint8(signal >= .22 * peak))
    if not .62 <= width / height <= 1.24:
        return []
    alpha = np.minimum(1., signal[y:y + height, x:x + width] / peak).astype(np.float32)
    target = np.pad(cv2.resize(alpha, (24, 24), interpolation=cv2.INTER_LINEAR), 2)
    low = cv2.GaussianBlur(target, (0, 0), .66)
    if simplified:
        options.update(_variant(char) for char in tuple(options))
    choices = {}
    for char in options:
        if char not in cache:
            raster = _letter_rasters(paths, char, 24, 24)
            cache[char] = (raster, _blur(raster)) if len(raster) else None
        if (templates := cache[char]) is None:
            continue
        original, softened = templates
        score = float((.62 * _error(target, original) + .38 * _error(low, softened)).min())
        key = _variant(char) if simplified else char
        choices[key] = min(score, choices.get(key, np.inf))
    return sorted((score, char) for char, score in choices.items())


def _run(signal, index, paths, simplified, cache):
    measured = _parts(signal)
    if measured is None:
        return None
    x, width, parts = measured
    edges = [x + (first[2] + second[0]) // 2 for first, second in zip(parts[:-1], parts[1:], strict=True)]
    bounds = [max(0, x - 1), *edges, min(signal.shape[1], x + width + 1)]
    votes = []
    for left, right in zip(bounds[:-1], bounds[1:], strict=True):
        ranks = _scores(signal[:, left:right], index, paths, simplified, cache)
        if len(ranks) < 2:
            return None
        score, label = ranks[0]
        difference = ranks[1][0] - score
        if score > .243 or difference < .009:
            return None
        votes.append((label, difference))
    return votes


def read_soft_rail_caption(tile, index, paths, peers):
    """Check whole glyphs against the locally observed font in two color views."""
    simplified = any(_variant(char) == char and _variant(char, _TRADITIONAL) != char
                     for label in peers for char in label)
    distance, paper = _contrast(tile)
    centered = tile.astype(np.float32) - paper
    maxima = np.percentile(centered, 98, axis=(0, 1))
    strongest, weakest = int(maxima.argmax()), int(maxima.argmin())
    if maxima[strongest] < 1.16 * maxima[weakest]:
        return None
    independent = np.maximum(centered[..., strongest], 0) - .45 * np.maximum(centered[..., weakest], 0)
    cache = {}
    first = _run(distance, index, paths, simplified, cache)
    if first is None:
        return None
    second = _run(independent, index, paths, simplified, cache)
    if second is None:
        return None
    name = "".join(char for char, _ in first)
    if name in peers or name != "".join(char for char, _ in second):
        return None
    if any((a[1] + b[1]) / 2 < .022 for a, b in zip(first, second, strict=True)):
        return None
    return name
