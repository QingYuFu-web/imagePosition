"""Read isolated printed glyphs by comparing rasterized system-font outlines."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
import os
import string
from threading import Lock

import cv2
import numpy as np
from PIL import Image, ImageFont


_SIZE = 32
_COARSE = 12
_PARALLEL_BANK_SIZE = 32768
_POOL_LOCK = Lock()
_POOL = None
_KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1, dtype=np.uint8)
_LATIN = string.ascii_letters + string.digits + "-+?!:./%()&"
_CJK_FONTS = ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "msjh.ttc", "msjhl.ttc")
_LATIN_FONTS = ("arial.ttf", "arialbd.ttf", "segoeui.ttf", "segoeuib.ttf", "simsun.ttc")
_CHOICE_LATIN_FONTS = (*_LATIN_FONTS, "Deng.ttf", "Dengb.ttf")


@lru_cache(maxsize=1)
def _cjk_charset():
    characters = set()
    repertoires = (("gb2312", range(0xA1, 0xF8), range(0xA1, 0xFF)),
                   ("big5", range(0xA1, 0xFA), (*range(0x40, 0x7F), *range(0xA1, 0xFF))))
    for codec, leaders, followers in repertoires:
        for a, b in product(leaders, followers):
            char = bytes((a, b)).decode(codec, errors="ignore")
            if len(char) == 1 and 0x3400 <= ord(char) <= 0x9FFF:
                characters.add(char)
    return "".join(sorted(characters))


def _normal(mask):
    left, top, width, height = cv2.boundingRect(mask)
    if width == 0 or height == 0:
        return None
    cropped = mask[top:top + height, left:left + width]
    method = cv2.INTER_AREA if max(width, height) > _SIZE else cv2.INTER_LINEAR
    raster = cv2.resize(cropped, (_SIZE, _SIZE), interpolation=method)
    ratio = np.log(width / height)
    return raster, ratio


def _mask_pixels(mask):
    # Export through Pillow's byte-backed image interface. Iterating over an
    # ImagingCore converts each pixel into a Python integer.
    return np.asarray(Image.Image()._new(mask), dtype=np.uint8)


def _font_raster(font, char):
    mask = font.getmask(char, mode="L")
    width, height = mask.size
    if width < 1 or height < 1:
        return None
    pixels = _mask_pixels(mask)
    return np.uint8(pixels >= 128) * 255


@dataclass(frozen=True)
class _Glyphs:
    rasters: np.ndarray
    coarse: np.ndarray
    ratios: np.ndarray
    chars: tuple[str, ...]
    fonts: tuple[str, ...]


def _rasterize_bank(paths, characters):
    rasters, coarse, ratios, names, fonts = [], [], [], [], []
    for path in paths:
        font = ImageFont.truetype(str(path), 52)
        missing = tuple(raster for char in ("\u0378", "\u0380", "\U0010ffff")
                        if (raster := _font_raster(font, char)) is not None)
        for char in characters:
            mask = _font_raster(font, char)
            if mask is None or any(other.shape == mask.shape and np.array_equal(other, mask) for other in missing):
                continue
            normalized = _normal(mask)
            if normalized is None:
                continue
            raster, ratio = normalized
            rasters.append(raster)
            coarse.append(cv2.resize(raster, (_COARSE, _COARSE), interpolation=cv2.INTER_AREA).ravel())
            ratios.append(ratio)
            names.append(char)
            fonts.append(path.name)
    if not rasters:
        return None
    return _Glyphs(np.stack(rasters), np.stack(coarse), np.float32(ratios), tuple(names), tuple(fonts))


@lru_cache(maxsize=8)
def _build(paths, characters):
    if len(characters) < 1000:
        return _rasterize_bank(paths, characters)
    from screen_glyph_bank_cache import load_glyph_bank

    payload = load_glyph_bank(paths, characters, raster_size=_SIZE, coarse_size=_COARSE,
                              build=lambda: _rasterize_bank(paths, characters))
    return _Glyphs(*payload) if payload is not None else None


def _pack_rows(mask):
    # One word per 32-pixel glyph row. Little-endian order keeps adjacent
    # pixels one bit apart on every host.
    packed = np.packbits(mask, axis=-1, bitorder="little")
    return packed.view("<u4").reshape(mask.shape[:-1]).astype(np.uint32, copy=False)


def _count_pixels(rows):
    return _POPCOUNT[rows.view(np.uint8)].sum(axis=1, dtype=np.int32)


def _shape_distances(query, query_rows, allowed_q, binary_area, fonts):
    records = _pack_rows(fonts >= .5)
    allowed_f = records | (records << 1) | (records >> 1)
    allowed_f[:, 1:] |= records[:, :-1]
    allowed_f[:, :-1] |= records[:, 1:]
    area1 = query.sum() + fonts.sum(axis=(1, 2))
    area2 = binary_area + _count_pixels(records)
    difference = np.subtract(fonts, query)
    np.abs(difference, out=difference)
    difference = difference.sum(axis=(1, 2)) / np.maximum(1, area1)
    outside_pixels = (records & ~allowed_q) | (query_rows & ~allowed_f)
    outside = _count_pixels(outside_pixels) / np.maximum(1, area2)
    return .62 * difference + .38 * outside


@lru_cache(maxsize=1)
def _coarse_workers():
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return min(4, max(1, count))


def _coarse_pool(workers):
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="screen-glyph")
        return _POOL


def _coarse_l1(query, templates, count):
    return cv2.batchDistance(query, templates, cv2.CV_32S, normType=cv2.NORM_L1,
                             K=min(count, len(templates)))


def _coarse_neighbors(query, templates, *, parallel=True):
    """Return the same ordered neighbors from contiguous OpenCV searches."""
    count = min(112, len(templates))
    workers = _coarse_workers() if parallel and len(templates) >= _PARALLEL_BANK_SIZE else 1
    if workers < 2:
        return _coarse_l1(query, templates, count)
    step = (len(templates) + workers - 1) // workers
    pool = _coarse_pool(workers)
    jobs = [(start, pool.submit(_coarse_l1, query, templates[start:start + step], count))
            for start in range(0, len(templates), step)]
    results = [(start, *job.result()) for start, job in jobs]
    distances = np.concatenate([values[0] for _, values, _ in results])
    indices = np.concatenate([indices[0] + start for start, _, indices in results])
    # Every global neighbor is among its shard's top K. Equal distances retain
    # the database order used by OpenCV's single scan.
    order = np.lexsort((indices, distances))[:count]
    return distances[order][None, :], indices[order][None, :]


def _shortlist(index, raster, ratio):
    coarse = cv2.resize(raster, (_COARSE, _COARSE), interpolation=cv2.INTER_AREA).reshape(1, -1)
    _, indexes = _coarse_neighbors(coarse, index.coarse)
    selected = indexes[0]
    query = raster.astype(np.float32) / 255
    fonts = index.rasters[selected].astype(np.float32) / 255
    qmask = np.uint8(query >= .5)
    query_rows = _pack_rows(qmask)
    allowed_q = _pack_rows(cv2.dilate(qmask, _KERNEL))
    binary_area = int(qmask.sum())
    distances = _shape_distances(query, query_rows, allowed_q, binary_area, fonts)
    mean = query.mean()
    if mean < .56:
        # Screen font weight and antialiasing vary with raster size. Only
        # test a stroke change when the glyph actually has a denser or sparser
        # silhouette, and keep the original template in the competition.
        density = fonts.mean(axis=(1, 2)) - mean
        for eligible, adjust in ((density >= .050, cv2.erode), (density <= -.050, cv2.dilate)):
            if not np.any(eligible):
                continue
            similar = fonts[eligible]
            transformed = adjust(np.moveaxis(similar, 0, -1), _KERNEL)
            adjusted = np.moveaxis(transformed.reshape(_SIZE, _SIZE, len(similar)), -1, 0)
            variant = .64 * adjusted + .36 * similar
            distances[eligible] = np.minimum(
                distances[eligible], _shape_distances(query, query_rows, allowed_q, binary_area, variant) + .021)
    aspect = np.minimum(.9, np.abs(index.ratios[selected] - ratio))
    distances += .095 * aspect
    return ((index.chars[pos], float(score), index.fonts[pos]) for pos, score in zip(selected, distances, strict=True))


class FontGlyphIndex:
    """Verify one swept-out foreground glyph using local fonts and image geometry."""

    def __init__(self, *, fonts=None, charset=None, max_distance=.19, min_margin=.018):
        folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        if isinstance(fonts, (str, os.PathLike)):
            fonts = (fonts,)
        self.fonts = tuple(Path(path).resolve() for path in fonts) if fonts is not None else None
        self.latin_sources = tuple(path for name in _LATIN_FONTS if (path := folder / name).is_file())
        self.choice_latin_sources = tuple(path for name in _CHOICE_LATIN_FONTS if (path := folder / name).is_file())
        self.cjk_sources = tuple(path for name in _CJK_FONTS if (path := folder / name).is_file())
        self.charset = "".join(dict.fromkeys(charset)) if charset is not None else None
        self.max_distance = max_distance
        self.min_margin = min_margin
        self._rank_cached = lru_cache(maxsize=2048)(self._rank_raster)

    def _alphabets(self, alphabet):
        if alphabet not in {"all", "latin", "cjk", "code", "choice_latin"}:
            raise ValueError(f"Unknown glyph alphabet {alphabet!r}")
        entries = []
        if alphabet in {"latin", "all", "code", "choice_latin"}:
            charset = _LATIN if self.charset is None else "".join(c for c in self.charset if ord(c) < 128)
            if alphabet == "code":
                charset = "".join(c for c in charset if c in string.ascii_uppercase + string.digits + "-")
            if alphabet == "choice_latin":
                entries.append((self.choice_latin_sources, charset + ("<>" if self.charset is None else "")))
            else:
                entries.append((self.latin_sources, charset))
        if alphabet in {"cjk", "all"}:
            charset = _cjk_charset() if self.charset is None else "".join(c for c in self.charset if 0x3400 <= ord(c) <= 0x9FFF)
            entries.append((self.cjk_sources, charset))
        for paths, chars in entries:
            if chars and (paths := self.fonts if self.fonts is not None else paths):
                bank = _build(paths, chars)
                if bank is not None:
                    yield bank

    def prepare(self, alphabet="all"):
        """Populate the lazy in-process index and return the template count."""
        return sum(len(bank.chars) for bank in self._alphabets(alphabet))

    def rank(self, mask, *, alphabet="all", limit=5):
        """Return exact-character alternatives and distances for a diagnostic crop."""
        image = np.asarray(mask)
        if image.ndim != 2 or not image.size:
            raise ValueError("The glyph mask must be a nonempty 2D image with nonzero foreground")
        normalized = _normal(np.uint8(image > 0) * 255)
        if normalized is None:
            return []
        raster, ratio = normalized
        choices = self._rank_cached(alphabet, raster.tobytes(), float(ratio))
        return [{"text": char, "distance": distance, "font": font} for char, distance, font in choices[:limit]]

    def _rank_raster(self, alphabet, raster_bytes, ratio):
        raster = np.frombuffer(raster_bytes, np.uint8).reshape(_SIZE, _SIZE)
        best = {}
        for bank in self._alphabets(alphabet):
            for char, distance, font in _shortlist(bank, raster, ratio):
                if char not in best or distance < best[char][0]:
                    best[char] = distance, font
        return tuple((char, round(value[0], 5), value[1])
                     for char, value in sorted(best.items(), key=lambda item: (item[1][0], item[0])))

    def recognize(self, mask, *, alphabet="all"):
        """Abstain when the closest shape is weak or its identity is ambiguous."""
        choices = self.rank(mask, alphabet=alphabet, limit=2)
        if not choices:
            return None
        best = choices[0]
        margin = choices[1]["distance"] - best["distance"] if len(choices) > 1 else 1.0
        if best["distance"] > self.max_distance or margin < self.min_margin:
            return None
        return {**best, "margin": round(margin, 5)}
