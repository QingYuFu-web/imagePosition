"""Read an artwork-overlaid title from aligned local font silhouettes."""

from collections import Counter

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts
from screen_wide_panel_checks import _letter_rasters


def _caption_samples(signal):
    base, bright = np.percentile(signal, (50, 99))
    if bright - base < 54:
        return []
    rows = []
    for cut in (.44, .49, .53, .57, .595, .62, .65, .675, .70):
        ink = np.uint8(signal >= base + cut * (bright - base))
        x, y, w, h = cv2.boundingRect(ink)
        if (not 12 <= h <= .79 * len(signal) or min(y, len(signal) - y - h) < 1
                or min(x, signal.shape[1] - x - w) < 3
                or abs(2 * x + w - signal.shape[1]) > .16 * w):
            continue
        line = ink[y:y + h, x:x + w]
        parts = _cjk_parts(line, h, join_radicals=True)
        if 3 <= len(parts) <= 7:
            rows.append([(x + a, y + b, x + c, y + d, g) for a, b, c, d, g in parts])
    return rows


def _centered_unit(array):
    centered = array - array.mean(axis=1, keepdims=True)
    mass = np.sqrt(np.square(centered).sum(axis=1, keepdims=True))
    return centered / np.maximum(1e-7, mass)


def _matching_outline(planes, box, candidates, paths):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    if top < 2 or left < 2 or bottom + 2 > planes.shape[1] or right + 2 > planes.shape[2]:
        return None
    queries = planes[:, top - 2:bottom + 2, left - 2:right + 2]
    full = _centered_unit(queries.reshape(len(planes), -1))
    core = _centered_unit(queries[:, 2:-2, 2:-2].reshape(len(planes), -1))
    votes = []
    for char in sorted(candidates):
        bank = _letter_rasters(paths, char, w, h)
        if not bank.size:
            continue
        whole = _centered_unit(bank.reshape(len(bank), -1))
        inside = _centered_unit(bank[:, 2:-2, 2:-2].reshape(len(bank), -1))
        scores = np.concatenate((np.einsum('ij,kj->ki', whole, full).max(axis=1),
                                 np.einsum('ij,kj->ki', inside, core).max(axis=1)))
        votes.append((char, scores))
    if len(votes) < 2:
        return None
    chars, values = zip(*votes, strict=True)
    scores = np.asarray(values, dtype=np.float32)
    order = np.argsort(scores, axis=0)
    best = order[-1]
    peaks = scores[best, np.arange(scores.shape[1])]
    gaps = peaks - scores[order[-2], np.arange(scores.shape[1])]
    if (np.any(best != best[0]) or min(peaks[:2]) < .72 or min(peaks[2:]) < .64
            or min(gaps[:2]) < .006 or min(gaps[2:]) < .012):
        return None
    return chars[best[0]]


def artwork_title(patch, index):
    """Confirm a word on both color treatments without guessing weak glyphs."""
    if not 18 <= len(patch) <= 36 or not 3.0 <= patch.shape[1] / len(patch) <= 6.2:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pale = patch.min(axis=2).astype(np.float32)
    suppressed = pale - .70 * np.ptp(patch, axis=2)
    planes = np.stack((gray, suppressed))
    masks = [_caption_samples(plane) for plane in planes]
    if any(not samples for samples in masks):
        return None
    shapes = Counter(len(line) for lines in masks for line in lines)
    n, count = shapes.most_common(1)[0]
    if count < 5 or count <= sum(shapes.values()) * .66:
        return None
    aligned = [[line for line in samples if len(line) == n] for samples in masks]
    if any(not samples for samples in aligned):
        return None
    flat = [sample for samples in aligned for sample in samples]
    positions = np.array([[glyph[:4] for glyph in sample] for sample in flat])
    if np.ptp(positions, axis=0).max() > 3:
        return None
    # Small recaptures can leave only one pixel outside the bright outline.
    # Extend that background edge before scoring the two-pixel template border.
    planes = np.pad(planes, ((0, 0), (2, 2), (2, 2)), mode='edge')
    boxes = (2 + np.rint(np.median(positions, axis=0))).astype(int).tolist()
    paths = tuple(index.fonts if index.fonts is not None else index.cjk_sources)
    title = []
    cache = {}
    for i, box in enumerate(boxes):
        candidates = Counter()
        seen = set()
        for sample in flat:
            raster = sample[i][4]
            key = raster.shape, raster.tobytes()
            if key in seen:
                continue
            seen.add(key)
            if key not in cache:
                cache[key] = index.rank(raster, alphabet='cjk', limit=4)
            candidates.update(row['text'] for row in cache[key])
        verified = {name for name, count in candidates.items() if count >= 2}
        char = _matching_outline(planes, box, verified, paths)
        if char is None:
            return None
        title.append(char)
    return ''.join(title)
