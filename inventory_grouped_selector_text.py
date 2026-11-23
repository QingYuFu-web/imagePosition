"""Read separately bounded names and values in a verified selector group."""

import unicodedata

import cv2
import numpy as np


def _key(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _crop(frame, box):
    return frame[max(0, box.top - 1):box.bottom + 1, max(0, box.left - 1):box.right + 1]


def _deviation(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.int16)
    ground = np.median(gray)
    return gray - ground if ground <= 127 else ground - gray


def _mnemonic_mask(ink, peer):
    _, components, stats, _ = cv2.connectedComponentsWithStats(np.uint8(ink), connectivity=8)
    height = ink.shape[0]
    detached = np.zeros_like(ink)
    for index, (x, y, w, h, area) in enumerate(stats[1:], start=1):
        if not (1 <= h <= max(1, round(.12 * height)) and max(3, .19 * height) <= w <= .74 * height
                and w >= 3 * h and y >= .61 * height and area >= .78 * w * h):
            continue
        mark = components[y:y + h, x:x + w] == index
        if np.count_nonzero(mark & peer[y:y + h, x:x + w]) <= .15 * area:
            detached[y:y + h, x:x + w] |= mark
    return detached


def _recaptured_patches(patches):
    first, second = map(_deviation, patches)
    difference = first - second
    masks, baseline = [], []
    for sign in (1, -1):
        core = sign * difference >= 60
        yy, xx = core.nonzero()
        if not yy.size:
            return None
        width, depth = np.ptp(xx) + 1, np.ptp(yy) + 1
        if (depth > 2 or not max(3, 2.4 * depth) <= width <= .74 * core.shape[0]
                or yy.min() < .58 * core.shape[0] or core.sum() < .55 * width * depth):
            return None
        baseline.append(np.median(yy))
        halo = cv2.dilate(np.uint8(core), np.ones((3, 3), np.uint8)).astype(bool)
        masks.append(halo & (sign * difference >= 14))
    if abs(baseline[0] - baseline[1]) > 1:
        return None
    recovered = []
    for patch, peer, mask in zip(patches, patches[::-1], masks, strict=True):
        restored = patch.copy()
        restored[mask] = peer[mask]
        recovered.append(restored)
    old = first >= 52, second >= 52
    clean = tuple(_deviation(patch) >= 52 for patch in recovered)
    before = (old[0] & old[1]).sum() / max(1, (old[0] | old[1]).sum())
    after = (clean[0] & clean[1]).sum() / max(1, (clean[0] | clean[1]).sum())
    if after < max(.91, before + .02):
        return None
    return [(patch, True, True) for patch in recovered]


def _peer_patches(frame, a, b, delta):
    left, right = max(0, min(a.left, b.left) - 1), min(frame.shape[1], max(a.right, b.right) + 1)
    for offset in (0, -1, 1):
        shift = delta + offset
        top, bottom = min(a.top, b.top - shift) - 1, max(a.bottom, b.bottom - shift) + 1
        if top < 0 or bottom + shift >= frame.shape[0] or bottom <= top:
            continue
        patches = (frame[top:bottom, left:right], frame[top + shift:bottom + shift, left:right])
        marks = tuple(_deviation(patch) >= 52 for patch in patches)
        if not all(np.count_nonzero(mark) for mark in marks):
            continue
        removed = tuple(_mnemonic_mask(mark, other) for mark, other in (marks, marks[::-1]))
        kept = tuple(mark & ~mask for mark, mask in zip(marks, removed, strict=True))
        if (not any(mask.any() for mask in removed) or (kept[0] & kept[1]).sum() < .97 * (kept[0] | kept[1]).sum()
                or any(mask.sum() > .12 * mark.sum() for mask, mark in zip(removed, marks, strict=True))):
            if (recaptured := _recaptured_patches(patches)) is not None:
                return recaptured
            continue
        plain = []
        for patch, mask in zip(patches, removed, strict=True):
            clean = patch.copy()
            clean[mask] = np.median(patch, axis=(0, 1)).astype(np.uint8)
            plain.append((clean, bool(mask.any()), False))
        return plain
    return None


def _views(patch, recaptured):
    contrast = np.clip(1.75 * _deviation(patch), 0, 255).astype(np.uint8)
    views = patch, cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
    if recaptured:
        # Give the OCR more samples across each glyph and the intervening spaces.
        return tuple(cv2.resize(view, None, fx=2.5, fy=2, interpolation=cv2.INTER_CUBIC) for view in views)
    return views


def _corroborated(first, second, *, strip=" "):
    text = first["text"].strip(strip)
    alternate = second["text"].strip(strip)
    if text == alternate and min(first["score"], second["score"]) >= 90 and any(c.isalnum() for c in text):
        return text
    return None


def _hint_matches(hint, title, *, mnemonic):
    if hint is None:
        return True
    observed, visible = _key(hint.strip(" :：")), _key(title)
    return (observed == visible or (mnemonic and len(observed) == len(visible)
            and sum(left != right for left, right in zip(observed, visible, strict=True)) == 1))


def _same_value_ink(first, second):
    if first.shape != second.shape:
        return False
    a, b = _deviation(first), _deviation(second)
    left, right = a >= 52, b >= 52
    return ((left & right).sum() >= .86 * (left | right).sum() and np.quantile(np.abs(a - b), .99) <= 39)


def _visible_value(observed, value, *, repaired=False):
    original, actual = _key(observed), _key(value)
    if repaired:
        for opening, closing in (("(", ")"), ("[", "]")):
            if (actual.endswith(closing) and original.endswith(closing * 2)
                    and actual.count(closing) == actual.count(opening)
                    and original.count(closing) == original.count(opening) + 1):
                original = original[:-1]
        if len(actual) >= 10 and len(original) + 1 == len(actual):
            return any(letter.isascii() and letter.isalpha() and original == actual[:i] + actual[i + 1:]
                       for i, letter in enumerate(actual))
    return original == actual


def read_grouped_selector_text(frame, rows, line_reader):
    """Return {field: (caption, value)} where both isolated texts are corroborated."""
    recognize = getattr(line_reader, "recognize_prepared_lines", None)
    if recognize is None or not rows:
        return {}
    labels = [(_crop(frame, row.title_box), False, False) for row in rows]
    for index, row in enumerate(rows):
        peer = row.peer
        if peer is not None and index < peer < len(rows) and rows[peer].peer == index:
            other = rows[peer]
            delta = round(other.arrow.center[1] - row.arrow.center[1])
            patches = _peer_patches(frame, row.title_box, other.title_box, delta)
            if patches is not None:
                labels[index], labels[peer] = patches
        elif peer is None and (reference := getattr(row, "title_reference", None)) is not None:
            delta = round(reference.center[1] - row.title_box.center[1])
            patches = _peer_patches(frame, row.title_box, reference, delta)
            if patches is not None:
                labels[index] = patches[0]
    requests = [(index, row) for index, row in enumerate(rows) if row.value_box.right <= row.arrow.left - 2]
    values = [_crop(frame, row.value_box) for row in rows]
    crops = [crop for i, _ in requests for patch in (labels[i][0], values[i]) for crop in _views(patch, labels[i][2])]
    if not crops:
        return {}
    try:
        readings = recognize(crops)
    except (RuntimeError, ValueError):
        return {}
    if len(readings) != len(crops):
        return {}
    titles, shown = {}, {}
    for index, (i, row) in enumerate(requests):
        first, second, earlier, later = readings[4 * index:4 * index + 4]
        separator = row.merged_text or ""
        peer = row.peer
        if peer is not None and 0 <= peer < len(rows):
            separator += rows[peer].merged_text or ""
        separator = ":" in unicodedata.normalize("NFKC", separator)
        title = _corroborated(first, second, strip=" :：." if labels[i][2] and separator else " :：")
        value = _corroborated(earlier, later)
        if title:
            titles[i] = title
        if value:
            shown[i] = value
    accepted = {}
    for i, row in requests:
        title, value = titles.get(i), shown.get(i)
        if not title or not value or not _hint_matches(row.title_hint, title, mnemonic=labels[i][1]):
            continue
        peer = row.peer
        repaired = labels[i][2] and peer in shown and value == shown[peer] and _same_value_ink(values[i], values[peer])
        if row.value_hint is not None and not _visible_value(row.value_hint, value, repaired=repaired):
            continue
        if row.merged_text and row.title_hint is None:
            merged = _key(row.merged_text)
            boundary = len(_key(title))
            if (not _hint_matches(merged[:boundary], title, mnemonic=labels[i][1])
                    or not _visible_value(merged[boundary:].removeprefix(":"), value, repaired=repaired)):
                continue
        accepted[i] = title, value
    return {rows[i].field: pair for i, pair in accepted.items()
            if rows[i].peer not in titles or titles[rows[i].peer] == pair[0]}
