"""Find paired inline labels and values in a common column of strict selectors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median

import numpy as np

from inventory_list_rows import _Box


@dataclass(frozen=True)
class SelectorRow:
    field: _Box
    arrow: _Box
    title_box: _Box
    value_box: _Box
    title_hint: str | None
    value_hint: str | None
    merged_text: str | None = None
    peer: int | None = None
    title_reference: _Box | None = None


def _tight_ink(gray, left, top, right, bottom, height):
    if right - left < .60 * height or bottom - top < .46 * height:
        return None
    patch = gray[top:bottom, left:right].astype(np.int16)
    contrast = np.abs(patch - np.median(patch))
    level = max(41, .30 * np.quantile(contrast, .97))
    ink = contrast >= level
    rows = np.flatnonzero(ink.sum(axis=1) >= 2)
    cols = np.flatnonzero(ink.sum(axis=0) >= 1)
    if (not rows.size or not cols.size or cols[-1] - cols[0] < .49 * height
            or not .23 * height <= rows[-1] - rows[0] + 1 <= .73 * height
            or not .021 <= ink.mean() <= .56):
        return None
    return _Box(left + max(0, int(cols[0]) - 2), top + max(0, int(rows[0]) - 2),
                left + min(right - left, int(cols[-1]) + 3), top + min(bottom - top, int(rows[-1]) + 3))


def _crops(gray, field, arrow, lines):
    h = field.height
    same = [line for line in lines if line.score >= 86 and .52 * h <= line.box.height <= 1.22 * h
            and abs(line.box.center[1] - field.center[1]) <= .24 * h]
    outer = [line for line in same if 1.35 * h <= field.left - line.box.left <= 8.2 * h
             and line.box.right <= arrow.left - .20 * h]
    if len(outer) != 1:
        return None
    heading = outer[0]
    if heading.box.right >= field.left + .60 * h:
        value = heading
        raw = heading.text
        title_hint, colon, value_hint = raw.replace("：", ":", 1).partition(":")
        title_hint, value_hint = ((title_hint.strip(), value_hint.strip()) if colon else (None, None))
    else:
        if heading.box.right > field.left + .19 * h:
            return None
        shown = [line for line in same if line is not heading
                 and field.left - .24 * h <= line.box.left <= field.left + .60 * h
                 and line.box.right <= arrow.left - .25 * h]
        if len(shown) != 1:
            return None
        value = shown[0]
        title_hint, value_hint, raw = heading.text, value.text, None
    top, bottom = field.top + 2, field.bottom - 2
    title = _tight_ink(gray, heading.box.left, top,
                       min(field.left - 2, heading.box.right + 1), bottom, h)
    shown = _tight_ink(gray, field.left + 2, top,
                       min(arrow.left - 2, value.box.right + 2), bottom, h)
    if title and shown:
        return SelectorRow(field, arrow, title, shown, title_hint, value_hint, raw)
    return None


def _title_match(gray, a, b, displacement, h):
    if abs(a.left - b.left) > .20 * h or abs(a.right - b.right) > .20 * h:
        return False
    # Align the actual glyphs, testing one pixel of capture-scale vertical variation.
    left = min(a.left, b.left)
    right = max(a.right, b.right)
    for offset in (0, -1, 1):
        shift = displacement + offset
        top, bottom = min(a.top, b.top - shift), max(a.bottom, b.bottom - shift)
        if min(top, top + shift) < 0 or max(bottom, bottom + shift) > gray.shape[0] or bottom <= top:
            continue
        first_patch = gray[top:bottom, left:right].astype(np.int16)
        peer_patch = gray[top + shift:bottom + shift, left:right].astype(np.int16)
        results = []
        for level in (47, 72):
            source = np.abs(first_patch - np.median(first_patch)) >= level
            target = np.abs(peer_patch - np.median(peer_patch)) >= level
            intersect, union = source & target, source | target
            extra = (source ^ target).sum(axis=1)
            union_count = int(union.sum())
            if not union_count:
                break
            exact = intersect.sum() / union_count
            row = int(np.argmax(extra))
            if row >= .58 * extra.size and union_count > union[row].sum():
                # A single thin mnemonic underline may differ below the recurring text.
                fit = (intersect.sum() - intersect[row].sum()) / (union_count - union[row].sum())
            else:
                fit = 0.0
            results.append(exact >= .91 or (exact >= .815 and fit >= .925))
        if len(results) == 2 and all(results):
            return True
        # When the text is antialiased, keep the continuous foreground strength
        # as another witness instead of forcing a binary edge at one cutoff.
        source = np.maximum(0.0, np.abs(first_patch - np.median(first_patch)) - 27)
        target = np.maximum(0.0, np.abs(peer_patch - np.median(peer_patch)) - 27)
        difference = np.abs(source - target).sum(axis=1)
        differing_row = int(np.argmax(difference))
        bands = [(source, target)]
        if differing_row >= .61 * len(difference) and difference[differing_row] >= .24 * difference.sum():
            keep = np.arange(len(difference)) != differing_row
            bands.append((source[keep], target[keep]))
        for x, y in bands:
            shared = np.minimum(x, y).sum() / max(1, np.maximum(x, y).sum())
            similarity = (x * y).sum() / max(1, np.linalg.norm(x) * np.linalg.norm(y))
            if shared >= .80 and similarity >= .96:
                return True
    return False


def _same_title(gray, first, second):
    shift = round(second.arrow.center[1] - first.arrow.center[1])
    return _title_match(gray, first.title_box, second.title_box, shift, min(first.field.height, second.field.height))


def _paired(gray, rows):
    h = median(row.field.height for row in rows)
    gaps = np.diff([row.arrow.center[1] for row in rows])
    small = min(gaps)
    if not .83 * h <= small <= 1.49 * h:
        return None
    if len(rows) == 4:
        first, separation, last = gaps
        if abs(first - last) > .23 * h or not 1.48 * small <= separation <= 3.0 * small:
            return None
        pairs = ((0, 2), (1, 3))
    else:
        if not 1.50 * small <= max(gaps) <= 3.55 * small:
            return None
        pairs = ((0, 2), (0, 1), (1, 2))
    supported = [(i, j) for i, j in pairs if _same_title(gray, rows[i], rows[j])]
    if len(supported) != (2 if len(rows) == 4 else 1):
        return None
    peers = {i: j for a, b in supported for i, j in ((a, b), (b, a))}
    return [replace(row, peer=peers.get(i)) for i, row in enumerate(rows)]


def _horizontal_edge(gray, left, right, first, stop):
    candidates = []
    for y in range(max(4, first), min(stop, gray.shape[0] - 4)):
        rim = gray[y, left:right].astype(np.int16)
        above = np.median(gray[y - 4:y - 1, left:right], axis=0)
        below = np.median(gray[y + 2:y + 5, left:right], axis=0)
        delta = np.minimum(above, below) - rim
        if (delta >= 10).mean() >= .74 and np.median(delta) >= 13:
            candidates.append((y, float(np.median(delta))))
    return candidates


def _vertical_edge(gray, top, bottom, guide, height):
    inset = max(2, round(.16 * height))
    margin = max(2, round(.24 * height))
    candidates = []
    for x in range(max(inset, guide - margin), min(gray.shape[1] - inset, guide + margin + 1)):
        rim = gray[top:bottom, x].astype(np.int16)
        outside = np.minimum(gray[top:bottom, x - inset], gray[top:bottom, x + inset])
        depth = float(np.median(outside - rim))
        if ((outside - rim) >= 9).mean() >= .72 and 13 <= depth <= 62:
            candidates.append((-abs(x - guide), depth, x))
    return max(candidates)[2] if candidates else None


def _measure_field(gray, anchor, y):
    h = anchor.height
    left = anchor.left + round(.48 * h)
    right = anchor.right - round(.92 * h)
    upper = _horizontal_edge(gray, left, right, round(y - .72 * h), round(y - .30 * h) + 1)
    lower = _horizontal_edge(gray, left, right, round(y + .27 * h), round(y + .73 * h) + 1)
    bands = [(top, bottom, high + low) for top, high in upper for bottom, low in lower
             if .76 * h <= bottom - top + 1 <= 1.28 * h]
    pad = max(2, round(.15 * h))
    for top, bottom, _ in sorted(bands, key=lambda band: (abs(band[1] - band[0] + 1 - h), -band[2])):
        a = _vertical_edge(gray, top + pad, bottom - pad, anchor.left, h)
        b = _vertical_edge(gray, top + pad, bottom - pad, anchor.right - 1, h)
        if a is not None and b is not None and .91 * anchor.width <= b - a + 1 <= 1.09 * anchor.width:
            return _Box(a, top, b + 1, bottom + 1)
    return None


def _recaptured_rows(gray, per_arrow, lines, read_arrow):
    # A matching OCR row can guide a local boundary search after JPEG splits a
    # contour. All four physical edges and that field's own strict chevron are required.
    seeds = [row for alternatives in per_arrow.values() for row in alternatives]
    for anchor in seeds:
        h = anchor.field.height
        partner = next((row for row in seeds if 2.1 * h <= abs(row.arrow.center[1] - anchor.arrow.center[1]) <= 4.8 * h
                        and abs(row.field.left - anchor.field.left) <= .18 * h
                        and abs(row.field.right - anchor.field.right) <= .18 * h
                        and .76 <= row.field.height / h <= 1.30 and _same_title(gray, anchor, row)), None)
        if partner is None:
            continue
        for line in lines:
            y = line.box.center[1]
            if (line.score < 86 or not .52 * h <= line.box.height <= 1.22 * h
                    or abs(line.box.left - anchor.title_box.left) > .26 * h
                    or min(abs(y - row.arrow.center[1]) for row in (anchor, partner)) > 2.0 * h
                    or any(abs(y - mark.center[1]) < .55 * h and abs(anchor.arrow.center[0] - mark.center[0]) <= .20 * h
                           for mark in per_arrow)):
                continue
            field = _measure_field(gray, anchor.field, y)
            if field is None:
                continue
            mark = read_arrow(gray, field)
            if (mark is None or mark.up or abs(mark.box.center[0] - anchor.arrow.center[0]) > .16 * h
                    or not .73 <= mark.box.width / anchor.arrow.width <= 1.35):
                continue
            row = _crops(gray, field, mark.box, lines)
            if row is not None:
                per_arrow.setdefault(mark.box, []).append(row)


def _with_reference(gray, rows, lines):
    if len(rows) != 3:
        return rows
    first = next(row for row in rows if row.peer is not None)
    other = rows[first.peer]
    missing = next(i for i, row in enumerate(rows) if row.peer is None)
    row = rows[missing]
    pitch = abs(round(first.arrow.center[1] - other.arrow.center[1]))
    h = row.field.height
    found = set()
    for shift in (-pitch, pitch):
        y = row.arrow.center[1] + shift
        if any(abs(y - current.arrow.center[1]) < .55 * h for current in rows):
            continue
        for line in lines:
            box = line.box
            if (line.score < 86 or abs(box.center[1] - y) > .25 * h
                    or abs(box.left - row.title_box.left) > .25 * h
                    or not .52 * h <= box.height <= 1.22 * h):
                continue
            reference = _tight_ink(gray, box.left, max(0, row.field.top + shift + 2),
                                   min(row.field.left - 2, box.right + 1),
                                   min(gray.shape[0], row.field.bottom + shift - 2), h)
            if reference is not None and _title_match(gray, row.title_box, reference, shift, h):
                found.add(reference)
    if len(found) == 1:
        rows[missing] = replace(row, title_reference=next(iter(found)))
    return rows


def grouped_selector_rows(gray, strict_fields, lines, *, read_arrow=None):
    """Return one unambiguous paired group, using only separately verified down arrows."""
    if len(strict_fields) < (2 if read_arrow is not None else 3):
        return []
    per_arrow = {}
    for field, arrow in strict_fields.items():
        if field.width < 3.8 * field.height:
            continue
        seen = per_arrow.setdefault(arrow, [])
        if len(seen) < 5 and (row := _crops(gray, field, arrow, lines)) is not None:
            seen.append(row)
    if read_arrow is not None and 2 <= sum(bool(options) for options in per_arrow.values()) < 4:
        _recaptured_rows(gray, per_arrow, lines, read_arrow)
    groups = {}
    for alternatives in per_arrow.values():
        for anchor in alternatives:
            h = anchor.field.height
            lane = []
            for arrow, peers in per_arrow.items():
                if abs(anchor.arrow.center[0] - arrow.center[0]) > .17 * h:
                    continue
                options = [row for row in peers if abs(row.field.left - anchor.field.left) <= .18 * h
                           and abs(row.field.right - anchor.field.right) <= .18 * h
                           and .78 <= row.field.height / h <= 1.26
                           and abs(row.title_box.left - anchor.title_box.left) <= .27 * h]
                if options:
                    lane.append(options[0])
            if len(lane) not in (3, 4):
                continue
            lane.sort(key=lambda item: item.arrow.center[1])
            key = tuple(row.arrow for row in lane)
            if key not in groups and (fit := _paired(gray, lane)) is not None:
                groups[key] = fit
    if not groups:
        return []
    maximum = max(map(len, groups))
    winners = [rows for arrows, rows in groups.items() if len(arrows) == maximum]
    return _with_reference(gray, winners[0], lines) if len(winners) == 1 else []
