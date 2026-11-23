"""Localize repeated choices from indicator and caption pixels."""

from dataclasses import dataclass

import cv2
import numpy as np

from inventory_choice_glyphs import ChoiceGlyph, find_choice_glyphs
from inventory_list_rows import _Box
from inventory_value_select import _arrow


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class ChoiceRowControl:
    region_id: int | None
    box: Box
    caption_area: Box
    indicator_box: Box
    click_point: tuple[int, int]
    kind: str
    selected: bool
    disabled: bool | None
    state: str


@dataclass(frozen=True)
class _Row:
    glyph: ChoiceGlyph
    caption: Box
    strength: float
    leading: bool


def _center(box):
    return round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)


def _runs(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2).tolist()


def _height(box):
    return box[3] - box[1]


def _isolated_radio(gray, bounds, glyph, leading):
    """Reject loops that are part of a word on the side opposite the caption."""
    left, top, right, bottom = bounds
    x0, y0, x1, y1 = glyph.box
    height = y1 - y0
    reach = round(.92 * height)
    gap = max(1, round(.035 * height))
    pad = round(.16 * height)
    start, stop = (x0 - reach, x0 - gap) if leading else (x1 + gap, x1 + reach)
    start, stop = max(left, start), min(right, stop)
    if stop <= start:
        return True
    first, last = max(top, y0 - pad), min(bottom, y1 + pad)
    patch = gray[first:last, start:stop].astype(np.int16)
    ink = np.uint8(np.abs(patch - np.median(patch)) >= max(18, .20 * glyph.contrast))
    _, _, components, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for x, y, width, depth, area in components[1:]:
        distance = x0 - start - x - width if leading else start + x - x1
        if (distance <= .31 * height and width >= .12 * height and .37 * height <= depth <= 1.12 * height
                and abs(first + y + depth / 2 - (y0 + y1) / 2) <= .30 * height and area >= 8):
            return False
    return True


def _caption(gray, surface, glyph, leading, *, text_scale=None):
    left, top, right, bottom = surface
    x0, y0, x1, y1 = glyph.box
    size = text_scale or y1 - y0
    cy = (y0 + y1) / 2
    start = max(top, round(cy - .59 * size))
    stop = min(bottom, round(cy + .59 * size))
    if leading:
        bound = x1 + max(2, round(.12 * size)), min(right, x1 + round(.70 * (right - left)))
    else:
        bound = left + max(2, round(.024 * (right - left))), x0 - max(2, round(.12 * size))
    a, b = bound
    if b - a < size or stop - start < size:
        return None
    patch = gray[start:stop, a:b].astype(np.int16)
    tone = np.median(patch)
    contrast = np.abs(patch - tone)
    ink = contrast >= 20
    if ink.mean() >= .30:
        return None
    runs = _runs(ink.sum(axis=0) >= max(2, round(.043 * size)))
    if not runs:
        return None
    if leading and runs[0][0] > 2.25 * size:
        return None
    if not leading and runs[0][0] > .13 * (right - left):
        return None
    first = runs[0][0]
    last = runs[0][1]
    count = 1
    for u, v in runs[1:]:
        if u - last > .43 * size:
            break
        last = v
        count += 1
    if count < 2 or not .96 * size <= last - first <= .69 * (right - left):
        return None
    crop = np.uint8(ink[:, first:last])
    x, y, w, h = cv2.boundingRect(crop)
    if (not .38 * size <= h <= 1.06 * size or abs(start + y + h / 2 - cy) > .29 * size
            or not .09 <= crop[y:y + h, x:x + w].mean() <= .67):
        return None
    pad = max(2, round(.095 * size))
    box = max(a, a + first + x - pad), max(top, start + y - pad), min(b, a + first + x + w + pad), min(bottom, start + y + h + pad)
    strengths = contrast[:, first:last][crop != 0]
    return _Row(glyph, box, float(np.percentile(strengths, 77)), leading)


def _same_lane(first, second):
    a, b = first.glyph, second.glyph
    size = min(_height(a.box), _height(b.box))
    dx = abs(a.center[0] - b.center[0])
    dy = abs(a.center[1] - b.center[1])
    spacing = 6.45 if a.kind == "radio" and {a.state, b.state} == {"on", "off"} else 4.8
    return (a.kind == b.kind and first.leading == second.leading and dx <= .28 * size
            and 1.43 * size <= dy <= spacing * size and max(_height(a.box), _height(b.box)) <= 1.24 * size
            and abs(first.caption[0] - second.caption[0]) <= .43 * size
            and max(_height(first.caption), _height(second.caption)) <= 1.47 * min(_height(first.caption), _height(second.caption)))


def _glyphs(gray, box, leading):
    left, top, right, bottom = box
    width = right - left
    a, b = (.018, .27) if leading else (.73, .982)
    x0, x1 = left + round(a * width), left + round(b * width)
    y0, y1 = top + round(.038 * min(width, bottom - top)), bottom - round(.028 * min(width, bottom - top))
    side = gray[y0:y1, x0:x1]
    scale = min(1., 790 / width)
    working = cv2.resize(side, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else side
    minimum, maximum = max(8, round(.017 * width * scale)), max(10, round(.079 * width * scale))
    for glyph in find_choice_glyphs(working, (0, 0, working.shape[1], working.shape[0]), (minimum, maximum)):
        l, t, r, b = glyph.box
        shape = x0 + round(l / scale), y0 + round(t / scale), x0 + round(r / scale), y0 + round(b / scale)
        if glyph.confidence >= .68:
            yield ChoiceGlyph(shape, glyph.kind, glyph.state, glyph.contrast, glyph.confidence)


def _owner(row, candidates, surface):
    gx, gy = _center(row.glyph.box)
    left, top, right, bottom = row.glyph.box
    l, t, r, b = row.caption
    h = bottom - top
    member = min(left, l), min(top, t), max(right, r), max(bottom, b)
    matches = [entry for entry in candidates if entry["box"][0] - .22 * h <= member[0]
               and entry["box"][1] <= gy <= entry["box"][3] and entry["box"][2] + .25 * h >= member[2]
               and .76 * h <= _height(entry["box"]) <= 3.3 * h
               and abs(entry["center"][1] - gy) <= .63 * h]
    if matches:
        anchor = min(matches, key=lambda entry: (_height(entry["box"]), entry["box"][2] - entry["box"][0]))
        region_id = int(anchor["id"])
    else:
        # Some detectors return the checkbox by itself instead of the whole row.
        anchor = next((entry for entry in candidates if entry["box"][0] <= gx < entry["box"][2]
                       and entry["box"][1] <= gy < entry["box"][3] and .68 * h <= _height(entry["box"]) <= 2.05 * h
                       and entry["box"][2] - entry["box"][0] <= 2.3 * h), None)
        region_id = int(anchor["id"]) if anchor is not None else None
    pad = max(3, round(.23 * h))
    owned = max(surface[0], member[0] - pad), max(surface[1], member[1] - pad), min(surface[2], member[2] + pad), min(surface[3], member[3] + pad)
    return region_id, owned


def _choices(gray, bounds, candidates):
    rows = [row for leading in (True, False) for glyph in _glyphs(gray, bounds, leading)
            if glyph.kind != "radio" or _isolated_radio(gray, bounds, glyph, leading)
            if (row := _caption(gray, bounds, glyph, leading)) is not None]
    result = []
    for row in rows:
        peers = [other for other in rows if _same_lane(row, other)]
        glyph = row.glyph
        if not peers and not _checkbox_beside_radios(row, rows):
            continue
        if glyph.kind == "radio" and not (len(peers) >= 2 or any(peer.glyph.state != glyph.state for peer in peers)):
            continue
        # Dim ink alone is also common on naturally outlined controls. Only a
        # jointly dim mark and caption relative to same-state peers is evidence.
        same_state = [p for p in peers if p.glyph.state == glyph.state]
        faded = (len(same_state) >= 2 and row.strength < .64 * np.median([p.strength for p in same_state])
                 and glyph.contrast < .54 * np.median([p.glyph.contrast for p in same_state]))
        region_id, box = _owner(row, candidates, bounds)
        result.append(ChoiceRowControl(region_id, box, row.caption, glyph.box, _center(glyph.box), glyph.kind,
                                      glyph.state == "on", True if faded else None, glyph.state))
    return result


def _checkbox_beside_radios(check, rows):
    """Allow an independently drawn checkbox in an aligned radio group."""
    mark = check.glyph
    h = _height(mark.box)
    x, y = mark.center
    if mark.kind != "checkbox" or mark.confidence < .81:
        return False
    aligned = [row for row in rows if row.glyph.kind == "radio" and row.leading == check.leading
               and abs(row.glyph.center[0] - x) <= .24 * h and abs(row.caption[0] - check.caption[0]) <= .34 * h
               and 1.50 * h <= abs(row.glyph.center[1] - y) <= 12.0 * h and
               .74 * h <= _height(row.glyph.box) <= 1.32 * h]
    return any(abs(radio.glyph.center[1] - y) <= 7.0 * h and radio != other and _same_lane(radio, other)
               and {radio.glyph.state, other.glyph.state} == {"on", "off"}
               for radio in aligned for other in aligned)


def _divider(gray, bounds, start, stop, estimated):
    left, top, right, bottom = bounds
    w = right - left
    start, stop = max(top, round(start)), min(bottom, round(stop))
    if stop - start < 7:
        return None
    patch = gray[start:stop, left + round(.16 * w):right - round(.16 * w):6].astype(np.int16)
    surface = np.median(patch, axis=0)
    rules = (np.abs(patch - surface) >= 9).mean(axis=1) >= .84
    separators = [start + b for a, b in _runs(rules) if 0 < a < b < len(rules) and b - a <= .013 * w]
    return min(separators, key=lambda y: abs(y - estimated), default=None)


def _disclosure_rows(gray, bounds, candidates, choices):
    left, top, right, bottom = bounds
    width = right - left
    rows = []
    for candidate in candidates:
        x0, y0, x1, y1 = candidate["box"]
        h = y1 - y0
        if not (left <= x0 < x1 <= right and top <= y0 < y1 <= bottom
                and x1 - x0 >= .73 * width and .105 * width <= h <= .23 * width):
            continue
        mark = _arrow(gray, _Box(x0, y0, x1, y1), relaxed=True)
        if mark is None:
            continue
        glyph_box = mark.box.xyxy()
        x, y = _center(glyph_box)
        if not (left + .83 * width <= x <= right - .029 * width and .018 * width <= mark.box.width <= .067 * width):
            continue
        text_scale = round(.29 * h)
        glyph = ChoiceGlyph(glyph_box, "collapse" if mark.up else "expand", "expanded" if mark.up else "collapsed", 0., 1.)
        row = _caption(gray, bounds, glyph, False, text_scale=text_scale)
        if row is None or row.caption[0] > left + .115 * width or row.caption[2] > left + .68 * width:
            continue
        above = _divider(gray, bounds, y - .65 * h, y - .22 * h, y0)
        below = _divider(gray, bounds, y + .24 * h, y + .76 * h, y1)
        if above is not None:
            rows.append((row, candidate, above, below))
    results = []
    for row, candidate, above, below in rows:
        glyph = row.glyph
        x, y = _center(glyph.box)
        h = _height(candidate["box"])
        peers = [peer for peer, anchor, _, lower in rows if peer != row and lower is not None
                 and peer.glyph.kind == "expand" and .76 * h <= _height(anchor["box"]) <= 1.33 * h
                 and abs(peer.glyph.center[0] - x) <= .031 * width and abs(peer.caption[0] - row.caption[0]) <= .023 * width
                 and .68 * h <= abs(peer.glyph.center[1] - y) <= 4.2 * h]
        if len(peers) < 2:
            continue
        if glyph.kind == "collapse" and below is None:
            # The first visible child closes the header without eating the
            # expanded subsection into the hit region.
            child_top = min((child.box[1] for child in choices if y + .26 * h < child.box[1] < y + 1.15 * h), default=None)
            if child_top is not None:
                below = (row.caption[3] + child_top) // 2
        if below is None or not above < row.caption[1] < row.caption[3] < below:
            continue
        x0, _, x1, _ = candidate["box"]
        results.append(ChoiceRowControl(int(candidate["id"]), (x0, above, x1, below), row.caption,
                                        glyph.box, (x, y), glyph.kind, False, None, glyph.state))
    return results


def choice_row_controls(frame, regions):
    """Return repeated choices supported by visible marks and adjacent captions."""
    height, width = frame.shape[:2]
    layer = regions["layer"]
    if layer == "page":
        bounds = (0, 0, width, height)
        visible = "page"
    elif layer in {"dialog", "bottom_sheet"} and regions["regions"]:
        bounds = tuple(regions["regions"][0])
        visible = "foreground"
    else:
        return []
    if bounds[2] - bounds[0] < 100 or bounds[3] - bounds[1] < 120:
        return []
    candidates = [row for row in regions["candidates"] if row["visibility"] == visible]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    choices = _choices(gray, bounds, candidates)
    disclosures = _disclosure_rows(gray, bounds, candidates, choices)
    return sorted(choices + disclosures, key=lambda record: (record.box[1], record.box[0]))
