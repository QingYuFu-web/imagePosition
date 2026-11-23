"""Propose bounded card context for controls on repeated horizontal rails."""

from itertools import pairwise
from statistics import median


def _width(box):
    return box[2] - box[0]


def _height(box):
    return box[3] - box[1]


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _row_bound(row):
    return (row[0][1][0], min(box[1] for _, box in row), row[-1][1][2], max(box[3] for _, box in row))


def _rails(elements, width, height):
    compact = [(identity, box) for identity, box in elements
               if .47 * width <= _center(box)[0] and .095 * height <= _center(box)[1] <= .91 * height
               and .048 * width <= _width(box) <= .24 * width and .045 * width <= _height(box) <= .13 * width]
    rows = {}
    for _, mark in compact:
        _, y = _center(mark)
        same_row = [(identity, box) for identity, box in compact
                    if abs(_center(box)[1] - y) <= .20 * min(_height(mark), _height(box))
                    and .65 <= _height(box) / _height(mark) <= 1.53]
        if not 3 <= len(same_row) <= 5:
            continue
        row = sorted(same_row, key=lambda item: item[1][0])
        bound = _row_bound(row)
        if not (.24 * width <= _width(bound) <= .59 * width and bound[2] >= .92 * width):
            continue
        if not all(-.10 * min(_width(first), _width(second)) <= second[0] - first[2] <= .082 * width
                   for (_, first), (_, second) in pairwise(row)):
            continue
        rows[tuple(identity for identity, _ in row)] = row
    return sorted(rows.values(), key=lambda row: _row_bound(row)[1])


def _repeated(earlier, later, width, height):
    first, second = _row_bound(earlier), _row_bound(later)
    span = max(_height(first), _height(second))
    gap = second[1] - first[3]
    if len(earlier) != len(later) or not 1.4 * span <= gap <= .53 * height:
        return False
    return all(.62 <= _width(after) / _width(before) <= 1.62
               and .65 <= _height(after) / _height(before) <= 1.54
               and abs(_center(after)[0] - _center(before)[0]) <= .36 * min(_width(before), _width(after))
               and abs(_center(after)[0] - _center(before)[0]) <= .078 * width
               for (_, before), (_, after) in zip(earlier, later, strict=True))


def _near_body(box, previous, row_top, scale):
    return previous <= box[1] and -.12 * scale <= row_top - box[3] <= .42 * scale


def _body_parts(boxes, previous, row, width):
    scale = median(_height(box) for _, box in row)
    row_top = _row_bound(row)[1]
    bodies = [box for box in boxes if _near_body(box, previous, row_top, scale)
              and 1.35 * scale <= _height(box) <= 6.8 * scale]
    left = [box for box in bodies if box[0] <= .10 * width and .42 * width <= _width(box) <= .79 * width]
    right = [box for box in bodies if .55 * width <= box[0] <= .83 * width
             and .12 * width <= _width(box) <= .39 * width and box[2] >= .90 * width]
    wide = [box for box in bodies if box[0] <= .095 * width and _width(box) >= .83 * width]
    return left, right, wide, scale


def _header_peers(boxes, body_top, previous, width, scale):
    return [box for box in boxes if previous <= box[1] < body_top
            and -.12 * scale <= body_top - box[3] <= .38 * scale
            and .52 * scale <= _height(box) <= 1.12 * scale
            and .44 * scale <= _width(box) <= 1.15 * scale and _width(box) <= .13 * width]


def _split_title(boxes, parts, previous, width):
    left, right, _, scale = parts
    if len(right) != 1:
        return None
    thumbnail = right[0]
    peers = _header_peers(boxes, thumbnail[1], previous, width, scale)
    menu = [box for box in peers if box[0] >= .85 * width and box[2] >= .93 * width]
    matching = [box for box in left if -.055 * width <= thumbnail[0] - box[2] <= .066 * width
                and max(abs(thumbnail[i] - box[i]) for i in (1, 3)) <= .28 * scale]
    if len(matching) == 1:
        lead = matching[0][0]
    elif not matching and len(menu) == 1:
        leading = [box for box in peers if box[0] <= .095 * width
                   and abs(_center(box)[1] - _center(menu[0])[1]) <= .18 * scale]
        if len(leading) != 1:
            return None
        lead = leading[0][0]
    else:
        return None
    # Leave the thumbnail out so the preview preserves the headline's pixels.
    top = max(previous, (menu[0][1] if len(menu) == 1 else thumbnail[1] - round(.91 * scale)) - round(.09 * scale))
    bottom = min(thumbnail[3], thumbnail[1] + round(1.78 * scale))
    return max(0, lead - round(.16 * scale)), top, thumbnail[0], bottom


def _wide_titles(first, second, width, before_rail, between_rails):
    if len(first[2]) != 1 or len(second[2]) != 1:
        return None
    before, after = first[2][0], second[2][0]
    if (abs(before[0] - after[0]) > .038 * width or abs(before[2] - after[2]) > .048 * width
            or not .61 <= _height(after) / _height(before) <= 1.62):
        return None

    def title(body, scale, previous):
        left, top, right, bottom = body
        start = max(previous + round(.16 * scale), top - round(1.30 * scale)) if previous else top
        return left, start, min(right, left + round(.79 * width)), min(bottom, top + round(1.89 * scale))

    return title(before, first[3], before_rail), title(after, second[3], between_rails)


def _bounded_title(box, width, height, floating):
    left, top, right, bottom = box
    return (0 <= left < right <= width and 0 <= top < bottom <= height
            and .36 * width <= right - left <= .82 * width and .052 * height <= bottom - top <= .22 * height
            and not any(max(a, left) < min(c, right) and max(b, top) < min(d, bottom) for a, b, c, d in floating))


def propose_feed_context(page):
    """Return title crops owned by verified repeated control geometry."""
    width, height = page["image_size"]
    layer = page["layer"]
    if layer["kind"] != "page" or layer.get("boxes") or width < 240 or not 1.25 * width <= height <= 3.0 * width:
        return []
    elements = [(row["id"], row["box"]) for row in page["elements"] if row["type"] == "control_candidate"]
    boxes = [box for _, box in elements]
    rails = _rails(elements, width, height)
    contexts = {}
    for index, (earlier, later) in enumerate(pairwise(rails)):
        if not _repeated(earlier, later, width, height):
            continue
        before = _row_bound(rails[index - 1])[3] if index > 0 else 0
        between = _row_bound(earlier)[3]
        first = _body_parts(boxes, before, earlier, width)
        second = _body_parts(boxes, between, later, width)
        titles = _wide_titles(first, second, width, before, between)
        if titles is None:
            titles = (_split_title(boxes, first, before, width), _split_title(boxes, second, between, width))
        for row, box in zip((earlier, later), titles, strict=True):
            if box is not None and _bounded_title(box, width, height, layer.get("floating_boxes", ())):
                contexts[tuple(identity for identity, _ in row)] = box
    return [{"box": list(box), "element_ids": list(identifiers)}
            for identifiers, box in sorted(contexts.items(), key=lambda item: item[1][1])]
