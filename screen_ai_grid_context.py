"""Find small context crops at unsupported slots in a repeated visual grid."""

from itertools import pairwise
from statistics import median


def _axis_lines(values, spacing):
    groups = []
    for value in sorted(values):
        if not groups or value - median(groups[-1]) > .24 * spacing:
            groups.append([])
        groups[-1].append(value)
    return [median(group) for group in groups]


def _regular_runs(lines, tile):
    for start, first in enumerate(lines[:-2]):
        pitch = lines[start + 1] - first
        if not 1.12 * tile <= pitch <= 2.20 * tile:
            continue
        end = start + 2
        while end < len(lines) and abs((lines[end] - lines[end - 1]) - pitch) <= .19 * tile:
            end += 1
        if end - start >= 3:
            yield lines[start:end]


def _fit_grid(boxes):
    width = median(box[2] - box[0] for box in boxes)
    height = median(box[3] - box[1] for box in boxes)
    points = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes]
    columns = list(_regular_runs(_axis_lines((x for x, _ in points), width), width))
    rows = list(_regular_runs(_axis_lines((y for _, y in points), height), height))
    for xs in columns:
        for ys in rows:
            occupied = set()
            for x, y in points:
                column = min(range(len(xs)), key=lambda index: abs(xs[index] - x))
                row = min(range(len(ys)), key=lambda index: abs(ys[index] - y))
                if abs(xs[column] - x) <= .23 * width and abs(ys[row] - y) <= .23 * height:
                    occupied.add((row, column))
            counts = [sum((row, column) in occupied for column in range(len(xs))) for row in range(len(ys))]
            density = len(occupied) / (len(xs) * len(ys))
            if (len(occupied) >= 8 and density >= .59 and max(counts) == len(xs)
                    and sum(4 * count >= 3 * len(xs) for count in counts) >= 2):
                score = (len(occupied), density)
                yield score, xs, ys, width, height, occupied


def _largest_grid(boxes, short_side):
    icons = [box for box in boxes if .068 * short_side <= box[2] - box[0] <= .26 * short_side
             and .80 <= (box[2] - box[0]) / (box[3] - box[1]) <= 1.25]
    sizes = [(box[2] - box[0], box[3] - box[1]) for box in icons]
    groups = {tuple(index for index, (w, h) in enumerate(sizes)
                    if .83 * width <= w <= 1.20 * width and .83 * height <= h <= 1.20 * height)
              for width, height in sizes}
    largest = sorted((group for group in groups if len(group) >= 8), key=lambda group: (-len(group), group))[:5]
    return max((grid for group in largest for grid in _fit_grid([icons[index] for index in group])),
               key=lambda grid: grid[0], default=None)


def _supports_box(box, outline):
    l, t, r, b = box
    loop = list(outline) + [outline[0]]
    for x, y in ((l, t), (r, t), (r, b), (l, b)):
        areas = [(x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
                 for (x0, y0), (x1, y1) in pairwise(loop)]
        if min(areas) < 0 < max(areas):
            return False
    return True


def _room_for(box, other_boxes):
    l, t, r, b = box
    return not any(max(l, left) < min(r, right) and max(t, top) < min(b, bottom)
                   for left, top, right, bottom in other_boxes)


def _slot_locations(xs, ys, occupied, top, bottom):
    for row in range(1, len(ys) - 1):
        for column, x in enumerate(xs):
            if (row, column) in occupied:
                continue
            if column in (0, len(xs) - 1):
                step = 1 if column == 0 else -1
                neighbors = [(row, column + step), (row, column + 2 * step),
                             (row - 1, column), (row + 1, column)]
                supported = all(cell in occupied for cell in neighbors)
            else:
                horizontal = (row, column - 1) in occupied and (row, column + 1) in occupied
                support = sum((peer, column) in occupied for peer in range(max(0, row - 2), min(len(ys), row + 3)))
                supported = horizontal and support >= 2
            if supported:
                yield x, ys[row]

    # A partially populated first row can reveal the columns of an obscured
    # row above it. A complete edge row or the natural trailing blanks of a
    # grid do not establish cells beyond the observed rows.
    first = [column for column in range(len(xs)) if (0, column) in occupied]
    pitch = median(b - a for a, b in pairwise(ys))
    cy = ys[0] - pitch
    if (len(ys) >= 4 and 2 <= len(first) < len(xs) and top <= cy <= top + .24 * (bottom - top)
            and all(b - a == 1 for a, b in pairwise(first))
            and all((1, column) in occupied for column in first)):
        yield from ((xs[column], cy) for column in first)


def propose_grid_context(page):
    """Return raster boxes for extra context, never inferred button labels."""
    layer = page["layer"]
    if layer["kind"] != "page" or layer.get("boxes") or layer.get("floating_boxes"):
        return []
    w, h = page["image_size"]
    outline = page.get("screen_corners") or [[0, 0], [w, 0], [w, h], [0, h]]
    left, right = min(point[0] for point in outline), max(point[0] for point in outline)
    top, bottom = min(point[1] for point in outline), max(point[1] for point in outline)
    boxes = [row["box"] for row in page["elements"] if row["type"] == "control_candidate"
             and row["box"][0] < row["box"][2] and row["box"][1] < row["box"][3]
             and _supports_box(row["box"], outline)]
    grid = _largest_grid(boxes, min(right - left, bottom - top))
    if grid is None:
        return []
    _, xs, ys, width, height, occupied = grid
    horizontal_gap = min(b - a for a, b in pairwise(xs)) - width
    vertical_gap = min(b - a for a, b in pairwise(ys)) - height
    pad_x = min(.08 * width, .23 * horizontal_gap)
    pad_y = min(.08 * height, .23 * vertical_gap)
    result = []
    for x, y in _slot_locations(xs, ys, occupied, top, bottom):
        for dx, dy in ((pad_x, pad_y), (0, 0)):
            half_w, half_h = width / 2 + dx, height / 2 + dy
            box = [round(x - half_w), round(y - half_h), round(x + half_w), round(y + half_h)]
            if _supports_box(box, outline) and _room_for(box, boxes):
                result.append({"box": box})
                break
    return sorted(result, key=lambda item: (item["box"][1], item["box"][0]))
