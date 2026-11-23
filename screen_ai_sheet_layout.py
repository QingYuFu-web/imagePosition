"""Plan bounded contact sheets with readable horizontal control crops."""

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class TilePlan:
    index: int
    label_origin: tuple[int, int]
    tile_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class SheetPlan:
    width: int
    height: int
    tiles: tuple[TilePlan, ...]


def _render_size(size, cell_width, cell_height):
    width, height = size
    gain = min(4., (cell_width - 12) / width, (cell_height - 28) / height)
    return max(1, round(width * gain)), max(1, round(height * gain))


def _tile(index, rendered, x, y, width, height):
    w, h = rendered
    left = x + (width - w) // 2
    top = y + 22 + (height - 25 - h) // 2
    return TilePlan(index, (x + 7, y + 16), (left, top, left + w, top + h))


def _regular(rendered, start, cell_width, cell_height):
    columns = min(6, len(rendered))
    tiles = tuple(_tile(start + i, size, (i % columns) * cell_width, (i // columns) * cell_height,
                        cell_width, cell_height) for i, size in enumerate(rendered))
    return SheetPlan(columns * cell_width, ceil(len(tiles) / columns) * cell_height, tiles)


def _shelf(regular, wide, start, width_limit, height_limit, pixel_limit):
    """Group wide crops into short rows without shrinking the other crops."""
    width = y = 0
    tiles = []
    for cards in (regular, wide):
        rows, row, span = [], [], 0
        for index, size in cards:
            tile_width = max(42, size[0] + 12)
            if tile_width > width_limit:
                return None
            if span + tile_width > width_limit:
                rows.append(row)
                row, span = [], 0
            row.append((span, tile_width, index, size))
            span += tile_width
        if row:
            rows.append(row)
        for row in rows:
            row_height = max(size[1] for _, _, _, size in row) + 28
            width = max(width, row[-1][0] + row[-1][1])
            if y + row_height > height_limit or width * (y + row_height) > pixel_limit:
                return None
            tiles.extend(_tile(start + i, size, x, y, cell_width, row_height) for x, cell_width, i, size in row)
            y += row_height
    return SheetPlan(width, y, tuple(tiles))


def _fit_group(dimensions, start, cell_width, cell_height, *, widen):
    compact = [_render_size(size, cell_width, cell_height) for size in dimensions]
    legacy = _regular(compact, start, cell_width, cell_height)
    if not widen:
        return legacy
    candidates = sorted((i for i, (w, h) in enumerate(dimensions) if w >= 3.75 * h and compact[i][0] < .40 * w),
                        key=lambda i: (compact[i][0] / dimensions[i][0], i))[:4]
    if not candidates:
        return legacy
    pixel_limit = legacy.width * legacy.height
    height_limit = max(6 * cell_width, legacy.height)
    for count in range(len(candidates), 0, -1):
        priority = set(candidates[:count])
        regular = [(i, size) for i, size in enumerate(compact) if i not in priority]
        for gain in (2., 1.75, 1.50, 1.35):
            wide = [(i, _render_size(dimensions[i], round(gain * cell_width), cell_height)) for i in sorted(priority)]
            if any(size == compact[i] for i, size in wide):
                continue
            fits = [_shelf(regular, wide, start, columns * cell_width, height_limit, pixel_limit)
                    for columns in range(2, 7)]
            valid = [sheet for sheet in fits if sheet is not None]
            if valid:
                return min(valid, key=lambda sheet: (sheet.width * sheet.height, sheet.height, sheet.width))
    return legacy


def plan_contact_sheets(dimensions, cell_w, cell_h, *, widen=True):
    """Return one replayable plan per regular sheet, within its original pixel budget."""
    return [_fit_group(dimensions[start:start + 36], start, cell_w, cell_h, widen=widen)
            for start in range(0, len(dimensions), 36)]


def plan_caption_sheets(dimensions, cell_w, cell_h, baseline, preferred):
    """Widen measured caption crops while retaining the existing sizes of their neighbors."""
    plans = []
    for start, previous in zip(range(0, len(dimensions), 36), baseline, strict=True):
        group = dimensions[start:start + 36]
        original = {tile.index - start: (tile.tile_box[2] - tile.tile_box[0], tile.tile_box[3] - tile.tile_box[1])
                    for tile in previous.tiles}
        wanted = {i for i in original if start + i in preferred}
        if not wanted:
            plans.append(previous)
            continue
        compact = [_render_size(size, cell_w, cell_h) for size in group]
        pixels = min(6, len(group)) * ceil(len(group) / 6) * cell_w * cell_h
        chosen = previous
        for gain in (3., 2.5, 2., 1.75, 1.50):
            scaled = [(i, max(original[i], _render_size(size, round(gain * cell_w), cell_h)) if i in wanted else original[i])
                      for i, size in enumerate(group)]
            if all(size == original[i] for i, size in scaled):
                continue
            regular = [(i, size) for i, size in scaled if size == compact[i]]
            wide = [(i, size) for i, size in scaled if size != compact[i]]
            mixed = sorted(scaled, key=lambda item: (-item[1][1], -item[1][0], item[0]))
            fits = [_shelf(first, second, start, columns * cell_w, max(6 * cell_w, previous.height), pixels)
                    for first, second in ((regular, wide), (mixed, ())) for columns in range(2, 7)]
            if valid := [sheet for sheet in fits if sheet is not None]:
                chosen = min(valid, key=lambda sheet: (sheet.width * sheet.height, sheet.height, sheet.width))
                break
        plans.append(chosen)
    return plans
