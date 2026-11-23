"""Store and match click examples using local screenshot evidence."""

from contextlib import closing, contextmanager
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
from math import ceil, floor, isfinite
import os
from pathlib import Path
from shutil import copyfile
import sqlite3
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2
import numpy as np

from screen_registration import PROFILES, known_screen_mm, photo_to_fixture_mm
from screen_photo_geometry import display_mapping, display_size, fixture_occlusions
from screen_ai_visibility import on_disabled_control, on_selection_marker


_CHIP_SIZE = 48
_ORIGIN_PRIORITY = {"manual": 2, "ai": 1, "detector": 0}
_EXPLICIT_PHOTO_SPACE = "explicit-screen-photo"
_BUNDLED_EXAMPLES = Path(__file__).resolve().parent / "learning" / "control-examples.seed.sqlite3"
_SCHEMA = """
    CREATE TABLE IF NOT EXISTS examples (
        frame_key TEXT NOT NULL,
        space TEXT NOT NULL,
        layer TEXT NOT NULL,
        aspect REAL NOT NULL,
        box TEXT NOT NULL,
        geometry TEXT NOT NULL,
        samples BLOB,
        PRIMARY KEY (frame_key, box)
    )
"""


def parse_annotations(text):
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as error:
        raise ValueError("请输入补标 JSON。") from error
    if (not isinstance(data, dict) or data.get("coordinate_space") != "original_image_px"
            or not isinstance(data.get("annotations"), list) or not data["annotations"]):
        raise ValueError("补标 JSON 需要原图像素坐标和 annotations 列表。")
    return [_annotation(row) for row in data["annotations"]]


def _annotation(row):
    if not isinstance(row, dict) or set(row) != {"function", "box", "click_point"}:
        raise ValueError("每条样本需要用途、方框和点击点。")
    name, box, point = row["function"], row["box"], row["click_point"]
    if name is not None and (not isinstance(name, str) or not 1 <= len(name.strip()) <= 50
                             or not name.strip().isprintable()):
        raise ValueError("请填写 1–50 字的用途，或标记为不可点击。")
    if (not isinstance(box, (list, tuple)) or len(box) != 4 or not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(type(value) is int or type(value) is float and isfinite(value) for value in (*box, *point))
            or not _contains(box, point)):
        raise ValueError("点击点必须在样本框内。")
    return {"function": name.strip() if name is not None else None, "box": list(box), "click_point": list(point)}


def _contains(box, point):
    left, top, right, bottom = box
    x, y = point
    return left <= x < right and top <= y < bottom


def _size(box):
    return box[2] - box[0], box[3] - box[1]


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _coverage(box, other):
    w, h = _size(box)
    x = max(0, min(box[2], other[2]) - max(box[0], other[0]))
    y = max(0, min(box[3], other[3]) - max(box[1], other[1]))
    return x * y / (w * h) if w > 0 and h > 0 else 0


def _same_owner(first, second):
    return (_contains(first["box"], second["click_point"]) and _contains(second["box"], first["click_point"])
            and min(_coverage(first["box"], second["box"]), _coverage(second["box"], first["box"])) >= .46)


def _normalized_purpose(purpose):
    return " ".join(purpose.split()).casefold() if purpose is not None else None


def _same_purpose(first, second, *observed_actions):
    first, second = _normalized_purpose(first), _normalized_purpose(second)
    if first == second:
        return True
    if first is None or second is None:
        return False
    actions = [action for action in observed_actions if action is not None]
    if any(role != "switch_tab" for role, _ in actions):
        return False
    prefix = "切换到"
    for _, purpose in actions:
        label = _normalized_purpose(purpose)
        if not label or not label.startswith(prefix) or len(label) == len(prefix):
            continue
        title = label[len(prefix):]
        aliases = {f"{verb}{title}{suffix}" for verb in (prefix, "打开", "进入") for suffix in ("", "标签", "标签页")}
        if first in aliases and second in aliases:
            return True
    return False


def _thumbnails(regions):
    return [{**item, "click_point": item["center"]} for item in regions["candidates"] if item.get("media_thumbnail")]


def _on_thumbnail_marker(tiles, record):
    return any(on_selection_marker(tile, record["click_point"]) and _same_owner(tile, record) for tile in tiles)


def _floating_panes_at(regions, box, point):
    return [pane for pane in regions.get("floating_controls", ()) if _contains(pane, point) or _coverage(box, pane) >= .12]


def _floating_control(regions, box, point):
    panes = _floating_panes_at(regions, box, point)
    if len(panes) != 1 or _coverage(box, panes[0]) < 1:
        return None
    record = {"box": box, "click_point": point}
    owners = [item for item in regions["candidates"] if item.get("floating_control")
              and item.get("visibility") == "foreground" and not item.get("review_only") and not item.get("disabled")
              and _coverage(item["box"], panes[0]) == 1
              and _same_owner(record, {"box": item["box"], "click_point": item["center"]})
              and ((outline := item.get("photo_outline_px")) is None
                   or cv2.pointPolygonTest(np.float32(outline), tuple(map(float, point)), False) >= 0)]
    return owners[0]["floating_control"] if len(owners) == 1 else None


def _automatic_examples(frame, outcome, annotations):
    height, width = frame.shape[:2]
    distinct, conflicts = [], []
    tiles = _thumbnails(outcome.regions)
    for annotation in annotations:
        try:
            row = _annotation(annotation)
        except ValueError:
            continue
        box = [round(value) for value in row["box"]]
        left, top, right, bottom = box
        point = row["click_point"]
        floating = _floating_control(outcome.regions, box, point)
        if (not 0 <= left < right <= width or not 0 <= top < bottom <= height
                or min(right - left, bottom - top) < 6 or (right - left) * (bottom - top) > .65 * width * height
                or not _contains(box, point) or not _visible(outcome.regions, box, point, floating=floating)):
            continue
        row["box"] = box
        if _on_thumbnail_marker(tiles, row):
            continue
        if any(_same_owner(item, row) for item in conflicts):
            continue
        owners = [item for item in distinct if _same_owner(item, row)]
        if not owners:
            distinct.append(row)
        else:
            observed = next((item for item in outcome.active if _same_owner(item, row)), None)
            if len(owners) != 1 or not _same_purpose(owners[0]["function"], row["function"], _prior_action(observed)):
                conflicts.extend([row, *owners])
                distinct = [item for item in distinct if item not in owners]
    sliders = [row for row in outcome.active if row["role"] == "adjust_slider"]
    return [row for row in distinct if row["function"] is None or not any(_same_owner(row, slider) for slider in sliders)]


def _sample_origin(record):
    return record.get("origin", "manual")


def _sample_priority(record):
    return _ORIGIN_PRIORITY[_sample_origin(record)]


def _visible(regions, box, point, *, floating=None):
    layer = regions["layer"]
    if layer in {"layer_uncertain", "voice_recording", "voice_stop_panel"}:
        return False
    if any(on_disabled_control(item, point) for item in regions.get("candidates", ())):
        return False
    if regions.get("regions") and not any(_contains(panel, point) and _coverage(box, panel) >= .8
                                          for panel in regions["regions"]):
        return False
    if floating is not None:
        if _floating_control(regions, box, point) != floating:
            return False
    elif _floating_panes_at(regions, box, point):
        return False
    if "screen_corners_px" in regions:
        corners = np.float32(regions["screen_corners_px"])
        if cv2.pointPolygonTest(corners, tuple(map(float, point)), False) < 0:
            return False
        if regions.get("calibration", {}).get("space") == "fixture_estimate_mm":
            blocked = fixture_occlusions()
            height, width = blocked.shape
            # Occlusion lives in the fixture's physical orientation. Learning
            # points have already been projected back onto the original photo.
            transform, _ = display_mapping(corners, (width, height))
            x, y = cv2.perspectiveTransform(np.float64([[point]]), transform)[0, 0]
            i, j = round(x), round(y)
            if not (0 <= i < width and 0 <= j < height) or blocked[j, i]:
                return False
    return True


def _explicit_photo(regions):
    return (regions.get("calibration", {}).get("registration") == "photo_screen_corners"
            and len(regions.get("screen_corners_px", ())) == 4)


def _space(regions):
    calibration = regions.get("calibration", {})
    return _EXPLICIT_PHOTO_SPACE if _explicit_photo(regions) else calibration.get("reference_image", "screenshot")


def _explicit_sample_size(regions):
    calibration = regions["calibration"]
    width, height = calibration.get("rectified_size_px") or display_size(regions["screen_corners_px"])
    scale = 1376 / max(width, height)
    return max(2, round(width * scale)), max(2, round(height * scale))


def _learning_aspect(regions, shape):
    if _explicit_photo(regions):
        width, height = _explicit_sample_size(regions)
        if regions["calibration"].get("content_rotation_cw") in (-90, 90):
            width, height = height, width
        return width / height
    return shape[1] / shape[0]


def _project_point(matrix, point):
    return cv2.perspectiveTransform(np.float64([[point]]), matrix)[0, 0].round(3).tolist()


def _project_box(matrix, box, size, outline=None):
    if outline is None:
        left, top, right, bottom = box
        outline = [[left, top], [right, top], [right, bottom], [left, bottom]]
    corners = cv2.perspectiveTransform(np.float64([outline]), matrix)[0]
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    return [max(0, floor(x0)), max(0, floor(y0)), min(size[0], ceil(x1)), min(size[1], ceil(y1))]


class _ScreenSampleView:
    """Compare controls on the rectified display and return results in original photo pixels."""

    def __init__(self, frame, regions, size, *, blocked=None, rotation=0):
        self.to_screen, self.size = display_mapping(regions["screen_corners_px"], size, rotation)
        self.to_photo = np.linalg.inv(self.to_screen)
        self.photo_size = frame.shape[1::-1]
        self.frame = cv2.warpPerspective(frame, self.to_screen, self.size)
        if blocked is not None:
            self.frame[blocked != 0] = 32
        self.regions = {"layer": regions["layer"], "candidates": []}
        for name in ("regions", "floating_controls"):
            self.regions[name] = [_project_box(self.to_screen, box, self.size) for box in regions.get(name, ())]
        for candidate in regions["candidates"]:
            if not candidate.get("review_only"):
                outline = candidate.get("photo_outline_px")
                box = _project_box(self.to_screen, candidate["box"], self.size, outline)
                row = {**candidate, "box": box}
                if outline is not None:
                    row["photo_outline_px"] = cv2.perspectiveTransform(np.float64([outline]), self.to_screen)[0].tolist()
                if row.get("floating_control"):
                    row["center"] = _project_point(self.to_screen, row["center"])
                self.regions["candidates"].append(row)

    def sample_geometry(self, record):
        box = _project_box(self.to_screen, record["box"], self.size)
        point = _project_point(self.to_screen, record["click_point"])
        if (min(_size(box)) >= 6 and _visible(self.regions, box, point, floating=record.get("floating_control"))
                and _contains(box, point)):
            return {"box": box, "click_point": point, "size": list(self.size)}
        return None

    def restore(self, record):
        box = _project_box(self.to_photo, record["box"], self.photo_size)
        point = _project_point(self.to_photo, record["click_point"])
        return {**record, "box": box, "click_point": point} if _contains(box, point) else None


def _screen_sample_view(frame, regions):
    if _explicit_photo(regions):
        return _ScreenSampleView(frame, regions, _explicit_sample_size(regions),
                                 rotation=regions["calibration"].get("content_rotation_cw", 0))
    if (regions.get("calibration", {}).get("space") == "fixture_estimate_mm"
            and len(regions.get("screen_corners_px", ())) == 4):
        blocked = fixture_occlusions()
        return _ScreenSampleView(frame, regions, blocked.shape[::-1], blocked=blocked)
    return None


def _frame_key(frame):
    # Include shape so that equally sized buffers with different image strides
    # or dimensions cannot refer to the same annotation.
    digest = sha256(repr(frame.shape).encode())
    digest.update(np.ascontiguousarray(frame))
    return digest.hexdigest()


def _learning_frame_key(frame_key, regions):
    if not _explicit_photo(regions):
        return frame_key
    calibration = regions["calibration"]
    geometry = [regions["screen_corners_px"], calibration.get("reference_size_mm"), calibration.get("origin"),
                calibration.get("axes"), calibration.get("rectified_size_px"),
                calibration.get("content_rotation_cw", 0)]
    fingerprint = sha256(json.dumps(geometry, separators=(",", ":")).encode()).hexdigest()
    return f"{frame_key}:{fingerprint}"


def _chip(frame, box):
    return _resize_chip(frame, tuple(map(round, box)))


def _resize_chip(frame, bounds):
    left, top, right, bottom = bounds
    height, width = frame.shape[:2]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height
            and right - left >= 5 and bottom - top >= 5):
        return None
    return cv2.resize(frame[top:bottom, left:right], (_CHIP_SIZE, _CHIP_SIZE), interpolation=cv2.INTER_AREA)


def _textured(chip):
    if chip is None:
        return False
    gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
    return gray.std() >= 11 and (np.abs(gray[:, 1:].astype(np.int16) - gray[:, :-1]) > 18).mean() >= .025


def _witnesses(frame, box):
    left, top, right, bottom = box
    w, h = _size(box)
    margin = max(12, round(.72 * min(w, h)))
    gap = max(3, round(.12 * min(w, h)))
    extent = [(left - margin, top - gap, left - gap, bottom + gap),
              (right + gap, top - gap, right + margin, bottom + gap),
              (left - gap, top - margin, right + gap, top - gap),
              (left - gap, bottom + gap, right + gap, bottom + margin)]
    area = _chip(frame, box)
    if not _textured(area):
        return [], None
    parts = [(list(bounds), chip) for bounds in extent if _textured(chip := _chip(frame, bounds))]
    if not parts:
        return [], None
    serialized = np.stack([area, *(chip for _, chip in parts)])
    return [bounds for bounds, _ in parts], serialized.tobytes()


def _close_visual(reference, current, *, tolerance):
    distance = _visual_distance(reference, current)
    return distance is not None and distance <= tolerance


def _scale_box(box, factors, delta=(0, 0)):
    return [value * factors[i % 2] + delta[i % 2] for i, value in enumerate(box)]


def _candidate_boxes(regions, expected, *, allow_review=True):
    box_w, box_h = _size(expected)
    for candidate in regions["candidates"]:
        if candidate.get("visibility") not in {"page", "foreground"}:
            continue
        if not allow_review and candidate.get("review_only"):
            continue
        box = candidate["box"]
        w, h = _size(box)
        if .63 * box_w <= w <= 1.55 * box_w and .63 * box_h <= h <= 1.55 * box_h:
            yield box


def _owner_offset(regions, record):
    bounds, point = record["box"], record["click_point"]
    owners = [(min(_coverage(bounds, box), _coverage(box, bounds)), box)
              for box in _candidate_boxes(regions, bounds) if _contains(box, point)]
    if not owners:
        return [0, 0]
    support, owner = max(owners)
    if support < .55:
        return [0, 0]
    return [a - b for a, b in zip(_center(bounds), _center(owner), strict=True)]


def _visual_distance(reference, current):
    if current is None:
        return None
    difference = cv2.absdiff(reference, current).reshape(-1, 1)
    if cv2.countNonZero(cv2.compare(difference, 24, cv2.CMP_GT)) > .12 * difference.size:
        return None
    return cv2.sumElems(difference)[0] / difference.size


def _stroke_mass(chip):
    gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
    dx = cv2.absdiff(gray[:, :-1], gray[:, 1:])
    dy = cv2.absdiff(gray[:-1, :], gray[1:, :])
    return sum(cv2.sumElems(cv2.threshold(delta, 21, 0, cv2.THRESH_TOZERO)[1])[0] for delta in (dx, dy))


def _paint_is_intact(reference, chip, mass, *, near):
    if _stroke_mass(chip) >= .75 * mass:
        return True
    if not near:
        return False
    source, current = (cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY) for piece in (reference, chip))
    # An edited value can change the ink density. A faint rendering of the
    # same glyph arrangement is evidence that the control changed state.
    return float(cv2.matchTemplate(current, source, cv2.TM_CCOEFF_NORMED)[0, 0]) < .82


def _surroundings(frame, box, nearby):
    """Collect disjoint evidence from beyond the immediate button edges."""
    height, width = frame.shape[:2]
    left, top, right, bottom = box
    span = min(_size(box))
    stride = round(1.48 * span)
    first = round(.85 * span)
    witnesses = []
    for hop in (0, 1, 2, 4):
        distance = first + hop * stride
        windows = ([left - distance - stride, top, left - distance, bottom],
                   [right + distance, top, right + distance + stride, bottom],
                   [left, top - distance - stride, right, top - distance],
                   [left, bottom + distance, right, bottom + distance + stride])
        for bounds in windows:
            clipped = [max(0, bounds[0]), max(0, bounds[1]), min(width, bounds[2]), min(height, bounds[3])]
            if _coverage(bounds, clipped) < .75 or any(_coverage(clipped, prior) > .12 for prior in nearby):
                continue
            chip = _chip(frame, clipped)
            if chip is not None and (_textured(chip) or _stroke_mass(chip) >= 2400):
                witnesses.append((clipped, chip))
    return witnesses


def _sample_pixels(frame, box):
    nearby, patterns = _witnesses(frame, box)
    chip = _chip(frame, box)
    wide = _surroundings(frame, box, nearby) if _textured(chip) else []
    if wide:
        if patterns is None:
            patterns = chip.tobytes()
        patterns += np.stack([part for _, part in wide]).tobytes()
    return nearby, [bounds for bounds, _ in wide], patterns


def _aligned_shift(frame, reference, box, delta):
    """Refine a detector center against the taught raster on a small local tile."""
    l, t, r, b = [round(value) for value in box]
    height, width = frame.shape[:2]
    if not (0 <= l < r <= width and 0 <= t < b <= height):
        return delta
    x0, y0 = max(0, l - max(3, round(.045 * (r - l)))), max(0, t - max(3, round(.045 * (b - t))))
    x1, y1 = min(width, r + l - x0), min(height, b + t - y0)
    sx, sy = _CHIP_SIZE / (r - l), _CHIP_SIZE / (b - t)
    tile = cv2.resize(frame[y0:y1, x0:x1], (round(sx * (x1 - x0)), round(sy * (y1 - y0))),
                      interpolation=cv2.INTER_AREA)
    correlation = cv2.matchTemplate(tile, reference, cv2.TM_SQDIFF)
    _, _, (x, y), _ = cv2.minMaxLoc(correlation)
    return (delta[0] + round(x0 + x / sx - l), delta[1] + round(y0 + y / sy - t))


def _context_support(frame, nearby, wide, reference, factors, delta):
    distances = [_visual_distance(pattern, _chip(frame, _scale_box(bounds, factors, delta)))
                 for bounds, pattern in zip((*nearby, *wide), reference[1:], strict=True)]
    local = sum(distance is not None and distance <= 7 for distance in distances[:len(nearby)])
    strong = sum(distance is not None and distance <= 6 for distance in distances)
    return (nearby and local >= min(2, len(nearby))) or strong >= 2, strong


def _rival_at(frame, regions, record, reference, factors, delta, mass):
    box = _scale_box(record["box"], factors, delta)
    chip = _chip(frame, box)
    return (_close_visual(reference[0], chip, tolerance=5) and _stroke_mass(chip) >= .75 * mass
            and _visible(regions, box, _scale_box(record["click_point"], factors, delta),
                         floating=record.get("floating_control"))
            and _context_support(frame, record["witnesses"], record.get("wide_witnesses", []),
                                 reference, factors, delta)[0])


def _raster_gray(frame, size):
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _raster_rival(frame, regions, record, reference, factors, chosen, mass, *, read_gray=None):
    """Reject a relocated match if another credible instance survived outside the proposals."""
    if chosen != (0, 0) and _rival_at(frame, regions, record, reference, factors, (0, 0), mass):
        return True
    expected = _scale_box(record["box"], factors)
    w, h = _size(expected)
    height, width = frame.shape[:2]
    scale = min(1, 1100 / max(width, height), 64 / max(w, h))
    shape = round(w * scale), round(h * scale)
    if min(shape) < 12:
        return True
    size = round(width * scale), round(height * scale)
    gray = _raster_gray(frame, size) if read_gray is None else read_gray(size)
    x_scale, y_scale = gray.shape[1] / width, gray.shape[0] / height
    ax = min(gray.shape[1] - shape[0], max(0, round((expected[0] + chosen[0]) * x_scale)))
    ay = min(gray.shape[0] - shape[1], max(0, round((expected[1] + chosen[1]) * y_scale)))
    glyph = gray[ay:ay + shape[1], ax:ax + shape[0]].copy()
    matches = cv2.matchTemplate(gray, glyph, cv2.TM_CCOEFF_NORMED)
    # The chosen rendition is already confirmed. The downsampled image only
    # has to suggest possible rivals, whose full-size pixels decide the veto.
    rx, ry = shape[0] // 2, shape[1] // 2
    matches[max(0, ay - ry):ay + ry + 1, max(0, ax - rx):ax + rx + 1] = -1
    checked = {(0, 0)}
    guard_distance = .40 * min(w, h)
    for _ in range(16):
        _, score, _, (x, y) = cv2.minMaxLoc(matches)
        if score < .62:
            return False
        matches[max(0, y - ry):y + ry + 1, max(0, x - rx):x + rx + 1] = -1
        delta = round(x / x_scale - expected[0]), round(y / y_scale - expected[1])
        if max(abs(delta[0] - chosen[0]), abs(delta[1] - chosen[1])) <= guard_distance:
            continue
        refined = _aligned_shift(frame, reference[0], _scale_box(record["box"], factors, delta), delta)
        shifts = [(refined[0] + dx, refined[1] + dy) for dx in (0, -1, 1, -2, 2, -3, 3)
                  for dy in (0, -1, 1, -2, 2, -3, 3)]
        for rival in (delta, *shifts):
            if rival in checked:
                continue
            checked.add(rival)
            if (max(abs(rival[0] - chosen[0]), abs(rival[1] - chosen[1])) > guard_distance
                    and _rival_at(frame, regions, record, reference, factors, rival, mass)):
                return True
    return True


def _may_relocate(frame, outcome, record, patterns):
    if len(record["witnesses"]) + len(record["wide_witnesses"]) < 2:
        return False
    previous = record["prior_action"]
    if previous is not None and sum(_prior_action(item) == previous for item in outcome.active) > 1:
        return False
    reference = np.frombuffer(patterns, np.uint8).reshape((-1, _CHIP_SIZE, _CHIP_SIZE, 3))
    return not _raster_rival(frame, outcome.regions, record, reference, (1, 1), (0, 0), _stroke_mass(reference[0]))


def _recover_nearby_example(frame, regions, record, reference, factors, expected, read_chip, read_gray):
    """Recover a missed proposal at the taught position from independent pixels."""
    if record.get("floating_control"):
        return None
    nearby = record["witnesses"]
    wide = record.get("wide_witnesses", [])
    manual = _sample_origin(record) == "manual"
    witness_count = len(nearby) + len(wide)
    required = 2 if manual else max(4, ceil(.8 * witness_count))
    if witness_count < required or (not manual and max(abs(factor - 1) for factor in factors) > .03):
        return None
    reference_chip = reference[0]
    mass = _stroke_mass(reference_chip)
    delta = (0, 0)
    for refined in (False, True) if manual else (False,):
        if refined:
            delta = _aligned_shift(frame, reference_chip, expected, (0, 0))
            if delta == (0, 0):
                return None
        box = _scale_box(record["box"], factors, delta)
        point = _scale_box(record["click_point"], factors, delta)
        if not _visible(regions, box, point):
            continue
        chip = read_chip(box)
        tolerance = 3 if manual else 2
        if not _close_visual(reference_chip, chip, tolerance=tolerance) or _stroke_mass(chip) < .82 * mass:
            continue
        supported, strong = _context_support(frame, nearby, wide, reference, factors, delta)
        if not supported or strong < required:
            continue
        if _raster_rival(frame, regions, record, reference, factors, delta, mass, read_gray=read_gray):
            return None
        bounds = [round(value) for value in box]
        landing = [round(value, 2) for value in point]
        if _contains(bounds, landing):
            return {**record, "box": bounds, "click_point": landing, "match": "near", "raster_recovery": True}
    return None


def _translated_example(frame, regions, record, patterns, *, read_chip=None, read_gray=None):
    if not patterns:
        return None
    if read_chip is None:
        def read_chip(box):
            return _chip(frame, box)
    height, width = frame.shape[:2]
    source_w, source_h = record["size"]
    factors = width / source_w, height / source_h
    expected = _scale_box(record["box"], factors)
    box_w, box_h = _size(expected)
    center = _center(expected)
    anchor = _scale_box(record.get("owner_offset", [0, 0]), factors)
    origin = _sample_origin(record)
    manual = origin == "manual"
    ai_relocation = (origin == "ai" and record["function"] is not None
                     and max(abs(factor - 1) for factor in factors) <= .03)
    allow_relocation = (manual or ai_relocation) and record.get("allow_relocation", False)
    reach = .38 if manual else .20
    reference = np.frombuffer(patterns, np.uint8).reshape((-1, _CHIP_SIZE, _CHIP_SIZE, 3))
    nearby = record["witnesses"]
    wide = record.get("wide_witnesses", [])
    if len(reference) != 1 + len(nearby) + len(wide):
        return None
    move_error = 4 if manual else 3
    # An AI-labeled icon can resemble another control. Require most of the
    # surrounding details to move with it before reusing the label.
    move_witnesses = 2 if manual else max(4, ceil(.70 * (len(nearby) + len(wide))))
    offsets = {}
    boxes = _candidate_boxes(regions, expected, allow_review=manual or not allow_relocation)
    for current in boxes:
        dx, dy = (a - b for a, b in zip(_center(current), center, strict=True))
        for shift in {(round(dx), round(dy)), (round(dx + anchor[0]), round(dy + anchor[1]))}:
            offsets.setdefault(shift, []).append(current)
        if abs(dx) <= reach * box_w and abs(dy) <= reach * box_h and _coverage(expected, current) >= .65:
            offsets.setdefault((0, 0), []).append(current)

    def is_near(delta):
        return abs(delta[0]) <= reach * box_w and abs(delta[1]) <= reach * box_h

    if allow_relocation:
        for delta, owners in list(offsets.items()):
            if is_near(delta):
                continue
            box = _scale_box(record["box"], factors, delta)
            chip = read_chip(box)
            if chip is not None:
                difference = cv2.absdiff(chip, reference[0]).reshape(-1, 1)
                if cv2.sumElems(difference)[0] < 12 * difference.size:
                    aligned = _aligned_shift(frame, reference[0], box, delta)
                    if aligned != delta:
                        offsets.setdefault(aligned, []).extend(owners)
    mass = None
    matches = []
    credible = []
    ordered = offsets.items()
    if not (manual and allow_relocation):
        ordered = sorted(ordered, key=lambda entry: not is_near(entry[0]))
    for delta, owners in ordered:
        near = is_near(delta)
        # Distant points can only veto a nearby match when relocation is disabled.
        if not allow_relocation and not near and not matches:
            break
        box = _scale_box(record["box"], factors, delta)
        point = _scale_box(record["click_point"], factors, delta)
        if not any(_contains(current, point) and _coverage(box, current) >= .55 for current in owners):
            continue
        chip = read_chip(box)
        error = _visual_distance(reference[0], chip)
        if error is None or error > (5 if manual else 3):
            continue
        if mass is None:
            mass = _stroke_mass(reference[0])
        intact = (_paint_is_intact(reference[0], chip, mass, near=near) if manual
                  else _stroke_mass(chip) >= .82 * mass)
        if not intact or not _visible(regions, box, point, floating=record.get("floating_control")):
            continue
        supported, strong = _context_support(frame, nearby, wide, reference, factors, delta)
        if not supported or (not manual and strong < 2):
            continue
        credible.append(delta)
        if near or (allow_relocation and error <= move_error and strong >= move_witnesses):
            matches.append((near, error, delta, box, point))
    if not matches:
        if not any(is_near(delta) for delta in offsets):
            return _recover_nearby_example(frame, regions, record, reference, factors, expected, read_chip, read_gray)
        return None
    near, _, delta, box, point = min(
        matches, key=lambda item: (not item[0], sum(value * value for value in item[2]) if item[0] else item[1], item[2]))
    if any(max(abs(a - b) for a, b in zip(other, delta, strict=True)) > .23 * min(box_w, box_h)
           for other in credible):
        return None
    if (not manual or not near) and _raster_rival(frame, regions, record, reference, factors, delta, mass,
                                                 read_gray=read_gray):
        return None
    return {**record, "box": [round(value) for value in box], "click_point": [round(value, 2) for value in point],
            "match": "near" if near else "relocated"}


def _sample_geometry(record):
    """Inputs to visual matching, apart from the stored pixels and current frame."""
    return (_sample_origin(record), record["function"] is not None, record.get("floating_control"),
            tuple(record["size"]), tuple(record["box"]), tuple(record["click_point"]),
            tuple(record.get("owner_offset", (0, 0))), bool(record.get("allow_relocation")),
            tuple(map(tuple, record["witnesses"])), tuple(map(tuple, record.get("wide_witnesses", ()))))


def _prior_action(item):
    if item is not None:
        return [item["role"], item["function"]]
    return None


def _matches_current_state(frame, record, outcome):
    if record["function"] is not None:
        if any((item.get("selected") or item.get("disabled")) and _contains(item["box"], record["click_point"])
               and min(_coverage(record["box"], item["box"]), _coverage(item["box"], record["box"])) >= .46
               for item in outcome.labels if "box" in item):
            return False
        if outcome.regions["layer"] == "page":
            from screen_light_filter_peers import selected_light_filter_at

            if selected_light_filter_at(frame, record["box"], record["click_point"]):
                return False
    if "prior_action" not in record and record.get("match") == "near":
        return True
    previous = record.get("prior_action")
    for item in outcome.active:
        if not _same_owner(item, record):
            continue
        current = _prior_action(item)
        if _same_purpose(item["function"], record["function"], current, previous):
            continue
        if (previous is None or current[0] != previous[0]
                or not _same_purpose(current[1], previous[1], current, previous)):
            return False
    return True


def _to_mm(regions, frame):
    calibration = regions.get("calibration", {})
    if calibration.get("space") == "screen_mm":
        if (corners := regions.get("screen_corners_px")) is not None:
            to_screen, (width, height) = display_mapping(corners, calibration.get("rectified_size_px"))
            return known_screen_mm((height, width), calibration["reference_size_mm"])[0] @ to_screen
        return known_screen_mm(frame.shape, calibration["reference_size_mm"])[0]
    if calibration.get("space") != "fixture_estimate_mm":
        return None
    if "fixture_corners_px" in calibration:
        return photo_to_fixture_mm(calibration["fixture_corners_px"])
    profile = next((item for item in PROFILES if item.source == calibration.get("reference_image")), None)
    if profile is None:
        return None
    photo = cv2.getPerspectiveTransform(np.float32(regions["screen_corners_px"]), np.float32(profile.screen_corners))
    return photo_to_fixture_mm(profile.case_corners) @ photo


def _action(record, matrix):
    action = {key: record[key] for key in ("function", "box", "click_point")}
    action.update(role=record["role"], review_source="confirmed_example", review_example=record["review_example"])
    if matrix is not None:
        point = np.float64([[action["click_point"]]])
        center = cv2.perspectiveTransform(point, matrix)[0, 0].round(1).tolist()
        action.update(center_mm=center, click_point_mm=center)
    return action


def _sample_id(row_id, encoded):
    return f"{row_id}:{sha256(encoded.encode()).hexdigest()[:24]}"


def _select_examples(matches):
    selected = []
    for exact, record in sorted(matches, key=lambda item: (-_sample_priority(item[1]), not item[0])):
        priority = _sample_priority(record)
        owners = [(same_frame, peer) for same_frame, peer in matches if _same_owner(record, peer)]
        if any(_sample_priority(peer) > priority for _, peer in owners):
            continue
        peers = [(same_frame, peer) for same_frame, peer in owners if _sample_priority(peer) == priority]
        if _sample_origin(record) == "manual":
            # Preserve explicit edits for this image over older manual matches.
            if not exact and any(same_frame for same_frame, _ in peers):
                continue
            peers = [(same_frame, peer) for same_frame, peer in peers if same_frame == exact]
        if (any(peer.get("conflicted") or not _same_purpose(
                peer["function"], record["function"], peer.get("prior_action"), record.get("prior_action"))
                for _, peer in peers)
                or any(_same_owner(record, peer) for peer in selected)):
            continue
        selected.append(record)
    return selected


class ScreenUserExamples:
    """Store decisions atomically and match new frames by current evidence."""

    def __init__(self, path=None):
        if path is not None:
            self.path = Path(path)
            return
        folder = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / "ImagePosition"
        self.path = folder / "control-examples.sqlite3"
        if not self.path.exists() and _BUNDLED_EXAMPLES.is_file():
            self._install_bundled_examples()

    def _install_bundled_examples(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=self.path.parent, prefix=".examples-") as directory:
                staged = Path(directory) / self.path.name
                copyfile(_BUNDLED_EXAMPLES, staged)
                try:
                    # Publish a complete copy without replacing another process's database.
                    if os.name == "nt":
                        staged.rename(self.path)
                    else:
                        os.link(staged, self.path)
                except FileExistsError:
                    pass
        except OSError as error:
            raise OSError(f"无法初始化本机学习库 {self.path}。") from error

    @contextmanager
    def _database(self, *, create=False):
        try:
            with closing(sqlite3.connect(self.path)) as database, database:
                if create:
                    database.execute(_SCHEMA)
                    database.execute("CREATE INDEX IF NOT EXISTS examples_scope ON examples (space, layer, aspect)")
                yield database
        except sqlite3.Error as error:
            raise OSError(f"无法读取或保存本地样本库 {self.path}。") from error

    def learn_detections(self, frame, outcome, *, source=None):
        """Save named single-click detections from the current, unlearned observation."""
        annotations = ({key: row[key] for key in ("function", "box", "click_point")} for row in outcome.active
                       if not row.get("review_only") and row["role"] != "adjust_slider" and row["function"] is not None)
        return self.teach(frame, outcome, annotations, source=source, origin="detector")

    def teach(self, frame, outcome, annotations, *, source=None, origin="manual"):
        if not isinstance(origin, str) or origin not in _ORIGIN_PRIORITY:
            raise ValueError("无效的学习样本来源。")
        if origin != "manual":
            annotations = _automatic_examples(frame, outcome, annotations)
            if not annotations:
                return 0
        regions = outcome.regions
        height, width = frame.shape[:2]
        key = _learning_frame_key(_frame_key(frame), regions)
        additions = []
        explicit_photo = _explicit_photo(regions)
        screen = _screen_sample_view(frame, regions)
        geometry_key = "screen_geometry" if explicit_photo else "fixture_geometry"
        thumbnails = _thumbnails(regions) if origin == "manual" else ()
        for row in annotations:
            record = _annotation(row)
            left, top, right, bottom = record["box"] = [round(value) for value in record["box"]]
            point = record["click_point"]
            if (not 0 <= left < right <= width or not 0 <= top < bottom <= height
                    or min(right - left, bottom - top) < 6 or not _contains(record["box"], point)
                    or (right - left) * (bottom - top) > .65 * width * height):
                raise ValueError("请在原图里框住一个控件，框内须包含点击点。")
            floating = _floating_control(regions, record["box"], point)
            if not _visible(regions, record["box"], point, floating=floating):
                raise ValueError("样本点必须位于当前可见的操作层。")
            if floating is not None:
                record["floating_control"] = floating
            if _on_thumbnail_marker(thumbnails, record):
                raise ValueError("照片框的点击点应放在照片本体上。")
            if any(_same_owner(record, item) for item, _ in additions):
                raise ValueError("一批补标中同一个控件只需确认一次。")
            previous = next((item for item in outcome.active if _same_owner(item, record)), None)
            if record["function"] is not None and previous is not None and previous["role"] == "adjust_slider":
                raise ValueError("可学习的样本须为单击控件。")
            role = (previous["role"] if previous is not None and previous["function"] == record["function"]
                    else "labeled_control")
            geometry = screen.sample_geometry(record) if screen is not None else None
            sample_frame, sample_outcome = frame, outcome
            if geometry is not None:
                witnesses, wide, patterns = _sample_pixels(screen.frame, geometry["box"])
                if patterns is not None:
                    sample_frame = screen.frame
                    sample_outcome = replace(outcome, regions=screen.regions)
                    record[geometry_key] = geometry
            if geometry_key not in record:
                geometry = record
                witnesses, wide, patterns = ([], [], None) if explicit_photo else _sample_pixels(frame, record["box"])
            record.update(witnesses=witnesses, wide_witnesses=wide, size=[width, height], role=role, origin=origin,
                          owner_offset=_owner_offset(sample_outcome.regions, geometry), prior_action=_prior_action(previous))
            can_move = origin == "manual" or (
                origin == "ai" and record["function"] is not None and len(witnesses) + len(wide) >= 4)
            record["allow_relocation"] = can_move and _may_relocate(
                sample_frame, sample_outcome, {**record, **geometry}, patterns)
            if source:
                record["source"] = Path(source).name
            additions.append((record, patterns))
        if not additions:
            raise ValueError("请至少确认一条样本。")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        priority = _ORIGIN_PRIORITY[origin]
        scope = _space(regions)
        aspect = _learning_aspect(regions, frame.shape)
        written = 0
        with self._database(create=True) as database:
            database.execute("BEGIN IMMEDIATE")
            prior = {box: json.loads(encoded) for box, encoded in database.execute(
                "SELECT box, geometry FROM examples WHERE frame_key = ? ORDER BY rowid", (key,))}
            for record, patterns in additions:
                owners = [(box, peer) for box, peer in prior.items() if _same_owner(record, peer)]
                if any(_sample_priority(peer) > priority for _, peer in owners):
                    continue
                equals = [(box, peer) for box, peer in owners if _sample_priority(peer) == priority]
                if origin != "manual" and equals:
                    if (not any(peer.get("conflicted") for _, peer in equals)
                            and any(not _same_purpose(
                                peer["function"], record["function"], peer.get("prior_action"), record.get("prior_action"))
                                for _, peer in equals)):
                        box, peer = equals[0]
                        peer.update(conflicted=True, revision=uuid4().hex)
                        database.execute("UPDATE examples SET geometry = ? WHERE frame_key = ? AND box = ?",
                                         (json.dumps(peer, ensure_ascii=False), key, box))
                        written += 1
                    continue
                for box, _ in owners:
                    database.execute("DELETE FROM examples WHERE frame_key = ? AND box = ?", (key, box))
                    del prior[box]
                box = json.dumps(record["box"])
                record["revision"] = uuid4().hex
                database.execute("INSERT OR REPLACE INTO examples VALUES (?, ?, ?, ?, ?, ?, ?)",
                                 (key, scope, regions["layer"], aspect,
                                  box, json.dumps(record, ensure_ascii=False), patterns))
                prior[box] = record
                written += 1
        return written

    def forget(self, frame):
        if not self.path.is_file():
            return 0
        key = _frame_key(frame)
        with self._database() as database:
            return database.execute("DELETE FROM examples WHERE frame_key = ? OR frame_key LIKE ?",
                                    (key, f"{key}:%")).rowcount

    def forget_sample(self, sample_id):
        if not self.path.is_file() or not isinstance(sample_id, str):
            return 0
        head, separator, fingerprint = sample_id.partition(":")
        if not separator or not head.isascii() or not head.isdecimal() or len(head) > 18 or len(fingerprint) != 24:
            return 0
        row_id = int(head)
        with self._database() as database:
            database.execute("BEGIN IMMEDIATE")
            row = database.execute("SELECT geometry FROM examples WHERE rowid = ?", (row_id,)).fetchone()
            if row is None or _sample_id(row_id, row[0]) != sample_id:
                return 0
            return database.execute("DELETE FROM examples WHERE rowid = ?", (row_id,)).rowcount

    def apply(self, frame, outcome, *, prefer_exact=False):
        if not self.path.is_file():
            return outcome
        regions = outcome.regions
        explicit_photo = _explicit_photo(regions)
        scope = _space(regions)
        aspect = _learning_aspect(regions, frame.shape)
        with self._database() as database:
            rows = database.execute(
                "SELECT rowid, frame_key, geometry, samples FROM examples"
                " WHERE space = ? AND layer = ? AND aspect BETWEEN ? AND ? ORDER BY rowid",
                (scope, regions["layer"], .965 * aspect, 1.035 * aspect)).fetchall()
        if not rows:
            return outcome

        @lru_cache(maxsize=512)
        def image_chip(bounds):
            return _resize_chip(frame, bounds)

        @lru_cache(maxsize=1)
        def screen_view():
            return _screen_sample_view(frame, regions)

        @lru_cache(maxsize=512)
        def screen_chip(bounds):
            return _resize_chip(screen_view().frame, bounds)

        @lru_cache(maxsize=8)
        def rival_gray(on_screen, size):
            return _raster_gray(screen_view().frame if on_screen else frame, size)

        visual_matches = {}

        def translate(record, samples):
            if not samples:
                return None
            screen = None
            geometry_key = "screen_geometry" if explicit_photo else "fixture_geometry"
            if geometry_key in record:
                screen = screen_view()
                if screen is None:
                    return None
                record = {**record, **record[geometry_key]}
            elif explicit_photo:
                return None
            geometry = (screen is not None, _sample_geometry(record))
            cached = visual_matches.get(geometry)
            if cached is not None and samples == cached[0]:
                found = None if cached[1] is None else {**record, **cached[1]}
            else:
                sample_frame = frame if screen is None else screen.frame
                sample_regions = regions if screen is None else screen.regions
                chip = image_chip if screen is None else screen_chip
                found = _translated_example(sample_frame, sample_regions, record, samples,
                                            read_chip=lambda bounds: chip(tuple(map(round, bounds))),
                                            read_gray=lambda size: rival_gray(screen is not None, size))
                if cached is not None or len(visual_matches) < 512:
                    position = None if found is None else {
                        key: found[key] for key in ("box", "click_point", "match", "raster_recovery") if key in found}
                    visual_matches[geometry] = samples, position
            if found is not None and screen is not None:
                return screen.restore(found)
            return found

        frame_key = _frame_key(frame)
        key = _learning_frame_key(frame_key, regions)
        matches = []
        thumbnails = _thumbnails(regions)
        if prefer_exact:
            exact_rows = [row for row in rows if row[1] == key]
            for _, _, encoded, _ in exact_rows:
                sample = json.loads(encoded)
                if (_sample_origin(sample) == "manual" and sample["function"] is not None
                        and not _on_thumbnail_marker(thumbnails, sample)
                        and _visible(regions, sample["box"], sample["click_point"],
                                     floating=sample.get("floating_control"))):
                    rows = exact_rows
                    break
        for row_id, source, encoded, samples in rows:
            exact = key == source
            if explicit_photo and not exact and source.startswith(f"{frame_key}:"):
                continue
            record = json.loads(encoded)
            if not exact:
                # Visual results can be reused, but every saved purpose is checked independently.
                record = translate(record, samples)
            if record is None or (record.get("raster_recovery") and record["function"] is None):
                continue
            origin = _sample_origin(record)
            if _on_thumbnail_marker(thumbnails, record):
                continue
            # Older automatic samples can describe artwork inside a thumbnail.
            # A measured thumbnail owns that area in the current image.
            if origin != "manual" and any(
                    _coverage(record["box"], tile["box"]) >= .75 and not _same_owner(record, tile)
                    for tile in thumbnails):
                continue
            if (not exact or origin != "manual") and not record.get("conflicted"):
                if not _matches_current_state(frame, record, outcome):
                    continue
            if _visible(regions, record["box"], record["click_point"], floating=record.get("floating_control")):
                record["review_example"] = {
                    "id": _sample_id(row_id, encoded), "source": record.get("source", "较早保存的截图"),
                    "match": "exact" if exact else record.get("match", "near"), "origin": origin,
                }
                if not exact and record.get("raster_recovery"):
                    record["review_example"]["raster_recovery"] = True
                matches.append((exact, record))
        selected = _select_examples(matches)
        if not selected:
            return outcome
        matrix = _to_mm(regions, frame)
        active = [item for item in outcome.active if not any(_same_owner(item, peer) for peer in selected)]
        review = [item for item in outcome.review if not any(_same_owner(item, peer) for peer in selected)]
        entries = [item for item in outcome.entries if not any(_same_owner(item, peer) for peer in selected)]
        confirmed = [_action(record, matrix) for record in selected if record["function"] is not None]
        fields = ("function", "box", "click_point", "floating_control")
        used = tuple({**record["review_example"], **{key: record[key] for key in fields if key in record}}
                     for record in selected)
        if matrix is not None:
            points = np.float64([[row["click_point"] for row in used]])
            millimeters = cv2.perspectiveTransform(points, matrix)[0].round(1).tolist()
            for row, point in zip(used, millimeters, strict=True):
                row["center_mm"] = point
        return replace(outcome, active=active + confirmed, review=review, entries=entries + confirmed,
                       review_examples=used)
