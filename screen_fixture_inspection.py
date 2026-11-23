"""Analyze registered fixture photographs and project results onto the source image."""

from dataclasses import replace
from math import ceil, floor
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from screen_boxes import _box_overlap, inspect_boxes
from screen_floating_ring import translucent_ring_boxes
from screen_photo_geometry import (ExtractionInput, display_mapping, display_size, fixture_occlusions,
                                   prepare_camera_screen, validate_photo_outline)
from screen_registration import FIXTURE_SIZE_MM, ScreenRegistrar, known_screen_mm, rectify_screen, validate_screen_size_mm


_IMAGE_DIR = Path(__file__).resolve().parent / "img"


def _project(matrix, points, decimals):
    mapped = cv2.perspectiveTransform(np.asarray([points], dtype=np.float64), matrix)[0]
    return np.round(mapped, decimals).tolist()


def _rotation_mapping(shape, code):
    """Map upright pixel centers back to the rectified fixture view."""
    height, width = shape
    if code == cv2.ROTATE_90_CLOCKWISE:
        return np.float64([[0, 1, 0], [-1, 0, height - 1], [0, 0, 1]]), 90
    if code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return np.float64([[0, -1, width - 1], [1, 0, 0], [0, 0, 1]]), -90
    return None, 0


def _add_floating_review(outcome, frame):
    regions = outcome.regions
    if regions["layer"] != "page" or regions["regions"]:
        return
    candidates = regions["candidates"]
    known = [row["box"] for row in candidates] + [row["box"] for row in outcome.active]
    identity = max((row["id"] for row in candidates), default=0)
    for box in translucent_ring_boxes(frame):
        left, top, right, bottom = box
        center = [(left + right) / 2, (top + bottom) / 2]
        if any(_box_overlap(box, earlier) >= .40 for earlier in known):
            continue
        if any(pane[0] <= center[0] < pane[2] and pane[1] <= center[1] < pane[3]
               for pane in regions.get("floating_controls", ())):
            continue
        identity += 1
        known.append(box)
        candidates.append({"id": identity, "box": box, "center": center, "visibility": "page", "review_only": True})


def _recover_rotated_back(frame, blocked, outcome, projection):
    if outcome.regions["layer"] != "page" or outcome.regions.get("regions") or any(
            row["role"] == "back" for row in outcome.active):
        return outcome
    from screen_fixture_gallery_orientation import gallery_back_box

    if (box := gallery_back_box(frame, blocked=blocked)) is None:
        from screen_fixture_settings_orientation import settings_back_box

        if (box := settings_back_box(frame, blocked=blocked)) is None:
            return outcome
    left, top, right, bottom = box
    point = [(left + right) / 2, (top + bottom) / 2]
    if (not projection.is_visible(box, point)
            or any(_box_overlap(box, pane) for pane in outcome.regions.get("floating_controls", ()))):
        return outcome
    action = {"role": "back", "function": "返回", "preview_text": "返回", "box": list(box),
              "click_point": point, "fixed_center": True}
    return replace(outcome, active=[*outcome.active, action], entries=[*outcome.entries, action])


class _PhotoProjection:
    def __init__(self, to_photo, to_mm, occluded, photo_shape, *, turned=False):
        self.to_photo = to_photo
        self.to_mm = to_mm
        self.height, self.width = occluded.shape
        self.photo_height, self.photo_width = photo_shape
        self.occluded = occluded
        self.turned = turned

    def outline(self, box):
        left, top, right, bottom = box
        corners = [[left, top], [right, top], [right, bottom], [left, bottom]]
        return _project(self.to_photo, corners, 2)

    def bounds(self, outline):
        xs, ys = zip(*outline, strict=True)
        return [max(0, floor(min(xs))), max(0, floor(min(ys))),
                min(self.photo_width, ceil(max(xs))), min(self.photo_height, ceil(max(ys)))]

    def is_visible(self, box, center):
        x, y = map(round, center)
        if not (0 <= x < self.width and 0 <= y < self.height) or self.occluded[y, x]:
            return False
        left, top, right, bottom = map(int, box)
        crop = self.occluded[max(0, top):min(self.height, bottom), max(0, left):min(self.width, right)]
        blocked = cv2.countNonZero(crop) if crop.size else 0
        if blocked < max(4, .025 * crop.size):
            return True
        if not self.turned or blocked >= .05 * crop.size:
            return False
        # On a landscape screen a long tile can touch the holder at its side.
        # Keep its evidenced point only when the area around the point is clear.
        radius = max(3, round(.12 * min(right - left, bottom - top)))
        touch = self.occluded[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
        return cv2.countNonZero(touch) == 0

    def item(self, row, box_key, center_key, *, keep_partial=False):
        center = row[center_key]
        review_only = not self.is_visible(row[box_key], center)
        if review_only:
            x, y = map(round, center)
            if not keep_partial or not (0 <= x < self.width and 0 <= y < self.height) or self.occluded[y, x]:
                return None
            left, top, right, bottom = map(int, row[box_key])
            visible = self.occluded[max(0, top):min(self.height, bottom), max(0, left):min(self.width, right)]
            if not visible.size or cv2.countNonZero(visible) > .35 * visible.size:
                return None
        outline = self.outline(row[box_key])
        result = {**row, box_key: self.bounds(outline), center_key: _project(self.to_photo, [center], 2)[0],
                  "photo_outline_px": outline, "screen_fraction": [center[0] / max(1, self.width - 1),
                                                                    center[1] / max(1, self.height - 1)]}
        if "context_box" in row:
            result["context_box"] = self.bounds(self.outline(row["context_box"]))
        if (marker := result.pop("selection_marker", None)) is not None:
            a, t, b, d = marker["box"]
            if self.is_visible(marker["box"], ((a + b) / 2, (t + d) / 2)):
                marker_outline = self.outline(marker["box"])
                result["selection_marker"] = {**marker, "box": self.bounds(marker_outline), "polygon": marker_outline}
        if review_only:
            result["review_only"] = True
        if self.to_mm is not None:
            left, top, right, bottom = row[box_key]
            result["center_mm"] = _project(self.to_mm, [center], 1)[0]
            result["outline_mm"] = _project(self.to_mm, [[left, top], [right, top], [right, bottom], [left, bottom]], 1)
        return result

    def actions(self, items):
        projected = []
        for row in items:
            if row.get("review_only"):
                continue
            if (item := self.item(row, "box", "click_point")) is not None:
                if "center_mm" in item:
                    item["click_point_mm"] = item["center_mm"]
                for key in ("slider_thumb_box", "slider_track_box", "component_artwork", "component_caption",
                            "component_box"):
                    if key in row:
                        item[key] = self.bounds(self.outline(row[key]))
                if "component_controls" in row:
                    item["component_controls"] = [self.bounds(self.outline(box)) for box in row["component_controls"]]
                if evidence := row.get("component_evidence"):
                    left, top, right, bottom = evidence["caption"]
                    if self.is_visible(evidence["caption"], ((left + right) / 2, (top + bottom) / 2)):
                        item["component_evidence"] = {key: self.bounds(self.outline(box)) for key, box in evidence.items()}
                    else:
                        item.pop("component_evidence")
                if (count := row.get("engagement_count_box")) is not None:
                    l, t, r, b = count
                    if self.is_visible(count, ((l + r) / 2, (t + b) / 2)):
                        item["engagement_count_box"] = self.bounds(self.outline(count))
                    else:
                        item.pop("engagement_count_box", None)
                        item.pop("context_box", None)
                projected.append(item)
        return projected

    def labels(self, rows):
        labels = []
        for row in rows:
            left, top, right, bottom = row["line_box"]
            center = [(left + right) / 2, (top + bottom) / 2]
            if (item := self.item({**row, "center": center}, "line_box", "center")) is not None:
                if "box" in row:
                    item["box"] = self.bounds(self.outline(row["box"]))
                labels.append(item)
        return labels

    def regions(self, source, corners, calibration):
        regions = {**source, "image_size": [self.photo_width, self.photo_height]}
        for name in ("regions", "floating_controls"):
            if name in regions:
                regions[name] = [self.bounds(self.outline(box)) for box in regions[name]]
        for name in ("detected_close", "recording_stop", "comment_input"):
            if (box := regions.get(name)) is not None:
                regions[name] = self.bounds(self.outline(box))
        regions["candidates"] = [item for row in source["candidates"]
                                 if (item := self.item(row, "box", "center", keep_partial=True)) is not None]
        regions["screen_corners_px"] = corners.astype(float).round(2).tolist()
        regions["calibration"] = calibration
        return regions


class LocalImageInspector:
    """Use the local detector on either a screenshot or a registered camera frame."""

    def __init__(self, detector, *, font_index=None, image_dir=_IMAGE_DIR, screen_size_mm=None,
                 phone_photo=False, screen_corners=None):
        self.detector = detector
        self.font_index = font_index
        self.set_calibration(screen_corners=screen_corners, screen_size_mm=screen_size_mm, phone_photo=phone_photo)
        try:
            self.registrar = ScreenRegistrar(image_dir)
        except (OSError, ValueError, cv2.error):
            self.registrar = None

    def set_calibration(self, *, screen_corners, screen_size_mm, phone_photo):
        self._screen_size_mm = validate_screen_size_mm(screen_size_mm)
        self._phone_photo = phone_photo
        self._screen_corners = screen_corners

    def inspect(self, image, *, regions_only=False):
        start = perf_counter()
        registration = (self.registrar.register(image) if self.registrar is not None and self._screen_corners is None
                        else None)
        corners = self._screen_corners
        if registration is None and corners is None and self._phone_photo:
            raise ValueError("这张照片未匹配到已标定的手机夹具，请指定原照片里的屏幕四角。")
        registered = perf_counter()
        if registration is None and corners is None:
            outcome = inspect_boxes(image, self.detector, font_index=self.font_index, regions_only=regions_only)
            if self._screen_size_mm is not None:
                to_mm, calibration = known_screen_mm(image.shape, self._screen_size_mm)

                def calibrated_actions(rows):
                    result = []
                    for row in rows:
                        point = _project(to_mm, [row["click_point"]], 1)[0]
                        result.append({**row, "center_mm": point, "click_point_mm": point})
                    return result

                outcome = replace(outcome, regions={**outcome.regions, "calibration": calibration},
                                  active=calibrated_actions(outcome.active), review=calibrated_actions(outcome.review),
                                  entries=calibrated_actions(outcome.entries))
            return replace(outcome, timing_ms={**outcome.timing_ms,
                           "registration": round((registered - start) * 1000, 3),
                           "total": round((perf_counter() - start) * 1000, 3)})
        if registration is not None:
            corners = registration.screen_corners
            frame, to_photo, to_mm = rectify_screen(image, registration)
            occluded = fixture_occlusions()
            calibration = {"space": "fixture_estimate_mm", "origin": "fixture_top_left",
                           "reference_size_mm": list(FIXTURE_SIZE_MM),
                           "registration": registration.method, "reference_image": registration.profile.source,
                           "fixture_corners_px": registration.case_corners.astype(float).round(2).tolist(),
                           "axes": "x_right_y_down", "robot_motion_ready": False}
        else:
            corners = validate_photo_outline(corners, image.shape)
            corners = corners.astype(float).round(2).astype(np.float32)
            size = display_size(corners, self._screen_size_mm)
            to_screen, size = display_mapping(corners, size)
            frame = cv2.warpPerspective(image, to_screen, size)
            to_photo = np.linalg.inv(to_screen)
            occluded = np.zeros(frame.shape[:2], np.uint8)
            to_mm = None
            calibration = {"space": "original_image_px", "origin": "image_top_left", "registration": "photo_screen_corners",
                           "robot_motion_ready": False}
            if self._screen_size_mm is not None:
                to_mm, calibration = known_screen_mm(frame.shape, self._screen_size_mm)
                calibration.update(registration="photo_screen_corners", rectified_size_px=list(size))
        ready = perf_counter()
        from screen_fixture_orientation import fixture_orientation

        direction = fixture_orientation(frame, index=self.font_index, blocked=occluded if registration is not None else None)
        undo, angle = _rotation_mapping(frame.shape[:2], direction)
        if undo is not None:
            frame = cv2.rotate(frame, direction)
            occluded = cv2.rotate(occluded, direction)
            to_photo = to_photo @ undo
            if to_mm is not None:
                to_mm = to_mm @ undo
        oriented = perf_counter()
        proposals = prepare_camera_screen(frame)
        prepared = perf_counter()
        projection = _PhotoProjection(to_photo, to_mm, occluded, image.shape[:2], turned=bool(angle))
        outcome = inspect_boxes(frame, self.detector, font_index=self.font_index, proposal_frame=proposals,
                                proposal_visibility=projection.is_visible, regions_only=regions_only)
        if angle:
            outcome = _recover_rotated_back(frame, occluded, outcome, projection)
        _add_floating_review(outcome, proposals)
        processed = perf_counter()
        extraction_input = None
        if (bounds := outcome.regions.pop("proposal_input_bounds", None)) is not None:
            left, top, right, bottom = bounds
            offset = np.float64([[1, 0, left], [0, 1, top], [0, 0, 1]])
            extraction_input = ExtractionInput(proposals[top:bottom, left:right], to_photo @ offset)
        regions = projection.regions(outcome.regions, corners, calibration)
        if angle:
            regions["calibration"]["content_rotation_cw"] = angle
        result = replace(outcome, regions=regions, extraction_input=extraction_input,
                         active=projection.actions(outcome.active), review=projection.actions(outcome.review),
                         labels=projection.labels(outcome.labels), entries=projection.actions(outcome.entries))
        finished = perf_counter()
        return replace(result, timing_ms={**outcome.timing_ms, "registration": round((registered - start) * 1000, 3),
                       "rectification": round((ready - registered) * 1000, 3),
                       "orientation": round((oriented - ready) * 1000, 3),
                       "preprocessing": round((prepared - oriented) * 1000, 3),
                       "projection": round((finished - processed) * 1000, 3),
                       "total": round((finished - start) * 1000, 3)})
