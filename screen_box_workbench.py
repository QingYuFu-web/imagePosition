"""Adapt local pixel detections to the existing desktop preview."""

from copy import deepcopy
from dataclasses import replace
from threading import Lock
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from screen_boxes import VisualControlDetector
from screen_fixture_inspection import LocalImageInspector
from screen_font_glyphs import FontGlyphIndex
from screen_page_handoff import build_page_analysis
from screen_registration import validate_screen_size_mm
from screen_scene import analyze_screen
from screen_user_examples import ScreenUserExamples


def _point(identity, kind, label, box, width, height, *, observed, action=False):
    left, top, right, bottom = box
    if action or "center_mm" in observed:
        x, y = observed.get("click_point", observed.get("center"))
    else:
        x, y = (left + right) / 2, (top + bottom) / 2
    outline = observed.get("photo_outline_px", [[left, top], [right, top], [right, bottom], [left, bottom]])
    return {"id": identity, "type": kind, "label": label,
            "photo_outline_px": outline, "photo_center_px": [x, y],
            "screen_fraction": observed.get("screen_fraction", [x / width, y / height]),
            **({"review_only": True} if observed.get("review_only") else {}),
            **{key: observed[key] for key in ("center_mm", "outline_mm") if key in observed},
            "geometry": {"method": "detector_proposal" if not action and kind == "control_candidate" else "pixel_edges"}}


class LocalBoxWorkbench:
    """Reuse the same detector and font index for successive opened screenshots."""

    def __init__(self, *, screen_size_mm=None, examples_path=None, phone_photo=False, screen_corners=None):
        self._calibration = self._calibration_options(screen_corners, screen_size_mm, phone_photo)
        self.inspector = LocalImageInspector(VisualControlDetector(), font_index=FontGlyphIndex(),
                                             **self._calibration)
        self.examples = ScreenUserExamples(examples_path)
        self._calibration_lock = Lock()
        self._calibration_revision = 0
        self._last_image = None
        self._detector_learning_pending = False

    @staticmethod
    def _calibration_options(screen_corners, screen_size_mm, phone_photo):
        corners = (tuple(map(tuple, np.asarray(screen_corners, dtype=float).reshape(4, 2)))
                   if screen_corners is not None else None)
        return {"screen_corners": corners, "screen_size_mm": validate_screen_size_mm(screen_size_mm),
                "phone_photo": bool(phone_photo)}

    def set_calibration(self, *, screen_corners=None, screen_size_mm=None, phone_photo=False):
        """Update the complete calibration and return the revision used by later views."""
        calibration = self._calibration_options(screen_corners, screen_size_mm, phone_photo)
        with self._calibration_lock:
            if calibration != self._calibration:
                self.inspector.set_calibration(**calibration)
                self._calibration = calibration
                self._calibration_revision += 1
                self._last_image = None
                self._detector_learning_pending = False
            return self._calibration_revision

    def _has_frame(self, frame, *, regions_only=None):
        return (self._last_image is not None and self._last_image[0].dtype == frame.dtype
                and (regions_only is None or self._last_image[2] == regions_only)
                and np.array_equal(self._last_image[0], frame))

    def teach(self, frame, annotations, *, source=None, origin="manual", calibration_revision=0):
        """Save an annotation from the analysis view's current calibration."""
        with self._calibration_lock:
            if calibration_revision != self._calibration_revision:
                raise ValueError("屏幕标定已变化，请重新分析当前图片，再确认样本。")
            if not self._has_frame(frame):
                raise ValueError("请先分析当前图片，再确认样本。")
            observation = self._last_image[1]
            return self.examples.teach(frame, observation, annotations, source=source, origin=origin)

    def forget(self, frame):
        with self._calibration_lock:
            count = self.examples.forget(frame)
            if count and self._has_frame(frame):
                self._detector_learning_pending = False
            return count

    def forget_sample(self, sample_id):
        with self._calibration_lock:
            count = self.examples.forget_sample(sample_id)
            if count:
                self._detector_learning_pending = False
            return count

    def analyze(self, frame, *, learn_detector=False, source=None, regions_only=False):
        """Analyze a frame with full labels or the optional faster region pass."""
        start = perf_counter()
        with self._calibration_lock:
            if self._has_frame(frame, regions_only=regions_only):
                outcome = deepcopy(self._last_image[1])
                elapsed = round((perf_counter() - start) * 1000, 3)
                outcome = replace(outcome, timing_ms={**dict.fromkeys(outcome.timing_ms, 0.),
                                                      "reuse": elapsed, "total": elapsed})
            else:
                outcome = self.inspector.inspect(frame, regions_only=regions_only)
                self._last_image = (frame.copy(), deepcopy(outcome), regions_only)
                self._detector_learning_pending = True
            began = perf_counter()
            detector_samples_saved = 0
            if learn_detector and self._detector_learning_pending:
                detector_samples_saved = self.examples.learn_detections(frame, outcome, source=source)
                self._detector_learning_pending = False
            calibration_revision = self._calibration_revision
        observation = outcome
        if self.examples.path.is_file():
            outcome = self.examples.apply(frame, outcome, prefer_exact=regions_only)
            elapsed = round((perf_counter() - began) * 1000, 3)
            outcome = replace(outcome, timing_ms={**outcome.timing_ms, "examples": elapsed,
                                                  "total": round(outcome.timing_ms["total"] + elapsed, 3)})
        page = build_page_analysis(frame, observation, confirmed_examples=outcome.review_examples)
        height, width = frame.shape[:2]
        actions = [{**item, **_point(f"action_{i}", "control_candidate", item["function"], item["box"],
                                     width, height, observed=item, action=True)}
                   for i, item in enumerate(outcome.active, 1)]
        review = [_point(f"review_{i}", "control_candidate", f"待复核 · {item['function']}", item["box"],
                         width, height, observed=item)
                  for i, item in enumerate(outcome.review, 1)]
        proposed = [_point(f"box_{row['id']}", "control_candidate", "候选控件", row["box"],
                           width, height, observed=row)
                    for row in outcome.regions["candidates"]]
        labels = [_point(f"text_{i}", "text", row["text"], row["line_box"], width, height, observed=row)
                  for i, row in enumerate(outcome.labels, 1)]
        calibration = outcome.regions.get("calibration", {"space": "original_image_px", "origin": "image_top_left"})
        result = {"calibration": calibration, "image_size": [width, height], "timing_ms": outcome.timing_ms,
                  "engines": {"yolo": {"status": "ok"}, "opencv": {"status": "ok"}, "font": {"status": "ok"}},
                  "keypoints": actions + review + proposed + labels, "page_analysis": page}
        if outcome.review_examples:
            result["review_examples"] = list(outcome.review_examples)
        if "screen_corners_px" in outcome.regions:
            result["screen_corners_px"] = outcome.regions["screen_corners_px"]
        if outcome.regions["layer"] != "page":
            result["active_layer"] = {"kind": outcome.regions["layer"], "regions_px": outcome.regions["regions"]}
        result["screen_context"] = analyze_screen(result)
        if outcome.regions["layer"] == "voice_recording":
            result["screen_context"]["recording_gesture"] = True
        return result, SimpleNamespace(calibration=calibration, calibration_revision=calibration_revision, review=outcome.review,
                                       extraction_input=outcome.extraction_input,
                                       detector_samples_saved=detector_samples_saved), actions
