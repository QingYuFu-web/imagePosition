"""Register nearby camera frames to one of the measured white fixtures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

import cv2
import numpy as np

from screen_fixture_edges import bezel_matches, bezel_reference, screen_edges_match


def validate_screen_size_mm(size):
    if size is not None and (len(size) != 2 or not all(isfinite(value) and value > 0 for value in size)):
        raise ValueError("屏幕宽和高必须是有限的正数，单位为毫米")
    return tuple(size) if size is not None else None


def known_screen_mm(shape, size_mm):
    """Map full-screen screenshot pixels to a measured screen."""
    height, width = shape[:2]
    mm_width, mm_height = size_mm
    transform = np.diag([mm_width / max(1, width - 1), mm_height / max(1, height - 1), 1])
    calibration = {"space": "screen_mm", "origin": "screen_top_left", "axes": "x_right_y_down",
                   "registration": "full_image_known_screen_size", "reference_size_mm": list(size_mm),
                   "robot_motion_ready": False}
    return transform, calibration


@dataclass(frozen=True)
class Profile:
    source: str
    screen_corners: tuple[tuple[int, int], ...]
    case_corners: tuple[tuple[int, int], ...]


PROFILES = (
    Profile(
        "img_v3_0217j_5a73c7b1-537d-4065-aa7c-b3c275bed6ix.jpg",
        ((270, 259), (922, 251), (917, 1648), (278, 1650)),
        ((91, 125), (1015, 114), (1008, 1783), (104, 1789)),
    ),
    Profile(
        "img_v3_0217j_73cd89a4-9284-4f5e-81ae-2187796b29ix.jpg",
        ((268, 247), (917, 240), (920, 1625), (287, 1632)),
        ((78, 128), (1011, 122), (990, 1795), (86, 1798)),
    ),
)


FIXTURE_SIZE_MM = (90, 165)


def photo_to_fixture_mm(case_corners):
    """Map the four ordered corners of the white fixture to its measured plane."""
    corners = np.asarray(case_corners, dtype=np.float32)
    if (corners.shape != (4, 2) or not np.isfinite(corners).all()
            or not cv2.isContourConvex(corners) or cv2.contourArea(corners, oriented=True) <= 1):
        return None
    width, height = FIXTURE_SIZE_MM
    return cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [width, 0], [width, height], [0, height]]))


@dataclass(frozen=True)
class Registration:
    profile: Profile
    screen_corners: np.ndarray
    case_corners: np.ndarray
    method: str
    inliers: int | None = None
    residual_px: float | None = None


def rectify_screen(image, registration, size=(640, 1376)):
    """Warp the display and retain mappings to the photograph and fixture millimeters."""
    width, height = size
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    screen = np.float32(registration.screen_corners)
    to_screen = cv2.getPerspectiveTransform(screen, corners)
    to_photo = cv2.getPerspectiveTransform(corners, screen)
    photo_to_mm = photo_to_fixture_mm(registration.case_corners)
    if photo_to_mm is None:
        raise ValueError("白色夹具的四角无法用于毫米标定。")
    return cv2.warpPerspective(image, to_screen, size), to_photo, photo_to_mm @ to_photo


@dataclass(frozen=True)
class _Reference:
    profile: Profile
    points: np.ndarray
    descriptors: np.ndarray
    scale: float
    on_fixture: np.ndarray
    bezel: tuple


def _image_probe(image: np.ndarray) -> bytes:
    """Cheap lookup for the exact path; the full image is compared before use."""
    return image[4::47, 7::43].tobytes()


class ScreenRegistrar:
    """Identify the exact reference or align stable scenery around the screen.

    Nearby frames can include a moderate change of viewing angle.
    Returning None means the fixture alignment could not be established.
    """

    def __init__(self, image_dir: Path):
        self._orb = cv2.ORB_create(nfeatures=550, fastThreshold=12, edgeThreshold=15)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._references = []
        self._exact = {}
        for profile in PROFILES:
            path = Path(image_dir) / profile.source
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape != (1920, 1080, 3):
                raise ValueError(f"Invalid fixture reference: {path}")
            self._exact[_image_probe(image)] = (profile, image)
            points, descriptors, scale = self._features(image)
            if descriptors is None:
                raise ValueError(f"No fixture features in {path}")
            fixture = np.float32(profile.case_corners) * scale
            on_fixture = np.array([cv2.pointPolygonTest(fixture, tuple(point), False) >= 0 for point in points])
            bezel = bezel_reference(image, np.float32(profile.screen_corners))
            self._references.append(_Reference(profile, points, descriptors, scale, on_fixture, bezel))

    def _features(self, image: np.ndarray):
        height, width = image.shape[:2]
        scale = min(1.0, 540 / width, 960 / height)
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mask = np.full((h, w), 255, np.uint8)
        # Keep the fixture and bezel, avoiding content and reflections in the display.
        mask[round(h * .109):round(h * .87), round(w * .218):round(w * .875)] = 0
        keypoints, descriptors = self._orb.detectAndCompute(gray, mask)
        return np.asarray([point.pt for point in keypoints], np.float32), descriptors, scale

    def register(self, image: np.ndarray) -> Registration | None:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            return None
        height, width = image.shape[:2]
        if width < 240 or not 1.6 <= height / width <= 2.0:
            return None
        if image.shape == (1920, 1080, 3):
            exact = self._exact.get(_image_probe(image))
            if exact is not None and cv2.norm(image, exact[1], cv2.NORM_INF) == 0:
                profile = exact[0]
                return Registration(
                    profile, np.float32(profile.screen_corners), np.float32(profile.case_corners),
                    "exact_reference",
                )

        points, descriptors, scale = self._features(image)
        if descriptors is None or len(descriptors) < 20:
            return None
        candidates = []
        for reference in self._references:
            candidate = self._align(reference, points, descriptors, scale, width, height)
            if candidate is None:
                continue
            if screen_edges_match(image, candidate.screen_corners):
                candidates.append(candidate)
            elif (candidate.method == "fixture_features" and candidate.inliers >= 100 and candidate.residual_px <= .5
                  and bezel_matches(image, candidate.screen_corners, np.float32(reference.profile.screen_corners),
                                    reference.bezel)):
                candidates.append(replace(candidate, method="fixture_bezel"))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item.inliers, item.residual_px))
        best = candidates[0]
        for other in candidates[1:]:
            conflict = max(
                np.linalg.norm(best.screen_corners - other.screen_corners, axis=1).max(),
                np.linalg.norm(best.case_corners - other.case_corners, axis=1).max(),
            )
            if other.inliers >= 0.7 * best.inliers and conflict > 0.012 * width:
                return None
        return best

    def _align(self, reference, points, descriptors, scale, width, height):
        matches = self._matcher.knnMatch(reference.descriptors, descriptors, k=2)
        good = {}
        for neighbors in matches:
            if len(neighbors) != 2:
                continue
            near, far = neighbors
            if near.distance < 0.73 * far.distance:
                previous = good.get(near.trainIdx)
                if previous is None or near.distance < previous.distance:
                    good[near.trainIdx] = near
        # Anchor every fit to the measured holder. The unchanged desk is not
        # evidence that the holder and screen are still in the same position.
        pairs = [(match.queryIdx, index) for index, match in good.items() if reference.on_fixture[match.queryIdx]]
        if len(pairs) < 20:
            return None
        from_indices, to_indices = zip(*pairs, strict=True)
        a = reference.points[list(from_indices)]
        b = points[list(to_indices)]
        transform, inliers = cv2.estimateAffinePartial2D(
            a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=500,
        )
        affine = None
        if transform is not None:
            matrix = np.vstack([transform, [0, 0, 1]])
            affine = self._validated_fit(reference, a, b, matrix, inliers, scale, width, height, projective=False)
        if affine is not None and affine.residual_px <= .0011 * width:
            return affine

        if len(a) < 26:
            return affine
        matrix, inliers = cv2.findHomography(a, b, cv2.RANSAC, 1.45, maxIters=700, confidence=.998)
        if matrix is None:
            return affine
        perspective = self._validated_fit(reference, a, b, matrix, inliers, scale, width, height, projective=True)
        if perspective is not None and (affine is None or perspective.residual_px < .78 * affine.residual_px):
            return perspective
        return affine

    def _validated_fit(self, reference, a, b, transform, inliers, scale, width, height, *, projective):
        mask = inliers.ravel().astype(bool)
        count = int(mask.sum())
        if count < (24 if projective else 20) or count / len(a) < (0.55 if projective else 0.48):
            return None
        # Require spatial coverage so repeating slats or one unchanged object cannot register a frame.
        quadrant = (b[mask, 0] >= width * scale / 2).astype(int)
        quadrant += 2 * (b[mask, 1] >= height * scale / 2)
        counts = np.bincount(quadrant, minlength=4)
        if np.count_nonzero(counts >= 3) < 3 or (projective and not counts.all()):
            return None
        minimum_coverage = .26 if projective else .17
        if cv2.contourArea(cv2.convexHull(b[mask])) < minimum_coverage * width * height * scale**2:
            return None

        fitted = cv2.perspectiveTransform(a[None, mask], transform)[0]
        residual_px = float(np.sqrt(np.mean(np.sum((fitted - b[mask])**2, axis=1))) / scale)
        if residual_px > (.002 if projective else .003) * width:
            return None
        profile = reference.profile
        points = np.float32([[*profile.screen_corners, *profile.case_corners]]) * reference.scale
        denominator = points[0] @ transform[2, :2] + transform[2, 2]
        if not np.isfinite(denominator).all() or denominator.min() <= .55 or denominator.max() >= 1.6:
            return None
        screen, case = np.split(cv2.perspectiveTransform(points, transform)[0] / scale, 2)
        frame = np.array([width, height])
        if (not np.isfinite(screen).all() or not np.isfinite(case).all()
                or np.any(screen < 0) or np.any(screen >= frame)
                or not cv2.isContourConvex(screen) or cv2.contourArea(screen, oriented=True) <= 0
                or not cv2.isContourConvex(case) or cv2.contourArea(case, oriented=True) <= 0):
            return None
        if np.any(case < -0.035 * frame) or np.any(case > 1.035 * frame):
            return None
        top_edge = case[1] - case[0]
        reference_edge = np.subtract(profile.case_corners[1], profile.case_corners[0])
        angle = np.degrees(np.arctan2(top_edge[1], top_edge[0]) - np.arctan2(reference_edge[1], reference_edge[0]))
        if not projective:
            zoom = np.linalg.norm(top_edge) / np.linalg.norm(reference_edge) * 1080 / width
            if abs(angle) > 8 or not .86 <= zoom <= 1.15:
                return None
            return Registration(profile, screen, case, "fixture_features", count, residual_px)

        reference_case = np.float32(profile.case_corners)
        lengths = np.linalg.norm(case - np.roll(case, -1, axis=0), axis=1)
        relative = lengths / np.linalg.norm(reference_case - np.roll(reference_case, -1, axis=0), axis=1) * 1080 / width
        if abs(angle) > 12 or relative.min() < .78 or relative.max() > 1.20 or relative.max() / relative.min() > 1.27:
            return None
        if not all(cv2.pointPolygonTest(case, tuple(point), False) >= 0 for point in screen):
            return None
        # Independent fits must agree at the calibration corners, not only
        # at the locations where feature descriptors happened to match.
        for start in (0, 1):
            fit, _ = cv2.findHomography(a[mask][start::2], b[mask][start::2], 0)
            if fit is None:
                return None
            predicted = cv2.perspectiveTransform(points, fit)[0] / scale
            if not np.isfinite(predicted).all() or np.linalg.norm(predicted - np.vstack([screen, case]), axis=1).max() > .005 * width:
                return None
        return Registration(profile, screen, case, "fixture_perspective", count, residual_px)
