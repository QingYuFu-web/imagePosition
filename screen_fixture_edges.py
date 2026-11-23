"""Verify the lit display's side edges independently of fixture matching."""

from __future__ import annotations

import cv2
import numpy as np


def _edge_strengths(image, start, end, toward, screen_width):
    direction = end - start
    normal = np.float32([-direction[1], direction[0]]) / np.linalg.norm(direction)
    if np.dot(normal, toward - start) < 0:
        normal *= -1

    radius = max(17, round(.063 * screen_width))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    centers = np.float32([.20, .40, .60, .80])
    positions = (centers[:, None] + np.linspace(-.07, .07, 11, dtype=np.float32)).reshape(-1)
    points = start + positions[:, None, None] * direction + offsets[None, :, None] * normal
    strip = cv2.remap(image, points[..., 0], points[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    values = strip.max(axis=2).astype(np.float32) * .55 + cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) * .45
    half_step = max(2, round(.004 * screen_width))
    smoothed = cv2.blur(values, (2 * half_step + 1, 1))
    outside = smoothed[:, :-4 * half_step]
    inside = smoothed[:, 4 * half_step:]
    support = np.maximum(0., inside - outside - np.maximum(0., outside - 92.))
    band = np.quantile(support.reshape(len(centers), len(positions) // len(centers), -1), .67, axis=1)
    return centers, offsets[2 * half_step:-2 * half_step], band


def _side_evidence(image, start, end, toward, screen_width):
    centers, offsets, band = _edge_strengths(image, start, end, toward, screen_width)
    peaks = band.argmax(axis=1)
    return centers, offsets[peaks], band[np.arange(len(centers)), peaks]


def screen_edges_match(image, corners):
    """Confirm both lateral edges and the lower edge of the visible display."""
    quad = np.asarray(corners, dtype=np.float32)
    if (image.ndim != 3 or image.shape[2] != 3 or quad.shape != (4, 2)
            or not np.isfinite(quad).all() or not cv2.isContourConvex(quad)):
        return False
    upper_left, upper_right, lower_right, lower_left = quad
    screen_width = (np.linalg.norm(upper_right - upper_left) + np.linalg.norm(lower_right - lower_left)) / 2
    if screen_width < 80:
        return False
    tolerance = max(3., .013 * screen_width)
    span = 0.
    for start, end, toward in ((upper_left, lower_left, upper_right), (upper_right, lower_right, upper_left)):
        centers, offsets, strengths = _side_evidence(image, start, end, toward, screen_width)
        near = centers[(np.abs(offsets) <= tolerance) & (strengths >= 28)]
        if len(near) < 2:
            return False
        span = max(span, np.ptp(near))
    # Dark display content can hide part of one edge. The opposite edge still
    # needs widely separated witnesses to tell displacement from a rotated phone.
    if span < .34:
        return False

    # Require a common transition across three lower-edge strips. On dark
    # displays the lit pixels may end just inside the glass. A transition
    # outside the predicted display can instead belong to the holder.
    _, offsets, bands = _edge_strengths(image, lower_left, lower_right, upper_left, screen_width)
    consensus = np.sort(bands, axis=0)[1]
    near = (offsets >= 0.) & (offsets <= .032 * screen_width)
    return bool(consensus[near].max() >= max(24., .82 * consensus.max()))


def _bezel_gray(image):
    return cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), .7)


def _sample_bezel(gray, points):
    return cv2.remap(gray, points[..., 0].astype(np.float32), points[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def bezel_reference(image, corners):
    """Sample hardware outside the display on three separate edges."""
    gray = _bezel_gray(image)
    width = np.linalg.norm(corners[1] - corners[0])
    offsets = np.arange(3, round(.11 * width), dtype=np.float32)
    patches = []
    for first, last, inside in ((0, 3, 1), (1, 2, 0), (3, 2, 0)):
        start, end = corners[[first, last]]
        direction = end - start
        normal = np.float32([-direction[1], direction[0]]) / np.linalg.norm(direction)
        if np.dot(normal, corners[inside] - start) > 0:
            normal *= -1
        for position in (.20, .35, .50, .65, .80):
            fractions = np.linspace(position - .063, position + .063, 49, dtype=np.float32)
            points = start + fractions[:, None, None] * direction + offsets[None, :, None] * normal
            bounds = np.array(gray.shape[::-1]) - 1
            if np.any(points < 1) or np.any(points >= bounds):
                return ()
            template = _sample_bezel(gray, points)[:, 8:-8]
            edge_profile = np.median(np.diff(template.astype(np.float32), axis=1), axis=0)
            if float(np.max(np.abs(edge_profile))) < 3:
                return ()
            patches.append((points[:, 3:], np.ascontiguousarray(template)))
    return tuple(patches)


def bezel_matches(image, corners, reference_corners, patches):
    """Recheck unchanged phone hardware when dark content hides display edges."""
    if len(patches) != 15:
        return False
    gray = _bezel_gray(image)
    bounds = np.array(gray.shape[::-1]) - 1
    transform = cv2.getPerspectiveTransform(reference_corners, corners)
    for points, template in patches:
        projected = cv2.perspectiveTransform(points.reshape(1, -1, 2).astype(np.float64), transform)
        projected = projected.reshape(points.shape)
        if not np.isfinite(projected).all() or np.any(projected < 1) or np.any(projected >= bounds):
            return False
        strip = _sample_bezel(gray, projected)
        scores = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED).ravel()
        peak = int(scores.argmax())
        if scores[peak] < .98 or abs(peak - 5) > 1:
            return False
    return True
