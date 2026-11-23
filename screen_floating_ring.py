"""Find visible translucent ring controls as review-only photo evidence."""

import cv2
import numpy as np


_ANGLES = np.linspace(0, 2 * np.pi, 32, endpoint=False, dtype=np.float32)
_RADII = np.float32([.12, .24, .34, .42, .54, .67, .78, .88, .98, 1.09, 1.18])[:, None]
_COS, _SIN = np.cos(_ANGLES), np.sin(_ANGLES)


def _floating_ring(lab, x, y, radius):
    radial = radius * _RADII
    probe_x, probe_y = x + radial * _COS, y + radial * _SIN
    height, width = lab.shape[:2]
    valid = ((probe_x[-1] >= 0) & (probe_x[-1] <= width - 1)
             & (probe_y[-1] >= 0) & (probe_y[-1] <= height - 1))
    if np.count_nonzero(valid) < 25:
        return False
    samples = cv2.remap(lab, probe_x[:, valid], probe_y[:, valid], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    light = samples[:, :, 0]
    center = np.median(light[:3], axis=0)
    ring = np.median(light[5:8], axis=0)
    exterior = np.median(light[-2:], axis=0)

    # A plain circle can have two colored rings, but a floating translucent
    # control also has a bright, nearly uniform center and a darker annulus.
    if (np.median(center) < 155 or np.ptp(np.quantile(center, [.10, .90])) > 23
            or np.quantile(center - ring, .25) < 15 or np.median(center - ring) < 26
            or np.quantile(exterior - ring, .25) < 2 or np.median(exterior - ring) < 14):
        return False

    # The colored background should still change in roughly the same way
    # around the ring and immediately outside it. Opaque icons rarely do.
    ring_color = np.median(samples[5:8, :, 1:], axis=0)
    outside_color = np.median(samples[-2:, :, 1:], axis=0)
    ring_color -= ring_color.mean(axis=0)
    outside_color -= outside_color.mean(axis=0)
    if np.median(np.linalg.norm(outside_color, axis=1)) < 12:
        return False
    correspondence = float(np.sum(ring_color * outside_color))
    alignment = correspondence / (1 + np.linalg.norm(ring_color) * np.linalg.norm(outside_color))
    transmission = correspondence / (1 + float(np.sum(outside_color**2)))
    return alignment >= .65 and .26 <= transmission <= 1.15


def translucent_ring_boxes(frame):
    """Return rectangles with visual evidence for a translucent floating ring."""
    height, width = frame.shape[:2]
    short = min(width, height)
    if short < 240:
        return []
    scale = min(1., 480 / short, 1280 / max(width, height))
    view = (frame if scale == 1 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    gray = cv2.GaussianBlur(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), (5, 5), .95)
    size = min(gray.shape)
    if size < 120:
        return []
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.10, minDist=.052 * size,
                              param1=72, param2=26, minRadius=round(.043 * size), maxRadius=round(.097 * size))
    if circles is None:
        return []
    lab = cv2.cvtColor(view, cv2.COLOR_BGR2Lab)
    h, w = gray.shape
    found = []
    for x, y, radius in circles[0]:
        if radius >= min(x, y, w - 1 - x, h - 1 - y) or not _floating_ring(lab, x, y, radius):
            continue
        found.append([round((x - radius) * width / w), round((y - radius) * height / h),
                      round((x + radius) * width / w), round((y + radius) * height / h)])
    return found
