"""Map an observed display quadrilateral to an upright pixel plane."""

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, sqrt

import cv2
import numpy as np


def prepare_camera_screen(frame):
    """Strengthen weak display edges without thresholding text or icon colors."""
    channels = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    brightness = channels[:, :, 0]
    smooth = cv2.GaussianBlur(brightness, (0, 0), 1.05)
    channels[:, :, 0] = cv2.addWeighted(brightness, 1.30, smooth, -.30, 0.)
    return cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)


@dataclass(frozen=True)
class ExtractionInput:
    """Processed pixels and their position in the original camera photo."""

    pixels: np.ndarray
    to_photo: np.ndarray

    def project(self, photo_shape):
        height, width = photo_shape
        return cv2.warpPerspective(self.pixels, self.to_photo, (width, height),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(27, 27, 27))


@lru_cache(maxsize=1)
def fixture_occlusions():
    """Calibrated fixture hardware seen in the two reference photos."""
    mask = np.zeros((1376, 640), np.uint8)
    clamp = np.int32([(307, 0), (449, 0), (441, 64), (426, 99), (395, 121), (349, 120), (322, 102), (306, 72)])
    wire = np.int32([(-8, 31), (20, 70), (59, 99), (105, 125), (136, 119), (151, 108), (172, 125),
                     (194, 129), (215, 116), (237, 94), (259, 97), (282, 107), (301, 93), (322, 72),
                     (349, 48), (376, 29)])
    cv2.fillConvexPoly(mask, clamp, 1)
    cv2.polylines(mask, [wire], False, 1, 34)
    mask.setflags(write=False)
    return mask


def display_size(corners, size_mm=None):
    """Keep the observed sampling budget while accounting for perspective foreshortening."""
    points = np.asarray(corners, dtype=np.float32)
    edges = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    width, height = (max(2, round((float(edges[a]) + float(edges[b])) / 2)) for a, b in ((0, 2), (1, 3)))
    if size_mm is None:
        return width, height
    ratio = size_mm[0] / size_mm[1]
    limit = 2 * max(width, height)
    if not 2 / limit <= ratio <= limit / 2:
        return width, height
    target_height = min(sqrt(width * height / ratio), limit / max(1, ratio))
    return max(2, round(target_height * ratio)), max(2, round(target_height))


def display_mapping(corners, size=None, rotation=0):
    """Return original photo to upright screen homography and target size."""
    points = np.asarray(corners, dtype=np.float32)
    width, height = display_size(points) if size is None else size
    straight = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(points, straight)
    if rotation == 90:
        return np.float64([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]]) @ matrix, (height, width)
    if rotation == -90:
        return np.float64([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]]) @ matrix, (height, width)
    return matrix, (width, height)


def validate_photo_outline(corners, shape):
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    height, width = shape[:2]
    if (not np.isfinite(points).all() or (points < 0).any() or (points >= [width, height]).any()
            or not cv2.isContourConvex(points) or cv2.contourArea(points, oriented=True) < .022 * width * height
            or min(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)) < 24):
        raise ValueError("请按左上、右上、右下、左下提供原照片里有效的屏幕四角。")
    return points


class PhotoCrops:
    """Straighten local crops and retain their inverse projection to the photo."""

    def __init__(self, page):
        calibration = page.get("calibration", {})
        fixture = calibration.get("space") == "fixture_estimate_mm"
        size = (640, 1376) if fixture else calibration.get("rectified_size_px")
        rotation = calibration.get("content_rotation_cw", 0)
        self.to_screen, self.size = display_mapping(page["screen_corners"], size, rotation)
        self.to_photo = np.linalg.inv(self.to_screen)
        self.blocked = fixture_occlusions() if fixture else None
        if self.blocked is not None and rotation in (-90, 90):
            code = cv2.ROTATE_90_CLOCKWISE if rotation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE
            self.blocked = cv2.rotate(self.blocked, code)

    def allows_point(self, point):
        if self.blocked is None:
            return True
        x, y = np.rint(cv2.perspectiveTransform(np.float64([[point]]), self.to_screen)[0, 0]).astype(int)
        width, height = self.size
        return 0 <= x < width and 0 <= y < height and not self.blocked[y, x]

    def crop_window(self, box):
        """Return the integer upright window for a source box, where right and bottom are exclusive."""
        left, top, right, bottom = box
        corners = np.float64([[[left, top], [right, top], [right, bottom], [left, bottom]]])
        outline = cv2.perspectiveTransform(corners, self.to_screen)[0]
        if not np.isfinite(outline).all() or np.any(np.abs(outline) > 3 * max(self.size)):
            return None
        x, y = (floor(value) for value in outline.min(axis=0))
        end_x, end_y = (ceil(value) for value in outline.max(axis=0))
        if end_x <= x or end_y <= y:
            return None
        return x, y, end_x, end_y

    def upright(self, pixels, box):
        window = self.crop_window(box)
        if window is None:
            return pixels, None
        left, top = box[:2]
        x, y, end_x, end_y = window
        local = np.float64([[1, 0, left], [0, 1, top], [0, 0, 1]])
        offset = np.float64([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
        upright = cv2.warpPerspective(pixels, offset @ self.to_screen @ local, (end_x - x, end_y - y),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(32, 32, 32))
        if self.blocked is not None:
            width, height = self.size
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(end_x, width), min(end_y, height)
            if x0 < x1 and y0 < y1:
                local = upright[y0-y:y1-y, x0-x:x1-x]
                local[self.blocked[y0:y1, x0:x1] != 0] = 32
        return upright, (x, y)

    def tile_quad(self, origin, width, height):
        """Return the four photo points corresponding to the tile rectangle edges."""
        left, top = origin
        right, bottom = left + width, top + height
        corners = np.float64([[[left, top], [right, top], [right, bottom], [left, bottom]]])
        return cv2.perspectiveTransform(corners, self.to_photo)[0].round(3).tolist()

    def outline_in_tile(self, outline, origin, pixel_size, tile_box):
        x, y, right, bottom = tile_box
        relative = cv2.perspectiveTransform(np.float64([outline]), self.to_screen)[0] - origin
        scaled = relative * [(right - x) / pixel_size[0], (bottom - y) / pixel_size[1]] + [x, y]
        return np.rint(scaled).astype(np.int32)
