"""Measure dark neutral fields against an independently tinted panel surface."""

from math import ceil

import cv2
import numpy as np

from inventory_list_rows import _Box


_CLOSE = np.ones((3, 3), np.uint8)


def dark_neutral_enclosures(frame, gray):
    """Yield bounded, near-neutral surfaces that grayscale edges can miss."""
    height, width = gray.shape
    span = min(height, width)
    minimum_height = max(11, ceil(.009 * span))
    minimum_inset = max(3, round(.13 * minimum_height))
    minimum_width = ceil(3.25 * minimum_height) - 2 * minimum_inset
    minimum_body_pixels = ceil(.80 * minimum_width * minimum_inset)
    dark = cv2.compare(gray, 37, cv2.CMP_LE)
    if cv2.countNonZero(dark) < minimum_body_pixels:
        return

    blue, green, red = cv2.split(frame)
    chroma = cv2.subtract(cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red))
    if cv2.minMaxLoc(chroma)[1] < 7:
        return
    surface = cv2.bitwise_and(dark, cv2.compare(chroma, 6, cv2.CMP_LE))
    if cv2.countNonZero(surface) < minimum_body_pixels:
        return
    closed = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, _CLOSE)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        left, top, w, h = cv2.boundingRect(contour)
        right, bottom = left + w, top + h
        if (not minimum_height <= h <= .11 * span or not 3.25 <= w / h <= 31
                or min(left, width - right) < 2 or top < 4 or bottom + 4 >= height
                or cv2.contourArea(contour) < .86 * w * h):
            continue
        inset = max(3, round(.13 * h))
        band = slice(left + inset, right - inset)
        body = surface[top + inset:top + 2 * inset, band]
        if cv2.countNonZero(body) < .80 * body.size:
            continue
        tint = np.median(chroma[top + inset:top + 2 * inset, band])
        above = np.median(chroma[top - 4:top - 1, band])
        below = np.median(chroma[bottom + 1:bottom + 4, band])
        if tint <= 4 and min(above, below) - tint >= 7:
            yield _Box(left, top, right, bottom)
