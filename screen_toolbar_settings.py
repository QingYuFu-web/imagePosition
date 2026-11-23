"""Locate a toothed settings glyph in a supported top toolbar."""

import cv2
import numpy as np

from screen_candidate_glyphs import _overlap


def _toothed_ring(contour, children, bounds):
    x, y, width, height = bounds
    area = cv2.contourArea(contour)
    if not .84 <= width / height <= 1.19 or not .50 <= area / (width * height) <= .78:
        return False
    holes = [hole for hole in children if cv2.contourArea(hole) >= .014 * area]
    if len(holes) != 1:
        return False
    hole = holes[0]
    hx, hy, hw, hh = cv2.boundingRect(hole)
    if not (.82 <= hw / hh <= 1.22 and .10 * area <= cv2.contourArea(hole) <= .53 * area
            and cv2.contourArea(hole) >= .63 * hw * hh
            and abs(hx + hw / 2 - x - width / 2) <= .075 * width
            and abs(hy + hh / 2 - y - height / 2) <= .075 * height):
        return False
    mask = cv2.drawContours(np.zeros((height, width), np.uint8), [contour], 0, 1,
                            cv2.FILLED, offset=(-x, -y))
    polar = cv2.warpPolar(mask, (48, 128), (hx - x + (hw - 1) / 2, hy - y + (hh - 1) / 2),
                          .76 * max(width, height), cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS)
    radii = (polar * np.arange(polar.shape[1])).max(axis=1)
    outer = float(radii.mean())
    if not .14 * outer <= np.ptp(radii) <= .57 * outer:
        return False
    energy = np.square(np.abs(np.fft.rfft(radii - outer)))
    teeth = int(np.argmax(energy))
    return 6 <= teeth <= 12 and energy[teeth] >= .64 * energy.sum()


def _rings(frame, box):
    left, top, right, bottom = box
    crop = frame[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    low, paint, high = np.percentile(gray, (3, 50, 97))
    found = []
    for sign, span in ((1, high - paint), (-1, paint - low)):
        if span < 72:
            continue
        cuts = sorted({int(round(paint + sign * max(37, level * span))) for level in (.36, .48, .60, .72, .84)})
        for level in cuts:
            ink = gray >= level if sign > 0 else gray <= level
            contours, hierarchy = cv2.findContours(np.uint8(ink), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is None:
                continue
            parents = hierarchy[0, :, 3]
            for index, contour in enumerate(contours):
                x, y, w, h = bounds = cv2.boundingRect(contour)
                if (parents[index] >= 0 or not 0 < x < x + w < crop.shape[1]
                        or not 0 < y < y + h < crop.shape[0]
                        or not max(13, .32 * crop.shape[0]) <= h <= .96 * crop.shape[0]
                        or w < .34 * crop.shape[1]):
                    continue
                children = [hole for hole, parent in zip(contours, parents, strict=True) if parent == index]
                if _toothed_ring(contour, children, bounds):
                    found.append((sign, level, (left + x, top + y, left + x + w, top + y + h)))
    stable = [box for sign, level, box in found if any(
        other_sign == sign and other_level != level and _overlap(box, peer) >= .82
        for other_sign, other_level, peer in found)]
    if stable and all(_overlap(stable[0], box) >= .82 for box in stable):
        return stable[len(stable) // 2]
    return None


def _toolbar_row(row, peers):
    x, y = row["center"]
    _, top, _, bottom = row["box"]
    height = bottom - top
    aligned = []
    for peer in peers:
        px, py = peer["center"]
        _, t, _, b = peer["box"]
        if (.70 * height <= b - t <= 1.50 * height and abs(y - py) <= .25 * height
                and .79 * height <= abs(px - x) <= 4.10 * height and all(abs(px - at) >= .58 * height for at in aligned)):
            aligned.append(px)
    return len(aligned) >= 2


def toolbar_settings_actions(frame, regions, known):
    """Require visible page ownership, two toolbar peers and one stable gear."""
    if regions["layer"] != "page":
        return []
    height, width = frame.shape[:2]
    short = min(height, width)
    peers = []
    for row in regions["candidates"]:
        _, cy = row["center"]
        l, t, r, b = row["box"]
        if (row["visibility"] == "page" and .025 * height <= cy <= .16 * height
                and max(18, .032 * short) <= min(r - l, b - t) <= max(r - l, b - t) <= .17 * short
                and .62 <= (r - l) / (b - t) <= 1.62):
            peers.append(row)
    found = []
    for row in peers:
        x, _ = row["center"]
        if .22 * width < x < .78 * width or not _toolbar_row(row, peers):
            continue
        box = _rings(frame, row["box"])
        if box is not None and not any(_overlap(box, prior) >= .65 for prior in found):
            found.append(box)
    if len(found) != 1:
        return []
    left, top, right, bottom = found[0]
    x, y = (left + right) / 2, (top + bottom) / 2
    if any(l <= x < r and t <= y < b for l, t, r, b in (item["box"] for item in known)):
        return []
    return [{"role": "settings", "function": "打开设置", "preview_text": "Settings", "box": list(found[0]),
             "click_point": [x, y]}]
