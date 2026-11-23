"""Ground navigation controls beside an already verified media toolbar."""

import cv2
import numpy as np

from screen_candidate_glyphs import _overlap
from screen_caption_transcript import read_caption
from screen_landscape_media_bar import _action, _triangle


def _gear_teeth(contour, holes, bounds):
    x, y, width, height = bounds
    area = cv2.contourArea(contour)
    if not (.83 <= width / height <= 1.20 and .50 <= area / (width * height) <= .79):
        return None
    chambers = [hole for hole in holes if .105 * area <= cv2.contourArea(hole) <= .33 * area]
    if len(chambers) != 3 or np.ptp([cv2.contourArea(part) for part in chambers]) > .087 * area:
        return None
    cx, cy = x + (width - 1) / 2, y + (height - 1) / 2
    angles = []
    for chamber in chambers:
        moments = cv2.moments(chamber)
        dx, dy = moments["m10"] / moments["m00"] - cx, moments["m01"] / moments["m00"] - cy
        if not .14 * height <= np.hypot(dx, dy) <= .33 * height:
            return None
        angles.append(np.arctan2(dy, dx))
    angles.sort()
    spacing = np.diff([*angles, angles[0] + 2 * np.pi])
    if spacing.min() < 1.54 or spacing.max() > 2.66:
        return None
    hub = []
    for hole in holes:
        if not .007 * area <= cv2.contourArea(hole) <= .075 * area:
            continue
        left, top, w, h = cv2.boundingRect(hole)
        if np.hypot(left + w / 2 - cx, top + h / 2 - cy) <= .114 * height:
            hub.append(hole)
    if len(hub) != 1:
        return None
    silhouette = cv2.drawContours(np.zeros((height, width), np.uint8), [contour], 0, 1,
                                  cv2.FILLED, offset=(-x, -y))
    polar = cv2.warpPolar(silhouette, (64, 256), (cx - x, cy - y), .76 * max(width, height),
                          cv2.WARP_POLAR_LINEAR | cv2.WARP_FILL_OUTLIERS)
    radius = (polar * np.arange(polar.shape[1])).max(axis=1)
    if not .13 * radius.mean() <= np.ptp(radius) <= .46 * radius.mean():
        return None
    spectrum = np.square(abs(np.fft.rfft(radius - radius.mean())))
    teeth = int(spectrum.argmax())
    return teeth if 10 <= teeth <= 24 and spectrum[teeth] >= .37 * spectrum.sum() else None


def _trailing_gear(frame, track):
    _, width = frame.shape[:2]
    left, top, right, bottom = track
    size = bottom - top
    center = (top + bottom) / 2
    x0, x1 = max(right, round(.70 * width)), round(.972 * width)
    y0, y1 = max(0, round(center - .83 * size)), round(center + .83 * size)
    if x1 <= x0 or y1 - y0 < 14:
        return None
    pixels = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    neutral = np.ptp(pixels, axis=2) <= 35
    paper, peak = np.percentile(gray, (50, 99.4))
    if peak - paper < 79:
        return None
    matches = []
    for ratio in (.31, .43, .54, .64, .74):
        threshold = round(paper + ratio * (peak - paper))
        mask = np.uint8(neutral & (gray >= threshold))
        outlines, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        parents = hierarchy[0, :, 3]
        for i, outline in enumerate(outlines):
            x, y, w, h = bounds = cv2.boundingRect(outline)
            if (parents[i] >= 0 or not .69 * size <= h <= 1.28 * size
                    or not 0 < x < x + w < gray.shape[1] or not 0 < y < y + h < gray.shape[0]
                    or abs(y0 + y + h / 2 - center) > .19 * size):
                continue
            holes = [hole for hole, parent in zip(outlines, parents, strict=True) if parent == i]
            if (teeth := _gear_teeth(outline, holes, bounds)) is not None:
                matches.append((threshold, teeth, (x0 + x, y0 + y, x0 + x + w, y0 + y + h)))
    stable = [box for cut, teeth, box in matches if any(
        other != cut and count == teeth and _overlap(box, peer) >= .76 for other, count, peer in matches)]
    if stable and all(_overlap(stable[0], box) >= .73 for box in stable):
        return stable[len(stable) // 2]
    return None


def _drop_mark(gray, box):
    left, top, right, bottom = box
    patch = gray[top:bottom, left:right].astype(np.int16)
    height, width = patch.shape
    tone = np.median(patch)
    below = gray[bottom:bottom + max(3, round(.26 * height)), left:right]
    if not below.size or (abs(patch - tone) <= 9).mean() < .67 or np.median(below) - tone < 7:
        return None
    matched = []
    for cut in (42, 64, 92):
        shapes, _ = cv2.findContours(np.uint8(patch - tone >= cut), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(shapes) != 1:
            continue
        outline = shapes[0]
        x, y, w, h = cv2.boundingRect(outline)
        if (not .23 * height <= h <= .49 * height or not 1.2 <= w / h <= 1.85
                or abs(x + w / 2 - width / 2) > .12 * width or abs(y + h / 2 - height / 2) > .12 * height
                or not .36 <= cv2.contourArea(outline) / (w * h) <= .62):
            continue
        vertices = cv2.approxPolyDP(outline, .065 * cv2.arcLength(outline, True), True).reshape(-1, 2)
        if len(vertices) == 3 and _triangle(((vertices - (x, y)) / (w - 1, h - 1))[:, ::-1]):
            matched.append((left + x, top + y, left + x + w, top + y + h))
    if len(matched) >= 2 and all(_overlap(matched[0], box) >= .73 for box in matched):
        return matched[len(matched) // 2]
    return None


def _leading_dropdown(gray, boxes, track):
    _, width = gray.shape
    center = (track[1] + track[3]) / 2
    depth = track[3] - track[1]
    row = [box for box in boxes if box[2] <= .38 * width
           and 1.22 * depth <= box[3] - box[1] <= 2.38 * depth
           and .76 <= (box[2] - box[0]) / (box[3] - box[1]) <= 1.47
           and abs((box[1] + box[3]) / 2 - center) <= .25 * depth]
    marks = []
    for left, top, right, bottom in row:
        if not (.040 * width <= left <= .135 * width and right <= .18 * width):
            continue
        neighbors = [peer for peer in row if 0 <= peer[0] - right <= 2.3 * (bottom - top)]
        if len(neighbors) >= 2 and (glyph := _drop_mark(gray, (left, top, right, bottom))) is not None:
            marks.append(glyph)
    return marks[0] if len(marks) == 1 else None


def _header_pill(gray, box):
    left, top, right, bottom = box
    patch = gray[top:bottom, left:right]
    rows, cols = patch.shape
    corners = np.r_[patch[:3, :7].ravel(), patch[:3, -7:].ravel(),
                     patch[-3:, :7].ravel(), patch[-3:, -7:].ravel()]
    dark = float(np.median(corners))
    masks = []
    for step in (12, 21):
        boundaries, _ = cv2.findContours(np.uint8(patch >= dark + step), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not boundaries:
            continue
        outline = max(boundaries, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(outline)
        if (not .82 * cols <= w <= cols or not .76 * rows <= h <= rows
                or not .826 <= cv2.contourArea(outline) / (w * h) <= .936 or h <= 13):
            continue
        fill = cv2.drawContours(np.zeros((h, w), np.uint8), [outline], 0, 1, cv2.FILLED, offset=(-x, -y))
        reach = round(.19 * h)
        corners = [fill[:reach, :reach], fill[:reach, -reach:], fill[-reach:, :reach], fill[-reach:, -reach:]]
        if max(corner.mean() for corner in corners) <= .76:
            masks.append((left + x, top + y, left + x + w, top + y + h))
    if len(masks) == 2 and _overlap(*masks) >= .84:
        return masks[0]
    return None


def landscape_media_navigation_actions(frame, regions, index, known):
    """Return measured settings, menu and a literal label on a paired header."""
    height, width = frame.shape[:2]
    if regions["layer"] != "page" or width < 1.46 * height:
        return []
    verified = {item["role"]: item for item in known}
    transport = {"stop_playback", "toggle_playback", "record"} <= verified.keys()
    close = verified.get("close")
    paired = (close is not None and close["function"] == "完成" and close["click_point"][0] >= .84 * width
              and .052 * height <= close["click_point"][1] <= .158 * height and index is not None)
    if transport == paired:
        return []
    boxes = [tuple(row["box"]) for row in regions["candidates"] if row.get("visibility") == "page"
             and 0 <= row["box"][0] < row["box"][2] <= width and 0 <= row["box"][1] < row["box"][3] <= .213 * height]
    gray = cv2.cvtColor(frame[:round(.23 * height)], cv2.COLOR_BGR2GRAY)
    if transport:
        actions = []
        track = verified["record"]["box"]
        if "settings" not in verified and (gear := _trailing_gear(frame, track)) is not None:
            actions.append(_action("settings", "打开设置", gear))
        if (button := _leading_dropdown(gray, boxes, track)) is not None:
            actions.append(_action("open_menu", "打开菜单", button))
        return actions
    _, top, _, bottom = close["box"]
    diameter = bottom - top
    opposite = [box for box in boxes if box[2] <= .28 * width and 2.2 * diameter <= box[2] - box[0] <= 4.5 * diameter
                and max(abs(box[1] - top), abs(box[3] - bottom)) <= .22 * diameter]
    if len(opposite) != 1 or (pill := _header_pill(gray, opposite[0])) is None:
        return []
    left, top, right, bottom = pill
    pad_x = round(.085 * (right - left))
    pad_y = round(.18 * (bottom - top))
    reading = read_caption(frame, (left + pad_x, top + pad_y, right - pad_x, bottom - pad_y), index, cache={})
    if reading is None or reading["min_margin"] < .026 or reading["max_distance"] > .18:
        return []
    return [_action("open_menu", reading["text"], pill)]
