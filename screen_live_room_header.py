"""Verify a live-room host action and its independently drawn exit."""

from itertools import combinations

import cv2
import numpy as np

from inventory_segmented_actions import _rounded_shape
from screen_caption_transcript import read_caption
from screen_component_captions import _bands
from screen_dialog_close import _cross_footprint
from screen_foreground import _centered_cross
from screen_landscape_settings_rail import _literal_word


_LIVE = {"更多直播", "热门直播", "直播广场", "直播列表", "附近直播"}
_CAPTIONS = tuple(sorted(_LIVE | {
    "更多视频", "更多内容", "更多资讯", "更多推荐", "热门推荐", "直播回放", "关注主播", "发送弹幕",
}))
_FOLLOW = {"关注", "關注", "follow"}
_FOLLOW_CAPTIONS = tuple(sorted(_FOLLOW | {
    "互关", "取关", "订阅", "訂閱", "登录", "更多", "粉丝", "打赏", "帮助", "分享", "返回", "设置",
    "直播", "已关注", "已關注",
}))


def _follow_pill(frame):
    _, width = frame.shape[:2]
    x0, x1 = round(.22 * width), round(.59 * width)
    y0, y1 = round(.008 * width), round(.139 * width)
    field = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    colored = cv2.inRange(field, (150, 133, 182), (179, 255, 255))
    contours, _ = cv2.findContours(colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (.044 * width <= h <= .094 * width and 1.25 <= w / h <= 2.66
                and 0 < x < x + w < field.shape[1] and 0 < y < y + h < field.shape[0]
                and _rounded_shape(contour, (x, y, w, h))):
            boxes.append((x0 + x, y0 + y, x0 + x + w, y0 + y + h))
    return boxes[0] if len(boxes) == 1 else None


def _close_mark(white, gray):
    _, width = white.shape
    x0, x1 = round(.898 * width), round(.985 * width)
    y0, y1 = round(.022 * width), round(.123 * width)
    field = white[y0:y1, x0:x1]
    boxes = []
    for level in (121, 159, 191, 223):
        n, marks, stats, _ = cv2.connectedComponentsWithStats(np.uint8(field >= level), connectivity=8)
        for part in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[part])
            if (not .020 * width <= min(w, h) <= max(w, h) <= .048 * width or not .79 <= w / h <= 1.26
                    or not .26 <= area / (w * h) <= .58 or not (0 < x < x + w < field.shape[1]
                    and 0 < y < y + h < field.shape[0])):
                continue
            if _cross_footprint(marks[y:y + h, x:x + w] == part):
                boxes.append((level, (x0 + x, y0 + y, x0 + x + w, y0 + y + h)))
    stable = [box for level, box in boxes if sum(level != other
              and max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= 2
              for other, peer in boxes) >= 2]
    if not stable or np.max(np.ptp(stable, axis=0)) > 2:
        return None
    x, y, r, b = stable[0]
    pads = [max(2, round(factor * (b - y))) for factor in (.24, .42, .50)]
    if any(_centered_cross(gray[y - pad:b + pad, x - pad:r + pad]) for pad in pads):
        return x, y, r, b
    return None


def _circles(gray, left, right, cy, size, radius, votes, *, sample_limit=82):
    reach = round(.79 * size)
    y0, y1 = max(0, round(cy) - reach), min(len(gray), round(cy) + reach)
    field = gray[y0:y1, left:right]
    if min(field.shape) < 18:
        return []
    scale = min(1., sample_limit / size)
    small = cv2.resize(field, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else field
    smoothed = cv2.GaussianBlur(small, (7, 7), 1.4)
    found = cv2.HoughCircles(smoothed, cv2.HOUGH_GRADIENT, 1.3, max(14, .46 * size * scale),
                             param1=75, param2=votes, minRadius=round(radius[0] * size * scale),
                             maxRadius=round(radius[1] * size * scale))
    if found is None:
        return []
    return [(left + x / scale, y0 + y / scale, r / scale) for x, y, r in found[0]
            if abs(y0 + y / scale - cy) <= .19 * size and x - r > 0 and x + r < small.shape[1]
            and y - r > 0 and y + r < small.shape[0]]


def _host_labels(white, face, cy, size, follow):
    x, _, radius = face
    left = round(x + radius + .10 * size)
    right = round(follow[0] - .18 * size) if follow else round(.32 * white.shape[1])
    top, bottom = max(0, round(cy - .61 * size)), round(cy + .57 * size)
    if right - left < .75 * size:
        return False
    light = np.uint8(white[top:bottom, left:right] >= 111)
    bands = _bands(light.sum(axis=1) >= max(4, round(.074 * size)))
    lines = []
    for a, b in bands:
        if lines and a - lines[-1][1] <= .055 * size:
            lines[-1] = lines[-1][0], b
        else:
            lines.append((a, b))
    text = []
    for a, b in lines:
        x, y, w, h = cv2.boundingRect(light[a:b])
        if .116 * size <= h <= .58 * size and .38 * size <= w <= .95 * light.shape[1]:
            text.append((x, top + a + y + h / 2))
    return any(abs(a[0] - b[0]) <= .20 * size and a[1] < cy < b[1] and .30 * size <= b[1] - a[1] <= .84 * size
               for a, b in combinations(text, 2))


def _avatar_content(gray, face):
    x, y, radius = face
    reach = max(2, round(.72 * radius))
    left, top = round(x) - reach, round(y) - reach
    right, bottom = round(x) + reach + 1, round(y) + reach + 1
    yy, xx = np.ogrid[top:bottom, left:right]
    core = gray[top:bottom, left:right][(xx - x) ** 2 + (yy - y) ** 2 <= reach * reach]
    low, high = np.percentile(core, (18, 92))
    return high - low >= 32


def _host_identity(gray, white, cy, size, follow):
    width = gray.shape[1]
    # Additional sampling resolutions preserve the same rim votes and text ownership proof.
    for limit in ((82,) if size <= 82 else (82, 112, 88)):
        face = _circles(gray, round(.011 * width), round(.194 * width), cy, size, (.46, .70), 23,
                        sample_limit=limit)
        if len(face) == 1 and _avatar_content(gray, face[0]) and _host_labels(white, face[0], cy, size, follow):
            return True
    return False


def _has_audience(gray, cy, size):
    width = gray.shape[1]
    faces = sorted(_circles(gray, round(.49 * width), round(.825 * width), cy, size, (.36, .58), 21))
    return any(.72 * size <= b[0] - a[0] <= 1.42 * size and abs(a[1] - b[1]) <= .14 * size
               and abs(a[2] - b[2]) <= .13 * size for a, b in combinations(faces, 2))


def _centered_follow(label, box):
    x, y, r, b = label
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    return (.33 * width <= r - x <= .81 * width and .25 * height <= b - y <= .64 * height
            and abs(x + r - left - right) <= .20 * width and abs(y + b - top - bottom) <= .23 * height)


def _read_follow(white, box, index):
    left, top, right, bottom = box
    dx, dy = round(.12 * (right - left)), round(.16 * (bottom - top))
    area = left + dx, top + dy, right - dx, bottom - dy
    tone = white[area[1]:area[3], area[0]:area[2]]
    ground, peak = np.percentile(tone, (45, 98))
    if peak - ground < 98:
        return False
    direct = cv2.subtract(white, int(round(ground + .51 * (peak - ground))))
    text = read_caption(cv2.cvtColor(direct, cv2.COLOR_GRAY2BGR), area, index, cache={}, choice=True,
                        join_radicals=True)
    if text and text["text"].casefold() in _FOLLOW and _centered_follow(text["line_box"], box):
        return True
    if text and text["text"] in _FOLLOW_CAPTIONS and text["text"] not in _FOLLOW:
        return False
    reads = [_literal_word(cv2.subtract(white, int(round(ground + cut * (peak - ground)))),
                           area, _FOLLOW_CAPTIONS, index) for cut in (.37, .43)]
    return (all(item and item[0] in _FOLLOW and _centered_follow(item[1], box) for item in reads)
            and reads[0][0] == reads[1][0])


def _action(role, function, box):
    left, top, right, bottom = box
    return {"role": role, "function": function, "preview_text": function, "box": list(box),
            "click_point": [(left + right) // 2, (top + bottom) // 2]}


def live_room_header_actions(frame, overlay, index):
    """Read visible header actions only with independent live-room context."""
    height, width = frame.shape[:2]
    if index is None or overlay is not None or width < 290 or not 1.42 * width <= height <= 2.85 * width:
        return []
    top = frame[:round(.255 * width)]
    follow = _follow_pill(top)
    blue, green, red = cv2.split(top)
    white = cv2.min(cv2.min(blue, green), red)
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    close = _close_mark(white, gray)
    if follow is None and close is None:
        return []
    if follow is not None:
        cy, size = (follow[1] + follow[3]) / 2, follow[3] - follow[1]
    else:
        cy, size = (close[1] + close[3]) / 2, 2.36 * (close[3] - close[1])
    if close is not None and abs((close[1] + close[3]) / 2 - cy) > .22 * size:
        close = None
    if not (_host_identity(gray, white, cy, size, follow) and _has_audience(gray, cy, size)):
        return []
    right = round(.95 * width)
    left = max(round(.735 * width), right - round(2.06 * size))
    label_y = cy + 1.30 * size
    area = left, round(label_y - .34 * size), right, round(label_y + .34 * size)
    live = _literal_word(white, area, _CAPTIONS, index)
    if not live or live[0] not in _LIVE:
        return []
    result = []
    if follow is not None and _read_follow(white, follow, index):
        result.append(_action("follow", "关注主播", follow))
    if close is not None:
        result.append(_action("close", "退出直播间", close))
    return result
