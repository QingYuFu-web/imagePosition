"""Name adjustable sliders only after confirming their rails, thumbs and labels."""

import cv2
import numpy as np

from inventory_slider_actions import _spans, _tracks_share_axis
from screen_colored_settings_panel import _printed
from screen_range_geometry import ranged_controls


_CAPTIONS = {
    "Sensitivity": "灵敏度", "Aim Sensitivity": "瞄准灵敏度", "Camera Sensitivity": "镜头灵敏度",
    "Music": "音乐", "Music Volume": "音乐音量", "Sound Effects": "音效", "SFX": "音效",
    "SFX Volume": "音效音量", "Sound Volume": "音量", "Volume": "音量", "Master Volume": "主音量",
    "Media Volume": "媒体音量", "Notification Volume": "通知音量", "Ring Volume": "铃声音量",
    "Voice Volume": "语音音量", "Alarm Volume": "闹钟音量", "Microphone": "麦克风",
    "Brightness": "亮度", "Contrast": "对比度", "Gamma": "伽马", "Opacity": "不透明度",
    "Saturation": "饱和度", "Exposure": "曝光", "Vibration": "振动强度", "Preamp": "前置增益",
    "Balance": "左右平衡", "Bass Boost": "低音增强", "Virtualizer": "环绕效果", "Amplifier": "放大增益",
    "Zoom": "缩放", "Gain": "增益", "Speed": "速度", "Intensity": "强度",
}
_HAN = tuple(dict.fromkeys((*_CAPTIONS.values(), "透明度", "背景音乐", "音效音量", "录音音量", "画面亮度")))
_LATIN = tuple(_CAPTIONS)


def _caption_windows(gray, box, glyph_height, *, side):
    left, top, right, bottom = box
    patch = gray[top:bottom, left:right]
    if min(patch.shape) < 9:
        return
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, round(.78 * glyph_height)) | 1,) * 2)
    for light, operation in ((True, cv2.MORPH_TOPHAT), (False, cv2.MORPH_BLACKHAT)):
        ink = cv2.morphologyEx(patch, operation, kernel)
        mask = ink >= max(34, .40 * float(np.percentile(ink, 99)))
        bands = [(0, len(mask))] if side else _joined_spans(mask.any(axis=1), .29 * glyph_height)[-3:]
        for y1, y2 in bands:
            groups = _joined_spans(mask[y1:y2].any(axis=0), .64 * glyph_height)
            for start, end in (groups[-3:] if side else groups[:2]):
                _, y, _, h = cv2.boundingRect(np.uint8(mask[y1:y2, start:end]))
                y += y1
                if not (.43 * glyph_height <= h <= 1.35 * glyph_height and 1.18 * h <= end - start <= 12.0 * h):
                    continue
                if y <= 0 or y + h >= patch.shape[0]:
                    continue
                x_pad, y_pad = max(1, round(.24 * h)), max(2, round(.33 * h))
                yield ((left + max(0, start - x_pad), top + max(0, y - y_pad),
                        left + min(patch.shape[1], end + x_pad), top + min(patch.shape[0], y + h + y_pad)), light)


def _joined_spans(line, max_gap):
    groups = []
    for start, end in _spans(line):
        if groups and start - groups[-1][1] <= max_gap:
            groups[-1] = groups[-1][0], end
        else:
            groups.append((start, end))
    return groups


def _caption(gray, slider, index, *, previous_bottom=0):
    height, width = gray.shape
    track, thumb = slider.track, slider.thumb
    glyph_height = max(9, .85 * thumb.height)
    ty = round(thumb.center[1])
    side = (max(0, track.left - round(13 * glyph_height)), max(0, ty - round(glyph_height)),
            track.left - max(2, round(.42 * glyph_height)), min(height, ty + round(glyph_height)))
    above = (max(0, track.left - round(.8 * glyph_height)), max(previous_bottom, track.top - round(4.7 * glyph_height)),
             min(width, track.left + round(13 * glyph_height)), track.top - max(3, round(.42 * glyph_height)))
    for box, side_caption in ((side, True), (above, False)):
        windows = list(_caption_windows(gray, box, glyph_height, side=side_caption))
        for vocabulary in (_LATIN, _HAN):
            readings = {reading for window, light in windows
                        if (reading := _printed(gray, window, index, vocabulary=vocabulary, light=light)) is not None}
            if readings:
                return next(iter(readings)) if len(readings) == 1 else None
    return None


def setting_range_actions(frame, panel, index, *, scoped=False):
    """Return only independently named rails supported by a peer on the same surface."""
    left, top, right, bottom = panel
    pixels = frame[top:bottom, left:right]
    sliders = ranged_controls(pixels, grouped=not scoped)
    related = [item for item in sliders if any(peer is not item and _tracks_share_axis(item.track, peer.track)
                                               for peer in sliders)]
    if len(related) < 2:
        return []
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    named = []
    for item in related:
        previous = max((other.thumb.bottom for other in related if other.thumb.bottom < item.thumb.top
                        and _tracks_share_axis(item.track, other.track)), default=0)
        if (printed := _caption(gray, item, index, previous_bottom=previous)) is not None:
            named.append((item, printed))
    actions = []
    for item, printed in named:
        neighbors = [entry for entry in related if _tracks_share_axis(item.track, entry.track)]
        if not any(other != printed and (peer in neighbors or any(_tracks_share_axis(neighbor.track, peer.track)
                                                                for neighbor in neighbors)) for peer, other in named):
            continue
        label = _CAPTIONS.get(printed, printed)
        if any(action["setting_label"] == label for action in actions):
            continue
        actions.append(range_action(item.thumb, item.track, label, printed, (left, top)))
    return sorted(actions, key=lambda item: item["click_point"][1])


def range_action(thumb, rail, label, printed, offset, *, axis="horizontal"):
    left, top = offset
    bounds = [min(rail.left, thumb.left) + left, min(rail.top, thumb.top) + top,
              max(rail.right, thumb.right) + left, max(rail.bottom, thumb.bottom) + top]
    x, y = thumb.center
    return {"role": "adjust_slider", "function": f"拖动{label}滑块", "setting_label": label,
            "setting_display_label": printed, "click_point": [x + left, y + top], "box": bounds,
            "slider_thumb_box": [thumb.left + left, thumb.top + top, thumb.right + left, thumb.bottom + top],
            "slider_track_box": [rail.left + left, rail.top + top, rail.right + left, rail.bottom + top], "slider_axis": axis}
