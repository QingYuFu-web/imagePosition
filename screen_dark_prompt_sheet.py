"""Verify a near-full-width biometric prompt on a dimmed phone screen."""

from dataclasses import dataclass

import cv2
import numpy as np

from screen_caption_transcript import _bands, read_caption
from screen_dark_dialog_footer import _read_button
from screen_dialog_search_input import _fold
from screen_native_tab_bar import _distance


@dataclass(frozen=True)
class DarkPromptSheet:
    panel: tuple[int, int, int, int]
    actions: tuple[dict, ...]


_HEADINGS = {
    "deviceunlockrecognition", "deviceunlock", "biometricauthentication", "biometricverification",
    "fingerprintauthentication", "fingerprintverification", "unlockwithfingerprint", "verifyfingerprint",
    "验证指纹", "驗證指紋", "指纹验证", "指紋驗證", "指纹认证", "生物识别验证", "生物辨識驗證",
}
_INSTRUCTIONS = {
    "touchthefingerprintsensor", "touchyourfingerprintsensor", "touchthefingerprintreader",
    "placeyourfingeronthefingerprintsensor", "useyourfingerprint", "scanyourfingerprint",
    "触摸指纹传感器", "觸摸指紋感應器", "请触摸指纹传感器", "請觸碰指紋感應器",
    "请验证指纹", "請驗證指紋",
}


def _corner_curves(tile, paint):
    _, width = tile.shape[:2]
    reach = max(8, round(.043 * width))
    centers = []
    for cap in (tile[:reach], tile[-reach:][::-1]):
        for edge in (cap[:, :reach], cap[:, -reach:][:, ::-1]):
            painted = _distance(edge, paint) <= 6
            gap = np.where(painted.any(axis=1), painted.argmax(axis=1), reach)
            if (not .004 * width <= gap[0] < reach or gap[-3:].max() > 1
                    or np.count_nonzero(np.diff(gap) < 0) < max(2, round(.003 * width))
                    or gap[reach // 3] > .54 * gap[0]):
                return False
            centers.append(int(gap[0]))
    return max(centers) - min(centers) <= .022 * width


def _surface(frame):
    height, width = frame.shape[:2]
    step = max(4, round(.011 * width))
    reference = frame[round(.78 * height):round(.904 * height):step,
                      round(.23 * width):round(.77 * width):step]
    paint = np.median(reference, axis=(0, 1))
    if not 16 <= paint.mean() <= 93 or np.ptp(paint) > 23 or (_distance(reference, paint) <= 5).mean() < .88:
        return None
    scan = round(.48 * height)
    inner = slice(round(.225 * width), round(.775 * width), 5)
    sample = frame[scan:, inner]
    profile = (_distance(sample, paint) <= 5).mean(axis=1)
    darker = (sample.mean(axis=2) <= paint.mean() - 6).mean(axis=1)
    bands = []
    for a, b in _bands(profile >= .875):
        if bands and a - bands[-1][1] <= .057 * width and darker[bands[-1][1]:a].max(initial=0) < .72:
            bands[-1] = (bands[-1][0], b)
        else:
            bands.append((a, b))
    matches = [(a + scan, b + scan) for a, b in bands if .205 * height <= b - a <= .44 * height
               and .55 * height <= a + scan <= .797 * height and .943 * height <= b + scan <= .997 * height]
    if len(matches) != 1:
        return None
    top, bottom = matches[0]
    size = bottom - top
    rim = max(3, round(.0065 * width))
    first = frame[top - 3 * rim:top - rim, inner]
    last = frame[bottom + 1:bottom + rim, inner]
    if min((part.mean(axis=2) <= paint.mean() - 6).mean() for part in (first, last)) < .83:
        return None
    # A dimmed toolbar cannot outshine the prompt's own text bands.
    context = frame[round(.036 * height):top - round(.045 * height):step, ::step]
    if np.percentile(context.max(axis=2), 97) > min(120, paint.mean() + 68):
        return None
    sides = (_distance(frame[top + round(.14 * size):bottom - round(.145 * size):5], paint) <= 5).mean(axis=0)
    spans = [(l, r) for l, r in _bands(sides >= .83) if r - l >= .028 * width]
    if not spans:
        return None
    left, right = spans[0][0], spans[-1][1]
    if not (.93 * width <= right - left <= .995 * width and abs(left + right - width) <= .025 * width):
        return None
    panel = left, top, right, bottom
    tile = frame[top:bottom, left:right]
    if not _corner_curves(tile, paint) or (_distance(tile[::3, ::3], paint) <= 9).mean() < .88:
        return None
    return panel, paint


def _arch(mask):
    height, width = mask.shape
    if not (.11 <= height / width <= .39 and mask.any(axis=0).mean() >= .95):
        return False
    y = mask.argmax(axis=0)
    end = max(1, round(.19 * width))
    a, b, c = np.median(y[:end]), np.median(y[2 * end:width - 2 * end]), np.median(y[-end:])
    return min(a - b, c - b) >= .215 * height and abs(a - c) <= .21 * height


def _fingerprint(ink):
    height, width = ink.shape
    x0, x1 = round(.35 * width), round(.65 * width)
    y0, y1 = round(.295 * height), round(.564 * height)
    tile = np.uint8(ink[y0:y1, x0:x1] >= 53)
    x, y, w, h = cv2.boundingRect(tile)
    if (not .043 * width <= w <= .119 * width or not .051 * width <= h <= .143 * width
            or abs(x0 + x + w / 2 - width / 2) > .025 * width):
        return False
    _, labels, stats, _ = cv2.connectedComponentsWithStats(tile, connectivity=8)
    parts = [(i, (x, y, a, b)) for i, (x, y, a, b, n) in enumerate(stats[1:], 1) if n >= .009 * w * h]
    if not 3 <= len(parts) <= 7:
        return False
    parts.sort(key=lambda part: part[1][1])
    first, second = parts[:2]
    _, y, _, _ = first[1]
    _, following, _, _ = second[1]
    if not .085 * h <= following - y <= .37 * h:
        return False
    for i, (x, y, a, b) in (first, second):
        if not (.53 * w <= a <= w and abs(x0 + x + a / 2 - width / 2) <= .018 * width
                and _arch(labels[y:y + b, x:x + a] == i)):
            return False
    return True


def _letter_bounds(ink, *, start, stop, width_range, height_range, center=True):
    height, width = ink.shape
    y0 = round(start * height)
    limit = width - round(.045 * width)
    block = np.uint8(ink[y0:round(stop * height), :limit] >= 65)
    x, y, w, h = cv2.boundingRect(block)
    if not (width_range[0] * width <= w <= width_range[1] * width
            and height_range[0] * width <= h <= height_range[1] * width
            and 0 < y < y + h < len(block)):
        return None
    if center and abs(x + w / 2 - width / 2) > .052 * width:
        return None
    if not center and not .046 * width <= x <= .15 * width:
        return None
    return x, y0 + y, x + w, y0 + y + h


def _reading(frame, absolute, index, cache):
    l, t, r, b = absolute
    pad = max(2, round(.105 * (b - t)))
    result = read_caption(frame, (l - pad, t - pad, r + pad, b + pad), index, cache=cache,
                          choice=True, sentence_case=True, join_letters=True)
    return _fold(result["text"]) if result else ""


def _is_cancel(frame, box, paint, font_index):
    seen = _read_button(frame, box, paint, font_index)
    if seen is not None:
        return seen == ("cancel", "取消")
    l, t, r, b = box
    height = b - t
    if not 14 <= height <= 26:
        return False
    pad = max(4, round(.28 * height))
    scale = 30 / height
    chip = cv2.resize(frame[t - pad:b + pad, l - pad:r + pad], None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_CUBIC)
    larger = (round(pad * scale), round(pad * scale), round((r - l + pad) * scale), round((height + pad) * scale))
    return _read_button(chip, larger, paint, font_index) == ("cancel", "取消")


def dark_prompt_sheet(frame, font_index):
    """Return the blocking pane and any confirmed opt-out in original pixels."""
    height, width = frame.shape[:2]
    if font_index is None or not 1.53 * width <= height <= 2.86 * width or width < 270:
        return None
    detected = _surface(frame)
    if detected is None:
        return None
    panel, paint = detected
    left, top, right, bottom = panel
    ink = _distance(frame[top:bottom, left:right], paint)
    if not _fingerprint(ink):
        return None
    title = _letter_bounds(ink, start=.028, stop=.236, width_range=(.181, .86), height_range=(.028, .080))
    note = _letter_bounds(ink, start=.567, stop=.742, width_range=(.132, .65), height_range=(.016, .044))
    if title is None or note is None:
        return None

    def absolute(box):
        a, b, c, d = box
        return left + a, top + b, left + c, top + d

    cache = {}
    if _reading(frame, absolute(title), font_index, cache) not in _HEADINGS:
        return None
    instruction = _reading(frame, absolute(note), font_index, cache)
    if instruction and instruction not in _INSTRUCTIONS:
        return None
    label = _letter_bounds(ink, start=.799, stop=.946, width_range=(.051, .215),
                           height_range=(.017, .052), center=False)
    if label is None or not _is_cancel(frame, absolute(label), paint, font_index):
        return DarkPromptSheet(panel, ())
    a, b, c, d = absolute(label)
    dx, dy = round(.025 * (right - left)), round(.025 * (right - left))
    action = {"role": "cancel", "function": "取消", "preview_text": "取消",
              "box": [a - dx, b - dy, c + dx, d + dy], "click_point": [(a + c) // 2, (b + d) // 2]}
    return DarkPromptSheet(panel, (action,))
