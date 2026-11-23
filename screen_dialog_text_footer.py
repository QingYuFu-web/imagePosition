"""Read paired text buttons at the bottom of a verified light dialog."""

from math import ceil, floor

import cv2
import numpy as np

from screen_dark_dialog_footer import _footer_words, _read_button
from screen_native_tab_bar import _distance


def _ink(frame, box, paper):
    left, top, right, bottom = box
    pixels = frame[top:bottom, left:right]
    delta = _distance(pixels, paper)
    strength = float(np.percentile(delta, 93))
    crisp = pixels[delta >= max(31, .74 * strength)]
    if len(crisp) < 8:
        return None
    pigment = np.median(crisp, axis=0)
    return strength, pigment - paper


def _known(word, existing, panel):
    a, t, b, z = word
    x, y = (a + b) / 2, (t + z) / 2
    width, height = panel[2] - panel[0], panel[3] - panel[1]
    for item in existing:
        left, top, right, bottom = item["box"]
        if (left <= x < right and top <= y < bottom and right - left <= .36 * width and bottom - top <= .26 * height
                and min(right, b) - max(left, a) >= .74 * (b - a)
                and min(bottom, z) - max(top, t) >= .74 * (z - t)):
            return True
    return False


def _padded(word, panel):
    left, top, right, bottom = panel
    a, t, b, z = word
    pad = max(4, round(.70 * (z - t)))
    return [max(left, a - pad), max(top, t - pad), min(right, b + pad), min(bottom, z + pad)]


def dialog_text_footer(frame, regions, index, existing):
    """Retain an unrecognized or faded footer button without assigning its action."""
    panels = regions.get("regions", ())
    if index is None or regions["layer"] != "dialog" or len(panels) != 1 or regions.get("floating_controls"):
        return [], []
    left, top, right, bottom = panel = panels[0]
    width, height = right - left, bottom - top
    if (not (0 <= left < right <= frame.shape[1] and 0 <= top < bottom <= frame.shape[0])
            or min(width, height) < 160 or width < .58 * frame.shape[1]):
        return [], []
    scale = min(1., 640 / width)
    view_width, view_height = round(width * scale), round(height * scale)
    surface = frame[top:bottom, left:right]
    if scale < 1:
        surface = cv2.resize(surface, (view_width, view_height), interpolation=cv2.INTER_AREA)
    sample = surface[view_height - round(.30 * view_height):view_height - round(.03 * view_height):4,
                     round(.04 * view_width):view_width - round(.04 * view_width):4]
    paper = np.median(sample, axis=(0, 1))
    if paper.min() < 194 or np.ptp(paper) > 29:
        return [], []
    words = _footer_words(surface, (0, 0, view_width, view_height), paper)
    if words is None:
        return [], []
    colors = [_ink(surface, box, paper) for box in words]
    if any(color is None for color in colors):
        return [], []
    originals = [(left + floor(a * width / view_width), top + floor(t * height / view_height),
                  left + ceil(b * width / view_width), top + ceil(z * height / view_height)) for a, t, b, z in words]
    uncovered = [not _known(word, existing, panel) for word in originals]
    if not any(uncovered):
        return [], []
    strong = [strength >= 69 and np.ptp(pigment) >= 36 for strength, pigment in colors]
    readings = []
    for i, word in enumerate(words):
        reading = _read_button(surface, word, paper, index) if strong[i] else None
        if reading is None and strong[i] and scale < 1.:
            reading = _read_button(frame, originals[i], paper, index)
        readings.append(reading)
    first, second = (color[1] for color in colors)
    norms = np.linalg.norm(first) * np.linalg.norm(second)
    alike = first @ second >= .95 * norms and min(color[0] for color in colors) >= .72 * max(color[0] for color in colors)
    matched = all(readings) and {reading[0] for reading in readings} == {"cancel", "confirm"}
    records = []
    for word in originals:
        a, t, b, z = word
        records.append({"box": _padded(word, panel), "click_point": [(a + b) // 2, (t + z) // 2]})
    if alike and all(strong) and matched:
        return [{**item, "role": role, "function": label, "preview_text": label}
                for item, (role, label), missing in zip(records, readings, uncovered, strict=True) if missing], []
    paired = _padded((originals[0][0], min(word[1] for word in originals), originals[1][2],
                      max(word[3] for word in originals)), panel)
    return [], [{**records[i], "context_box": paired, "review_only": True} for i in range(2)
                if uncovered[i] and readings[1 - i] is not None and strong[1 - i]
                and (readings[i] is None or not strong[i]) and colors[i][0] >= 43]
