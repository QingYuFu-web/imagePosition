"""Read a purchase or cart verb on a separately painted action button."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_component_captions import _bands, _caption_center, _contrast, _rectangular_button, _tight


_ACTIONS = {
    "加入购物车": ("cart_add", "加入购物车"),
    "加入購物車": ("cart_add", "加入购物车"),
    "addtocart": ("cart_add", "加入购物车"),
    "addtobag": ("cart_add", "加入购物袋"),
    "现在买": ("checkout", "现在买"),
    "立即购买": ("checkout", "立即购买"),
    "立即購買": ("checkout", "立即购买"),
    "buynow": ("checkout", "立即购买"),
}


def _contains(outer, inner):
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def _ink_lines(frame, box):
    left, top, right, bottom = _caption_center(box)
    patch = frame[top:bottom, left:right]
    contrast, paint = _contrast(patch)
    if np.ptp(paint) < 70 or (contrast <= 15).mean() < .78:
        return []
    gap = max(3, round(.16 * (bottom - top)))
    if box[1] < gap or box[3] + gap > frame.shape[0]:
        return []
    outside = (frame[box[1] - gap:box[1] - 1, left:right], frame[box[3] + 1:box[3] + gap, left:right])
    if any((np.max(abs(band.astype(np.int16) - paint), axis=2) >= 48).mean() < .85 for band in outside):
        return []
    ink = np.uint8(contrast >= 44)
    bands = _bands(ink.sum(axis=1) >= max(3, round(.008 * (right - left))))
    areas = []
    for start, stop in bands:
        if not .14 * (box[3] - box[1]) <= stop - start <= .48 * (box[3] - box[1]):
            continue
        measured = _tight(ink[start:stop], left, top + start)
        x0, y0, x1, y1 = measured
        if abs((x0 + x1) - (left + right)) > .23 * (right - left):
            continue
        pad = max(2, round(.08 * (y1 - y0)))
        areas.append((max(left, x0 - pad), max(top, y0 - pad), min(right, x1 + pad), min(bottom, y1 + pad)))
    return areas


def commerce_button_actions(frame, regions, index, known):
    """Keep a distinct button separate from the product or sheet that owns it."""
    height, width = frame.shape[:2]
    layer = regions["layer"]
    if layer == "page":
        scope = (0, 0, width, height)
    elif layer in {"bottom_sheet", "dialog"} and regions["regions"]:
        scope = regions["regions"][0]
    else:
        return []
    sw = scope[2] - scope[0]
    candidates = []
    for item in regions["candidates"]:
        box = item["box"]
        w, h = box[2] - box[0], box[3] - box[1]
        if (item["visibility"] in {"page", "foreground"} and _contains(scope, box) and
                .67 * sw <= w <= sw and .041 * sw <= h <= .16 * sw and
                item["center"][1] >= max(.60 * height, scope[1] + .44 * (scope[3] - scope[1]))):
            candidates.append(item)
    if not candidates:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    actions = []
    cache = {}
    for item in candidates:
        box = _rectangular_button(gray, item["box"])
        if box is None or not _contains(scope, box):
            continue
        x, y = item["center"]
        if any(other["box"][0] <= x < other["box"][2] and other["box"][1] <= y < other["box"][3]
               and other["role"] != "open_product" for other in (*known, *actions)):
            continue
        reads = []
        for area in _ink_lines(frame, box):
            recognized = read_caption(frame, area, index, choice=True, cache=cache)
            if recognized is not None:
                text = recognized["text"]
                action = _ACTIONS.get("".join(text.casefold().split()))
                if action is not None:
                    reads.append((text, *action))
        if len(reads) == 1:
            text, role, function = reads[0]
            preview = "Add to cart" if role == "cart_add" else "Buy"
            actions.append({"role": role, "function": function, "text": text, "box": list(box),
                            "click_point": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], "preview_text": preview})
    return actions
