"""Verify the paperclip and recipient expander on an established mail form."""

import cv2
import numpy as np

from screen_component_captions import _bands
from screen_landscape_search_field import _available


def _paperclip(mask):
    h, w = mask.shape
    if not 1.43 <= w / h <= 2.49 or not .39 <= mask.mean() <= .73:
        return False
    for view in (mask, mask[:, ::-1]):
        def rail(position):
            index = round(position * (w - 1))
            return _bands(view[:, index] > 0)

        if not any(len(bands) == 1 and bands[0][1] - bands[0][0] >= .55 * h
                   for bands in (rail(p) for p in (.075, .115, .15))):
            continue
        if not any(len(bands) == 2 and bands[1][0] - bands[0][1] >= .39 * h
                   for bands in (rail(p) for p in (.17, .21, .25))):
            continue
        middle = [bands for bands in (rail(p) for p in (.43, .51, .62, .70))
                  if 3 <= len(bands) <= 4 and bands[0][0] <= .10 * h and bands[-1][1] >= .88 * h
                  and all(.086 * h <= b - a <= .29 * h for a, b in bands)]
        if len(middle) < 3:
            continue
        if any(bands and bands[0][0] <= .16 * h and bands[-1][1] <= .83 * h
               for bands in (rail(p) for p in (.84, .89, .94))):
            return True
    return False


def _down_chevron(mask):
    h, w = mask.shape
    if not 1.24 <= w / h <= 2.27 or not .22 <= mask.mean() <= .52:
        return False
    values = mask.sum(axis=0)
    columns = np.flatnonzero(values)
    if len(columns) < .91 * w:
        return False
    centers = (np.arange(h) @ mask[:, columns]) / values[columns]
    slopes = 1 - abs(2 * columns / max(1, w - 1) - 1)
    scale = centers[np.argmax(slopes)] - np.median(np.r_[centers[:max(1, w // 10)],
                                                        centers[-max(1, w // 10):]])
    model = centers.mean() + scale * (slopes - slopes.mean())
    return (.54 * h <= scale <= .86 * h and np.mean(abs(centers - model)) <= .101 * h
            and abs(centers[0] - centers[-1]) <= .16 * h
            and values.max() <= .49 * h)


def _stable_glyph(gray, box, check, extent, *, cuts=(49, 72, 96)):
    left, top, right, bottom = box
    chip = gray[top:bottom, left:right]
    if min(chip.shape) < 10:
        return None
    paint = float(np.median(chip))
    ink = paint - chip.astype(np.int16)
    if paint < 173 or (abs(ink) <= 18).mean() < .68:
        return None
    seen = []
    for level in cuts:
        mask = np.uint8(ink >= level)
        _, parts, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i, (x, y, w, h, _) in enumerate(stats[1:], 1):
            if (0 < x < x + w < chip.shape[1] and 0 < y < y + h < chip.shape[0]
                    and extent(w, h) and check(parts[y:y + h, x:x + w] == i)):
                seen.append((level, tuple(map(int, (left + x, top + y, left + x + w, top + y + h)))))
    if len({level for level, _ in seen}) < 2:
        return None
    boxes = [b for _, b in seen]
    if np.ptp(boxes, axis=0).max() > max(2, .11 * (boxes[0][3] - boxes[0][1])):
        return None
    return boxes[len(boxes) // 2]


def _attachments(gray, layout):
    l, t, r, b = layout["toolbar"]
    w, h = r - l, b - t
    surface = l + round(.53 * w), t, r - round(.095 * w), b
    return _stable_glyph(gray, surface, _paperclip,
                         lambda a, d: .42 * h <= a <= .85 * h and .23 * h <= d <= .44 * h)


def _recipient_menu(gray, layout):
    left, top, right, bottom = layout["recipient_row"]
    w, h = right - left, bottom - top
    x0, x1 = right - round(.122 * w), right - round(.029 * w)
    verify = lambda a, b: .125 * h <= a <= .274 * h and .062 * h <= b <= .161 * h
    below = _stable_glyph(gray, (x0, top + round(.17 * h), x1, bottom - round(.17 * h)),
                          _down_chevron, verify)
    above = _stable_glyph(gray, (x0, top - round(.95 * h), x1, top - round(.153 * h)),
                          _down_chevron, verify)
    if below is None or above is None:
        return None
    x, y, r, b = below
    l, t, a, d = above
    if (max(abs(x - l), abs(r - a)) > .052 * (r - x)
            or abs((b - y) - (d - t)) > max(2, .13 * (b - y))
            or not .82 * h <= (y + b - t - d) / 2 <= 1.26 * h):
        return None
    return below


def mail_compose_controls(frame, regions, layout):
    """Read two glyphs independently once the parent compose form is established."""
    if regions["layer"] != "page" or layout is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    actions = []
    for mark, role, caption in ((_attachments(gray, layout), "attach", "添加附件"),
                               (_recipient_menu(gray, layout), "dropdown", "展开抄送")):
        if mark is None:
            continue
        l, t, r, b = mark
        pad_x, pad_y = max(3, round(.25 * (r - l))), max(3, round(.37 * (b - t)))
        box = [l - pad_x, t - pad_y, r + pad_x, b + pad_y]
        if _available(box, regions):
            actions.append({"role": role, "function": caption, "preview_text": caption, "box": box,
                            "click_point": [(l + r) // 2, (t + b) // 2]})
    return actions
