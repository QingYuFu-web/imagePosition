"""Read independently placed textual dismissals on an already verified sheet."""

import cv2
import numpy as np

from screen_caption_transcript import read_caption
from screen_component_captions import _bands, _contrast, _tight
from screen_composer import _outlined_field
from screen_foreground import (COMMENT_SHEET_TITLES, BlockingOverlay, _comment_sheet_surfaces, _working_view,
                               account_comment_field, is_comment_sheet_heading)


_FOOTER_WORDS = {"取消": "取消", "cancel": "取消", "关闭": "关闭", "關閉": "关闭", "close": "关闭"}
_HEADER_WORDS = {**_FOOTER_WORDS, "done": "完成", "完成": "完成"}


def _caption(frame, area, *, min_height, max_height, contrast=53, max_aspect=7.8):
    left, top, right, bottom = area
    if right <= left or bottom <= top:
        return None
    crop = frame[top:bottom, left:right]
    delta, paint = _contrast(crop)
    if np.min(paint) < 224 or np.ptp(paint) > 18 or (delta <= 15).mean() < .72:
        return None
    ink = np.uint8(delta >= contrast)
    lines = [(start, stop) for start, stop in _bands(ink.sum(axis=1) >= 4)
             if min_height <= stop - start <= max_height]
    if len(lines) != 1:
        return None
    start, stop = lines[0]
    tight = _tight(ink[start:stop], left, top + start)
    x0, y0, x1, y1 = tight
    height = y1 - y0
    if not 1.20 * height <= x1 - x0 <= max_aspect * height:
        return None
    pad = max(3, round(.10 * height))
    return tight, (max(left, x0 - pad), max(top, y0 - pad), min(right, x1 + pad), min(bottom, y1 + pad))


def _footer(frame, panel):
    left, top, right, bottom = panel
    width = right - left
    near = round(.20 * width)
    area = (left + near, max(top + round(.26 * (bottom - top)), bottom - round(.211 * width)),
            right - near, bottom - round(.042 * width))
    observed = _caption(frame, area, min_height=.020 * width, max_height=.065 * width)
    if observed is None:
        return None
    (x0, y0, x1, y1), caption = observed
    if abs(x0 + x1 - left - right) > .10 * width:
        return None
    x = left + round(.11 * width)
    end = right - round(.11 * width)
    start, stop = max(top, y0 - round(.112 * width)), y0 - round(.023 * width)
    if stop <= start:
        return None
    stripe = cv2.cvtColor(frame[start:stop, x:end], cv2.COLOR_BGR2GRAY)
    ink = (stripe >= 220) & (stripe <= 248)
    edges = [(a, b) for a, b in _bands(ink.mean(axis=1) >= .89) if .002 * width <= b - a <= .027 * width]
    if not edges:
        return None
    pad = max(3, round(.017 * width))
    below = frame[y1 + pad:bottom - round(.035 * width), x:end].astype(np.int16)
    if not below.size or (np.min(below, axis=2) < 202).mean() >= .012:
        return None
    divider = start + edges[-1][1]
    if y0 - divider < .026 * width:
        return None
    row_end = min(bottom, y1 + round(.065 * width))
    return (left, divider, right, row_end), (x0, y0, x1, y1), caption


def _header(frame, panel):
    left, top, right, bottom = panel
    width = right - left
    area = (right - round(.211 * width), top + round(.019 * width),
            right - round(.025 * width), min(bottom, top + round(.093 * width)))
    observed = _caption(frame, area, min_height=.016 * width, max_height=.052 * width, contrast=54)
    if observed is None:
        return None
    (x0, y0, x1, y1), caption = observed
    pixels = frame[y0:y1, x0:x1].astype(np.int16)
    colors = pixels[(np.min(pixels, axis=2) <= 195) & (np.ptp(pixels, axis=2) >= 35)]
    if len(colors) < .15 * pixels.shape[0] * pixels.shape[1]:
        return None
    same_row = frame[y0:y1, left + round(.055 * width):left + round(.69 * width)]
    if same_row.size == 0 or not .020 <= (np.min(same_row, axis=2) <= 172).mean() <= .43:
        return None
    return area, (x0, y0, x1, y1), caption


def sheet_dismiss_actions(frame, regions, index, existing):
    if regions["layer"] != "bottom_sheet" or not regions["regions"]:
        return []
    panel = regions["regions"][0]
    if panel[2] - panel[0] < 200:
        return []
    output = []
    for layout, words in ((_footer, _FOOTER_WORDS), (_header, _HEADER_WORDS)):
        measured = layout(frame, panel)
        if measured is None:
            continue
        box, (left, top, right, bottom), area = measured
        x, y = (left + right) / 2, (top + bottom) / 2
        if any(item["box"][0] <= x < item["box"][2] and item["box"][1] <= y < item["box"][3]
               for item in existing):
            continue
        label = read_caption(frame, area, index, cache={}, choice=True)
        if label is None or (purpose := words.get("".join(label["text"].casefold().split()))) is None:
            continue
        preview = {"取消": "Cancel", "关闭": "Close", "完成": "Done"}[purpose]
        output.append({"role": "close", "function": purpose, "box": list(box),
                       "click_point": [x, y], "preview_text": preview})
    return output


def _comment_heading(frame, panel, index):
    left, top, right, bottom = panel
    width = right - left
    area = (left + round(.14 * width), top + round(.047 * width),
            right - round(.14 * width), min(bottom, top + round(.139 * width)))
    observed = _caption(frame, area, min_height=.018 * width, max_height=.054 * width, max_aspect=19)
    if observed is None:
        return False
    (x0, y0, x1, y1), caption = observed
    if abs(x0 + x1 - left - right) > .13 * width:
        return False
    cache = {}
    height = y1 - y0
    roi = frame[y0:y1, x0:x1]
    ink, _ = _contrast(roi)
    pieces = _bands((ink >= 54).any(axis=0))
    splits = [first + x0 for (_, last), (first, _) in zip(pieces[:-1], pieces[1:], strict=True)
              if first - last >= .24 * height]
    # The final word can establish the sheet when a preceding account or app
    # name has ambiguous glyphs. The panel and composer are measured separately.
    modes = [(caption, is_comment_sheet_heading)]
    if splits and x1 - splits[-1] >= 1.35 * height:
        _, y0, x1, y1 = caption
        modes.insert(0, ((splits[-1] - max(2, round(.10 * height)), y0, x1, y1),
                         lambda text: text.casefold() in COMMENT_SHEET_TITLES))
    for crop, verifies in modes:
        label = read_caption(frame, crop, index, cache=cache, choice=True)
        if label is not None and verifies(label["text"]):
            return True
    return False


def captioned_comment_sheet(frame, index, *, working_view=None, surfaces=None):
    """Verify a foreground comment panel using its grip, title, and composer."""
    height, width = frame.shape[:2]
    if not 1.55 * width <= height <= 2.70 * width:
        return None
    _, gray = _working_view(frame) if working_view is None else working_view
    if gray is None:
        return None
    small_h, small_w = gray.shape
    panels = []
    for panel in _comment_sheet_surfaces(gray) if surfaces is None else surfaces:
        field = account_comment_field(gray, panel)
        if field is None:
            continue
        scale = (width / small_w, height / small_h)
        bounds, composer = [tuple(round(value * scale[i % 2]) for i, value in enumerate(box))
                            for box in (panel, field)]
        if _comment_heading(frame, bounds, index):
            panels.append((BlockingOverlay("bottom_sheet", bounds), composer))
    return panels[0] if len(panels) == 1 else None


def _input_action(box):
    left, top, right, bottom = box
    return {"role": "comment_input", "function": "输入评论", "box": list(box),
            "click_point": [(left + right) / 2, (top + bottom) / 2], "preview_text": "Write comment"}


def _inline_input(frame, index):
    from inventory_inline_comment_input import _surface_contour
    from inventory_list_rows import _Box

    height, width = frame.shape[:2]
    if width < 180 or not 1.49 * width <= height <= 2.75 * width:
        return []
    area = (round(.02 * width), height - round(.112 * width),
            round(.48 * width), height - round(.013 * width))
    observed = _caption(frame, area, min_height=.018 * width, max_height=.061 * width, max_aspect=15)
    if observed is None:
        return []
    box, crop = observed
    label = read_caption(frame, crop, index, cache={}, choice=True)
    if label is None or "".join(label["text"].casefold().split()) not in {
        "addacomment", "writeacomment", "添加评论", "写评论", "新增留言", "新增評論",
    }:
        return []
    surfaces = list(_surface_contour(frame, _Box(*box)))
    if len(surfaces) != 1:
        return []
    field = surfaces[0]
    return [_input_action((field.left, field.top, field.right - round(1.15 * field.height), field.bottom))]


def _dark_composer_heading(frame, gray, index):
    height, width = frame.shape[:2]
    sh, sw = gray.shape
    x0, x1 = round(.35 * sw), round(.65 * sw)
    y0, y1 = round(.16 * sh), round(.63 * sh)
    strip = gray[y0:y1, x0:x1]
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(strip >= 29))
    grips = []
    for x, y, w, h, mass in stats[1:]:
        if not (.047 * sw <= w <= .145 * sw and .004 * sw <= h <= .020 * sw and
                abs(x0 + x + w / 2 - sw / 2) <= .023 * sw and mass >= .69 * w * h and y >= 3):
            continue
        paint = strip[y - 3:y + h + 3, :max(4, x - 2)]
        if paint.size and np.median(paint) <= 24 and (paint <= 26).mean() >= .91:
            grips.append(round((y0 + y + h) * height / sh))
    if len(grips) != 1:
        return False
    left, right = round(.024 * width), round(.35 * width)
    top, bottom = grips[0] + round(.019 * width), grips[0] + round(.105 * width)
    delta, paint = _contrast(frame[top:bottom, left:right])
    if np.max(paint) > 45 or (delta <= 10).mean() < .78:
        return False
    glyph = _tight(delta >= 38, left, top)
    if glyph is None:
        return False
    a, b, c, d = glyph
    if not .022 * width <= d - b <= .061 * width or not 1.3 * (d - b) <= c - a <= 8.0 * (d - b):
        return False
    pad = max(2, round(.09 * (d - b)))
    label = read_caption(frame, (a - pad, b - pad, c + pad, d + pad), index, cache={}, choice=True)
    return label is not None and label["text"].casefold() in COMMENT_SHEET_TITLES


def _send_box(gray, field, frame_width):
    _, top, right, bottom = field
    fh = bottom - top
    x0, x1 = right + round(.016 * frame_width), frame_width - round(.019 * frame_width)
    y0, y1 = top + round(.15 * fh), bottom - round(.055 * fh)
    chip = gray[y0:y1, x0:x1]
    if chip.size == 0 or np.median(chip) > 67:
        return None
    found = []
    for level in (168, 197, 222):
        mask = np.uint8(chip >= level)
        outlines, parents = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if parents is None:
            continue
        parent_index = parents[0][:, 3]
        for i, contour in enumerate(outlines):
            if parent_index[i] >= 0:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if not (.028 * frame_width <= h <= .078 * frame_width and .87 <= w / h <= 1.45 and
                    .30 <= cv2.contourArea(contour) / (w * h) <= .62 and .11 <= mask[y:y + h, x:x + w].mean() <= .42):
                continue
            yy, xx = np.nonzero(mask[y:y + h, x:x + w])
            tip = yy[xx >= .89 * w]
            tail = xx[yy < .16 * h], xx[yy >= .84 * h]
            holes = [cv2.contourArea(outline) for outline, parent in zip(outlines, parent_index, strict=True) if parent == i]
            if (tip.size and .29 * h <= tip.mean() <= .70 * h and all(len(p) and p.mean() < .36 * w for p in tail)
                    and len(holes) in (1, 2) and .16 <= sum(holes) / (h * w) <= .48):
                found.append((level, (x0 + x, y0 + y, x0 + x + w, y0 + y + h)))
    stable = [box for level, box in found if any(level != other and max(abs(a - b) for a, b in zip(box, peer, strict=True)) <= 4
                                              for other, peer in found)]
    if not stable or any(max(abs(a - b) for a, b in zip(stable[0], box, strict=True)) > 4 for box in stable):
        return None
    return stable[len(stable) // 2]


def _active_composer(frame, regions, index, *, working_view=None):
    from screen_visible_draft import has_visible_draft

    height, width = frame.shape[:2]
    _, gray = _working_view(frame) if working_view is None else working_view
    if gray is None:
        return []
    sh, sw = gray.shape
    top = regions["regions"][0][1]
    field = _outlined_field(gray, round(top * sh / height))
    if field is None or not _dark_composer_heading(frame, gray, index):
        return []
    scale = (width / sw, height / sh)
    a, b, c, d = (round(value * scale[i % 2]) for i, value in enumerate(field))
    actions = [_input_action((a, b, c - round(.74 * (d - b)), d))]
    native_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    box = _send_box(native_gray, (a, b, c, d), width)
    if box is not None and has_visible_draft(frame, (a, b, c, d), index):
        left, top, right, bottom = box
        actions.append({"role": "comment_send", "function": "发送评论", "box": list(box),
                        "click_point": [(left + right) / 2, (top + bottom) / 2], "preview_text": "Send"})
    return actions


def comment_input_actions(frame, regions, index, *, working_view=None):
    if (box := regions.get("comment_input")) is not None:
        return [_input_action(box)]
    if regions["layer"] == "composer" and regions["regions"]:
        return _active_composer(frame, regions, index, working_view=working_view)
    if regions["layer"] == "page":
        return _inline_input(frame, index)
    return []
