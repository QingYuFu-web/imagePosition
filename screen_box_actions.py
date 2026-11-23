"""Name conventional glyphs when their source pixels and surroundings agree."""

from functools import lru_cache
from itertools import product
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inventory_back_evidence import suggest_wide_toolbar_back
from screen_candidate_glyphs import CandidateGlyphs
from screen_dialog_close import dialog_close_box
from screen_foreground import _centered_cross


FUNCTIONS = {"back": "返回上一级", "search": "搜索", "comments": "打开评论区",
             "share": "打开分享", "close": "关闭", "voice_stop": "停止录音"}
PREVIEW = {"back": "Back", "search": "Search", "comments": "Comments", "share": "Share", "close": "Close",
           "voice_stop": "Stop recording"}


def _center(box):
    left, top, right, bottom = box
    return [(left + right) / 2, (top + bottom) / 2]


def _result(role, box):
    return {"role": role, "function": FUNCTIONS[role], "box": list(box), "click_point": _center(box)}


def _visible_cross(frame, box):
    left, top, right, bottom = box
    return min(right - left, bottom - top) >= 8 and _centered_cross(frame[top:bottom, left:right])


def _foreground_close(frame, regions):
    if (box := regions.get("detected_close")) and (regions["layer"] in {"side_drawer", "wide_panel"}
           or regions.get("close_style") in {"sheet_down", "literal"} or _visible_cross(frame, box)):
        return [_result("close", box)]
    if regions["layer"] == "dialog" and (box := dialog_close_box(frame, regions)) is not None:
        return [_result("close", box)]
    if regions["layer"] == "page":
        from screen_fullscreen_close import fullscreen_close_action

        if (action := fullscreen_close_action(frame, regions)) is not None:
            return [action]
    if regions["layer"] != "bottom_sheet" or not regions["regions"]:
        return []
    width = frame.shape[1]
    _, top, right, bottom = regions["regions"][0]
    reach = min(.16 * width, .28 * (bottom - top))
    matches = []
    for item in regions["candidates"]:
        x, y = item["center"]
        x0, y0, x1, y1 = item["box"]
        w, h = x1 - x0, y1 - y0
        if (right - reach <= x < right and top <= y <= top + reach and .012 * width <= min(w, h)
                and max(w, h) <= .16 * width and .65 <= w / h <= 1.5 and _visible_cross(frame, item["box"])):
            matches.append(_result("close", item["box"]))
    return matches if len(matches) == 1 else []


def _flat_toolbar(frame, mark, candidates):
    height, width = frame.shape[:2]
    x, y = mark["center"]
    for candidate in candidates:
        left, top, right, bottom = candidate["box"]
        bar_height = bottom - top
        if (right - left < .86 * width or not .06 * width <= bar_height <= .19 * width or
                not .023 * height <= top <= .12 * height or not top + .11 * bar_height <= y <= bottom - .11 * bar_height
                or x > left + .14 * width or bottom + .15 * bar_height >= height):
            continue
        pad = max(4, round(.14 * bar_height))
        inside = frame[top + pad:bottom - pad:3, left + pad:right - pad:4].astype(np.float32)
        after = frame[bottom:bottom + pad:2, left + pad:right - pad:4].astype(np.float32)
        fill = np.median(inside, axis=(0, 1))
        if ((np.max(abs(inside - fill), axis=2) <= 11).mean() >= .85 and
                (np.max(abs(after - fill), axis=2) >= 35).mean() >= .85):
            return True
    return False


def _toolbar_marks(frame, marks, candidates):
    height, width = frame.shape[:2]
    left = [mark for mark in marks if mark["role"] == "back" and mark["center"][0] < .20 * width
            and mark["center"][1] < .19 * height]
    right = [mark for mark in marks if mark["role"] in {"search", "share"} and mark["center"][0] > .70 * width
             and mark["center"][1] < .19 * height]
    pairs = [(a, b) for a, b in product(left, right)
             if abs(a["center"][1] - b["center"][1]) <= .48 * max(a["height"], b["height"])]
    accepted = [mark for pair in pairs for mark in pair]
    if width > height:
        nodes = [{"kind": "icon", "box": mark["box"], "center": mark["center"]} for mark in left]
        region = suggest_wide_toolbar_back(frame, nodes)
        if region is not None:
            accepted.extend(mark for mark in left if max(abs(a - b) for a, b in zip(mark["box"], region, strict=True))
                            <= max(3, .12 * mark["height"]))
    else:
        accepted.extend(mark for mark in left if mark not in accepted and _flat_toolbar(frame, mark, candidates))
    return accepted


def _video_rail(frame, marks, candidates):
    height, width = frame.shape[:2]
    if height < 1.6 * width:
        return []

    def at_side(x, y):
        return (x <= .18 * width or x >= .82 * width) and .37 * height <= y <= .90 * height

    bubbles = [mark for mark in marks if mark["role"] == "comments" and at_side(*mark["center"])]
    arrows = [mark for mark in marks if mark["role"] == "share" and at_side(*mark["center"])]
    peers = [item for item in candidates if at_side(*item["center"]) and
             .037 * width <= min(item["box"][2] - item["box"][0], item["box"][3] - item["box"][1]) <= .19 * width]
    selected = []
    for first, second in product(bubbles, arrows):
        x0, y0 = first["center"]
        x1, y1 = second["center"]
        if (abs(x0 - x1) > .026 * width or not .045 * height <= abs(y0 - y1) <= .20 * height
                or max(first["height"], second["height"]) > 1.65 * min(first["height"], second["height"])):
            continue
        if any(abs(item["center"][0] - (x0 + x1) / 2) <= .036 * width and
               min(y0, y1) - .085 * height <= item["center"][1] <= max(y0, y1) + .085 * height and
               min(abs(item["center"][1] - y0), abs(item["center"][1] - y1)) >= .042 * height for item in peers):
            selected.extend((first, second))
    if selected:
        # A magnifier with an adjacent toolbar candidate on the same video
        # surface is independent of the buttons running along the side.
        selected.extend(mark for mark in marks if mark["role"] == "search" and mark["center"][0] >= .68 * width
                        and .042 * height <= mark["center"][1] <= .17 * height and any(
                            .04 * width <= abs(item["center"][0] - mark["center"][0]) <= .21 * width and
                            abs(item["center"][1] - mark["center"][1]) <= .55 * mark["height"] for item in candidates))
    return selected


def suggest_box_actions(frame, regions, *, glyphs=None):
    """Return supported actions and a separate list for unclear layer ownership."""
    if regions["layer"] == "voice_stop_panel" and (stop := regions.get("recording_stop")) is not None:
        return [_result("voice_stop", stop)], []
    active = _foreground_close(frame, regions)
    if regions["layer"] not in {"page", "layer_uncertain"}:
        return active, []
    candidates = regions["candidates"]
    glyphs = glyphs or CandidateGlyphs(frame)
    marks = [{**mark, "center": _center(mark["box"]), "height": mark["box"][3] - mark["box"][1]}
             for mark in glyphs(candidates)]
    rows = {row["id"]: row for row in candidates}
    review = []
    for mark in _toolbar_marks(frame, marks, candidates) + _video_rail(frame, marks, candidates):
        result = _result(mark["role"], mark["box"])
        uses = review if rows[mark["id"]]["visibility"] == "layer_uncertain" else active
        x, y = result["click_point"]
        if any(item["role"] == result["role"] and abs(item["click_point"][0] - x) <= .012 * frame.shape[1]
               and abs(item["click_point"][1] - y) <= .012 * frame.shape[1] for item in uses):
            continue
        uses.append(result)
    return active, review


@lru_cache(maxsize=1)
def _label_font():
    folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        try:
            return ImageFont.truetype(str(folder / name), 18)
        except OSError:
            continue
    return None


def _label_image(label, color):
    if not label.isascii() and (font := _label_font()) is not None:
        l, t, r, b = font.getbbox(label)
        tile = Image.new("RGB", (r - l + 8, b - t + 6), (25, 28, 32))
        ImageDraw.Draw(tile).text((4 - l, 3 - t), label, font=font, fill=color)
        return np.asarray(tile)
    if not label.isascii():
        label = "Action?" if label.endswith("?") else "Action"
    (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .64, 2)
    tile = np.full((height + baseline + 6, width + 8, 3), (25, 28, 32), dtype=np.uint8)
    cv2.putText(tile, label, (4, height + 3), cv2.FONT_HERSHEY_SIMPLEX, .64, color, 2, cv2.LINE_AA)
    return tile


def _label_position(tile, box, image_shape, occupied, other_controls):
    image_height, image_width = image_shape[:2]
    height, width = tile.shape[:2]
    left, top, right, bottom = box
    gap = 5
    left_aligned = min(max(left, 0), image_width - width)
    right_aligned = min(max(right - width, 0), image_width - width)
    centered = min(max((left + right - width) // 2, 0), image_width - width)
    around = []
    if bottom - top >= height:
        # Edge rails have adjoining rows. Align the label beside its own row.
        y = (top + bottom - height) // 2
        if image_width - right <= gap and left - width - gap >= 0:
            around.append((left - width - gap, y, left - gap, y + height))
        elif left <= gap and right + gap + width <= image_width:
            around.append((right + gap, y, right + gap + width, y + height))
    for shift in (0, height + gap, 2 * (height + gap)):
        for y in (top - height - gap - shift, bottom + gap + shift):
            if 0 <= y <= image_height - height:
                for x in dict.fromkeys((left_aligned, right_aligned, centered)):
                    around.append((x, y, x + width, y + height))
        if shift == 0 and right - left >= 2 * width + 3 * gap and bottom - top >= height + 2 * gap:
            x = right - width - gap
            y = (top + bottom - height) // 2
            if 0 <= x <= image_width - width and 0 <= y <= image_height - height:
                around.append((x, y, x + width, y + height))
    if not around:
        y = min(max(top - height, 0), image_height - height)
        return left_aligned, y, left_aligned + width, y + height

    closest, least_cover = around[0], float("inf")
    for x, y, r, b in around:
        cover = sum(max(0, min(r, peer_right) - max(x, peer_left)) * max(0, min(b, peer_bottom) - max(y, peer_top))
                    for peer_left, peer_top, peer_right, peer_bottom in (*occupied, *other_controls))
        if cover == 0:
            return x, y, r, b
        if cover < least_cover:
            closest, least_cover = (x, y, r, b), cover
    return closest


def isolate_action_pixels(frame, active):
    """Keep only confirmed control pixels in their original image positions."""
    image = np.full_like(frame, 27)
    height, width = frame.shape[:2]
    for item in active:
        left, top, right, bottom = item["box"]
        left, top, right, bottom = max(0, left), max(0, top), min(width, right), min(height, bottom)
        if left >= right or top >= bottom:
            continue
        crop = np.s_[top:bottom, left:right]
        if (outline := item.get("photo_outline_px")) is not None:
            mask = np.zeros((bottom - top, right - left), np.uint8)
            points = np.rint(np.asarray(outline) - [left, top]).astype(np.int32)
            cv2.fillConvexPoly(mask, points, 1)
            np.copyto(image[crop], frame[crop], where=mask[..., None] != 0)
        else:
            image[crop] = frame[crop]
    return image


def draw_actions(frame, active, review):
    image = frame.copy()
    drawings, occupied = [], []
    for items, default_color in ((active, (90, 231, 145)), (review, (85, 182, 245))):
        for item in items:
            left, top, right, bottom = item["box"]
            drag = item.get("role") == "adjust_slider"
            color = (239, 193, 109) if drag and items is active else default_color
            label = item.get("preview_text")
            if not label or label.startswith("Text ("):
                label = item.get("function") or PREVIEW.get(item["role"], "Action")
            if items is review:
                label += "?"
            point = tuple(round(value) for value in item["click_point"])
            marker = cv2.MARKER_DIAMOND if drag else cv2.MARKER_CROSS
            drawings.append((item["box"], label, color, point, item.get("photo_outline_px"), marker))
            x, y = point
            occupied.extend(((x - 11, y - 11, x + 12, y + 12),
                             (left, top - 2, right, top + 2), (left, bottom - 2, right, bottom + 2),
                             (left - 2, top, left + 2, bottom), (right - 2, top, right + 2, bottom)))
    boxes = [row[0] for row in drawings]
    for number, (box, label, color, _, _, _) in enumerate(drawings):
        tile = _label_image(label, color)
        factor = min(1., image.shape[0] / tile.shape[0], image.shape[1] / tile.shape[1])
        if factor < 1:
            size = tuple(max(1, round(side * factor)) for side in tile.shape[1::-1])
            tile = cv2.resize(tile, size, interpolation=cv2.INTER_AREA)
        x, y, right, bottom = _label_position(tile, box, image.shape, occupied, boxes[:number] + boxes[number + 1:])
        image[y:bottom, x:right] = tile
        occupied.append((x, y, right, bottom))
    for (left, top, right, bottom), _, color, point, outline, marker in drawings:
        if outline is not None:
            cv2.polylines(image, [np.rint(outline).astype(np.int32)], True, color, 3, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (left, top), (right - 1, bottom - 1), color, 3)
        cv2.drawMarker(image, point, color, marker, 16, 2, cv2.LINE_AA)
    return image
