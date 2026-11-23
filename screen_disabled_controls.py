"""Measure unavailable affordances from their own visible state cues."""

from itertools import combinations

import cv2
import numpy as np

from inventory_compact_video_rail import _curved_arrow
from inventory_counted_reaction_column import _number_band
from inventory_list_rows import _Box
from screen_colored_settings_panel import _printed
from screen_landscape_segmented_choices import _centered_ink, _lattice, _selected_faces, _transcribe
from screen_user_examples import _contains, _coverage


def _paint(gray, box):
    left, top, right, bottom = box
    inset = round(.14 * (bottom - top))
    return np.median(gray[top + inset:bottom - inset, left + inset:right - inset:2])


def _dimmed_choice_boxes(frame, outcome, index):
    if index is None:
        return ()
    choices = [item for item in outcome.active if item["role"] == "select_option"]
    if len(choices) < 4:
        return ()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = [box for box in _selected_faces(gray) if _centered_ink(gray, box, selected=True)]
    if (geometry := _lattice(faces)) is None:
        return ()
    faces, pitch = geometry
    result = []
    for l, t, r, b in faces:
        w, h = r - l, b - t
        peers = [item["box"] for item in choices if abs(item["click_point"][1] - (t + b) / 2) <= .13 * h
                 and .91 * w <= item["box"][2] - item["box"][0] <= 1.10 * w
                 and abs((item["box"][0] - l) / pitch - round((item["box"][0] - l) / pitch)) <= .11 * h / pitch]
        style = [(box, ink[1], ink[3]) for box in peers if (ink := _centered_ink(gray, box)) is not None]
        if len(style) < 2:
            continue
        brightest = min(peak for _, peak, _ in style)
        glyph_height = float(np.median([height for _, _, height in style]))
        painted = float(np.median([_paint(gray, box) for box, _, _ in style]))
        for column in sorted({round((left - l) / pitch) for left, top, _, _ in faces if abs(top - t) >= h}):
            if column == 0:
                continue
            x = l + round(column * pitch)
            box = x, t, x + w, b
            if (not 0 <= x < x + w <= frame.shape[1]
                    or any(_coverage(box, item["box"]) >= .38 for item in outcome.active)):
                continue
            owners = [item for item in outcome.regions["candidates"] if item["visibility"] == "page"
                      and not item.get("review_only") and not item.get("disabled") and _contains(box, item["center"])
                      and _coverage(box, item["box"]) >= .85 and _coverage(item["box"], box) >= .43]
            if len(owners) != 1 or abs(_paint(gray, box) - painted) > 17:
                continue
            ink = _centered_ink(gray, box)
            if (ink is not None and 57 <= ink[1] <= .51 * brightest
                    and .78 * glyph_height <= ink[3] <= 1.25 * glyph_height and _transcribe(ink, index)):
                result.append(box)
    return tuple(result)


_SLASH_COLS = np.arange(10, 54)
_SLASH_LINES = np.rint([_SLASH_COLS * slope + offset for slope in np.arange(.60, 1.21, .05)
                        for offset in range(4, 19, 2)]).astype(int)


def _has_slash(mask):
    glyph = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST)
    # Read paint on both sides of the stroke. Optical sampling can widen the
    # gap by about one original pixel without changing the underlying shape.
    for flank in (7, 9):
        for line in _SLASH_LINES:
            valid = (line >= flank) & (line < 64 - flank)
            if valid.sum() < 31:
                continue
            y, x = line[valid], _SLASH_COLS[valid]
            corridor = (glyph[y, x] == 0) & (glyph[y - flank, x] != 0) & (glyph[y + flank, x] != 0)
            if corridor.mean() >= .78:
                return True
    return False


def _bubble_at_level(patch, spread, level, target_x, scale):
    _, labels, stats, centers = cv2.connectedComponentsWithStats(np.uint8((patch >= level) & (spread <= 42)))
    components = [(i, (x, y, w, h, area)) for i, (x, y, w, h, area) in enumerate(stats[1:], 1)
                  if .38 * scale <= min(w, h) <= max(w, h) <= 1.23 * scale
                  and abs(centers[i][0] - target_x) <= .33 * scale and area >= .12 * scale ** 2]
    found = []
    for (a, first), (b, second) in combinations(components, 2):
        if not .31 <= min(first[4], second[4]) / max(first[4], second[4]) <= .68:
            continue
        top, lower = sorted((centers[a], centers[b]), key=lambda center: center[1])
        if top[0] - lower[0] < .16 * scale or lower[1] - top[1] < .19 * scale:
            continue
        pair = np.uint8((labels == a) | (labels == b))
        l, t, w, h = cv2.boundingRect(pair)
        if (not .68 * scale <= min(w, h) <= max(w, h) <= 1.23 * scale
                or not .84 <= w / h <= 1.20 or abs(l + w / 2 - target_x) > .24 * scale
                or not .52 <= pair[t:t + h, l:l + w].mean() <= .76):
            continue
        reach = 2 * max(2, round(.09 * scale)) + 1
        closed = cv2.morphologyEx(pair, cv2.MORPH_CLOSE, np.fliplr(np.eye(reach, dtype=np.uint8)))
        boundaries, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(boundaries) != 1 or not .68 <= cv2.contourArea(boundaries[0]) / (w * h) <= .86:
            continue
        shape = closed[t:t + h, l:l + w]
        bottom = np.flatnonzero(shape[-1])
        mid = shape[round(.27 * h):round(.66 * h)]
        if (not .11 * w <= len(bottom) <= .50 * w or abs(bottom.mean() - w / 2) < .075 * w
                or np.median(mid.sum(axis=1)) < .74 * w or not _has_slash(pair[t:t + h, l:l + w])):
            continue
        found.append((l, t, l + w, t + h))
    return found[0] if len(found) == 1 else None


def _local_share_anchor(light, candidates):
    height, width = light.shape
    anchors = []
    for candidate in candidates:
        cx, cy = candidate["center"]
        if not .72 * height <= cy <= .95 * height or not (cx <= .19 * width or cx >= .81 * width):
            continue
        peers = sorted(item["center"][1] for item in candidates
                       if abs(cx - item["center"][0]) <= .03 * width and item["center"][1] <= cy)
        if len(peers) < 4:
            continue
        gaps = np.diff(peers[-4:])
        pitch = float(np.median(gaps))
        if not .072 * height <= pitch <= .16 * height or max(abs(gaps - pitch)) > .13 * pitch:
            continue
        left, top, right, bottom = candidate["box"]
        patch = light[top:min(bottom, top + round(.58 * pitch)), left:right]
        shapes = []
        for level in (241, 243, 244, 246, 248):
            contours, _ = cv2.findContours(np.uint8(patch >= level), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                l, t, w, h = cv2.boundingRect(contour)
                if not .22 * pitch <= min(w, h) <= max(w, h) <= .49 * pitch:
                    continue
                mask = cv2.drawContours(np.zeros((h, w), np.uint8), [contour], 0, 1, cv2.FILLED, offset=(-l, -t))
                if _curved_arrow(mask) and abs(left + l + w / 2 - cx) <= .14 * pitch:
                    shapes.append((level, (left + l, top + t, left + l + w, top + t + h)))
        stable = [b for level, b in shapes if any(level != other
                  and max(abs(x - y) for x, y in zip(b, peer, strict=True)) <= .043 * pitch for other, peer in shapes)]
        if stable and np.ptp(stable, axis=0).max() <= .052 * pitch:
            anchors.append(_Box(*map(int, np.median(stable, axis=0))))
    return anchors[0] if len(anchors) == 1 else None


def _crossed_comment_boxes(frame, outcome):
    height, width = frame.shape[:2]
    side = [item for item in outcome.active if item["role"] == "share"
            and (item["click_point"][0] >= .81 * width or item["click_point"][0] <= .19 * width)]
    if len(side) > 1:
        return ()
    light = cv2.min(cv2.min(frame[..., 0], frame[..., 1]), frame[..., 2])
    candidates = [item for item in outcome.regions["candidates"]
                  if item["visibility"] == "page" and not item.get("review_only") and not item.get("disabled")]
    share = _Box(*side[0]["box"]) if side else _local_share_anchor(light, candidates)
    if share is None:
        return ()
    x, y = share.center
    span = (share.width + share.height) / 2
    if not .034 * width <= span <= .14 * width:
        return ()
    peers = [item for item in candidates if abs(x - item["center"][0]) <= .043 * width]
    owners = [item for item in peers if .067 * height <= y - item["center"][1] <= .17 * height
              and any(.75 * (y - item["center"][1]) <= item["center"][1] - above["center"][1]
                      <= 1.54 * (y - item["center"][1]) for above in peers)]
    if len(owners) != 1:
        return ()
    left, top, right, bottom = owners[0]["box"]
    stop = min(bottom, top + round(1.63 * span))
    if min(right - left, stop - top) < span:
        return ()
    luminous = np.percentile(light[share.top:share.bottom, share.left:share.right], 92)
    found = []
    spread = np.ptp(frame[top:stop, left:right], axis=2)
    for level in (round(.62 * luminous), round(.655 * luminous), round(.69 * luminous)):
        if (box := _bubble_at_level(light[top:stop, left:right], spread, level, x - left, span)) is not None:
            found.append(box)
    if len(found) < 2 or np.ptp(found, axis=0).max() > .09 * span:
        return ()
    l, t, r, b = map(int, np.median(found, axis=0))
    bubble = _Box(left + l, top + t, left + r, top + b)
    reach = round(.78 * bubble.width)
    center = round(bubble.center[0])
    x0, x1 = max(left, center - reach), min(right, center + reach)
    # A bright edge of the photographed glass may run through the caption
    # rows. Restrict the count witness to the same centered rail column.
    local = _Box(bubble.left - x0, bubble.top, bubble.right - x0, bubble.bottom)
    following = _Box(share.left - x0, share.top, share.right - x0, share.bottom)
    number = _number_band(light[:, x0:x1], local, following, y - bubble.center[1])
    if number is None:
        return ()
    pad = max(2, round(.11 * span))
    return ((max(left, min(bubble.left, x0 + number.left) - pad), max(top, bubble.top - pad),
             min(right, max(bubble.right, x0 + number.right) + pad), min(bottom, number.bottom + pad)),)


_COMMIT_WORDS = ("发表", "發表", "发布", "發佈", "发送", "發送", "提交", "Post", "Publish", "Send", "Submit")
_OTHER_TOOLBAR_WORDS = ("确认", "完成", "下一步", "取消", "返回", "放弃", "保存", "分享", "更多", "选择",
                        "Save", "Next", "Done", "Cancel", "Close", "Back", "Share")


def _muted_caption(gray, color, box, paper, index):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    # The sides of the fill are separated from both the outline and the label.
    fill_rows = slice(top + round(.28 * h), bottom - round(.28 * h))
    inner, outer = max(2, round(.08 * w)), max(4, round(.19 * w))
    strips = (slice(left + inner, left + outer), slice(right - outer, right - inner))
    face = np.concatenate([gray[fill_rows, strip].ravel() for strip in strips])
    face_color = np.concatenate([color[fill_rows, strip].reshape(-1, 3) for strip in strips])
    dark, fill, light = np.percentile(face, (12, 50, 88))
    if (not 8 <= paper - fill <= 38 or light - dark > 8
            or np.percentile(np.ptp(face_color, axis=1), 86) > 11):
        return None
    side, vertical = max(2, round(.065 * w)), max(2, round(.075 * h))
    chip = gray[top + vertical:bottom - vertical, left + side:right - side]
    strokes = cv2.subtract(np.full_like(chip, round(fill)), chip)
    if not 17 <= np.percentile(strokes, 98) <= 58:
        return None
    strokes = cv2.add(strokes, strokes)
    for scale in (50 / len(chip), min(2, 104 / len(chip))):
        bright = cv2.resize(strokes, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        reading = _printed(bright, (0, 0, bright.shape[1], bright.shape[0]), index,
                           vocabulary=(*_COMMIT_WORDS, *_OTHER_TOOLBAR_WORDS), minimum_score=.84, minimum_margin=.025,
                           with_box=True)
        if reading is None:
            continue
        name, (x0, y0, x1, y1) = reading
        if (name in _COMMIT_WORDS and abs((x0 + x1) / 2 - bright.shape[1] / 2) <= scale * .12 * w
                and abs((y0 + y1) / 2 - bright.shape[0] / 2) <= scale * .21 * h
                and scale * .28 * h <= y1 - y0 <= scale * .69 * h):
            return name
        break
    return None


def disabled_draft_commit(frame, font_index, empty_draft):
    """Find the literal commit action whose own paint confirms it is unavailable."""
    if empty_draft is None or font_index is None:
        return None
    _, width = frame.shape[:2]
    _, top, right, bottom = empty_draft.toolbar_box
    _, title_top, title_right, title_bottom = empty_draft.title_box
    origin = max(right - round(.045 * width), title_right + round(.038 * width))
    color = frame[top:bottom, origin:]
    if color.shape[0] < 12 or color.shape[1] < 25:
        return None
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    paper = float(np.percentile(gray, 83))
    if paper < 232:
        return None
    # Search only this measured toolbar, using the fill-to-paper edge instead
    # of the patch rectangle supplied by the proposal detector.
    contours, _ = cv2.findContours(np.uint8(gray <= paper - 8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (not .078 * width <= w <= .245 * width or not .048 * width <= h <= .112 * width
                or not 1.25 <= w / h <= 3.6 or cv2.contourArea(contour) < .79 * w * h
                or not .012 * width <= width - (origin + x + w) <= .095 * width
                or not (0 < x < x + w < gray.shape[1] and 0 < y < y + h < gray.shape[0])
                or abs(top + y + h / 2 - (title_top + title_bottom) / 2) > .34 * h):
            continue
        box = (x, y, x + w, y + h)
        if (name := _muted_caption(gray, color, box, paper, font_index)) is not None:
            found.append(((origin + x, top + y, origin + x + w, top + y + h), name))
    return found[0] if len(found) == 1 else None


def disabled_control_boxes(frame, outcome, font_index):
    """Return only regions with independent, visible unavailable-state evidence."""
    if outcome.regions["layer"] != "page" or outcome.regions.get("regions"):
        return ()
    height, width = frame.shape[:2]
    if width >= 1.53 * height and height >= 220:
        return _dimmed_choice_boxes(frame, outcome, font_index)
    if 1.60 * width <= height <= 2.75 * width and width >= 230:
        return _crossed_comment_boxes(frame, outcome)
    return ()
