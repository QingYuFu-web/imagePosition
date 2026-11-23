"""Read isolated control labels with a local font index and verify their painted carrier."""

from itertools import combinations
import re
import unicodedata

import cv2
import numpy as np

from screen_button_text_windows import text_windows
from screen_calendar_choices import calendar_choices
from screen_caption_transcript import read_caption
from screen_choice_grid import grid_choice_controls
from screen_choice_row_geometry import choice_row_controls
from screen_component_captions import CaptionControl, caption_controls
from screen_font_consensus import verify_window
from screen_label_ownership import owned_label_box, supplement_command_cells


_VISIBLE_VERBS = {
    "开": "打开", "開": "打开", "打开": "打开", "打開": "打开", "open": "打开",
    "关闭": "关闭", "關閉": "关闭", "close": "关闭", "dismiss": "关闭",
    "取消": "取消", "cancel": "取消", "确定": "确认", "確認": "确认", "确认": "确认",
    "確定": "确认", "confirm": "确认", "ok": "确认", "继续": "继续", "繼續": "继续", "continue": "继续",
    "重试": "重试", "重試": "重试", "retry": "重试", "搜索": "搜索", "搜尋": "搜索", "search": "搜索",
    "分享": "打开分享", "share": "打开分享",
}
_CODE = re.compile(r"[A-Z]{1,5}(?:-[A-Z]{1,5})?-[0-9]{1,4}\Z")
_OWNED_LABELS = {
    "设置": ("打开设置", "capsule"), "設定": ("打开设置", "capsule"), "settings": ("打开设置", "capsule"),
    "评价": ("查看评价", "icon_row"), "評價": ("查看评价", "icon_row"), "reviews": ("查看评价", "icon_row"),
    "排序": ("打开排序", "sort"), "sort": ("打开排序", "sort"),
    "筛选": ("打开筛选", "filter"), "篩選": ("打开筛选", "filter"), "filter": ("打开筛选", "filter"),
    "搜索": ("搜索", "search"), "搜尋": ("搜索", "search"), "search": ("搜索", "search"),
}


def _center(box):
    left, top, right, bottom = box
    return [(left + right) / 2, (top + bottom) / 2]


def _intersect(first, second):
    left, top, right, bottom = first
    a, b, c, d = second
    return max(0, min(right, c) - max(left, a)) * max(0, min(bottom, d) - max(top, b))


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _text_key(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _tight_lines(readouts):
    retained = []
    for item in sorted(readouts, key=lambda row: (-row["support"], -row["min_margin"], row["max_distance"])):
        if not any(_intersect(peer["line_box"], item["line_box"]) >= .72 * min(_area(peer["line_box"]), _area(item["line_box"]))
                   for peer in retained):
            retained.append(item)
    return sorted(retained, key=lambda row: (row["line_box"][1], row["line_box"][0]))


def _code_like(masks):
    return 4 <= len(masks) <= 7 and any(
        mask.shape[1] >= 2.3 * mask.shape[0] and mask.mean() >= .62 for mask in masks[1:-1])


def _shortlist_action(frame, window, index):
    """Use agreement across contrast cuts for an isolated printed action."""
    if (window["support"] < 5 or len(window["masks"]) != 1
            or not _painted_button(frame, window["box"], window["line_box"])):
        return None
    mask = window["masks"][0]
    ranked = index.rank(mask, limit=2)
    if len(ranked) != 2:
        return None
    best, other = ranked
    margin = other["distance"] - best["distance"]
    if _text_key(best["text"]) not in _VISIBLE_VERBS or best["distance"] > .205 or margin < .025:
        return None
    alternatives = [index.rank(masks[0], limit=2) for masks in window["variants"]]
    if not any(len(sample) == 2 and sample[0]["text"] == best["text"] and sample[0]["distance"] <= .225
               and sample[1]["distance"] - sample[0]["distance"] >= .016 for sample in alternatives):
        return None
    return {**best, "margin": round(margin, 5)}


def read_short_labels(frame, regions, index, *, controls=None, dates=None):
    """Read short labels from proposed regions, leaving ambiguous matches empty."""
    readouts = []
    windows = text_windows(frame, regions)
    cache = {}
    for window in windows:
        glyphs = []
        alphabet = "code" if _code_like(window["masks"]) else "all"
        for mask in window["masks"]:
            key = (alphabet, mask.shape, mask.tobytes())
            if key not in cache:
                cache[key] = index.recognize(mask, alphabet=alphabet)
            glyph = cache[key]
            if glyph is None:
                if (literal := _shortlist_action(frame, window, index)) is not None:
                    glyphs.append(literal)
                break
            glyphs.append(glyph)
        if not glyphs or len(glyphs) != len(window["masks"]):
            if alphabet != "all" or (confirmed := verify_window(window, index)) is None:
                continue
            glyphs = confirmed
        text = "".join(result["text"] for result in glyphs)
        if alphabet == "code" and _CODE.fullmatch(text) is None:
            continue
        if not any(char.isalnum() for char in text) or not all(char.isalnum() or char in "-+·" for char in text):
            continue
        readouts.append({"id": window["id"], "box": list(window["box"]), "line_box": list(window["line_box"]),
                         "text": text, "polarity": window["polarity"], "support": window["support"],
                         "recognition": alphabet,
                         "min_margin": min(glyph["margin"] for glyph in glyphs),
                         "max_distance": max(glyph["distance"] for glyph in glyphs)})
    retained = _tight_lines(readouts)
    if controls is None:
        controls = _caption_controls(frame, regions)
    for control in controls:
        kind = control.purpose
        verified = read_caption(frame, control.caption_area, index, cache=cache,
                                choice=kind in {"select", "expand", "collapse", "checkbox", "radio", "uncheck"},
                                numeric=kind == "select")
        if verified is not None and _centered_caption(verified["line_box"], control.caption_area):
            retained.append({**verified, "id": control.region_id, "box": list(control.box),
                             "component_purpose": kind, "selected": control.selected,
                             **({"disabled": True} if control.disabled else {}),
                             **({"component_point": list(control.click_point)} if control.click_point else {})})
    if dates is None:
        dates = calendar_choices(frame, regions)
    for date in dates:
        year = f"{date.year}年" if date.year is not None else ""
        retained.append({"id": date.region_id, "box": list(date.box), "line_box": list(date.label_box),
                         "text": date.text, "recognition": "component", "component_purpose": "date",
                         "component_point": list(date.click_point),
                         "component_function": f"选择{year}{date.month}月{date.day}日", "selected": date.selected})
    return retained, len(windows) + len(controls) + len(dates)


def _centered_caption(line, area):
    x, y = _center(line)
    cx, cy = _center(area)
    return abs(x - cx) <= .16 * (area[2] - area[0]) and abs(y - cy) <= .22 * (area[3] - area[1])


def _caption_controls(frame, regions):
    rows = []
    for row in choice_row_controls(frame, regions):
        checked = row.kind == "checkbox" and row.state == "on"
        kind = "uncheck" if checked else row.kind
        selected = row.selected and not checked
        rows.append(CaptionControl(row.region_id, row.box, row.caption_area, kind, selected,
                                   bool(row.disabled), row.click_point))
    return caption_controls(frame, regions) + grid_choice_controls(frame, regions) + rows


_COMPONENT_ACTIONS = {"switch": "切换到", "select": "选择", "checkbox": "勾选", "radio": "选择",
                      "expand": "展开", "collapse": "收起", "uncheck": "取消勾选"}


def _painted_button(frame, box, label):
    left, top, right, bottom = box
    w, h = right - left, bottom - top
    if not .62 <= w / h <= 6.5:
        return False
    x, y = _center(label)
    if (abs(x - (left + right) / 2) > .15 * w or abs(y - (top + bottom) / 2) > .16 * h
            or not .10 * w <= label[2] - label[0] <= .79 * w
            or not .12 * h <= label[3] - label[1] <= .63 * h):
        return False
    pad = max(3, round(.12 * min(w, h)))
    x0, y0 = left - pad, top - pad
    if x0 < 0 or y0 < 0 or right + pad > frame.shape[1] or bottom + pad > frame.shape[0]:
        return False
    crop = frame[y0:bottom + pad, x0:right + pad]
    factor = min(1., 192 / max(crop.shape[:2]))
    small = cv2.resize(crop, (round(crop.shape[1] * factor), round(crop.shape[0] * factor)), interpolation=cv2.INTER_AREA)
    colors = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.int16)
    width, height = w * factor, h * factor
    cx, cy = colors.shape[1] / 2, colors.shape[0] / 2
    yy, xx = np.ogrid[:colors.shape[0], :colors.shape[1]]
    radius = np.hypot((xx - cx) / width, (yy - cy) / height)
    ring = colors[(radius >= .33) & (radius <= .39)]
    corners = colors[(abs(xx - cx) > .53 * width) & (abs(yy - cy) > .53 * height)]
    if not ring.size or not corners.size:
        return False
    paint = np.median(ring, axis=0)
    if (np.mean(np.max(abs(ring - paint), axis=1) <= 16) < .84
            or np.mean(np.linalg.norm(corners - paint, axis=1) >= 25) < .79):
        return False
    ink = np.uint8(np.max(abs(colors - paint), axis=2) <= 17)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x1, y1, sw, sh = cv2.boundingRect(contour)
        if (.73 * width <= sw <= 1.16 * width and .73 * height <= sh <= 1.16 * height
                and abs(x1 + sw / 2 - cx) <= .09 * width and abs(y1 + sh / 2 - cy) <= .09 * height
                and cv2.contourArea(contour) >= .62 * sw * sh):
            return True
    return False


def _code_plate(light, item):
    x0, y0, x1, y1 = item["box"]
    left, top, right, bottom = item["line_box"]
    w, h = x1 - x0, y1 - y0
    em = bottom - top
    pad = max(2, round(.19 * em))
    band = light[top - pad:bottom + pad, left - pad:right + pad]
    return (item["polarity"] == "dark" and 2 <= w / h <= 5.4 and .29 * h <= em <= .67 * h
            and left - pad >= x0 and right + pad <= x1 and top - pad >= y0 and bottom + pad <= y1
            and (band[:pad].mean() >= .82 or band[-pad:].mean() >= .82)
            and min(band[:, :pad].mean(), band[:, -pad:].mean()) >= .77)


def _near(box, x, y, reach):
    left, top, right, bottom = box
    return left - reach <= x <= right + reach and top - reach <= y <= bottom + reach


def _graph_codes(frame, labels):
    codes = [item for item in labels if item["recognition"] == "code"]
    if len(codes) < 3:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    light = (gray >= 212) & (np.ptp(frame, axis=2) <= 37)
    plates = [item for item in codes if _code_plate(light, item)]
    if len(plates) < 3:
        return []
    short = min(gray.shape)
    # Only links crossing empty space between label plates are useful.
    links = cv2.HoughLinesP(np.uint8(light) * 255, 1, np.pi / 180, max(14, round(.047 * short)),
                           minLineLength=max(12, round(.057 * short)), maxLineGap=max(3, round(.010 * short)))
    if links is None:
        return []
    diagonals = [tuple(map(int, line)) for line in links.reshape(-1, 4)
                 if min(abs(line[2] - line[0]), abs(line[3] - line[1])) >= .027 * short]
    reach = .044 * short
    edges = [set() for _ in plates]
    for a, b in combinations(range(len(plates)), 2):
        first, second = plates[a], plates[b]
        box1, box2 = first["box"], second["box"]
        h1, h2 = box1[3] - box1[1], box2[3] - box2[1]
        if (first["text"].split("-", 1)[0] != second["text"].split("-", 1)[0]
                or first["text"] == second["text"] or _intersect(box1, box2) or not .67 <= h1 / h2 <= 1.49):
            continue
        if any((_near(box1, x0, y0, reach) and _near(box2, x1, y1, reach)) or
               (_near(box1, x1, y1, reach) and _near(box2, x0, y0, reach)) for x0, y0, x1, y1 in diagonals):
            edges[a].add(b)
            edges[b].add(a)
    supported = []
    visited = set()
    for i, peers in enumerate(edges):
        if not peers or i in visited:
            continue
        cluster = {i}
        frontier = set(peers)
        while frontier:
            j = frontier.pop()
            cluster.add(j)
            frontier.update(edges[j] - cluster)
        visited.update(cluster)
        if len(cluster) >= 3:
            supported.extend(plates[j] for j in sorted(cluster))
    return supported


def propose_label_actions(frame, regions, labels):
    """Recognize a literal action only on a separately visible control surface."""
    if regions["layer"] not in {"page", "dialog", "bottom_sheet", "layer_uncertain"}:
        return []
    candidates = {row["id"]: row for row in regions["candidates"]}
    actions = []
    gray = None
    for item in labels:
        if item["recognition"] == "component":
            if not item["selected"] and not item.get("disabled"):
                box = item["box"]
                text = item["text"]
                verb = _COMPONENT_ACTIONS.get(item["component_purpose"])
                function = item.get("component_function") or (f"{verb}「{text}」" if verb else text)
                if not any(_intersect(box, prior["box"]) >= .67 * min(_area(box), _area(prior["box"]))
                           for prior in actions):
                    point = item.get("component_point") or _center(box)
                    actions.append({"role": "caption_control", "function": function, "click_point": point,
                                    "box": box, "text": text, "preview_text": text if text.isascii()
                                    else f"Text ({item['id']})"})
            continue
        candidate = candidates.get(item["id"])
        if candidate is None or candidate["visibility"] == "layer_uncertain":
            continue
        key = _text_key(item["text"])
        function = _VISIBLE_VERBS.get(key)
        if function and _painted_button(frame, candidate["box"], item["line_box"]):
            box = candidate["box"]
        elif (region := _OWNED_LABELS.get(key)) and regions["layer"] == "page":
            if gray is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            function, family = region
            box = owned_label_box(gray, candidate, regions["candidates"], item["line_box"], family)
            if box is None:
                continue
        else:
            continue
        if not any(_intersect(box, prior["box"]) >= .67 * min(_area(box), _area(prior["box"])) for prior in actions):
            actions.append({"role": "labeled_control", "function": function, "click_point": _center(box),
                            "box": list(box), "text": item["text"], "preview_text": f"Text ({item['id']})"})
    if regions["layer"] == "page":
        for item in _graph_codes(frame, labels):
            box = item["box"]
            if not any(_intersect(box, prior["box"]) >= .67 * min(_area(box), _area(prior["box"])) for prior in actions):
                actions.append({"role": "select_connected_entry", "function": f"选择 {item['text']}",
                                "click_point": _center(box), "box": box, "text": item["text"],
                                "preview_text": item["text"]})
    return actions


def additional_label_actions(frame, regions, icon_actions, labels):
    """Keep the icon path's established click when both paths find one control."""
    radius = .025 * min(frame.shape[:2])
    return [action for action in propose_label_actions(frame, regions, labels)
            if all(max(abs(x - y) for x, y in zip(action["click_point"], prior["click_point"], strict=True)) > radius
                   for prior in icon_actions)]


def resolve_short_labels(frame, regions, index, icon_actions):
    """Resolve captions and apply visual selected-state suppression to all click sources."""
    controls = _caption_controls(frame, regions)
    dates = calendar_choices(frame, regions)
    labels, count = read_short_labels(frame, regions, index, controls=controls, dates=dates)
    selected = [control.box for control in controls if control.selected or control.disabled]
    selected.extend(date.box for date in dates if date.selected)

    def active(items):
        return [action for action in items if not any(left <= action["click_point"][0] < right and
                                                      top <= action["click_point"][1] < bottom
                                                      for left, top, right, bottom in selected)]

    base = active(icon_actions)
    additions = active(additional_label_actions(frame, regions, base, labels))
    return base, additions, labels, count
