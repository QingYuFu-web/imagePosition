"""Locate visible icon-and-label destinations on a separated bottom bar."""

from collections import Counter
from difflib import SequenceMatcher
import unicodedata

import cv2
import numpy as np

from inventory_back_evidence import _elements
from inventory_toolbar_actions import _spans


def _normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


# A footer slot never assigns a destination's name.
_DESTINATIONS = {
    "首页": ("nav_home", "切换到首页", "Home"),
    "home": ("nav_home", "切换到首页", "Home"),
    "发现": ("tab", "切换到发现", "Discover"),
    "discover": ("tab", "切换到发现", "Discover"),
    "多多视频": ("tab", "切换到多多视频", "Video"),
    "视频": ("tab", "切换到视频", "Video"),
    "video": ("tab", "切换到视频", "Video"),
    "分类": ("tab", "切换到分类", "Category"),
    "categories": ("tab", "切换到分类", "Category"),
    "聊天": ("tab", "打开聊天", "Chat"),
    "chat": ("tab", "打开聊天", "Chat"),
    "消息": ("tab", "打开消息", "Messages"),
    "messages": ("tab", "打开消息", "Messages"),
    "收件箱": ("tab", "打开收件箱", "Inbox"),
    "inbox": ("tab", "打开收件箱", "Inbox"),
    "朋友": ("tab", "切换到好友", "Friends"),
    "好友": ("tab", "切换到好友", "Friends"),
    "friends": ("tab", "切换到好友", "Friends"),
    "购物车": ("cart", "购物车", "Cart"),
    "cart": ("cart", "购物车", "Cart"),
    "通讯录": ("contacts", "通讯录", "Contacts"),
    "contacts": ("contacts", "通讯录", "Contacts"),
    "个人中心": ("account_center", "个人中心", "Profile"),
    "我的": ("account_center", "个人中心", "Profile"),
    "profile": ("nav_profile", "打开个人主页", "Profile"),
    "account": ("account_center", "个人中心", "Profile"),
    "搜索": ("search", "搜索", "Search"),
    "search": ("search", "搜索", "Search"),
    "library": ("tab", "打开媒体库", "Library"),
    "yourlibrary": ("tab", "打开媒体库", "Your Library"),
    "premium": ("tab", "查看会员服务", "Premium"),
}
_NAV_PEERS = {"首页", "home", "个人中心", "我的", "profile", "account", "朋友", "好友", "friends"}
_COMPACT_DESTINATIONS = {"home", "friends", "inbox", "profile"}


def _surface(gray, *, compact=False):
    height, width = gray.shape
    margin = round(.07 * width)
    start = max(0, height - round(.28 * width))
    bottom = gray[start:, margin:width - margin].astype(np.int16)
    panel = float(np.median(bottom[-max(3, round(.034 * width)):]))
    if 65 < panel < 184 or (compact and panel > 8):
        return None
    flat = (np.abs(bottom - panel) <= 18).mean(axis=1)
    gap = max(2, round(.012 * width))
    for y in range(1, len(bottom) - round(.147 * width)):
        if flat[y] >= .89 and flat[y - 1] <= .55 and flat[y:y + gap].mean() >= .88:
            if np.mean(flat[y:] >= .64) >= (.88 if compact else .92):
                return (start + y), panel
    return None


def _trimmed_box(mask, left, top):
    x, y, w, h = cv2.boundingRect(np.uint8(mask))
    return (left + x, top + y, left + x + w, top + y + h)


def _label_regions(gray, top, panel):
    height, width = gray.shape
    y0, y1 = top + round(.082 * width), min(height, top + round(.134 * width))
    rows = np.abs(gray[y0:y1].astype(np.int16) - panel)
    ink = rows >= (11 if 11 <= panel <= 65 else 33)
    # Join letters belonging to the same caption while preserving spacing between tabs.
    occupancy = np.uint8(ink.sum(axis=0) >= max(1, round(.0022 * width)))[None, :]
    kernel = np.ones((1, max(3, round(.019 * width) | 1)), np.uint8)
    columns = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, kernel)[0]
    for left, right in _spans(columns):
        if not .026 * width <= right - left <= .133 * width:
            continue
        box = _trimmed_box(ink[:, left:right], left, y0)
        if .012 * width <= box[3] - box[1] <= .038 * width:
            yield box


def _icon_for_label(gray, panel, top, label):
    width = gray.shape[1]
    left, label_top, right, _ = label
    center = (left + right) / 2
    a, b = max(0, round(center - .044 * width)), min(width, round(center + .044 * width))
    y0 = top + round(.015 * width)
    y1 = min(label_top - max(2, round(.008 * width)), top + round(.083 * width))
    if y1 - y0 < .034 * width or b <= a:
        return None
    patch = np.abs(gray[y0:y1, a:b].astype(np.int16) - panel)
    ink = patch >= (11 if 11 <= panel <= 65 else 32)
    if not ink.any():
        return None
    lx, ty, rx, by = _trimmed_box(ink, a, y0)
    w, h = rx - lx, by - ty
    if (not .031 * width <= w <= .088 * width or not .028 * width <= h <= .070 * width
            or abs((lx + rx) / 2 - center) > .026 * width
            or not .11 <= np.count_nonzero(ink) / (w * h) <= .91):
        return None
    return lx, ty, rx, by


def _gradient_choices(gray, elements, origin):
    """Pair observed captions with isolated bright icons on a dark fading footer."""
    height, width = gray.shape
    labels = [(left, top - origin, right, bottom - origin)
              for element in elements
              for left, top, right, bottom in (element.box,)
              if element.kind == "text" and element.confidence >= .90
              and _normalize(element.label) in _DESTINATIONS
              and .012 * width <= bottom - top <= .045 * width
              and .020 * width <= right - left <= .176 * width
              and 0 <= height + origin - element.center[1] <= .073 * width]
    if not 3 <= len(labels) <= 6:
        return []
    depth = round(.087 * width)
    start = min(label[1] for label in labels) - depth
    margin = round(.07 * width)
    band = max(3, round(.012 * width))
    if start < 0:
        return []
    background = np.median(gray[start:, margin:width - margin], axis=1).astype(np.int16)
    if np.percentile(background, 90) > 45 or np.median(background[:band]) - np.median(background[-band:]) < 4:
        return []
    contrast = np.maximum(gray[start:].astype(np.int16) - background[:, None], 0)
    choices = []
    for label in sorted(labels):
        left, top, right, bottom = label
        icon = _icon_for_label(contrast, 0, top - start - depth, (left, top - start, right, bottom - start))
        if icon is None:
            return []
        lx, ty, rx, by = icon
        choices.append((label, (lx, ty + start, rx, by + start)))
    if np.ptp([(icon[1] + icon[3]) / 2 for _, icon in choices]) > .014 * width:
        return []
    return choices


def _text_at(points, elements, label, width):
    left, top, right, bottom = label
    center = (left + right) / 2
    found = {}
    for element in elements:
        l, t, r, b = element.box
        if abs(element.center[0] - center) > .053 * width or b < top or t > bottom:
            continue
        if element.kind == "text":
            variants = ((element.label, element.confidence),)
        else:
            variants = [(item.get("text", ""), item.get("confidence", 0))
                        for item in points[element.id].get("text_evidence", ())]
        for text, confidence in variants:
            key = _normalize(text)
            if confidence >= .88 and key in _DESTINATIONS:
                anchor = element.id if element.kind == "control_candidate" else None
                if key not in found or anchor is not None:
                    found[key] = anchor
    return next(iter(found.items())) if len(found) == 1 else (None, None)


def _person_glyph(gray, icon, panel):
    left, top, right, bottom = icon
    ink = abs(gray[top:bottom, left:right].astype(np.int16) - panel) >= 51
    h, w = ink.shape
    head = np.uint8(ink[:round(.60 * h)])
    contours, hierarchy = cv2.findContours(head, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    has_head = any(.019 * h * w <= cv2.contourArea(contour) <= .24 * h * w
                   and abs(x + size / 2 - w / 2) <= .18 * w and .26 <= size / th <= 1.7
                   for contour, (_, _, _, parent) in zip(contours, hierarchy[0], strict=True) if parent != -1
                   for x, _, size, th in (cv2.boundingRect(contour),))
    shoulders = ink[-max(2, round(.21 * h)):]
    side = max(1, round(.20 * w))
    return (has_head and shoulders[:, :side].sum() >= 2 and shoulders[:, -side:].sum() >= 2
            and shoulders[:, side:-side].mean() <= .14)


def _compact_icon(gray, box, panel):
    left, top, right, bottom = box
    patch = gray[top:bottom, left:right].astype(np.float32) - panel
    peak = patch.max()
    if peak < 105:
        return None
    roles = set()
    for fraction in (.31, .40, .52, .63):
        ink = np.uint8(patch >= fraction * peak)
        x, y, w, h = cv2.boundingRect(ink)
        ink = ink[y:y + h, x:x + w]
        if min(h, w) < 7:
            continue
        contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [(cv2.contourArea(contour) / ink.size, cv2.boundingRect(contour))
                 for contour, entry in zip(contours, hierarchy[0], strict=True) if entry[3] >= 0]
        heads = [(l + shape_w / 2, t + shape_h / 2) for area, (l, t, shape_w, shape_h) in holes
                 if .012 <= area <= .18 and .55 <= shape_w / shape_h <= 1.55 and t + shape_h / 2 < .61 * h]
        shoulder = ink[round(.69 * h):]
        side = max(2, round(.24 * w))
        outside_left = shoulder[:, :side].mean()
        outside_right = shoulder[:, -side:].mean()
        middle = shoulder[:, round(.36 * w):round(.64 * w)].mean()
        if (len(heads) == 1 and abs(heads[0][0] - w / 2) <= .12 * w
                and min(outside_left, outside_right) >= .16 and middle <= .28):
            roles.add("profile")
        if (any(abs(a[0] - b[0]) >= .27 * w and abs(a[1] - b[1]) <= .32 * h
                for i, a in enumerate(heads) for b in heads[i + 1:])
                and ink[round(.65 * h):].mean() >= .12):
            roles.add("friends")
        if any(area >= .31 and t <= .25 * h and shape_w >= .63 * w
               for area, (_, t, shape_w, _) in holes):
            tail = ink[-max(2, round(.15 * h)):]
            _, _, stats, centers = cv2.connectedComponentsWithStats(ink)
            dash = any(.22 * w <= sw <= .62 * w and sh <= .30 * h and sw >= 1.45 * sh
                       and abs(cx - w / 2) <= .15 * w and .32 * h <= cy <= .67 * h
                       for (_, _, sw, sh, _), (cx, cy) in zip(stats[1:], centers[1:], strict=True))
            if dash and max(tail[:, :side].mean(), tail[:, -side:].mean()) <= .24 and tail[:, side:-side].any():
                roles.add("inbox")
        if not holes and peak >= 215:
            widths = ink.mean(axis=1)
            floor = cv2.resize(patch[y:y + h, x:x + w], (24, 24), interpolation=cv2.INTER_LINEAR)[17:22]
            doorway = np.mean(floor[:, 10:14]) + 30 < min(floor[:, 6:10].mean(), floor[:, 14:18].mean())
            if widths[:max(2, round(.20 * h))].mean() <= .43 and widths.max() >= .85 and doorway:
                roles.add("home")
    return next(iter(roles)) if len(roles) == 1 else None


def _recover(frame, pending, line_reader):
    if not pending or line_reader is None:
        return {}
    crops = []
    for _, (left, top, right, bottom), _ in pending:
        patch = frame[max(0, top - 1):bottom + 1, max(0, left - 1):right + 1]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if min(h, w) < 3:
            return {}
        if np.median(gray) < 124:
            gray = 255 - gray
        variants = [cv2.resize(gray, (max(4, round(factor * w * 38 / h)), 38), interpolation=cv2.INTER_LINEAR)
                    for factor in (1., 1.15)]
        smaller = cv2.resize(gray, (max(4, round(w * 27 / h)), 27), interpolation=cv2.INTER_CUBIC)
        variants.append(cv2.copyMakeBorder(smaller, 3, 3, 5, 5, cv2.BORDER_CONSTANT, value=255))
        crops.extend(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in variants)
    try:
        readings = line_reader.recognize_prepared_lines(crops)
    except (RuntimeError, cv2.error):
        return {}
    if len(readings) != len(crops):
        return {}
    results = {}
    for index, (slot, _, profile_shape) in enumerate(pending):
        consistent = {}
        for reading in readings[3 * index:3 * index + 3]:
            key = _normalize(reading["text"])
            if key in _DESTINATIONS:
                consistent.setdefault(key, []).append(reading["score"])
        accepted = [key for key, scores in consistent.items()
                    if (sum(score >= 91 for score in scores) >= 2 and max(scores) >= 94
                        or key == "profile" and profile_shape and sum(score >= 85 for score in scores) >= 2
                        and max(scores) >= 90)]
        if len(accepted) == 1:
            results[slot] = accepted[0]
    return results


def _recover_compact(gray, choices, answers, line_reader, panel, create):
    roles = [_compact_icon(gray, icon, panel) for _, icon in choices]
    roles = {i: role for i, role in enumerate(roles) if role and roles.count(role) == 1}
    if len(roles) < 3 or line_reader is None or not set(roles).difference(answers):
        return {}

    crops = []
    variants = ((2, 20, 1.50), (1, 20, 1.15), (2, 42, 1.15), (2, 30, 1.50))
    for i in roles:
        left, top, right, bottom = choices[i][0]
        for pad, height, widen in variants:
            patch = 255 - gray[top - pad:bottom + pad, max(0, left - 1):right + 1]
            h, w = patch.shape
            resized = cv2.resize(patch, (round(widen * w * height / h), height), interpolation=cv2.INTER_LINEAR)
            padded = cv2.copyMakeBorder(resized, 3, 3, 4, 4, cv2.BORDER_CONSTANT, value=255)
            crops.append(cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR))
    try:
        readings = line_reader.recognize_prepared_lines(crops)
    except (RuntimeError, cv2.error):
        return {}
    if len(readings) != len(crops):
        return {}

    exact, similar, candidates = set(), set(), {}
    for offset, (i, role) in enumerate(roles.items()):
        if i in answers:
            if answers[i][0] == role:
                exact.add(role)
            continue
        texts = [(_normalize(row["text"]), row["score"])
                 for row in readings[offset * len(variants):(offset + 1) * len(variants)]]
        if any(key in _DESTINATIONS and key != role and score >= 76 for key, score in texts):
            continue
        if any(role == key and score >= 75 for key, score in texts):
            exact.add(role)
        elif any(score >= 70 and SequenceMatcher(None, role, key).ratio() >= .72 for key, score in texts):
            similar.add(role)
        candidates[i] = role
    # Worn text can support a shape, while the color badge and at least one
    # exact caption identify this specific footer before shape-only recovery.
    if len(exact) < 2 and (create is None or not exact or not similar):
        return {}
    return {i: (role, None) for i, role in candidates.items()}


def _create_bubble(frame, gray, top, panel, center, *, compact=False):
    width = gray.shape[1]
    left, right = round(center - .067 * width), round(center + .067 * width)
    y0, y1 = top + round(.019 * width), top + round(.118 * width)
    chip = frame[y0:y1, left:right]
    light = gray[y0:y1, left:right]
    if panel > 25 or min(light.shape) < 8:
        return None
    core = light >= 198
    if not core.any():
        return None
    l, t, r, b = _trimmed_box(core, 0, 0)
    w, h = r - l, b - t
    if (not .089 * width <= w <= .126 * width or not .050 * width <= h <= .085 * width
            or not 1.23 <= w / h <= 1.93 or core[t:b, l:r].mean() < .65
            or abs(left + (l + r) / 2 - center) > .022 * width):
        return None
    yy, xx = (t + b) // 2, (l + r) // 2
    radius = max(2, round(.21 * h))
    dark = light[yy - radius:yy + radius + 1, xx - radius:xx + radius + 1] < (120 if compact else 80)
    near_center = slice(radius - 1, radius + 2)
    if not (dark[near_center].mean(axis=1).max() >= .77 and dark[:, near_center].mean(axis=0).max() >= .77):
        return None
    # Cyan and pink side accents distinguish the video-create button from a lone plus.
    blue, green, red = np.moveaxis(chip.astype(np.int16), 2, 0)
    cyan = (green - red >= 40) & (blue - red >= 36)
    pink = (red - green >= 52) & (red - blue >= 18)
    if not cyan[:, :xx].any() or not pink[:, xx:].any():
        return None
    return left + l, y0 + t, left + r, y0 + b


def suggest_labeled_bottom_navigation(frame, observation, *, line_reader=None):
    """Propose only icon-and-caption pairs on a physically separated footer."""
    if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
            or observation.get("active_layer") or observation.get("mode") == "home"):
        return []
    height, width = frame.shape[:2]
    if width < 180 or not 1.62 * width <= height <= 2.65 * width:
        return []
    origin = height - round(.29 * width)
    gray = cv2.cvtColor(frame[origin:], cv2.COLOR_BGR2GRAY)
    surface = _surface(gray)
    if surface is None and width <= 250:
        surface = _surface(gray, compact=True)
    elements = list(_elements(observation, width, height))
    if surface is None:
        choices = _gradient_choices(gray, elements, origin)
    else:
        top, panel = surface
        choices = [(label, icon) for label in _label_regions(gray, top, panel)
                   if (icon := _icon_for_label(gray, panel, top, label)) is not None]
    if not 3 <= len(choices) <= 6:
        return []
    centers = [(label[0] + label[2]) / 2 for label, _ in choices]
    counts = (len(choices),) if surface is None else range(3, 7)
    tolerance = .023 if surface is None else .036
    for count in counts:
        pitch = width / count
        indices = [round(center / pitch - .5) for center in centers]
        if (len(set(indices)) == len(indices) and max(indices) - min(indices) >= count - 2
                and all(abs(center - (slot + .5) * pitch) <= tolerance * width
                        for center, slot in zip(centers, indices, strict=True))):
            break
    else:
        return []
    points = {row["id"]: row for row in observation.get("keypoints", ())}
    answers = {}
    pending = []
    for i, (label, icon) in enumerate(choices):
        absolute = (label[0], label[1] + origin, label[2], label[3] + origin)
        key, anchor = _text_at(points, elements, absolute, width)
        if key is not None:
            answers[i] = (key, anchor)
        elif surface is not None:
            pending.append((i, absolute, _person_glyph(gray, icon, panel)))
    compact = surface is not None and width <= 250 and panel <= 8 and count == 5 and 2 not in indices
    create = None
    if compact and len(answers) < len(choices):
        create = _create_bubble(frame[origin:], gray, top, panel, width / 2, compact=True)
        answers.update(_recover_compact(gray, choices, answers, line_reader, panel, create))
        pending = [item for item in pending if item[0] not in answers]
    if pending:
        answers.update({i: (key, None) for i, key in _recover(frame, pending, line_reader).items()})
    role_counts = Counter(_DESTINATIONS[key][1] for key, _ in answers.values())
    supported = [(i, key, anchor) for i, (key, anchor) in answers.items()
                 if role_counts[_DESTINATIONS[key][1]] == 1]
    if (len(supported) < 3 or (surface is None and len(supported) != len(choices))
            or not any(key in _NAV_PEERS for _, key, _ in supported)):
        return []
    align = np.asarray([(choices[i][0][1] + choices[i][0][3]) / 2 for i, _, _ in supported])
    if np.ptp(align) > .011 * width:
        return []

    def proposal(box, role, function, caption, anchor):
        left, upper, right, lower = box
        sx, sy = width - 1, height - 1
        return {"role": role, "function": function, "anchor_id": anchor, "preview_text": caption,
                "box_fraction": [left / sx, (upper + origin) / sy, right / sx, (lower + origin) / sy],
                "screen_fraction": [(left + right) / 2 / sx, ((upper + lower) / 2 + origin) / sy]}

    result = [proposal(choices[i][1], *_DESTINATIONS[key], anchor) for i, key, anchor in supported]
    create_context = {key for _, key, _ in supported}
    if surface is not None and count == 5 and len(create_context & _COMPACT_DESTINATIONS) >= 3 and 2 not in indices:
        if create is None:
            create = _create_bubble(frame[origin:], gray, top, panel, width / 2, compact=compact)
        if create is not None:
            result.append(proposal(create, "create_video", "创作视频", "Create", None))
    return sorted(result, key=lambda row: row["screen_fraction"][0])
