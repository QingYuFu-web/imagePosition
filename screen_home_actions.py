"""Read app tiles from a measured home-screen grid and their printed captions."""

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from PIL import ImageFont

from inventory_segmented_actions import _rounded_shape
from screen_home_dock_actions import _dock_icons


_SYSTEM_APPS = (
    "相机 日历 照片 邮件 备忘录 提醒事项 时钟 视频 游戏 地图 健康 钱包 设置 计算器 天气 指南针 查找 "
    "录音 通讯录 图书 快捷指令 播客 商店 短信 电话 相簿 翻译 信息 音乐 股市 家庭 健身 文稿 手表 文件 "
    "新闻 邮箱 游览 笔记 通话 账单 运动 密码 浏览器 照相机 麦克风 联系人 搜索 控制中心 屏幕使用 查找设备 "
    "手机管家 微信 支付宝 抖音 小红书 拼多多 淘宝 京东 美团 百度 夸克 哔哩哔哩 快手"
).split() + [
    "App Store", "TikTok", "FaceTime 通话", "FaceTime", "Safari", "Photos", "Calendar", "Camera", "Mail",
    "Reminders", "Settings", "Clock", "Maps", "Games", "Health", "Wallet", "Notes", "Music", "Messages",
    "Calculator", "Weather", "Books", "Files", "News", "Contacts", "Phone", "Passwords", "Translate", "Stocks",
    "Shortcuts", "WhatsApp", "Instagram", "YouTube", "Gmail", "Facebook", "Chrome", "WeChat", "Telegram", "Spotify",
    "Home", "Watch", "Find My", "Fitness", "Voice Memos", "Compass", "Magnifier", "Tips", "Measure", "Utilities",
    "Freeform", "Journal", "Podcasts", "TV", "iTunes Store", "iMovie", "Pages", "Numbers", "Keynote", "Clips",
    "Apple Store", "Support", "Activity", "GarageBand", "Signal", "Skype", "Chat", "Discord",
    "Zoom", "Slack", "Teams", "Outlook", "Edge", "Drive", "OneDrive", "Dropbox", "Meet", "LINE", "Phone Link",
    "Messenger", "Threads", "Snapchat", "Pinterest", "Reddit", "Netflix", "Twitch", "FireFox", "KKInput",
]
_EXCLUDE_AS_APP = {"搜索", "Utilities"}
_QR_CAPTION = "KKInput"


@dataclass(frozen=True)
class _HomeGrid:
    columns: tuple[float, ...]
    rows: tuple[tuple[float, dict[int, tuple[int, ...]]], ...]
    tile_size: float
    pitch: float


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _dock_columns(dock, width):
    if len(dock) == 4:
        return dock, None
    if len(dock) != 3:
        return None, None
    centers = np.asarray([_center(box) for box in dock])
    gaps = np.diff(centers[:, 0])
    pitch = float(np.min(gaps))
    strides = np.rint(gaps / pitch).astype(int)
    candidates = []
    if abs(gaps / pitch - strides).max() > .09 or strides.sum() > 3:
        return None, None
    for start in range(4 - strides.sum()):
        occupied = start + np.r_[0, np.cumsum(strides)]
        anchor = np.median(centers[:, 0] - occupied * pitch)
        if abs(anchor + 1.5 * pitch - width / 2) <= .25 * pitch:
            missing, = set(range(4)) - set(occupied)
            shift = round(anchor + missing * pitch - centers[0, 0])
            left, top, right, bottom = dock[0]
            supplement = (left + shift, top, right + shift, bottom)
            candidates.append((sorted([*dock, supplement]), missing))
    return candidates[0] if len(candidates) == 1 else (None, None)


def _rounded_grid_boxes(chip, columns, size, pitch, *, origin):
    left, top = origin
    channels = (*cv2.split(chip), cv2.cvtColor(chip, cv2.COLOR_BGR2HSV)[:, :, 1])
    kernel = np.ones((5, 5), np.uint8)
    for channel in channels:
        smooth = cv2.GaussianBlur(channel, (3, 3), .6)
        for low, high in ((13, 36), (28, 73)):
            edges = cv2.morphologyEx(cv2.Canny(smooth, low, high), cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = bounds = cv2.boundingRect(contour)
                if (.83 * size <= min(w, h) <= max(w, h) <= 1.15 * size and .88 <= w / h <= 1.14
                        and min(abs(column - left - x - w / 2) for column in columns) <= .08 * pitch
                        and _rounded_shape(contour, bounds)):
                    yield left + x, top + y, left + x + w, top + y + h


def _grid_contours(frame, dock):
    _, width = frame.shape[:2]
    col_x = np.asarray([_center(box)[0] for box in dock])
    pitch = float(np.median(np.diff(col_x)))
    size = float(np.median([(box[2] - box[0] + box[3] - box[1]) / 2 for box in dock]))
    bottom = round(_center(dock[0])[1] - .72 * pitch)
    top = round(.10 * width)
    chip = frame[top:bottom]
    if len(chip) < 3 * pitch:
        return (), col_x, size, pitch
    boxes = set(_rounded_grid_boxes(chip, col_x, size, pitch, origin=(0, top)))
    ordered = sorted(boxes, key=lambda b: (abs(b[2] - b[0] - size) + abs(b[3] - b[1] - size), b[1], b[0]))
    unique = []
    for box in ordered:
        x, y = _center(box)
        if all(abs(x - xx) + abs(y - yy) >= .15 * size for xx, yy, _ in unique):
            unique.append((x, y, box))
    return unique, col_x, size, pitch


def _body_rows(boxes, columns, size, pitch):
    rows = []
    for x, y, box in sorted(boxes, key=lambda item: item[1]):
        col = int(np.argmin(abs(columns - x)))
        if rows and abs(y - rows[-1][0]) <= .15 * size:
            rows[-1][1].setdefault(col, box)
        else:
            rows.append((y, {col: box}))
    dense = [(np.median([_center(b)[1] for b in row.values()]), row) for _, row in rows if len(row) >= 2]
    if len(dense) < 3:
        return ()
    centers = [point for point, _ in dense]
    gaps = [gap for gap in np.diff(centers) if .87 * pitch <= gap <= 1.30 * pitch]
    if len(gaps) < 2:
        return ()
    pitch_y = float(np.median(gaps))
    aligned = [row for y, row in rows
               if abs((y - centers[0]) / pitch_y - round((y - centers[0]) / pitch_y)) <= .10]
    if len(aligned) < 3 or len({col for row in aligned for col in row}) < 3:
        return ()
    return aligned


def _edge_face(frame, x, y, size):
    edges = []
    for image, expected, extent in ((frame, y - size / 2, x), (frame, y + size / 2, x),
                                    (frame.transpose(1, 0, 2), x - size / 2, y),
                                    (frame.transpose(1, 0, 2), x + size / 2, y)):
        lo, hi = round(extent - .26 * size), round(extent + .26 * size)
        gap = max(2, round(.03 * size))
        indices = np.arange(round(expected - .09 * size), round(expected + .09 * size) + 1)
        if lo < 0 or hi > image.shape[1] or indices[0] - gap < 0 or indices[-1] + gap >= len(image):
            return None
        delta = image[indices + gap, lo:hi] - image[indices - gap, lo:hi]
        magnitude = np.max(abs(delta), axis=2)
        central = np.median(magnitude, axis=1)
        aligned = np.max(abs(np.median(delta, axis=1)), axis=1)
        eligible = (central >= 16) & (aligned >= 13) & (np.mean(magnitude >= 13, axis=1) >= .62)
        if not np.any(eligible):
            return None
        edges.append(int(indices[np.argmax(central * eligible)]))
    top, bottom, left, right = edges
    w, h = right - left, bottom - top
    if .88 * size <= min(w, h) <= max(w, h) <= 1.14 * size and .88 <= w / h <= 1.15:
        return left, top, right, bottom
    return None


@lru_cache(maxsize=4)
def _font_words(cjk, latin):
    templates = []
    for source in dict.fromkeys((*cjk, *latin)):
        try:
            font = ImageFont.truetype(str(source), 48)
        except OSError:
            continue
        for word in _SYSTEM_APPS:
            if not word.isascii() and source not in cjk:
                continue
            mask = font.getmask(word, mode="L")
            raster = np.asarray(mask, np.uint8).reshape(mask.size[1], mask.size[0])
            x, y, w, h = cv2.boundingRect(np.uint8(raster >= 38))
            if w and h:
                templates.append((word, w / h, raster[y:y + h, x:x + w]))
    return templates


@lru_cache(maxsize=194)
def _word_bank(cjk, latin, width):
    height = 22
    words, ratios, rasters = [], [], []
    for word, ratio, raster in _font_words(cjk, latin):
        if not .80 <= ratio * height / width <= 1.24:
            continue
        image = cv2.resize(raster, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        image = cv2.GaussianBlur(image, (5, 5), .80).ravel()
        words.append(word)
        ratios.append(abs(np.log(ratio * height / width)))
        rasters.append(image / np.linalg.norm(image))
    return words, np.asarray(ratios, np.float32), np.asarray(rasters, np.float32)


def _top_words(residual, cjk, latin):
    h, w = residual.shape
    width = round(22 * w / h)
    words, ratios, bank = _word_bank(cjk, latin, width)
    if len(words) < 2:
        return None
    strength = max(1, np.percentile(residual[residual > 0], 87))
    patch = np.minimum(1, cv2.resize(residual, (width, 22)).astype(np.float32) / strength)
    patch = cv2.GaussianBlur(patch, (3, 3), .55).ravel()
    scores = np.einsum("ij,j->i", bank, patch) / max(1e-5, np.linalg.norm(patch)) - .09 * ratios
    ranking = np.argsort(scores)[::-1]
    first = ranking[0]
    other = next((i for i in ranking[1:] if words[i] != words[first]), None)
    if other is None:
        return None
    return words[first], float(scores[first]), float(scores[first] - scores[other])


def _caption_crop(signal, center, box, size, pitch):
    left = max(0, round(center - .48 * pitch))
    right = min(signal.shape[1], round(center + .48 * pitch))
    top = box[3] + round(.038 * size)
    bottom = min(signal.shape[0], box[3] + round(.32 * size))
    return left, signal[top:bottom, left:right]


def _caption_masks(signal, center, box, size, pitch):
    left_edge, crop = _caption_crop(signal, center, box, size, pitch)
    if not crop.size:
        return
    baseline = np.percentile(crop[:, round(.15 * crop.shape[1]):round(.85 * crop.shape[1])], 97)
    expected = .16 * size
    for ratio in (.27, .39, .53, .66):
        _, ids, stats, _ = cv2.connectedComponentsWithStats(np.uint8(crop >= max(7, ratio * baseline)))
        parts = sorted((x, y, w, h, i) for i, (x, y, w, h, n) in enumerate(stats[1:], 1)
                       if 2 <= n and .11 * expected <= h <= 1.39 * expected and w <= 2.2 * expected
                       and .12 * crop.shape[0] <= y + h / 2 <= .89 * crop.shape[0])
        groups = []
        right = 0
        for part in parts:
            x, _, w, _, _ = part
            if groups and x <= right + .37 * expected:
                groups[-1].append(part)
                right = max(right, x + w)
            else:
                groups.append([part])
                right = x + w
        for group in groups:
            left = group[0][0]
            right = max(x + w for x, _, w, _, _ in group)
            upper = min(y for _, y, _, _, _ in group)
            lower = max(y + h for _, y, _, h, _ in group)
            w, h = right - left, lower - upper
            if (.70 * expected <= h <= 1.39 * expected and .94 <= w / h <= 9.0
                    and abs(left_edge + (left + right) / 2 - center) <= .62 * expected):
                owned = np.isin(ids[upper:lower, left:right], [part[-1] for part in group])
                yield crop[upper:lower, left:right] * owned


@lru_cache(maxsize=22)
def _long_bank(cjk, latin, size):
    templates = []
    for name, aspect, raster in _font_words(cjk, latin):
        if aspect < 3.4:
            continue
        for ratio in (.14, .17, .19):
            h = round(size * ratio)
            w = round(aspect * h)
            shaped = cv2.resize(raster, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            templates.append((name, cv2.GaussianBlur(shaped, (3, 3), .50)))
    return templates


def _long_caption(signal, center, box, size, pitch, cjk, latin):
    left, raw = _caption_crop(signal, center, box, size, pitch)
    if not raw.size:
        return None
    bias = np.max(raw, axis=0) >= max(8, .34 * np.percentile(raw, 98))
    pos = np.flatnonzero(bias)
    if len(pos) < .55 * size or not bias[round(center - left - .29 * size):round(center - left + .29 * size)].any():
        return None
    query = cv2.GaussianBlur(np.minimum(1, raw.astype(np.float32) / max(3, np.percentile(raw, 98))), (3, 3), .50)
    scores = {}
    limit = round(.07 * size)
    for name, mask in _long_bank(cjk, latin, round(size)):
        h, w = mask.shape
        if h >= query.shape[0] or w >= query.shape[1]:
            continue
        midpoint = round(center - left - w / 2)
        lower, upper = max(0, midpoint - limit), min(query.shape[1], midpoint + limit + w + 1)
        if upper - lower >= w:
            score = float(cv2.matchTemplate(query[:, lower:upper], mask, cv2.TM_CCOEFF_NORMED).max())
            scores[name] = max(score, scores.get(name, 0.0))
    candidates = sorted(scores.items(), key=lambda item: -item[1])
    if len(candidates) >= 2 and candidates[0][1] >= .67 and candidates[0][1] - candidates[1][1] >= .13:
        return candidates[0][0]
    return None


def _read_name(signals, center, box, size, pitch, cjk, latin):
    reads = [[(match, mask) for mask in _caption_masks(signal, center, box, size, pitch)
              if (match := _top_words(mask, cjk, latin)) is not None] for signal in signals]
    per_signal = [[match for match, _ in group] for group in reads]
    verified = [(name, score, margin) for matches in per_signal for name, score, margin in matches
                if score >= .75 and margin >= .035]
    if verified:
        name, score, margin = max(verified, key=lambda item: item[1])
        count = sum(item[0] == name for item in verified)
        competing = max((value for other, value, _ in verified if other != name), default=0)
        paired_latin = not name.isascii() or all(any(word == name and confidence >= .75 and gap >= .035
                                                    for word, confidence, gap in group) for group in per_signal)
        if (name not in _EXCLUDE_AS_APP and paired_latin
                and (not name.isascii() or _letters_consistent(name, reads))
                and ((count >= 2 and competing == 0) or (score >= .85 and margin >= .07 and competing == 0)
                     or (count >= 2 and margin >= .055 and score >= .86 and score - competing >= .026))):
            return name
    repeated = [{name for name, score, margin in matches if score >= .78 and margin >= .015}
                for matches in per_signal]
    if len(repeated[0]) == 1 and repeated[0] == repeated[1]:
        name = next(iter(repeated[0]))
        if (name not in _EXCLUDE_AS_APP and all(sum(match[0] == name for match in matches) >= 2 for matches in per_signal)
                and (not name.isascii() or (_letters_consistent(name, reads)
                                            and max(score for matches in per_signal for word, score, _ in matches
                                                    if word == name) >= .85))):
            return name
    # Multiple printed words can be split into disconnected groups by the
    # first pass. Maintain the full dictionary as competitors, and use the
    # raw correlation only for a caption that actually contains a word gap.
    for signal in reversed(signals):
        name = _long_caption(signal, center, box, size, pitch, cjk, latin)
        if name and " " in name:
            return name
    return None


def _letters_consistent(name, reads):
    if name == _QR_CAPTION:
        return True  # Its independent QR-shaped tile proof is applied before emitting.
    letters = sum(char.isalnum() for char in name)
    level = .18 if letters >= 7 else .32
    for group in reads:
        for (word, confidence, _), mask in group:
            if word != name or confidence < .75:
                continue
            columns = (mask >= level * mask.max()).sum(axis=0) >= max(1, round(.10 * len(mask)))
            runs = np.count_nonzero(np.diff(np.r_[False, columns, False])) // 2
            if runs <= letters:
                return True
    return False


def _qr_face(frame, box):
    left, top, right, bottom = box
    side = min(right - left, bottom - top)
    pad = round(.12 * side)
    x0, y0 = max(0, left - pad), max(0, top - pad)
    raster = frame[y0:bottom + pad, x0:right + pad]
    visible, points = cv2.QRCodeDetector().detect(raster)
    if not visible:
        return False
    points = points[0] + (x0, y0)
    gaps = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
    return (left <= points[:, 0].min() <= points[:, 0].max() <= right
            and top <= points[:, 1].min() <= points[:, 1].max() <= bottom
            and .52 * side <= gaps.min() <= gaps.max() <= .94 * side and gaps.max() / gaps.min() <= 1.20)


def refine_home_app_box(prepared_frame, grid, proposal_box):
    """Tighten a proposal when one measured home-grid cell contains the app tile."""
    if grid is None:
        return None
    left, top, right, bottom = proposal_box
    x, y = _center(proposal_box)
    size = grid.tile_size
    if right - left > 1.78 * size or bottom - top > 2.15 * size:
        return None
    matches = [(column_index, column, row_center, boxes) for row_center, boxes in grid.rows
               for column_index, column in enumerate(grid.columns)
               if abs(x - column) <= .32 * size and abs(y - row_center) <= .58 * size]
    if len(matches) != 1:
        return None
    column_index, column, row_center, boxes = matches[0]
    face = boxes.get(column_index)
    if face is None:
        height, width = prepared_frame.shape[:2]
        radius = .69 * size
        x0, x1 = max(0, round(column - radius)), min(width, round(column + radius))
        y0, y1 = max(0, round(row_center - radius)), min(height, round(row_center + radius))
        chip = prepared_frame[y0:y1, x0:x1]
        if min(chip.shape[:2]) < size:
            return None
        outlines = _rounded_grid_boxes(chip, (column,), size, grid.pitch, origin=(x0, y0))
        centered = (box for box in outlines if abs(_center(box)[1] - row_center) <= .15 * size)
        face = min(centered, key=lambda b: (abs(b[2] - b[0] - size) + abs(b[3] - b[1] - size), b), default=None)
    if face is None:
        return None
    a, b, c, d = face
    overlap = max(0, min(right, c) - max(left, a)) * max(0, min(bottom, d) - max(top, b))
    area = (c - a) * (d - b)
    if overlap >= .82 * area and (right - left) * (bottom - top) >= 1.20 * area:
        return list(face)
    return None


def inspect_home_apps(frame, overlay, index):
    """Return caption-backed app actions and observed tile geometry in frame pixels."""
    height, width = frame.shape[:2]
    if overlay is not None or index is None or not 1.65 <= height / width <= 2.55 or width < 310:
        return [], None
    source = frame
    if width > 680:
        frame = cv2.resize(frame, (640, round(height * 640 / width)), interpolation=cv2.INTER_AREA)
    size_y, size_x = frame.shape[:2]
    dock = [box for _, box in _dock_icons(frame)]
    if not dock and frame is not source:
        if width > 900:
            source = cv2.resize(source, (720, round(height * 720 / width)), interpolation=cv2.INTER_AREA)
        source_h, source_w = source.shape[:2]
        scale = (size_x / source_w, size_y / source_h) * 2
        dock = [tuple(round(v * s) for v, s in zip(box, scale, strict=True))
                for _, box in _dock_icons(source)]
    dock, recovered = _dock_columns(dock, size_x)
    if dock is None:
        return [], None
    boxes, columns, size, pitch = _grid_contours(frame, dock)
    rows = _body_rows(boxes, columns, size, pitch)
    if not rows or recovered is not None and sum(recovered in row for row in rows) < 2:
        return [], None
    cjk = tuple(source for source in index.cjk_sources if source.name.lower() in {"msyh.ttc", "msyhbd.ttc", "msjh.ttc"})
    latin = tuple(source for source in index.choice_latin_sources if source.name.lower() in {"arial.ttf", "segoeui.ttf", "segoeuib.ttf"})
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(.065 * size) | 1,) * 2)
    blue, green, red = cv2.split(frame)
    minimum = cv2.min(cv2.min(blue, green), red)
    signals = (cv2.morphologyEx(minimum, cv2.MORPH_TOPHAT, kernel),
               cv2.subtract(minimum, cv2.GaussianBlur(minimum, (0, 0), .055 * size)))
    edges = cv2.GaussianBlur(frame, (3, 3), .7).astype(np.int16)
    sx, sy = width / size_x, height / size_y
    actions = []
    measured_rows = []
    for row in rows:
        y = np.median([_center(box)[1] for box in row.values()])
        measured_rows.append((float(y * sy), {col: tuple(round(value * scale) for value, scale in
                                                        zip(box, (sx, sy, sx, sy), strict=True))
                                              for col, box in row.items()}))
        for col, x in enumerate(columns):
            box = row.get(col)
            if box is None:
                box = _edge_face(edges, x, y, size)
            if not box:
                continue
            name = _read_name(signals, _center(box)[0], box, size, pitch, cjk, latin)
            if name and (name != _QR_CAPTION or _qr_face(frame, box)):
                x, y = _center(box)
                usage = name.replace(" 通话", "通话")
                actions.append({"role": "open_app", "function": f"打开{usage}", "preview_text": name,
                                "box": [round(value * scale) for value, scale in zip(box, (sx, sy, sx, sy), strict=True)],
                                "click_point": [round(x * sx), round(y * sy)]})
    grid = _HomeGrid(tuple(float(column * sx) for column in columns), tuple(measured_rows), size * min(sx, sy), pitch * sx)
    return actions, grid


def home_app_actions(frame, overlay, index):
    """Return names and points supported by a visible tile and legible caption."""
    actions, _ = inspect_home_apps(frame, overlay, index)
    return actions
