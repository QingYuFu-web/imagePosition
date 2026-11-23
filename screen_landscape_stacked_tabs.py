"""Prove a landscape settings rail before reading its low-resolution labels."""

from functools import lru_cache
from itertools import combinations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from screen_landscape_edit_actions import _features


_SETTINGS = (
    "设置", "設定", "基本设置", "基础设置", "操作设置", "操控设置", "控制设置",
    "按键设置", "触控设置", "按键布局", "灵敏度设置", "灵敏度", "陀螺仪",
    "画质设置", "画面设置", "图像设置", "显示设置", "视觉设置", "界面设置",
    "音效设置", "声音设置", "音频设置", "语音设置", "麦克风设置", "音乐设置",
    "战斗设置", "游戏设置", "录像设置", "网络设置", "新手设置", "性能设置",
    "隐私设置", "辅助设置", "拾取设置", "载具设置", "语言设置", "通知设置",
    "账户设置", "账号设置", "设备设置", "其他设置", "下载中心", "资源管理",
    "幀率設定", "基礎設定", "操作設定", "靈敏度設定", "畫面設定", "音效設定",
    "聲音設定", "語言設定", "下載中心", "自定义", "恢复默认", "返回设置",
    "按键自定义", "自定义按键", "自定义布局", "自定义操作", "按钮自定义",
    "自定义设置", "操作自定义", "重新设置",
)
_ALIASES = {
    "設定": "设置", "幀率設定": "帧率设置", "基礎設定": "基础设置", "操作設定": "操作设置",
    "靈敏度設定": "灵敏度设置", "畫面設定": "画面设置", "音效設定": "音效设置",
    "聲音設定": "声音设置", "語言設定": "语言设置", "下載中心": "下载中心",
}
_CUSTOMIZE = {"按键自定义", "自定义按键", "自定义布局", "自定义操作", "按钮自定义", "操作自定义"}


@lru_cache(maxsize=6)
def _shapes(paths):
    result = []
    for path in paths:
        font = ImageFont.truetype(str(path), 44, layout_engine=ImageFont.Layout.BASIC)
        for word in _SETTINGS:
            canvas = Image.new("L", (310, 80))
            ImageDraw.Draw(canvas).text((2, 0), word, font=font, fill=255)
            raster = np.asarray(canvas)
            x, y, w, h = cv2.boundingRect(raster)
            if w and h:
                result.append((_ALIASES.get(word, word), raster[y:y+h, x:x+w]))
    return tuple(result)


@lru_cache(maxsize=32)
def _bank(paths, w, h):
    labels, rasters = [], []
    for word, image in _shapes(paths):
        if not .68 <= w / h / (image.shape[1] / image.shape[0]) <= 1.35:
            continue
        for dw, dh in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1)):
            rw, rh = w + dw, h + dh
            resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA)
            window = np.zeros((h + 2, w + 2), np.float32)
            x, y = (w - rw) // 2 + 1, (h - rh) // 2 + 1
            window[y:y + rh, x:x + rw] = resized
            labels.append(word)
            rasters.append(window)
    weight = _weight(w)
    return tuple(labels), _features(np.stack(rasters) * weight) if rasters else None


def _weight(width):
    return np.interp(np.arange(width + 2), (0, .59 * width, .67 * width, width + 1),
                     (1.4, 1.4, .83, .83)).astype(np.float32)


def _ranked_views(signal, paths):
    peak = float(np.percentile(signal, 99.4))
    if peak < 34:
        return []
    normalized = np.clip(signal / peak, 0, 1)
    reads = []
    for level in (.30, .42, .55):
        x, y, w, h = cv2.boundingRect(np.uint8(normalized >= level))
        if not 5 <= h <= 26 or not 1.5 * h <= w <= 6.0 * h:
            continue
        labels, bank = _bank(paths, w, h)
        if bank is None:
            continue
        vector = _features(np.pad(normalized[y:y+h, x:x+w], 1) * _weight(w))
        scores = np.einsum("ij,j->i", bank, vector, optimize=False)
        best = {}
        for word, score in zip(labels, scores, strict=True):
            best[word] = max(best.get(word, 0.0), float(score))
        choices = sorted(best.items(), key=lambda pair: -pair[1])
        if len(choices) >= 2:
            reads.append(choices[:3])
    return reads


def _winner(reads, score=.80, gap=.022):
    supported = [(rank[0][0], rank[0][1], rank[0][1] - rank[1][1])
                 for rank in reads if rank[0][1] >= score and rank[0][1] - rank[1][1] >= gap]
    if len(supported) < 2 or len({hit[0] for hit in supported}) != 1:
        return None
    return supported[0][0]


def _read(signal, paths, *, diagnostic=False):
    reads = _ranked_views(signal, paths)
    return reads if diagnostic else _winner(reads)


def _read_heading(signal, paths):
    reads = _ranked_views(signal, paths)
    native = _winner(reads)
    if native is not None:
        return native == "设置"
    # Isolate a tiny light title from JPEG ringing and the nearby sloped backdrop.
    edge = signal[:, -max(2, round(signal.shape[1]/5)):]
    corrected = signal - np.median(edge, axis=1)[:, None]
    if any(view[0][0] == "设置" and view[0][1] >= .82 and view[0][1] - view[1][1] >= .11
           for view in reads) and _winner(_ranked_views(corrected, paths), .80, .06) == "设置":
        return True
    return all(_winner(_ranked_views(cv2.GaussianBlur(signal, (0, 0), sigma), paths), .80, .08) == "设置"
               for sigma in (.70, 1.15))


def _read_row(signal, paths):
    reads = _ranked_views(signal, paths)
    native = _winner(reads)
    if native is not None:
        return native, True
    finalist = reads[0][0][0] if reads else None
    coherent = len(reads) == 3 and all(rank[0][0] == finalist and rank[0][1] >= .89 for rank in reads)
    if coherent and all(rank[0][1]-rank[1][1] >= .018 for rank in reads):
        softened = _ranked_views(cv2.GaussianBlur(signal, (0, 0), .70), paths)
        if _winner(softened, .88, .018) == finalist:
            return finalist, True
    return None, coherent


@lru_cache(maxsize=8)
def _phase_bank(paths, width, height, finalists):
    # Render a small bank at quarter-pixel phase to model screenshot resampling.
    scale = 4
    size = width+2, height+2
    cols, rows = (n*scale for n in size)
    stretch = ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2))
    shift = ((0, 0), (-2, -2), (-2, 2), (2, -2), (2, 2))
    rasters, labels = [], []
    for word, glyph in _shapes(paths):
        if word not in finalists:
            continue
        for dw, dh in stretch:
            w, h = width*scale+dw, height*scale+dh
            proto = cv2.resize(glyph, (w, h), interpolation=cv2.INTER_AREA)
            for dx, dy in shift:
                x, y = (cols-w)//2+dx, (rows-h)//2+dy
                canvas = np.zeros((rows, cols), np.uint8)
                canvas[y:y+h, x:x+w] = proto
                raster = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA).astype(np.float32)
                rasters.extend((raster, cv2.GaussianBlur(raster, (3, 3), .50)))
                labels.extend((word, word))
    return tuple(labels), _features(np.stack(rasters) * _weight(width))


def _supported_phases(signal, paths, finalists, expected):
    peak = float(np.percentile(signal, 99.4))
    if peak < 34:
        return 0
    signal = np.clip(signal/peak, 0, 1)
    bounds = {cv2.boundingRect(np.uint8(signal >= level)) for level in (.30, .42, .55)}
    matches = 0
    for x, y, w, h in bounds:
        if not (5 <= h <= 26 and 1.5*h <= w <= 6.0*h):
            continue
        labels, bank = _phase_bank(paths, w, h, finalists)
        vector = _features(np.pad(signal[y:y+h, x:x+w], 1) * _weight(w))
        scores = np.einsum("ij,j->i", bank, vector, optimize=False)
        own = max(score for label, score in zip(labels, scores, strict=True) if label == expected)
        other = max(score for label, score in zip(labels, scores, strict=True) if label != expected)
        if other - own >= .02:
            return 0
        matches += int(own >= .81 and own-other >= .04)
    return matches


def _read_outlined(signal, paths):
    reads = _ranked_views(signal, paths)
    native = _winner(reads)
    if native is not None or len(reads) < 3:
        return native
    finalist = reads[0][0][0]
    if finalist not in _CUSTOMIZE or not all(rank[0][0] == finalist and rank[0][1] >= .725
                                            and rank[0][1]-rank[1][1] >= .035 for rank in reads):
        return None
    if signal.shape[0] >= 24:
        reduced = cv2.resize(signal, None, fx=.86, fy=.86, interpolation=cv2.INTER_AREA)
        if _winner(_ranked_views(reduced, paths), .80, .04) == finalist:
            return finalist
    finalists = tuple(sorted({word for rank in reads for word, score in rank}))
    native_matches = _supported_phases(signal, paths, finalists, finalist)
    if native_matches and _supported_phases(cv2.GaussianBlur(signal, (0, 0), .65), paths, finalists, finalist) >= 2:
        return finalist
    return None


def _selected_line(frame, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    a, r = left + round(.13 * width), right - round(.13 * width)
    first = top + round(.74 * height)
    strip = frame[first:bottom, a:r]
    if not strip.size:
        return False
    chroma = np.ptp(strip, axis=2)
    if ((chroma >= 47).mean(axis=1) >= .69).sum() in (1, 2, 3):
        return True
    reference = frame[top + round(.31 * height):top + round(.61 * height), a:r]
    if not reference.size:
        return False
    row_colors = np.median(strip, axis=1)
    pivot = np.argmax(np.ptp(row_colors, axis=1))
    hue = row_colors[pivot]
    high, low = int(hue.argmax()), int(hue.argmin())
    signal = strip[:, :, high].astype(np.int16) - strip[:, :, low].astype(np.int16)
    background = np.median(reference[:, :, high].astype(np.int16) - reference[:, :, low].astype(np.int16))
    strength = hue[high] - hue[low]
    if strength < 20 or strength - background < 15:
        return False
    quorum = (signal >= max(background + 11, .74 * strength)).mean(axis=1) >= .76
    hits = np.flatnonzero(quorum)
    return 0 < len(hits) <= max(2, round(.19 * height)) and np.ptp(hits) == len(hits) - 1


def _signal(gray, box, header=False):
    a, t, r, b = box
    w, h = r-a, b-t
    field = gray[t:b, a:r].astype(np.float32)
    if field.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    baseline = np.median(field[:, max(1, round(.08*w)):round(.41*w)], axis=1)
    if header:
        baseline = np.float32(np.percentile(field, 48))
        return field - baseline
    return (field-baseline[:, None])[round(.18*h):round(.81*h), round(.40*w):round(.95*w)]


def _pane_peers(boxes, edge, width, height):
    interior = [p for p in boxes if edge+.085*width <= p[0] < .94*width
                and .068*width <= p[2]-p[0] <= .115*width
                and .063*height <= p[3]-p[1] <= .235*height]
    if any(abs(left[1]-right[1]) <= .038*height and right[0]-left[0] >= .10*width
           for left, right in combinations(sorted(interior), 2)):
        return True
    cards = [p for p in boxes if edge+.025*width <= p[0]
             and .17*width <= p[2]-p[0] <= .31*width and .38*height <= p[3]-p[1] <= .79*height]
    for group in combinations(sorted(cards), 3):
        points = np.array(group)
        gaps = points[1:, 0] - points[:-1, 2]
        if (.006*width <= min(gaps) <= max(gaps) <= .068*width
                and np.ptp(gaps) <= .04*width and np.ptp(points[:, 1]) <= .052*height
                and np.ptp(points[:, 3]) <= .046*height and np.ptp(points[:, 2]-points[:, 0]) <= .046*width):
            return True
    return False


def stacked_setting_tabs(frame, regions, index):
    """Name one physical inactive tab per consistently painted settings row."""
    height, width = frame.shape[:2]
    if (index is None or regions.get("layer") != "page" or regions.get("regions")
            or height < 210 or width < 1.55*height):
        return []
    boxes = [tuple(p["box"]) for p in regions["candidates"] if p.get("visibility") == "page"
             and 0 <= p["box"][0] < p["box"][2] <= width and 0 <= p["box"][1] < p["box"][3] <= height]
    rows = sorted((p for p in boxes if .012*width <= p[0] and p[2] <= .25*width
                  and .108*width <= p[2]-p[0] <= .215*width
                  and .05*height <= p[3]-p[1] <= .102*height), key=lambda p: p[1])
    if not 5 <= len(rows) <= 8:
        return []
    locations = np.array(rows)
    centers = (locations[:, 1] + locations[:, 3]) / 2
    pitch = np.diff(centers)
    span = locations[:, 3] - locations[:, 1]
    if not (.082*height <= min(pitch) <= max(pitch) <= .155*height
            and np.ptp(pitch) <= .15 * np.median(pitch)
            and np.ptp(locations[:, 0]) <= .015 * width and np.ptp(locations[:, 2]) <= .015*width
            and np.count_nonzero(abs(span-np.median(span)) <= .23*np.median(span)) >= len(rows)-1
            and .07*height <= rows[0][1] <= .23*height
            and _pane_peers(boxes, int(np.median(locations[:, 2])), width, height)):
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    selected = []
    for i, (a, t, r, b) in enumerate(rows):
        w, h = r-a, b-t
        if _selected_line(frame, (a, t, r, b)):
            selected.append(i)
        paint = gray[t+round(.28*h):b-round(.25*h), a+round(.14*w):round(a+.39*w)]
        outer = gray[b:min(b+max(2, round(.17*h)), height), a+round(.14*w):round(a+.39*w)]
        if not paint.size or not outer.size or (i not in selected and np.median(paint)-np.median(outer) < 7):
            return []
    if len(selected) != 1:
        return []
    paths = tuple(index.fonts if index.fonts is not None else index.cjk_sources)
    if not paths:
        return []
    a, _, r, _ = rows[0]
    typ = np.median(span)
    header = (max(0, round(a-.30*typ)), max(0, round(centers[0]-1.65*typ)),
              a+round(.38*(r-a)), round(centers[0]-.72*typ))
    if not _read_heading(_signal(gray, header, header=True), paths):
        return []
    evidence = [_read_row(_signal(gray, box), paths) if i not in selected else (None, True)
                for i, box in enumerate(rows)]
    readings = [word for word, coherent in evidence]
    words = [word for word in readings if word]
    # An ambiguous row can prove the rail is present but never supplies a point.
    minimum = len(rows)-3 if all(coherent for word, coherent in evidence) else len(rows)-2
    if len(words) < minimum or len(set(words)) != len(words):
        return []
    return [{"role": "game_settings_tab", "function": word, "preview_text": word, "box": list(row),
             "click_point": [(row[0]+row[2])//2, (row[1]+row[3])//2]}
            for row, word in zip(rows, readings, strict=True) if word and word.endswith("设置")]


def _outlined_owner(gray, box):
    a, t, r, b = box
    w, h = r-a, b-t
    delta = max(2, round(.072*h))
    around = max(4, round(.18*h))
    if min(a, t, gray.shape[1]-r, gray.shape[0]-b) < delta:
        return False
    field = gray.astype(np.int16, copy=False)

    def ridge(rows, start, stop):
        begin = max(delta, start)
        end = min(rows.shape[0]-delta, stop)
        return any((rows[i]-np.maximum(rows[i-delta], rows[i+delta]) >= 6).mean() >= .72
                   for i in range(begin, end))

    horizontal = field[:, a+round(.15*w):r-round(.15*w)]
    vertical = field[t+round(.19*h):b-round(.19*h)].T
    return (ridge(horizontal, t-1, t+around) and ridge(horizontal, b-around, b+1)
            and ridge(vertical, a-1, a+around) and ridge(vertical, r-around, r+1))


def outlined_control_actions(frame, regions, index):
    """Confirm a labeled customization button by its own rim and settings controls."""
    height, width = frame.shape[:2]
    if (index is None or regions.get("layer") != "page" or regions.get("regions")
            or height < 220 or width < 1.55*height):
        return []
    boxes = [tuple(p["box"]) for p in regions["candidates"] if p.get("visibility") == "page"
             and 0 <= p["box"][0] < p["box"][2] <= width and 0 <= p["box"][1] < p["box"][3] <= height]
    # A separate heading and settings options must share the page with the button.
    titles = [box for box in boxes if .029*width <= box[0] <= .19*width
              and box[3] <= .11*height and .026*width <= box[2]-box[0] <= .067*width]
    if not titles:
        return []
    short = [p for p in boxes if .089*width <= p[2]-p[0] <= .21*width and
             .048*height <= p[3]-p[1] <= .112*height and p[1] >= .072*height and p[3] <= .34*height]
    if not short:
        return []
    paths = tuple(index.fonts if index.fonts is not None else index.cjk_sources)
    if not paths:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    outcome = []
    for owner in short:
        a, t, r, b = owner
        w, h = r-a, b-t
        parallels = [box for box in boxes if box[2] <= a and abs((box[1]+box[3]-t-b)/2) <= .14*h
                     and .72*h <= box[3]-box[1] <= 1.20*h and box[2]-box[0] >= 1.24*w]
        if not any(abs((left[2]-left[0])-(right[2]-right[0])) <= .14*w
                   and -.05*w <= right[0]-left[2] <= .14*w for left, right in combinations(sorted(parallels), 2)):
            continue
        if not _outlined_owner(gray, owner):
            continue
        roi = gray[t+round(.16*h):b-round(.18*h), a+round(.092*w):r-round(.092*w)].astype(np.float32)
        text = _read_outlined(roi - np.median(roi, axis=1)[:, None], paths)
        if text not in _CUSTOMIZE:
            continue
        if not any(_read_heading(_signal(gray, (x, y+max(1, (stop-y)//10), end, stop-max(1, (stop-y)//10)),
                                          header=True), paths) for x, y, end, stop in titles):
            continue
        outcome.append({"role": "customize_game_controls", "function": f"打开{text}",
                        "preview_text": text, "box": list(owner), "click_point": [(a+r)//2, (t+b)//2]})
    return outcome if len(outcome) == 1 else []
