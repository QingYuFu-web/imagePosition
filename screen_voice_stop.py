"""Find an explicit stop icon on a visible bottom recording surface."""

import cv2
import numpy as np


def _surface(gray):
    height, width = gray.shape
    tone = int(np.median(gray[round(.84 * height):round(.96 * height), round(.18 * width):round(.33 * width)]))
    if not 16 <= tone <= 75:
        return None
    tint = np.uint8(abs(gray.astype(np.int16) - tone) <= 5)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(tint, connectivity=8)
    choices = []
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if (not .87 * width <= w <= .99 * width or not .23 * height <= h <= .44 * height
                or not .53 * height <= y <= .78 * height or not .967 * height <= y + h < height
                or not .008 * width <= min(x, width - x - w)
                or abs(width - w - 2 * x) > .022 * width or area < .78 * w * h):
            continue
        lead = max(2, round(.015 * w))
        sides = max(2, round(.047 * w))
        body = labels[y:y + h, x:x + w] == i
        mid = body[round(.14 * h):round(.86 * h)]
        if min(mid[:, sides:2 * sides].mean(), mid[:, w - 2 * sides:w - sides].mean()) < .90:
            continue
        if any(np.any(corner) for corner in (
            body[:lead, :lead], body[:lead, -lead:], body[-lead:, :lead], body[-lead:, -lead:]
        )):
            continue
        above = gray[y - lead - 1:y - 1, x + sides:x + w - sides]
        guard = max(2, round(.18 * h))
        outside = np.column_stack((gray[y + guard:y + h - guard, :x],
                                   gray[y + guard:y + h - guard, x + w:]))
        if min((above <= tone - 11).mean(), (outside <= tone - 11).mean()) < .78:
            continue
        if _grip(gray, (x, y, x + w, y + h), tone):
            choices.append(tuple(map(int, (x, y, x + w, y + h))))
    return choices[0] if len(choices) == 1 else None


def _grip(gray, box, tone):
    left, top, right, _ = box
    w = right - left
    column = left + round(.35 * w)
    field = gray[top:top + round(.061 * w), column:right - round(.35 * w)]
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(field >= tone + 28), connectivity=8)
    return sum(.042 * w <= ww <= .126 * w and .005 * w <= hh <= .021 * w and 4.0 <= ww / hh <= 15
               and .003 * w <= y <= .031 * w and abs(column + x + ww / 2 - (left + right) / 2) <= .027 * w
               and area >= .66 * ww * hh for x, y, ww, hh, area in stats[1:]) == 1


def _waveform(stats, width, height):
    bars = [(x + w / 2, y + h / 2, h) for x, y, w, h, area in stats
            if (.002 * width <= w <= .012 * width and .007 * width <= h <= .18 * width
                and h >= 3.1 * w and area >= .65 * w * h and .25 * height <= y + h / 2 <= .66 * height)]
    if len(bars) < 24:
        return False
    midline = np.median([y for _, y, _ in bars])
    line = sorted((x, h) for x, y, h in bars if abs(y - midline) <= .021 * height)
    if len(line) < .79 * len(bars) or len(line) < 24:
        return False
    x, bars_height = np.array(line).T
    if np.ptp(x) < .65 * width or np.ptp(bars_height) < .026 * width:
        return False
    spacing = np.diff(x)
    step = np.median(spacing)
    return .0035 * width <= step <= .021 * width and (.71 * step <= spacing).all() and (
        (spacing <= 1.30 * step).mean() >= .69
    )


def _round_square(mask):
    h, w = mask.shape
    a, b = max(1, round(.10 * h)), max(1, round(.10 * w))
    return min(mask[a].mean(), mask[-a - 1].mean(), mask[:, b].mean(), mask[:, -b - 1].mean()) >= .77


def _stop_ring(field, x, y, span):
    points = np.linspace(0, 2 * np.pi, 96, endpoint=False, dtype=np.float32)
    radii = np.arange(.96 * span, 1.45 * span, max(.45, .012 * span), dtype=np.float32)
    pad = max(3, round(.09 * span))
    if min(x, y, field.shape[1] - x, field.shape[0] - y) < radii[-1] + pad:
        return None
    span_x = np.float32(np.cos(points))[None]
    span_y = np.float32(np.sin(points))[None]

    def sample(r):
        return cv2.remap(field, x + r[:, None] * span_x, y + r[:, None] * span_y,
                         cv2.INTER_LINEAR).astype(np.int16)

    edge = sample(radii)
    neighbors = np.maximum(sample(radii - pad), sample(radii + pad))
    support = (edge - neighbors >= 9).mean(axis=1)
    best = support.argmax()
    if support[best] < .64 or ((edge - neighbors >= 9)[best].reshape(4, -1).mean(axis=1) < .43).any():
        return None
    return float(radii[best])


def voice_stop_surface(frame, working, gray):
    """Return the proven sheet and its stop control in original pixel coordinates."""
    height, width = gray.shape
    if width < 180 or height < 1.50 * width or working.shape[:2] != gray.shape:
        return None
    if (bounds := _surface(gray)) is None:
        return None
    src_height, src_width = frame.shape[:2]
    sx, sy = src_width / width, src_height / height
    left, top, right, bottom = tuple(round(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(bounds))
    panel = left, top, right, bottom
    crop = frame[top:bottom, left:right]
    pw, ph = right - left, bottom - top
    red = crop[..., 2].astype(np.int16)
    accent = red - np.maximum(crop[..., 0], crop[..., 1])
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8((red >= 112) & (accent >= 55)), connectivity=8)
    if not _waveform(stats[1:], pw, ph):
        return None
    gray_panel = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    matches = []
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        cx, cy = x + (w - 1) / 2, y + (h - 1) / 2
        if (not .033 * pw <= min(w, h) <= max(w, h) <= .078 * pw or not .86 <= w / h <= 1.16
                or abs(cx - pw / 2) > .095 * pw or not .68 * ph <= cy <= .90 * ph
                or not .84 <= area / (w * h) <= .996):
            continue
        if not _round_square(labels[y:y + h, x:x + w] == i):
            continue
        radius = _stop_ring(gray_panel, cx, cy, (w + h) / 2)
        if radius is not None:
            matches.append((round(left + cx - radius), round(top + cy - radius),
                            round(left + cx + radius) + 1, round(top + cy + radius) + 1))
    return (panel, matches[0]) if len(matches) == 1 else None
