"""Keep high-resolution dialog proposals inside a verified foreground crop."""

from math import sqrt

from general_visual import VisualControlDetector


def _propose_in_crop(frame, detector, bounds, budget):
    left, top, right, bottom = bounds
    return [{**item, "polygon": [[x + left, y + top] for x, y in item["polygon"]]}
            for item in detector.detect(frame[top:bottom, left:right], pixel_budget=budget)]


def crop_dialog_proposals(frame, detector, overlay):
    """Return mapped proposals for a proven centered dialog, or defer to full view."""
    if (overlay is None or overlay.kind != "dialog" or len(overlay.regions) != 1 or overlay.close_button is not None
            or not isinstance(detector, VisualControlDetector) or detector._size is not None):
        return None
    height, width = frame.shape[:2]
    left, top, right, bottom = overlay.panel
    if not 0 < left < right < width or not 0 < top < bottom < height:
        return None
    pad = round(.025 * min(right - left, bottom - top))
    l, t, r, b = max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad)
    ratio = ((r - l) * (b - t)) / (width * height)
    if ratio >= .81:
        return None
    budget = max(256, round(640 * sqrt(ratio)))
    bounds = (l, t, r, b)
    candidates = _propose_in_crop(frame, detector, bounds, budget)
    return (candidates, bounds) if candidates else None


def crop_photo_foreground_proposals(frame, detector, overlay):
    """Extract proposals inside confirmed photo foregrounds at useful detail."""
    if (overlay is None or overlay.kind not in {"dialog", "bottom_sheet"}
            or not isinstance(detector, VisualControlDetector) or detector._size is not None):
        return None
    if overlay.kind == "dialog" and overlay.attached_close is not None:
        # The bright-rail check needs exterior proposals for this dialog type.
        return None
    height, width = frame.shape[:2]
    left, top, right, bottom = overlay.panel
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return None
    if overlay.kind == "bottom_sheet" and top < .18 * height:
        return None
    ratio = (right - left) * (bottom - top) / (width * height)
    if ratio > .70 or min(right - left, bottom - top) < 80:
        return None
    budget = max(256, round(640 * 1.12 * sqrt(ratio)))
    candidates = overlay.filter_controls(_propose_in_crop(frame, detector, overlay.panel, budget))
    if not candidates:
        return None
    # A detached dismiss is kept separately as detected_close. Its glyph is
    # checked on the original frame, so the dimmed gap never enters the crop.
    return candidates, overlay.panel


def crop_comment_proposals(frame, detector, overlay):
    """Scan a verified native comment sheet at the full-view sampling density."""
    if not isinstance(detector, VisualControlDetector) or detector._size is not None:
        return None
    height, width = frame.shape[:2]
    left, top, right, bottom = overlay.panel
    ratio = (right - left) * (bottom - top) / (width * height)
    if not .30 <= ratio <= .84:
        return None
    budget = max(256, round(640 * sqrt(ratio)))
    proposals = _propose_in_crop(frame, detector, overlay.panel, budget)
    return (proposals, overlay.panel) if proposals else None
