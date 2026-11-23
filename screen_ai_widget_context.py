"""Retain a small widget control when it falls outside the normal crops."""

import cv2

from screen_page_handoff import _intersection


def widget_context_box(page, frame, panels, covered, photo, *, handle_only=False):
    """Show a measured dismiss handle and the adjacent widget boundary."""
    if page["layer"]["kind"] != "page":
        return None
    width, height = photo.size if photo is not None else frame.shape[1::-1]
    if width < 380 or not 1.55 * width <= height <= 2.95 * width:
        return None

    from screen_ai_gap_context import _mapped_box
    from screen_live_producer_actions import _selected_picture

    view = frame if photo is None else cv2.warpPerspective(frame, photo.to_screen, (width, height))
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    if photo is not None and photo.blocked is not None:
        gray[photo.blocked != 0] = 32
    found = _selected_picture(gray)
    if found is None:
        return None
    _, close = found
    original = list(close) if photo is None else _mapped_box(photo.to_photo, close)
    excluded = [row["box"] for row in page["elements"] if row.get("disabled")]
    excluded.extend(sample["box"] for row in page["elements"]
                    if (sample := row.get("reviewed_sample")) is not None and sample["purpose"] is None)
    if (any(_intersection(original, box) == original for box in covered)
            or any(_intersection(original, box) for box in excluded)):
        return None

    left, top, right, bottom = close
    size = max(right - left, bottom - top)
    context = [max(0, left - round(1.65 * size)), max(0, top - round(.48 * size)),
               min(width, right + round(.30 * size)), min(height, bottom + round(.95 * size))]
    if photo is not None:
        x0, y0, x1, y1 = context
        if photo.blocked is not None and photo.blocked[y0:y1, x0:x1].any():
            return None
        context = _mapped_box(photo.to_photo, context)
    visible = next((clipped for panel in panels if (clipped := _intersection(context, panel)) is not None
                    and _intersection(original, clipped) == original), None)
    return original if visible is not None and handle_only else visible
