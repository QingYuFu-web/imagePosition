"""Find a single large product card above its separately recognized cart button."""

import cv2
import numpy as np

from screen_product_rows import _Card, _artwork_click, _price_evidence, _price_index, _spans


def _edge(profile, *, ending):
    length = len(profile)
    if ending:
        profile = profile[::-1]
    # The picture occupies most of the inner side. The exterior margin must
    # stay clear immediately before that edge.
    margin = round(.19 * length)
    runs = [(a, b) for a, b in _spans(profile >= .75, 1)
            if .13 * length <= a <= .79 * length and b - a >= .20 * length
            and (profile[max(0, a - margin):a] <= .22).mean() >= .89]
    if len(runs) != 1:
        return None
    position = runs[0][0]
    return length - position if ending else position


def _picture_boundary(frame, box, paper):
    a, t, b, d = box
    height, width = frame.shape[:2]
    span = b - a
    pad = round(.048 * span)
    inset = round(.14 * span)
    contours = []
    for mark, horizontal, ending in ((a, False, False), (t, True, False), (b, False, True), (d, True, True)):
        stop = height if horizontal else width
        begin, end = max(0, mark - pad), min(stop, mark + pad)
        strip = (frame[begin:end, a + inset:b - inset:4] if horizontal
                 else frame[t + inset:d - inset:4, begin:end])
        profile = (np.max(abs(strip.astype(np.int16) - paper), axis=2) >= 6).mean(axis=int(horizontal))
        offset = _edge(profile, ending=ending)
        if offset is None:
            return None
        contours.append(begin + offset)
    left, top, right, bottom = contours
    if not (.91 * span <= right - left <= 1.05 * span and .77 * span <= bottom - top <= 1.24 * span):
        return None
    texture = frame[top + inset:bottom - inset:5, left + inset:right - inset:5]
    return tuple(contours) if texture.std(axis=(0, 1)).max() >= 26 else None


def _title_line(ink, left, span):
    x, y, w, h = cv2.boundingRect(ink)
    if not (.015 * span <= h <= .060 * span and .21 * span <= w <= .965 * span and x <= .074 * span):
        return None
    runs = [(a, b) for a, b in _spans(ink[y:y + h, x:x + w].any(axis=0), 0) if b - a >= .07 * h]
    # A proof of a label needs separately positioned glyph islands, not a
    # similarly sized strip of painted decoration or an underline.
    heights = [cv2.boundingRect(ink[y:y + h, x + a:x + b])[3] for a, b in runs]
    if sum(.47 * h <= height <= h and .15 * h <= b - a <= 1.42 * h
           for (a, b), height in zip(runs, heights, strict=True)) < 7:
        return None
    return left + x, y, left + x + w, y + h


def _owned_price(frame, artwork, cart_top, paper):
    left, _, right, bottom = artwork
    span = right - left
    panel_top = bottom + max(3, round(.009 * span))
    panel = frame[panel_top:cart_top - round(.020 * span), left:right]
    ink = np.uint8((paper - panel.astype(np.int16)).min(axis=2) >= 49)
    spans = _spans(ink.mean(axis=1) >= .014, max(1, round(.002 * span)))
    if len(spans) < 2:
        return None
    first, last = spans[0]
    title = _title_line(ink[first:last], left, span)
    if first > .041 * span or title is None:
        return None
    bank = _price_index()
    if bank is None:
        return None
    prices, cache = [], {}
    for a, b in spans[1:]:
        if not (.010 * span <= b - a <= .052 * span and .009 * span <= a - last <= .075 * span):
            continue
        x, _, w, _ = cv2.boundingRect(ink[a:b])
        if not (.021 * span <= w <= .42 * span and x <= .08 * span):
            continue
        band = panel_top + a, panel_top + b
        evidence = _price_evidence(frame, left, right, band, span, bank, cache)
        if evidence is not None and evidence[1] is not None:
            title_left, title_top, title_right, title_bottom = title
            pad = max(3, round(.29 * (title_bottom - title_top)))
            caption = [max(left, min(title_left, left + x) - pad), panel_top + first + title_top - pad,
                       min(right, max(title_right, left + x + w) + pad), band[1] + pad]
            prices.append((evidence[1], caption))
    return prices[0] if len(prices) == 1 else None


def single_product_card_actions(frame, regions, known):
    """Add the detail click only when one photo, title and amount own the cart bar."""
    height, width = frame.shape[:2]
    if (regions.get("layer") != "page" or regions.get("regions") or height < 1.42 * width
            or width < 260):
        return []
    carts = [item["box"] for item in known if item["role"] == "cart_add"
             and .76 * width <= item["box"][2] - item["box"][0] <= .97 * width]
    if len(carts) != 1:
        return []
    cart_left, cart_top, cart_right, cart_bottom = carts[0]
    proposals = set()
    for item in regions["candidates"]:
        a, t, b, d = map(int, item["box"])
        span = b - a
        if (item["visibility"] == "page" and .75 * width <= span <= .98 * width
                and .76 * span <= d - t <= 1.24 * span and .16 * height <= t < d < cart_top < height
                and .125 * span <= cart_top - d <= .32 * span
                and max(abs(a - cart_left), abs(b - cart_right)) <= .058 * span):
            proposals.add((a, t, b, d))
    if len(proposals) != 1:
        return []
    proposal = proposals.pop()
    a, _, b, d = proposal
    paper = np.median(frame[d + 3:cart_top - 3:5, a:b:5], axis=(0, 1)).astype(np.int16)
    art = _picture_boundary(frame, proposal, paper)
    if art is None or cart_top - art[3] < .11 * (art[2] - art[0]):
        return []
    box = art[0], art[1], art[2], cart_top
    if any(item["role"] == "open_product" and box[0] <= item["click_point"][0] < box[2]
           and box[1] <= item["click_point"][1] < box[3] for item in known):
        return []
    if (printed := _owned_price(frame, art, cart_top, paper)) is None:
        return []
    price, caption = printed
    card = _Card(box, art, price)
    point = _artwork_click(card, (*regions["candidates"], *known))
    return [{"role": "open_product", "function": "打开商品详情", "preview_text": f"商品详情 {price}",
             "box": list(box), "click_point": point, "component_artwork": list(art),
             "component_evidence": {"artwork": list(art), "caption": caption},
             "component_box": [min(box[0], cart_left), box[1], max(box[2], cart_right), cart_bottom],
             "component_controls": [list(carts[0])]}] if point is not None else []
