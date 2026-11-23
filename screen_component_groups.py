"""Keep measured cards together while retaining separately evidenced controls."""

from math import ceil, floor

import numpy as np

from screen_user_examples import _contains, _coverage, _visible


_CARD_ROLES = {"open_product", "game_offer_card", "game_equipment_item"}


def _same_extent(box, other, minimum=.70):
    return min(_coverage(box, other), _coverage(other, box)) >= minimum


def _centered_on_control(bounds, center):
    left, top, right, bottom = bounds
    x, y = center
    return (abs(x - (left + right) / 2) <= .12 * (right - left)
            and abs(y - (top + bottom) / 2) <= .12 * (bottom - top))


def _inline_caption(frame, box, caption):
    left, top, right, bottom = box
    depth = bottom - top
    reach = round(.8 * depth)
    top, bottom = top + round(.23 * depth), bottom - round(.16 * depth)
    for x0, x1 in ((max(caption[0], left - reach), left), (right, min(caption[2], right + reach))):
        if x1 - x0 < .35 * depth:
            return False
        strip = frame[top:bottom:2, x0:x1:2]
        paper = (strip.min(axis=2) >= 213).mean()
        ink = (strip.max(axis=2) <= 160).mean()
        if paper < .46 or not .09 <= ink <= .55:
            return False
    return True


def _caption_fragment(frame, box, caption):
    left, top, right, bottom = box
    # Portrait paired cards have separately measured bright, printed caption bands.
    # A compact icon or painted button can still be a separate control.
    if right - left < 1.65 * (bottom - top) or _coverage(box, caption) < .87:
        return False
    x0, y0 = ceil(max(left, caption[0])), ceil(max(top, caption[1]))
    x1, y1 = floor(min(right, caption[2])), floor(min(bottom, caption[3]))
    if x0 >= x1 or y0 >= y1:
        return False
    paper = frame[y0:y1:3, x0:x1:3].min(axis=2)
    bright = paper >= 213
    if _inline_caption(frame, (x0, y0, x1, y1), caption):
        return True
    if (bright.mean() < .56 or np.mean(bright, axis=1).max() < .85
            or np.mean(bright, axis=0).max() < .75):
        return False
    inset = round(.13 * (x1 - x0))
    reach = max(1, round(.24 * (y1 - y0)))
    # Printed lines lack a continuous outline around the caption. A long
    # outlined button can otherwise have the same proportions and white fill.
    edges = (frame[y0:y0 + reach, x0 + inset:x1 - inset:2],
             frame[y1 - reach:y1, x0 + inset:x1 - inset:2])
    return not all(((edge.min(axis=2) <= 165).mean(axis=1) >= .72).any() for edge in edges)


def group_card_candidates(frame, observation, confirmed_examples):
    """Return the explicit owners and the raw candidates they do not replace."""
    regions = observation.regions
    candidates = regions["candidates"]
    if regions["layer"] != "page" or regions.get("regions"):
        return (), candidates
    cards = []
    height, width = frame.shape[:2]
    for item in observation.active:
        box, point = item["box"], item["click_point"]
        left, top, right, bottom = box
        if (item.get("role") not in _CARD_ROLES or item.get("review_only")
                or not (0 <= left < right <= width and 0 <= top < bottom <= height)
                or not _contains(box, point) or not _visible(regions, box, point)):
            continue
        if all(max(_coverage(box, prior["box"]), _coverage(prior["box"], box)) < .20 for prior in cards):
            cards.append(item)
    if not cards:
        return (), candidates

    group_ids = {id(item) for item in cards}
    independent = [item["box"] for item in (*observation.active, *observation.review)
                   if id(item) not in group_ids]
    independent.extend(box for card in cards for box in card.get("component_controls", ()))
    independent.extend(row["box"] for row in confirmed_examples
                       if not any(_same_extent(row["box"], card["box"], .86) for card in cards))

    def belongs_to_card(box):
        for item in cards:
            if _same_extent(box, item["box"]):
                return True
            if (art := item.get("component_artwork")) is not None and _same_extent(box, art, .80):
                return True
            if (caption := item.get("component_caption")) is not None and _caption_fragment(frame, box, caption):
                return True
        contained = [item["box"] for item in cards if _coverage(item["box"], box) >= .90]
        return len(contained) >= 2 and sum(_coverage(box, bounds) for bounds in contained) >= .80

    def redundant(row):
        box = row["box"]
        return (row.get("visibility") == "page" and not row.get("review_only") and belongs_to_card(box)
                and not any(_same_extent(box, bounds, .62) for bounds in independent))

    return cards, [row for row in candidates if not redundant(row)]


def link_card_controls(cards, elements, regions, image_size):
    """Connect an explicitly measured separate control to its card."""
    links = []
    assigned = {id(parent) for _, parent in cards}
    width, height = image_size
    for action, parent in cards:
        group = list(action.get("component_box", parent["box"]))
        found = False
        for box in action.get("component_controls", ()):
            left, top, right, bottom = box
            center = [(left + right) / 2, (top + bottom) / 2]
            if (not (0 <= left < right <= width and 0 <= top < bottom <= height)
                    or _coverage(box, group) < .95 or not _visible(regions, box, center)):
                continue
            nearby = [row for row in elements if row["type"] == "control_candidate"
                      and id(row) not in assigned and _contains(row["box"], center)
                      and _coverage(box, row["box"]) >= .72]
            matches = [row for row in nearby if _coverage(row["box"], box) >= .36]
            if len(matches) > 1:
                # Multiple boxes may describe different controls at this spot.
                # Keep the measured button as the one with proven ownership.
                matches = [row for row in matches if _same_extent(row["box"], box, .72)]
            if len(matches) > 1:
                continue
            if matches:
                child, = matches
            else:
                centered = [row for row in nearby if not row.get("review_only")
                            and _coverage(row["box"], box) >= .16 and _centered_on_control(row["box"], center)]
                if len(centered) == 1:
                    child, = centered
                    child.update(box=list(box), center=center)
                    child.pop("polygon", None)
                else:
                    child = {"type": "control_candidate", "box": list(box), "center": center}
                    elements.append(child)
            assigned.add(id(child))
            links.append((child, parent))
            found = True
            l, t, r, b = child["box"]
            group = [min(group[0], l), min(group[1], t), max(group[2], r), max(group[3], b)]
        if found and group != parent["box"]:
            parent["component_box"] = group
    return links
