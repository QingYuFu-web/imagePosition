"""Keep one candidate for a measured navigation control and its nested glyph."""

from math import prod

import cv2
import numpy as np

from screen_user_examples import _contains, _coverage, _size


def _contains_point(element, point):
    if not _contains(element["box"], point):
        return False
    outline = element.get("polygon")
    return outline is None or cv2.pointPolygonTest(np.float32(outline), tuple(map(float, point)), False) >= 0


def _shares_anchor(candidate, action):
    point = action["click_point"]
    box, measured = candidate["box"], action["box"]
    if not _contains_point(candidate, point):
        return False
    close = all(abs(a - b) <= max(3, .19 * span)
                for a, b, span in zip(candidate["center"], point, _size(box), strict=True))
    return close and max(_coverage(measured, box), _coverage(box, measured)) >= .84


def merge_navigation_candidates(elements, actions, image_size, *, protected=()):
    """Collapse coincident Back or Close proposals using a measured action point."""
    navigation = [item for item in actions if item["role"] in {"back", "close"} and not item.get("review_only")]
    if not navigation:
        return elements, {}
    excluded = set(protected)
    short = min(image_size)
    assignments = {}
    for item in elements:
        if (item["type"] != "control_candidate" or id(item) in excluded or item.get("review_only")
                or item.get("disabled") or item.get("media_thumbnail") or item.get("control_scope")
                or max(_size(item["box"])) > .21 * short):
            continue
        owners = [action for action in navigation if _shares_anchor(item, action)]
        if len(owners) != 1:
            continue
        owner, = owners
        # A separate measured action must keep its own clickable candidate.
        if any(action is not owner and min(_coverage(item["box"], action["box"]),
                                           _coverage(action["box"], item["box"])) >= .58
               and not _contains(owner["box"], action["click_point"]) for action in actions):
            continue
        assignments.setdefault(id(owner), []).append(item)

    removed, aliases = set(), {}
    for action in navigation:
        candidates = assignments.get(id(action), ())
        if len(candidates) < 2:
            continue
        point = action["click_point"]
        large = [item for item in candidates if _coverage(action["box"], item["box"]) >= .84]
        if not large:
            continue
        # Require one actual containing candidate. Overlapping, non-nested
        # rectangles are insufficient evidence to merge different objects.
        owner = max(large, key=lambda item: prod(_size(item["box"])))
        nested = [item for item in candidates if _coverage(item["box"], owner["box"]) >= .91]
        if len(nested) != len(candidates):
            continue
        owner["center"] = point
        owner["fixed_center"] = True
        aliases[id(owner)] = [{"box": item["box"], "click_point": point} for item in nested if item is not owner]
        aliases[id(owner)].append({"box": action["box"], "click_point": point})
        removed.update(id(item) for item in nested if item is not owner)
    return [item for item in elements if id(item) not in removed], aliases
