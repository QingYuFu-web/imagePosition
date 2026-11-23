"""Keep independently visible selector values in scope for value-only queries."""

from collections import defaultdict

from inventory_action_search import _event_key
from inventory_selector_side_titles import _stepper_arrow


def mark_unlisted_value_peers(evidence, rows):
    """Attach measured points for same-value fields that did not yield an action."""
    from inventory_value_select import _ink_strength, _overlap, _shown_value

    known_values = defaultdict(list)
    for _, action in rows:
        value = action.get("setting_value")
        if isinstance(value, str) and (key := _event_key(value)):
            known_values[key].append(action)
    if not known_values:
        return

    occupied = [field for field, _ in rows]
    points = defaultdict(list)
    for field, arrow in evidence.strict_closed.items():
        if field.width < 1.70 * field.height or any(_overlap(field, prior) for prior in occupied):
            continue
        valid, value = _shown_value(evidence.gray, field, arrow, evidence.lines)
        if not valid or value is None or value.score < 90:
            continue
        key = _event_key(value.text)
        if key not in known_values or _ink_strength(evidence.gray, value) < 74:
            continue
        if _stepper_arrow(evidence.gray, field, arrow, evidence.rectangles):
            continue
        occupied.append(field)
        points[key].append(arrow.center)

    for key, centers in points.items():
        for action in known_values[key]:
            action["setting_value_peer_centers_px"] = sorted(centers)
