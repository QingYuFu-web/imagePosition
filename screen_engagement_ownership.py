"""Keep measured engagement icons and their attached counts in one crop."""

from screen_navigation_ownership import _contains_point
from screen_user_examples import _contains, _coverage, _size


def _eligible(row, protected):
    return (row["type"] == "control_candidate" and id(row) not in protected
            and not any(row.get(key) for key in ("disabled", "review_only", "media_thumbnail", "control_scope")))


def _alias(row, glyph, number, vicinity, point):
    box = row["box"]
    if _coverage(box, vicinity) < .86:
        return False
    if _coverage(box, number) >= .82 and _coverage(number, box) >= .60:
        return True
    # A proposal around the glyph can stop part way through its measured count.
    return (_contains_point(row, point) and _coverage(glyph, box) >= .84
            and _coverage(number, box) >= .25)


def merge_counted_candidates(elements, actions, review, *, protected=()):
    """Collapse proposals only where an icon recognizer measured an owned count."""
    counted = [row for row in actions if row.get("engagement_count_box") and not row.get("review_only")]
    if not counted:
        return elements, {}
    protected = set(protected)
    removed, aliases = set(), {}
    for row in counted:
        glyph, number, context, point = (row[key] for key in ("box", "engagement_count_box", "context_box", "click_point"))
        if _coverage(glyph, context) < 1 or _coverage(number, context) < 1 or not _contains(glyph, point):
            continue
        separate = [other["click_point"] for other in (*actions, *review) if other is not row]
        if any(_contains(context, other) for other in separate):
            continue
        pad = round(.74 * max(_size(glyph)))
        left, top, right, bottom = context
        vicinity = [left - pad, top - pad, right + pad, bottom + pad]
        eligible = [item for item in elements if id(item) not in removed and _eligible(item, protected)]
        owners = [item for item in eligible if _contains_point(item, point)
                  and min(_coverage(item["box"], glyph), _coverage(glyph, item["box"])) >= .73]
        if len(owners) != 1:
            continue
        owner, = owners
        if any(_contains(owner["box"], other) for other in separate):
            continue
        duplicates = [item for item in eligible if item is not owner and not item.get("fixed_center")
                      and _alias(item, glyph, number, vicinity, point)
                      and not any(_contains(item["box"], other) for other in separate)]
        previously_offered = [owner["box"], *(item["box"] for item in duplicates if _contains_point(item, point))]
        owner.update(box=list(glyph), center=point, context_box=context, fixed_center=True)
        if (outline := row.get("photo_outline_px")) is not None:
            owner["polygon"] = outline
        else:
            owner.pop("polygon", None)
        aliases[id(owner)] = [{"box": box, "click_point": point} for box in previously_offered if box != glyph]
        removed.update(id(item) for item in duplicates)
        protected.add(id(owner))
    return [item for item in elements if id(item) not in removed], aliases
