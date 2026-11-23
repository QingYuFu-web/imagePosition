"""Export visible screenshot evidence for a separate AI decision."""

import json
from math import ceil, floor

import cv2
import numpy as np

from screen_ai_visibility import on_selection_marker
from screen_component_groups import group_card_candidates, link_card_controls
from screen_engagement_ownership import merge_counted_candidates
from screen_navigation_ownership import merge_navigation_candidates
from screen_user_examples import _contains, _coverage, _frame_key, _same_owner, _sample_priority, _to_mm, _visible


def _intersection(first, second):
    box = [max(first[0], second[0]), max(first[1], second[1]),
           min(first[2], second[2]), min(first[3], second[3])]
    return box if box[0] < box[2] and box[1] < box[3] else None


def _bounded(box, width, height):
    return _intersection([floor(box[0]), floor(box[1]), ceil(box[2]), ceil(box[3])], [0, 0, width, height])


def _visible_box(box, panels, floating, visibility):
    if box is None:
        return None
    if visibility != "foreground" and any(_coverage(box, pane) >= .20 for pane in floating):
        return None
    if not panels:
        return box
    # Keep the visible extent of an element crossing the edge of a panel.
    portions = [piece for panel in panels if (piece := _intersection(box, panel)) is not None]
    if not portions:
        return None
    return max(portions, key=lambda piece: (piece[2] - piece[0]) * (piece[3] - piece[1]))


def _visual_element(kind, box, bounds, row):
    left, top, right, bottom = box
    result = {"type": kind, "box": box, "center": [(left + right) / 2, (top + bottom) / 2]}
    if row.get("review_only"):
        result["review_only"] = True
    if kind == "control_candidate" and row.get("disabled"):
        result["disabled"] = True
    if kind == "control_candidate" and box == bounds:
        if (row.get("control_role") == "text_input" and row.get("input_state") == "empty"
                and row.get("placeholder") and not row.get("review_only")):
            result.update(control_role="text_input", input_state="empty", placeholder=row["placeholder"])
        if row.get("disabled") and "control_label" in row:
            result["control_label"] = row["control_label"]
    if kind == "control_candidate" and "control_scope" in row:
        result["control_scope"] = row["control_scope"]
    if kind == "control_candidate" and row.get("media_thumbnail"):
        result["media_thumbnail"] = True
        if box == bounds and "selection_marker" in row:
            result["selection_marker"] = row["selection_marker"]
    outline = row.get("photo_outline_px")
    if outline is not None and box == bounds:
        result["polygon"] = outline
    if kind == "control_candidate" and box == bounds and "context_box" in row:
        if (context := row["context_box"]) != box and _intersection(context, box) == box:
            result["context_box"] = context
    if outline is not None or kind == "control_candidate":
        point = row.get("click_point", row.get("center"))
        if point is not None and left <= point[0] < right and top <= point[1] < bottom:
            if outline is None or cv2.pointPolygonTest(np.asarray(outline, dtype=np.float32), tuple(point), False) >= 0:
                result["center"] = point
                if kind == "control_candidate" and row.get("fixed_center"):
                    result["fixed_center"] = True
    return result


def _sample_owner(elements, row, regions, image_size, aliases):
    box = _bounded(row["box"], *image_size)
    point = row["click_point"]
    floating = row.get("floating_control")
    if box is None or box != list(row["box"]) or not _visible(regions, box, point, floating=floating):
        return
    owners = [item for item in elements if item["type"] == "control_candidate"
              and (_same_owner(row, {"box": item["box"], "click_point": item["center"]})
                   or any(_same_owner(row, other) for other in aliases.get(id(item), ())))]
    if len(owners) > 1:
        return
    if owners:
        owner, = owners
        if owner.get("disabled") or not _contains(owner["box"], point) or on_selection_marker(owner, point):
            return
        if (outline := owner.get("polygon")) is not None:
            if cv2.pointPolygonTest(np.float32(outline), tuple(map(float, point)), False) < 0:
                return
        return owner
    if not floating and (row["match"] == "exact" or row.get("raster_recovery")) and row["function"] is not None:
        visible = _visible_box(box, regions.get("regions", ()), regions.get("floating_controls", ()), None)
        if visible is None or not (visible[0] <= point[0] < visible[2] and visible[1] <= point[1] < visible[3]):
            return
        owner = _visual_element("control_candidate", visible, box, row)
        if any(item.get("review_only") and _coverage(visible, item["box"]) >= .40 for item in elements):
            owner["review_only"] = True
        return owner


def _apply_checked_samples(elements, samples, regions, image_size, aliases):
    """Use confirmed points and purposes only on unambiguous current owners."""
    matched = {}
    for row in samples:
        if (owner := _sample_owner(elements, row, regions, image_size, aliases)) is not None:
            matched.setdefault(id(owner), []).append((owner, row))
    existing = {id(item) for item in elements}
    for key, pairs in matched.items():
        if key in aliases and len(pairs) > 1:
            best = max(_sample_priority(item) for _, item in pairs)
            pairs = [(owner, item) for owner, item in pairs if _sample_priority(item) == best]
            if len(pairs) > 1 and all((item["function"], item["click_point"]) ==
                                      (pairs[0][1]["function"], pairs[0][1]["click_point"]) for _, item in pairs):
                pairs = pairs[:1]
        if len(pairs) != 1:
            continue
        (owner, row), = pairs
        purpose = row["function"]
        owner["reviewed_sample"] = {"purpose": purpose, "match": row["match"],
                                    **({"origin": row["origin"]} if row.get("origin") in {"ai", "detector"} else {}),
                                    **({"box": row["box"]} if purpose is None else {})}
        if purpose is not None:
            owner["center"] = row["click_point"]
        if key not in existing:
            elements.append(owner)


def build_page_analysis(frame, observation, *, confirmed_examples=()):
    """Assemble outlines and literal text from the currently visible layer."""
    height, width = frame.shape[:2]
    regions = observation.regions
    layer = regions["layer"]
    panels = [box for panel in regions.get("regions", ()) if (box := _bounded(panel, width, height))]
    floating = [box for panel in regions.get("floating_controls", ()) if (box := _bounded(panel, width, height))]
    scope_known = layer == "page" or bool(panels)
    elements = []
    links = []
    if scope_known:
        confirmed_examples = tuple(confirmed_examples)
        cards, candidates = group_card_candidates(frame, observation, confirmed_examples)
        review_candidates = []
        close_boxes = ([item["box"] for item in observation.active if item.get("role") == "close"]
                       if regions.get("proposal_scope") == "foreground_crop" else [])
        for row in candidates:
            visibility = row.get("visibility")
            if visibility not in {"page", "foreground"} or (layer == "layer_uncertain" and visibility != "foreground"):
                continue
            bounds = _bounded(row["box"], width, height)
            box = _visible_box(bounds, panels, floating, visibility)
            if box and close_boxes:
                x, y = row["center"]
                if any(close[0] <= x < close[2] and close[1] <= y < close[3]
                       and max(_coverage(box, close), _coverage(close, box)) >= .72
                       for close in close_boxes):
                    continue
            if box and row.get("review_only"):
                review_candidates.append(_visual_element("control_candidate", box, bounds, row))
            elif box and not any(min(_coverage(box, other["box"]), _coverage(other["box"], box)) >= .70
                                 for other in elements):
                elements.append(_visual_element("control_candidate", box, bounds, row))
        grouped = []
        for item in cards:
            box = list(item["box"])
            component = _visual_element("control_candidate", box, box, item)
            component["center"] = item["click_point"]
            if (evidence := item.get("component_evidence")) and all(_intersection(part, box) == part
                                                                   for part in evidence.values()):
                component["component_evidence"] = evidence
            elements.append(component)
            grouped.append((item, component))
        # Pixel recognizers also outline controls that the detector missed.
        # Only use their visual extent. Their inferred action is not evidence.
        if layer != "layer_uncertain":
            for row in (*observation.active, *observation.review):
                bounds = _bounded(row["box"], width, height)
                box = _visible_box(bounds, panels, floating, None)
                if not box or (panels and _coverage(bounds, box) < .60):
                    continue
                owners = (other for other in elements
                          if min(_coverage(box, other["box"]), _coverage(other["box"], box)) >= .70)
                owner = next(owners, None)
                if owner is None:
                    elements.append(_visual_element("control_candidate", box, bounds, row))
                elif box == bounds and (context := row.get("context_box")) is not None and next(owners, None) is None:
                    if (_contains(owner["box"], row["click_point"]) and _intersection(context, owner["box"]) == owner["box"]
                            and _visible_box(context, panels, floating, None) == context):
                        owner.setdefault("context_box", context)
        for item in review_candidates:
            box = item["box"]
            if not any(min(_coverage(box, other["box"]), _coverage(other["box"], box)) >= .70 for other in elements):
                elements.append(item)
        elements, aliases = merge_navigation_candidates(elements, observation.active, (width, height),
                                                         protected=(id(item) for _, item in grouped))
        links = link_card_controls(grouped, elements, regions, (width, height))
        if layer == "page" and not panels and not floating:
            protected = {id(item) for _, item in grouped}
            protected.update(id(item) for link in links for item in link)
            elements, counted_aliases = merge_counted_candidates(elements, observation.active, observation.review,
                                                                  protected=protected)
            aliases.update(counted_aliases)
        if confirmed_examples:
            _apply_checked_samples(elements, confirmed_examples, regions, (width, height), aliases)
        seen_lines = set()
        for row in observation.labels:
            label = row.get("text")
            if not isinstance(label, str) or not label.strip():
                continue
            bounds = _bounded(row["line_box"], width, height)
            box = _visible_box(bounds, panels, floating, None)
            if not box or box != bounds:
                continue
            key = tuple(box), label.strip()
            if key not in seen_lines:
                seen_lines.add(key)
                elements.append({**_visual_element("text", box, bounds, row), "label": label.strip()})
    elements.extend({**_visual_element("region", box, box, {}), "label": "前景范围"} for box in dict.fromkeys(
        tuple(region) for region in (*panels, *floating)))
    for row in elements:
        row["box"] = list(row["box"])
    elements.sort(key=lambda row: (row["box"][1], row["box"][0], row["type"], row["box"][3], row["box"][2]))
    for index, element in enumerate(elements, 1):
        element["id"] = f"E{index}"
    for child, parent in links:
        child["parent_id"] = parent["id"]

    calibration = regions.get("calibration")
    matrix = _to_mm(regions, frame) if calibration else None
    if matrix is not None and elements:
        points = np.float64([[element["center"] for element in elements]])
        positions = cv2.perspectiveTransform(points, matrix)[0].round(1).tolist()
        for element, point in zip(elements, positions, strict=True):
            element["center_mm"] = point

    words = [row for row in elements if row["type"] == "text"]
    for item in elements:
        if item["type"] == "control_candidate":
            linked = [word["id"] for word in words if _coverage(word["box"], item["box"]) >= .75]
            if linked:
                item["text_ids"] = linked
    page = {
        "schema": "screen-page/1", "frame_id": _frame_key(frame),
        "image_size": [width, height], "coordinate_space": "original_image_px", "origin": "image_top_left",
        "layer": {"kind": layer, "boxes": panels, **({"floating_boxes": floating} if floating else {})},
        **({"calibration": calibration} if calibration else {}),
        **({"screen_corners": regions["screen_corners_px"]} if "screen_corners_px" in regions else {}),
        "elements": elements,
    }
    if confirmed_examples:
        from screen_ai_visibility import visible_point

        millimeters = matrix is not None
        page["click_coordinate_space"] = calibration["space"] if millimeters else "original_image_px"
        page["click_targets"] = [
            {"function": sample["purpose"], "click_point": item["center_mm"] if millimeters else item["center"]}
            for item in elements if not item.get("review_only") and (sample := item.get("reviewed_sample"))
            and sample["purpose"] is not None and visible_point(page, item, item["center"])
        ]
    return page


def page_payload(result):
    return result["page_analysis"]


def isolate_page_pixels(frame, page):
    """Clip only known non-visible areas, keeping unboxed display content."""
    panels = page["layer"]["boxes"]
    corners = page.get("screen_corners")
    if not panels and not corners:
        return frame.copy()

    mask = np.zeros(frame.shape[:2], np.uint8)
    if panels:
        for left, top, right, bottom in panels:
            mask[top:bottom, left:right] = 255
    else:
        mask[:] = 255
    if corners:
        display = np.zeros_like(mask)
        outline = np.rint(np.asarray(corners) * 16).astype(np.int32)
        cv2.fillConvexPoly(display, outline, 255, shift=4)
        cv2.bitwise_and(mask, display, dst=mask)
    isolated = np.full_like(frame, 27)
    cv2.copyTo(frame, mask, isolated)
    return isolated


def draw_page(frame, page, *, simplified=False, target_point=None):
    from screen_ai_visibility import visible_point

    image = isolate_page_pixels(frame, page) if simplified else frame.copy()
    colors = {"region": (174, 159, 139), "text": (225, 178, 112), "control_candidate": (161, 218, 86)}
    words_in_controls = {word for row in page["elements"] for word in row.get("text_ids", ())}
    control_boxes = {tuple(row["box"]) for row in page["elements"] if row["type"] == "control_candidate"}
    width = frame.shape[1]
    size = min(.86, max(.34, width / 1600))
    stroke = max(1, round(width / 620))
    for row in page["elements"]:
        if row["id"] in words_in_controls or (row["type"] == "region" and tuple(row["box"]) in control_boxes):
            continue
        left, top, right, bottom = row["box"]
        color = colors[row["type"]]
        if row.get("disabled"):
            color = (160, 160, 160)
        elif row.get("review_only"):
            color = (88, 185, 238)
        elif "parent_id" in row:
            color = (105, 168, 255)
        if (bounds := row.get("component_box")) is not None:
            l, t, r, b = bounds
            cv2.rectangle(image, (l, t), (r - 1, b - 1), (170, 153, 134), stroke, cv2.LINE_AA)
        if (outline := row.get("polygon")) is None:
            cv2.rectangle(image, (left, top), (right - 1, bottom - 1), color, stroke, cv2.LINE_AA)
        else:
            cv2.polylines(image, [np.rint(outline).astype(np.int32)], True, color, stroke, cv2.LINE_AA)
        if ((sample := row.get("reviewed_sample")) is not None and sample["purpose"] and not row.get("review_only")
                and visible_point(page, row, row["center"])):
            x, y = map(round, row["center"])
            cv2.circle(image, (x, y), max(3, 2 * stroke), (24, 33, 44), cv2.FILLED, cv2.LINE_AA)
            cv2.circle(image, (x, y), max(2, stroke), color, cv2.FILLED, cv2.LINE_AA)
        name = row["id"]
        (label_w, label_h), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, size, stroke)
        x, y = max(0, min(left, width - label_w - 6)), max(label_h + 4, top - 4)
        cv2.rectangle(image, (x, y - label_h - 3), (x + label_w + 5, y + baseline), (22, 31, 43), cv2.FILLED)
        cv2.putText(image, name, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, stroke, cv2.LINE_AA)
    if target_point is not None:
        x, y = map(round, target_point)
        length = max(9, round(.011 * width))
        for start, end in (((x - length, y), (x + length, y)), ((x, y - length), (x, y + length))):
            cv2.line(image, start, end, (27, 33, 46), stroke + 3, cv2.LINE_AA)
            cv2.line(image, start, end, (104, 224, 255), stroke + 1, cv2.LINE_AA)
    return image


def save_page_handoff(destination, source, frame, result, *, extraction_input=None, target_point=None,
                      include_review_images=True):
    page = page_payload(result)
    if page["frame_id"] != _frame_key(frame):
        raise ValueError("当前图片与页面分析不一致，请重新分析。")
    destination.mkdir(parents=True, exist_ok=True)
    images = {"annotated": (".page.jpg", draw_page(frame, page, target_point=target_point))}
    if include_review_images:
        images.update(simplified=(".page.simplified.png", draw_page(frame, page, simplified=True,
                                                                   target_point=target_point)),
                      original=(".page.source.png", frame))
        if extraction_input is not None:
            images["extraction_input"] = (".page.input.png", extraction_input.pixels)
    paths = [destination / f"{source.stem}{suffix}" for suffix in
             (".page.json", *(suffix for suffix, _ in images.values()))]
    if source.resolve() in {path.resolve() for path in paths}:
        raise ValueError("输出路径与原图重复")
    files = dict(zip(images, (path.name for path in paths[1:]), strict=True))
    payload = {**page, "source": source.name, "files": files}
    encoded = []
    for path, (_, image) in zip(paths[1:], images.values(), strict=True):
        ok, data = cv2.imencode(path.suffix, image)
        if not ok:
            raise OSError(f"无法保存 {path}")
        encoded.append(data)
    for path, data in zip(paths[1:], encoded, strict=True):
        data.tofile(path)
    paths[0].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths
