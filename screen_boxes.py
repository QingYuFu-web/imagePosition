"""Extract candidate regions with the existing OpenCV and YOLO pipeline."""

from dataclasses import dataclass, replace
from functools import cache, partial
from itertools import combinations
import json
from math import ceil, floor
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from general_visual import VisualControlDetector
from screen_box_actions import draw_actions, isolate_action_pixels, suggest_box_actions
from screen_candidate_glyphs import CandidateGlyphs
from screen_foreground import BlockingOverlay, _comment_sheet_surfaces, _working_view, detect_blocking_overlay
from screen_photo_geometry import ExtractionInput


_UNMEASURED_OVERLAY = object()


def read_frame(path):
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"无法读取图片 {path}")
    return frame


def _box(control, width, height):
    polygon = np.asarray(control["polygon"])
    left, top = polygon.min(axis=0)
    right, bottom = polygon.max(axis=0)
    return [max(0, floor(left)), max(0, floor(top)), min(width, ceil(right)), min(height, ceil(bottom))]


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _box_overlap(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    shared = max(0, right - left) * max(0, bottom - top)
    return shared / (_area(first) + _area(second) - shared) if shared else 0.


def _bright_component(image):
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = np.uint8((luminance >= 222) & (np.ptp(image, axis=2) <= 32))
    _, _, components, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if len(components) <= 1:
        return False
    return any(area >= .075 * ink.size and .26 * image.shape[1] <= w <= .88 * image.shape[1]
               and .20 * image.shape[0] <= h <= .76 * image.shape[0]
               for _, _, w, h, area in components[1:])


def _separate_bright_rail(frame, rows, overlay):
    """Keep visible exterior proposals for review when a translucent panel is ambiguous."""
    if overlay.kind != "dialog" or overlay.attached_close is None:
        return False
    height, width = frame.shape[:2]
    panel_left, _, panel_right, _ = overlay.panel
    lanes = [[], []]
    for row in rows:
        left, top, right, bottom = row["box"]
        x, y = row["center"]
        w, h = right - left, bottom - top
        on_left = right <= panel_left and x <= .17 * width
        on_right = left >= panel_right and x >= .83 * width
        if ((on_left or on_right) and .12 * height <= y <= .89 * height and
                .043 * width <= min(w, h) <= max(w, h) <= .19 * width and
                _bright_component(frame[top:bottom, left:right])):
            lanes[on_right].append((x, y, top, bottom, w, h))
    for lane in lanes:
        for a, b, c in combinations(sorted(lane, key=lambda row: row[1]), 3):
            first, second = b[1] - a[1], c[1] - b[1]
            if (a[3] <= b[2] and b[3] <= c[2] and .057 * height <= first <= .17 * height and
                    .82 <= second / first <= 1.22 and max(a[0], b[0], c[0]) - min(a[0], b[0], c[0]) <= .027 * width and
                    max(a[4], b[4], c[4]) <= 1.34 * min(a[4], b[4], c[4]) and
                    max(a[5], b[5], c[5]) <= 1.32 * min(a[5], b[5], c[5])):
                return True
    return False


def extract_boxes(frame, detector, *, label_index=None, _overlay=_UNMEASURED_OVERLAY, _crop_dialog=False,
                  _crop_comment=False, _view=None, proposal_frame=None):
    """Return raw candidate geometry and visible-layer scope without assigning functions."""
    height, width = frame.shape[:2]
    if _overlay is _UNMEASURED_OVERLAY:
        _view = _working_view(frame) if _view is None else _view
        overlay = detect_blocking_overlay(frame, working_view=_view, camera=proposal_frame is not None,
                                          label_index=label_index)
    else:
        overlay = _overlay
    if overlay is not None and overlay.kind in {"voice_recording", "voice_stop_panel", "wide_panel"}:
        result = {"image_size": [width, height], "coordinate_space": "original_image_px", "origin": "image_top_left",
                  "layer": overlay.kind, "regions": list(overlay.regions), "candidates": []}
        key, box = (("recording_stop", overlay.recording_stop) if overlay.kind == "voice_stop_panel"
                    else ("detected_close", overlay.close_button))
        if box is not None:
            left, top, right, bottom = box
            result[key] = box
            result["candidates"].append({"id": 1, "box": list(box), "center": [(left + right) / 2, (top + bottom) / 2],
                                         "visibility": "foreground"})
        return result
    comment_input = None
    if (label_index is not None and (overlay is None or overlay.kind == "bottom_sheet")
            and 1.55 * width <= height <= 2.73 * width):
        from screen_sheet_actions import captioned_comment_sheet

        _view = _working_view(frame) if _view is None else _view
        surfaces = tuple(_comment_sheet_surfaces(_view[1])) if _view[1] is not None else ()
        confirmed = captioned_comment_sheet(frame, label_index, working_view=_view, surfaces=surfaces)
        if confirmed is not None:
            observed, composer = confirmed
            if overlay is None:
                overlay, comment_input = observed, composer
            elif (_box_overlap(overlay.panel, observed.panel) >= .82 and
                  overlay.panel[0] <= composer[0] < composer[2] <= overlay.panel[2] and
                  overlay.panel[1] <= composer[1] < composer[3] <= overlay.panel[3]):
                comment_input = composer
        if comment_input is None:
            from screen_comment_sheet_modal import comment_sheet_modal

            if (sheet := comment_sheet_modal(frame, label_index, overlay=overlay, working_view=_view,
                                             surfaces=surfaces)) is not None:
                observed, controls = sheet
                close = next((tuple(map(int, item["box"])) for item in controls if item["role"] == "close"), None)
                if overlay is None:
                    overlay = BlockingOverlay("bottom_sheet", observed.panel, close)
                comment_input = next((tuple(map(int, item["box"])) for item in controls
                                      if item["role"] == "comment_input"), None)
    cropped = None
    visual = frame if proposal_frame is None else proposal_frame
    if proposal_frame is not None and overlay is not None:
        from screen_foreground_proposals import crop_photo_foreground_proposals

        cropped = crop_photo_foreground_proposals(visual, detector, overlay)
    if (_crop_comment and proposal_frame is None and overlay is not None and overlay.kind == "bottom_sheet"
            and (not overlay.comment_sheet or width >= 640)):
        from screen_comment_composer_icons import comment_composer_icons
        from screen_foreground_proposals import crop_comment_proposals

        # Small sheets resolved from their own headings already have useful full-view proposals.
        if (overlay.comment_sheet or comment_input is not None
                or {item["role"] for item in comment_composer_icons(frame, overlay.panel)} >= {"mention", "emoji"}):
            cropped = crop_comment_proposals(visual, detector, overlay)
    cropped_foreground = cropped is not None
    if cropped is None and _crop_dialog:
        from screen_foreground_proposals import crop_dialog_proposals

        cropped = crop_dialog_proposals(visual, detector, overlay)
    raw, input_bounds = cropped if cropped is not None else (detector.detect(visual), (0, 0, width, height))
    native_count = len(raw)
    if (overlay is None or overlay.include_coarse_proposals
            or (overlay.kind == "bottom_sheet" and overlay.panel[1] < .12 * height)):
        # Different scales produce different proposal sizes. Preserve the fine
        # pass on a shared component and use coarse proposals to fill gaps.
        if cropped is not None and _crop_comment and proposal_frame is None:
            from screen_foreground_proposals import _propose_in_crop

            coarse = _propose_in_crop(visual, detector, input_bounds, 320)
        else:
            coarse = detector.detect(visual, pixel_budget=384)
        raw.extend(sorted(coarse,
                          key=lambda proposal: _area(_box(proposal, width, height)), reverse=True))
    rows = []
    for index, proposal in enumerate(raw):
        left, top, right, bottom = _box(proposal, width, height)
        if right <= left or bottom <= top:
            continue
        box = [left, top, right, bottom]
        if index >= native_count and any(_box_overlap(box, row["box"]) >= .40 for row in rows):
            continue
        rows.append({"id": index + 1, "box": box,
                     "center": [(left + right) / 2, (top + bottom) / 2], "score": proposal["score"],
                     "polygon": proposal["polygon"], "pass": "fine" if index < native_count else "coarse"})
    state = "page"
    if overlay is not None:
        front = {row["id"] for row in overlay.filter_controls(rows)}
        if overlay.attached_close is not None:
            left, top, right, bottom = overlay.attached_close
            matches = [row for row in rows if left <= row["center"][0] < right and top <= row["center"][1] < bottom]
            close_center = ((left + right) / 2, (top + bottom) / 2)
            if matches:
                close = min(matches, key=lambda row: (row["center"][0] - close_center[0]) ** 2 +
                            (row["center"][1] - close_center[1]) ** 2)
                front.add(close["id"])
        ambiguous = _separate_bright_rail(frame, rows, overlay)
        state = "layer_uncertain" if ambiguous else overlay.kind
        rows = [{**row, "visibility": "foreground" if row["id"] in front else "layer_uncertain"}
                for row in rows if row["id"] in front or (ambiguous and not overlay.contains(row["center"]))]
    else:
        for row in rows:
            row["visibility"] = "page"
    for row in rows:
        del row["polygon"]
    if overlay is not None and overlay.share_icons:
        next_id = max((row["id"] for row in rows), default=0) + 1
        retain_overlap = .32 if overlay.include_coarse_proposals else .65
        for left, top, right, bottom in overlay.share_icons:
            box = [left, top, right, bottom]
            center = [(left + right) / 2, (top + bottom) / 2]
            if any(_box_overlap(box, row["box"]) >= retain_overlap for row in rows):
                continue
            rows = [row for row in rows if not (_box_overlap(box, row["box"]) >= .28 and
                    row["box"][0] <= center[0] < row["box"][2] and row["box"][1] <= center[1] < row["box"][3])]
            rows.append({"id": next_id, "box": box, "center": center, "visibility": "foreground"})
            next_id += 1
    result = {"image_size": [width, height], "coordinate_space": "original_image_px", "origin": "image_top_left",
              "layer": state, "regions": list(overlay.regions) if overlay else [], "candidates": rows}
    if cropped_foreground:
        result["proposal_scope"] = "foreground_crop"
    if proposal_frame is not None:
        result["proposal_input_bounds"] = input_bounds
    if overlay is not None and overlay.close_button is not None:
        result["detected_close"] = overlay.close_button
        if overlay.close_style is not None:
            result["close_style"] = overlay.close_style
    if comment_input is not None:
        result["comment_input"] = comment_input
    if overlay is not None and (overlay.comment_sheet or comment_input is not None):
        result["comment_sheet"] = True
    return result


def draw_boxes(frame, regions, *, simplified=False):
    result = np.full_like(frame, 27) if simplified else frame.copy()
    if simplified:
        if regions["layer"] == "voice_recording":
            left, top, right, bottom = regions["regions"][0]
            result[top:bottom, left:right] = frame[top:bottom, left:right]
        for row in regions["candidates"]:
            left, top, right, bottom = row["box"]
            preview = result[top:bottom, left:right]
            pixels = frame[top:bottom, left:right]
            if (outline := row.get("photo_outline_px")) is not None:
                mask = np.zeros(preview.shape[:2], np.uint8)
                contour = np.rint(np.asarray(outline) - [left, top]).astype(np.int32)
                cv2.fillConvexPoly(mask, contour, 1)
                np.copyto(preview, pixels, where=mask[:, :, None] != 0)
            else:
                preview[:] = pixels
    for row in regions["candidates"]:
        left, top, right, bottom = row["box"]
        color = (90, 177, 239) if row["visibility"] == "layer_uncertain" else (94, 218, 128)
        if (outline := row.get("photo_outline_px")) is not None:
            cv2.polylines(result, [np.rint(outline).astype(np.int32)], True, color, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(result, (left, top), (right - 1, bottom - 1), color, 2)
        cv2.putText(result, str(row["id"]), (left, max(15, top - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, color, 2, cv2.LINE_AA)
    return result


def save_boxes(destination, source, frame, regions):
    destination.mkdir(parents=True, exist_ok=True)
    outputs = [destination / f"{source.stem}{suffix}" for suffix in (".candidates.json", ".simplified.png", ".boxes.jpg")]
    if any(path.resolve() == source.resolve() for path in outputs):
        raise ValueError("输出路径与原图重复")
    payload = {"source": source.name, **regions}
    outputs[0].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path, image in zip(outputs[1:], (draw_boxes(frame, regions, simplified=True), draw_boxes(frame, regions)), strict=True):
        ok, data = cv2.imencode(path.suffix, image)
        if not ok:
            raise OSError(f"无法保存标注图 {path}")
        data.tofile(path)
    return outputs


def action_payload(source, active, review, *, calibration=None):
    calibrated = calibration is not None and calibration["space"] in {"fixture_estimate_mm", "screen_mm"}
    point_key = "click_point_mm" if calibrated else "click_point"
    drags = [{"function": item["function"], "start_point": item[point_key]} for item in active
             if item.get("role") == "adjust_slider"]
    return {"source": source.name, "annotated_image": f"{source.stem}.actions.jpg",
            "coordinate_space": calibration["space"] if calibrated else "original_image_px",
            "origin": calibration["origin"] if calibrated else "image_top_left",
            **{key: [{"function": item["function"], "click_point": item[point_key]} for item in rows
                     if item.get("role") != "adjust_slider"]
               for key, rows in (("click_targets", active), ("review_targets", review))},
            **({"drag_targets": drags} if drags else {})}


def save_actions(destination, source, frame, active, review, *, calibration=None, simplified=False):
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{source.stem}.actions.json"
    payload = action_payload(source, active, review, calibration=calibration)
    preview = destination / payload["annotated_image"]
    paths = [json_path, preview]
    if simplified:
        paths.append(destination / f"{source.stem}.actions.simplified.png")
    if source.resolve() in {path.resolve() for path in paths}:
        raise ValueError("输出路径与原图重复")
    images = [draw_actions(frame, active, review)]
    if simplified:
        images.append(draw_actions(isolate_action_pixels(frame, active), active, []))
    encoded = []
    for path, image in zip(paths[1:], images, strict=True):
        ok, data = cv2.imencode(path.suffix, image)
        if not ok:
            raise OSError(f"无法保存标注图 {path}")
        encoded.append(data)
    for path, data in zip(paths[1:], encoded, strict=True):
        data.tofile(path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


@dataclass(frozen=True)
class BoxInspection:
    regions: dict
    active: list
    review: list
    labels: list
    entries: list
    timing_ms: dict
    review_examples: tuple = ()
    extraction_input: ExtractionInput | None = None


def _unseen_controls(additions, existing, width):
    reach = .034 * width
    return [item for item in additions if not any(
        other["role"] == item["role"] and
        max(abs(a - b) for a, b in zip(item["click_point"], other["click_point"], strict=True)) <= reach
        for other in existing)]


def _comment_control_candidates(frame, regions, index, working_view, visibility):
    from screen_comment_composer_icons import comment_composer_icons
    from screen_comment_row_controls import comment_row_controls, has_comment_sheet_header

    if index is None or not regions["regions"] or regions.get("floating_controls"):
        return []
    panel = regions["regions"][0]
    composer = comment_composer_icons(frame, panel, index)
    verified = regions.get("comment_sheet")
    row_actions = None
    if not verified and {item["role"] for item in composer} >= {"mention", "emoji"}:
        row_actions = comment_row_controls(frame, regions, index, confirmed_comment=True, working_view=working_view)
        hearts = [item["click_point"][1] for item in row_actions if item["role"] == "comment_like"]
        verified = len(hearts) >= 2 and max(hearts) - min(hearts) >= .08 * (panel[2] - panel[0])
    if not verified and not has_comment_sheet_header(frame, panel, index, working_view=working_view):
        return []
    if row_actions is None:
        row_actions = comment_row_controls(frame, regions, index, confirmed_comment=True, working_view=working_view)
    actions = row_actions + composer
    left, top, right, bottom = panel

    def contains(box, point):
        return box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]

    accepted = []
    for item in actions:
        native = item["box"]
        point = item["click_point"]
        box = native
        if item["role"] in {"mention", "emoji"}:
            # Retain a separate touch target measured around the confirmed glyph.
            owners = [row["box"] for row in regions["candidates"]
                      if not (row.get("disabled") or row.get("review_only"))
                      and .98 * _area(native) <= _area(row["box"]) <= 2.85 * _area(native)
                      and _box_overlap(native, row["box"]) >= .32 and contains(row["box"], point)
                      and contains(native, row["center"])
                      and not any(other is not item and contains(row["box"], other["click_point"])
                                  for other in actions)]
            if owners:
                box = max(owners, key=lambda owner: _box_overlap(owner, native))
        if (left <= box[0] < box[2] <= right and top <= box[1] < box[3] <= bottom and contains(box, point)
                and (visibility is None or visibility(box, point))):
            accepted.append({**item, "box": box})
    return accepted


def _focused_keyboard_controls(frame, regions, *, font_index=None):
    from screen_focused_search_capsule import focused_search_capsule_actions
    from screen_ime_completion import ime_completion_actions
    from screen_keyboard_footer_geometry import keyboard_footer_bounds

    @cache
    def get_keyboard():
        return keyboard_footer_bounds(frame)

    return (focused_search_capsule_actions(frame, regions, get_keyboard=get_keyboard, font_index=font_index)
            + ime_completion_actions(frame, regions, get_keyboard=get_keyboard))


def _editor_surface_actions(frame, regions):
    from screen_keyboard_footer_geometry import keyboard_footer_bounds
    from screen_message_composer import message_composer_actions
    from screen_retained_search_focus import retained_search_focus_actions
    from screen_stroke_ime import stroke_ime_actions, stroke_ime_layout

    actions = retained_search_focus_actions(frame, regions)
    stroke = stroke_ime_layout(frame)
    if stroke is not None:
        actions.extend(stroke_ime_actions(frame, regions, layout=stroke))
        actions.extend(message_composer_actions(frame, regions, stroke["keyboard"]))
    else:
        actions.extend(message_composer_actions(frame, regions, get_keyboard=lambda: keyboard_footer_bounds(frame)))
    return actions


def _keyboard_type_actions(frame, regions, font_index):
    from screen_handwriting_ime import handwriting_ime_actions
    from screen_native_search_prompt import native_search_prompt_actions
    from screen_symbol_ime import symbol_ime_actions

    return (native_search_prompt_actions(frame, regions, font_index) + symbol_ime_actions(frame, regions)
            + handwriting_ime_actions(frame, regions, font_index))


def _dialog_search_actions(frame, regions, font_index, existing):
    from screen_dialog_checkbox_rows import dialog_checkbox_actions
    from screen_dialog_search_input import dialog_search_field, dialog_search_input_actions
    from screen_dialog_search_mode import dialog_search_mode_actions

    field = dialog_search_field(frame, regions, font_index)
    if field is None:
        return []
    return (dialog_search_input_actions(frame, regions, font_index, field=field)
            + dialog_search_mode_actions(frame, regions, font_index, field=field)
            + dialog_checkbox_actions(frame, regions, font_index, field.panel, field.input_box[3], existing))


def _dark_dialog_form(frame, index):
    from screen_native_dialog_fields import dark_dialog_fields

    return dark_dialog_fields(frame, index)


def _dark_dialog_accessories(frame, regions, index, fields):
    from screen_archive_inline_selectors import archive_inline_selector_actions
    from screen_dark_dialog_footer import dark_dialog_footer_actions
    from screen_dialog_checkbox_rows import dialog_checkbox_actions
    from screen_dialog_field_accessories import dialog_field_accessory_actions

    first_input = min(item["box"][3] for item in fields.actions if item["role"] == "input")
    return (dialog_field_accessory_actions(frame, regions, index, fields=fields)
            + dark_dialog_footer_actions(frame, regions, index, fields=fields)
            + dialog_checkbox_actions(frame, regions, index, fields.panel, first_input, join_letters=True)
            + archive_inline_selector_actions(frame, regions, index, fields))


def _dark_radio_scope(frame, index):
    from screen_dark_radio_dialog import dark_radio_dialog

    return dark_radio_dialog(frame, index)


def _dark_prompt_scope(frame, index):
    from screen_dark_prompt_sheet import dark_prompt_sheet

    prompt = dark_prompt_sheet(frame, index)
    return (prompt.panel, list(prompt.actions)) if prompt is not None else None


def _native_numeric_controls(frame, regions, index):
    from screen_native_numeric_dialog import native_numeric_dialog_actions

    return native_numeric_dialog_actions(frame, regions, index)


def _timeline_sheet_controls(frame, regions, index):
    from screen_timeline_sheet import timeline_sheet_actions

    return timeline_sheet_actions(frame, regions, index)


def _covered_by_pane(box, point, pane):
    left, top, right, bottom = pane
    x, y = point
    if left <= x < right and top <= y < bottom:
        return True
    x0, y0, x1, y1 = box
    shared = max(0, min(right, x1) - max(left, x0)) * max(0, min(bottom, y1) - max(top, y0))
    return shared > .5 * _area(box)


def _duplicate_tab_captions(actions):
    """Prefer a proven tab cell over a second action on its printed caption."""
    caption_ids = set()
    tabs = {}
    for action in actions:
        if action["role"] == "switch_tab":
            name = action["function"].replace("「", "").replace("」", "").casefold()
            tabs.setdefault(name, []).append(action["box"])
    for action in actions:
        if action["role"] != "caption_control":
            continue
        name = action["function"].replace("「", "").replace("」", "").casefold()
        x, y = action["click_point"]
        a, t, b, d = action["box"]
        for left, top, right, bottom in tabs.get(name, ()):
            if not (left <= x < right and top <= y < bottom):
                continue
            shared = max(0, min(b, right) - max(a, left)) * max(0, min(d, bottom) - max(t, top))
            if shared >= .80 * _area(action["box"]):
                caption_ids.add(id(action))
                break
    return caption_ids


def inspect_boxes(frame, detector, *, font_index=None, proposal_frame=None, proposal_visibility=None,
                  regions_only=False):
    """Inspect visible controls, with an optional pass that skips the generic text sweep."""
    outcome = _inspect_boxes(frame, detector, font_index=font_index, proposal_frame=proposal_frame,
                             proposal_visibility=proposal_visibility, regions_only=regions_only)
    if font_index is None:
        return outcome
    if outcome.regions["layer"] == "page":
        media = _with_media_picker(frame, outcome, font_index, proposal_visibility)
        if media is not outcome:
            return media
        photo = _with_photo_selection(frame, outcome, font_index, proposal_visibility)
        if photo is not outcome:
            return photo
        from screen_empty_draft_field import empty_draft_field

        start = perf_counter()
        draft = None if outcome.regions.get("regions") or outcome.regions.get("floating_controls") else (
            empty_draft_field(frame, font_index))
        if draft is not None:
            outcome = _with_empty_draft_input(outcome, draft, proposal_visibility)
        elapsed = (perf_counter() - start) * 1000
        timing = {**outcome.timing_ms, "draft_input": round(elapsed, 3),
                  "total": round(outcome.timing_ms["total"] + elapsed, 3)}
        return _with_disabled_controls(frame, replace(outcome, timing_ms=timing), font_index, proposal_visibility, draft)
    if outcome.regions["layer"] != "dialog":
        return outcome
    from screen_dialog_text_footer import dialog_text_footer

    start = perf_counter()
    foreground = [item for item in outcome.regions["candidates"] if item.get("visibility") == "foreground"]
    actions, visual = dialog_text_footer(frame, outcome.regions, font_index, (*outcome.active, *outcome.review, *foreground))
    if proposal_visibility is not None:
        actions = [item for item in actions if proposal_visibility(item["box"], item["click_point"])]
        visual = [item for item in visual if proposal_visibility(item["box"], item["click_point"])]
    regions = outcome.regions
    if visual:
        candidates = regions["candidates"]
        next_id = max((item["id"] for item in candidates), default=0) + 1
        extra = [{"id": identity, "box": item["box"], "center": item["click_point"], "visibility": "foreground",
                  "review_only": True, "context_box": item["context_box"]}
                 for identity, item in enumerate(visual, next_id)]
        regions = {**regions, "candidates": candidates + extra}
    elapsed = (perf_counter() - start) * 1000
    timing = {**outcome.timing_ms, "footer": round(elapsed, 3),
              "total": round(outcome.timing_ms["total"] + elapsed, 3)}
    return replace(outcome, regions=regions, active=outcome.active + actions, entries=outcome.entries + actions,
                   timing_ms=timing)


def _with_empty_draft_input(outcome, draft, proposal_visibility):
    from screen_user_examples import _contains, _coverage

    candidates = outcome.regions["candidates"]
    ink = draft.placeholder_box
    left, top, right, bottom = ink
    body_left, body_top, body_right, body_bottom = draft.body_box
    w, h = right - left, bottom - top
    owners = [(i, row) for i, row in enumerate(candidates)
              if row.get("visibility") == "page" and not row.get("disabled") and not row.get("review_only")
              and _contains(draft.body_box, row["center"]) and _coverage(ink, row["box"]) >= .91
              and row["box"][2] - row["box"][0] <= 2.35 * w and row["box"][3] - row["box"][1] <= 3.6 * h]
    owners.sort(key=lambda pair: _area(pair[1]["box"]))
    if owners:
        owner, candidate = owners[0]
    else:
        owner = len(candidates)
        box = [max(body_left, left - round(.42 * h)), max(body_top, top - round(.39 * h)),
               min(body_right, right + round(.42 * h)), min(body_bottom, bottom + round(.39 * h))]
        candidate = {"id": max((row["id"] for row in candidates), default=0) + 1, "box": box,
                     "center": [(left + right) / 2, (top + bottom) / 2], "visibility": "page"}
    box, point = candidate["box"], candidate["center"]
    if proposal_visibility is not None and not proposal_visibility(box, point):
        return outcome
    evidence = {"control_role": "text_input", "input_state": "empty", "placeholder": draft.placeholder}
    marked = {**candidate, **evidence}
    regions = {**outcome.regions, "candidates": [*candidates[:owner], marked, *candidates[owner + 1:]]}
    action = {**evidence, "role": "text_input", "function": "输入正文", "box": box,
              "click_point": point, "preview_text": draft.placeholder}
    return replace(outcome, regions=regions, active=outcome.active + [action], entries=outcome.entries + [action])


def _with_disabled_controls(frame, outcome, font_index, proposal_visibility, empty_draft=None):
    from screen_disabled_controls import disabled_control_boxes, disabled_draft_commit
    from screen_user_examples import _contains, _coverage

    start = perf_counter()
    panes = outcome.regions.get("floating_controls", ())
    existing = outcome.regions["candidates"]
    identity = max((item["id"] for item in existing), default=0)
    disabled = []
    controls = [(box, None) for box in disabled_control_boxes(frame, outcome, font_index)]
    if (commit := disabled_draft_commit(frame, font_index, empty_draft)) is not None:
        controls = [row for row in controls if _box_overlap(commit[0], row[0]) < .65] + [commit]
    for (left, top, right, bottom), label in controls:
        box = [left, top, right, bottom]
        point = [(left + right) / 2, (top + bottom) / 2]
        if any(_box_overlap(box, pane) for pane in panes):
            continue
        if proposal_visibility is not None and not proposal_visibility(box, point):
            continue
        identity += 1
        disabled.append({"id": identity, "box": box, "center": point, "visibility": "page", "disabled": True,
                         **({"control_label": label} if label is not None else {})})
    if not disabled:
        return outcome

    def keep_candidate(item):
        return not any(_contains(control["box"], item["center"]) and (
                           _coverage(item["box"], control["box"]) >= .46
                           or (_coverage(control["box"], item["box"]) >= .84
                               and _area(item["box"]) <= 3.0 * _area(control["box"])))
                       for control in disabled)

    def keep_action(item):
        return not any(_contains(control["box"], item["click_point"]) for control in disabled)

    regions = {**outcome.regions, "candidates": [*disabled, *filter(keep_candidate, existing)]}
    elapsed = (perf_counter() - start) * 1000
    timing = {**outcome.timing_ms, "disabled_controls": round(elapsed, 3),
              "total": round(outcome.timing_ms["total"] + elapsed, 3)}
    return replace(outcome, regions=regions, active=list(filter(keep_action, outcome.active)),
                   review=list(filter(keep_action, outcome.review)), entries=list(filter(keep_action, outcome.entries)),
                   timing_ms=timing)


def _with_media_picker(frame, outcome, index, proposal_visibility):
    from screen_media_picker_surface import media_picker_layout

    start = perf_counter()
    if (layout := media_picker_layout(frame)) is None:
        return outcome
    tray, tile_boxes = layout
    from screen_dark_dialog_footer import _read_button
    from screen_media_picker_footer import media_picker_footer
    from screen_user_examples import _coverage

    controls = media_picker_footer(frame, tray)
    if not controls:
        return outcome
    search, *other = controls
    actions = [{**search, "role": "search_input", "function": "编辑媒体搜索词", "control_scope": "media_picker"}]
    visuals = []
    for item in other:
        left, top, right, bottom = item["box"]
        paper = np.median(frame[top:bottom, left:right], axis=(0, 1))
        marked = {"box": item["box"], "click_point": item["click_point"], "control_scope": "media_picker"}
        if _read_button(frame, item["caption_box"], paper, index) == ("cancel", "取消"):
            actions.append({**marked, "role": "cancel", "function": "关闭媒体选择器", "fixed_center": True})
        else:
            visuals.append(marked)

    from screen_media_composer_header import media_composer_header

    actions.extend({**item, "control_scope": "composer"} for item in media_composer_header(frame, index))
    panes = outcome.regions.get("floating_controls", ())

    def visible(item):
        box, click = item["box"], item["click_point"]
        return not any(_box_overlap(box, pane) for pane in panes) and (
            proposal_visibility is None or proposal_visibility(box, click))

    actions = list(filter(visible, actions))
    visuals = list(filter(visible, visuals))
    tiles = []
    for left, top, right, bottom in tile_boxes:
        tile = {"box": [left, top, right, bottom], "click_point": [(left + right) / 2, (top + bottom) / 2],
                "control_scope": "media_picker", "media_thumbnail": True, "fixed_center": True}
        if visible(tile):
            tiles.append(tile)
    new_controls = [*actions, *visuals, *tiles]
    if not new_controls:
        return outcome

    def other_owner(item):
        box = item["box"]
        # A separately detected magnifier belongs to the enclosing input.
        if any(_box_overlap(box, control["box"]) >= .50 or
               (control.get("role") == "search_input" and _coverage(box, control["box"]) >= .75)
               for control in (*actions, *visuals)):
            return False
        # Remove artwork and text fragments enclosed by a measured tile.
        return not any(_coverage(box, tile["box"]) >= .75 for tile in tiles)

    existing = outcome.regions["candidates"]
    next_id = max((row["id"] for row in existing), default=0) + 1
    appended = [{"id": i, "box": item["box"], "center": item["click_point"], "visibility": "page",
                 **{key: item[key] for key in ("control_scope", "fixed_center", "context_box", "review_only",
                                               "media_thumbnail") if key in item}}
                for i, item in enumerate(new_controls, next_id)]
    regions = {**outcome.regions, "candidates": [*filter(other_owner, existing), *appended]}
    elapsed = (perf_counter() - start) * 1000
    timing = {**outcome.timing_ms, "media_picker": round(elapsed, 3),
              "total": round(outcome.timing_ms["total"] + elapsed, 3)}
    return replace(outcome, regions=regions, active=[*filter(other_owner, outcome.active), *actions],
                   review=list(filter(other_owner, outcome.review)),
                   entries=[*filter(other_owner, outcome.entries), *actions], timing_ms=timing)


def _with_photo_selection(frame, outcome, font_index, proposal_visibility):
    from screen_photo_selection_grid import photo_selection_grid

    start = perf_counter()
    if (layout := photo_selection_grid(frame)) is None:
        return outcome
    from screen_photo_selection_footer import photo_selection_footer
    from screen_user_examples import _coverage

    tiles, footer = layout
    height, width = frame.shape[:2]
    panes = outcome.regions.get("floating_controls", ())

    def visible(box, point):
        return not any(_box_overlap(box, pane) for pane in panes) and (
            proposal_visibility is None or proposal_visibility(box, point))

    measured = []
    for box, marker, state in tiles:
        left, top, right, bottom = box
        point = [(left + right) / 2, (top + bottom) / 2]
        if not visible(box, point):
            continue
        tile = {"box": list(box), "center": point, "media_thumbnail": True, "fixed_center": True}
        if marker is not None and visible(marker, [(marker[0] + marker[2]) / 2, (marker[1] + marker[3]) / 2]):
            tile["selection_marker"] = {"box": list(marker), "state": state}
        measured.append(tile)
    if not measured:
        return outcome

    actions = [{**row, "control_scope": "photo_picker"} for row in photo_selection_footer(frame, footer, font_index)
               if visible(row["box"], row["click_point"])]
    existing = outcome.regions["candidates"]
    next_id = max((row["id"] for row in existing), default=0) + 1
    proposals = measured + [{"box": row["box"], "center": row["click_point"],
                              **({"fixed_center": True} if row.get("fixed_center") else {})} for row in actions]
    added = [{**row, "id": index, "visibility": "page", "control_scope": "photo_picker"}
             for index, row in enumerate(proposals, next_id)]

    def retain(row):
        box = row["box"]
        if any(_box_overlap(box, action["box"]) >= .46 or _coverage(box, action["box"]) >= .70 for action in actions):
            return False
        # A detector can outline the toolbar containing a measured button.
        if ("visibility" in row and _coverage(box, footer) >= .86
                and any(_coverage(action["box"], box) >= .88 for action in actions)):
            return False
        left, top, right, bottom = box
        if ("visibility" in row and top >= footer[3] - .018 * width
                and right - left >= .55 * width and bottom - top >= .10 * height):
            # Ignore a large blank bar below the verified footer, as in a
            # letterboxed camera view. Small proposals still get evaluated.
            dx, dy = round(.10 * (right - left)), round(.10 * (bottom - top))
            interior = frame[top + dy:bottom - dy:4, left + dx:right - dx:4]
            if interior.size and (interior.max(axis=2) <= 16).mean() >= .99:
                return False
        tiles_in_box = [tile["box"] for tile in measured if _coverage(tile["box"], box) >= .85]
        if len(tiles_in_box) >= 2 and sum(_coverage(box, tile) for tile in tiles_in_box) >= .62:
            return False
        return not any(_coverage(box, tile["box"]) >= .75 or _box_overlap(box, tile["box"]) >= .60 for tile in measured)

    regions = {**outcome.regions, "candidates": [*filter(retain, existing), *added]}
    elapsed = (perf_counter() - start) * 1000
    timing = {**outcome.timing_ms, "photo_picker": round(elapsed, 3),
              "total": round(outcome.timing_ms["total"] + elapsed, 3)}
    return replace(outcome, regions=regions, active=[*filter(retain, outcome.active), *actions],
                   review=list(filter(retain, outcome.review)),
                   entries=[*filter(retain, outcome.entries), *actions], timing_ms=timing)


def _inspect_boxes(frame, detector, *, font_index=None, proposal_frame=None, proposal_visibility=None,
                   regions_only=False):
    """Share a single local recognition path between CLI and workbench."""
    from screen_back_control import suggest_back_action
    from screen_game_marked_choices import game_marked_settings
    from screen_game_preview_settings import game_preview_settings
    from screen_icon_create import icon_create_actions
    from screen_ime_sheet import ime_search_key_actions
    from screen_keyboard_actions import keyboard_search_actions
    from screen_landscape_app_bar import landscape_app_bar_actions
    from screen_landscape_camera_toolbar import landscape_camera_toolbar_actions
    from screen_landscape_header import landscape_header_actions
    from screen_landscape_hub_header import landscape_hub_header_actions
    from screen_landscape_media_bar import landscape_media_actions
    from screen_landscape_media_navigation import landscape_media_navigation_actions
    from screen_landscape_pip import landscape_pip_actions
    from screen_landscape_split_bar import landscape_split_bar_actions
    from screen_landscape_tutorial import landscape_tutorial_actions
    from screen_native_drawer_outline import native_drawer_outline_actions
    from screen_portrait_drawer import portrait_drawer_actions
    from screen_product_rows import product_card_actions
    from screen_vertical_product_tiles import vertical_tile_card_actions
    from screen_sheet_search import sheet_search_actions
    from screen_toolbar_mail_actions import toolbar_mail_actions
    from screen_toolbar_settings import toolbar_settings_actions
    from screen_video_controls import video_control_actions

    start = perf_counter()
    layer = "tutorial"
    scoped = landscape_tutorial_actions(frame, None, font_index)
    if scoped is None:
        layer = "side_drawer"
        scoped = portrait_drawer_actions(frame, font_index)
        if scoped is None:
            scoped = native_drawer_outline_actions(frame, font_index)
    working_view = _working_view(frame)
    overlay = detect_blocking_overlay(frame, working_view=working_view, camera=proposal_frame is not None,
                                      label_index=font_index)
    if scoped is None and font_index is not None:
        height, width = frame.shape[:2]
        page_body = (overlay is not None and overlay.kind == "bottom_sheet" and overlay.close_button is None
                     and 0 <= overlay.panel[1] <= .16 * height and overlay.panel[0] <= .025 * width
                     and overlay.panel[2] >= .975 * width and overlay.panel[3] >= .985 * height)
        if overlay is None or page_body:
            from screen_empty_comment_footer import empty_comment_footer_actions

            footer = empty_comment_footer_actions(frame, {"layer": "page"}, font_index)
            if {"back", "comment_input"} <= {item["role"] for item in footer}:
                scoped = (0, 0, width, height), footer
                layer = "page"
                overlay = None
    if scoped is None and overlay is None and font_index is not None:
        from screen_reply_draft_page import reply_draft_page

        if (scoped := reply_draft_page(frame, overlay, font_index)) is not None:
            layer = "page"
    if scoped is None and font_index is not None and (overlay is None or overlay.kind == "bottom_sheet"):
        from screen_permission_prompt import permission_prompt

        if (prompt := permission_prompt(frame, font_index)) is not None:
            if overlay is None or _box_overlap(overlay.panel, prompt.panel) >= .74:
                layer = "layer_uncertain" if prompt.review_only else "bottom_sheet"
                scoped = prompt.panel, prompt.controls
                overlay = None
    if scoped is None:
        scoped = game_preview_settings(frame, overlay, font_index) or game_marked_settings(frame, overlay, font_index)
        if scoped is not None:
            layer = "page"
            overlay = None
    if scoped is None and font_index is not None and frame.shape[1] >= 1.30 * frame.shape[0]:
        from screen_colored_settings_panel import colored_settings_panel
        from screen_grouped_range_grid import grouped_range_grid
        from screen_wide_guide_tabs import wide_guide_tabs

        if (scoped := grouped_range_grid(frame, overlay, font_index)) is not None:
            layer = "page"
        elif (scoped := colored_settings_panel(frame, overlay, font_index)) is not None:
            layer = "dialog"
            overlay = None
        elif overlay is None or overlay.kind == "dialog":
            if (guide := wide_guide_tabs(frame, font_index)) is not None:
                guide_overlay, guide_actions = guide
                if overlay is None or _box_overlap(overlay.panel, guide_overlay.panel) >= .70:
                    scoped = guide_overlay.panel, guide_actions
                    layer = "dialog"
                    overlay = None
    if scoped is None and font_index is not None:
        from screen_equalizer_bands import equalizer_band_actions
        from screen_media_information_dialog import media_information_dialog

        if not regions_only:
            scoped = equalizer_band_actions(frame, overlay, font_index)
        scoped = scoped or media_information_dialog(frame, overlay, font_index)
        if scoped is not None:
            layer = "dialog"
            overlay = None
    if scoped is None and font_index is not None and (overlay is None or overlay.kind == "dialog"):
        from screen_grouped_option_dialog import grouped_option_dialog

        if (options := grouped_option_dialog(frame, {"layer": overlay.kind if overlay else "page"}, font_index)) is not None:
            if overlay is None or _box_overlap(overlay.panel, options[0]) >= .75:
                scoped = options
                layer = "dialog"
                overlay = None
    if scoped is None and (overlay is None or (overlay.kind == "dialog" and overlay.close_button is None)):
        from screen_live_producer_actions import live_producer_surface

        producer = live_producer_surface(frame, font_index)
        if producer and (overlay is None or producer.owns_widget_layer(overlay.panel)):
            height, width = frame.shape[:2]
            layer = "page"
            overlay = None
            scoped = (0, 0, width, height), list(producer.actions)
    if overlay is None and scoped is None and font_index is not None:
        from screen_dark_barrage_input import dark_barrage_composer

        if (scoped := dark_barrage_composer(frame, font_index)) is not None:
            layer = "composer"
    photo_home = False
    home_grid = None
    dock = []
    if overlay is None and scoped is None:
        from screen_home_dock_actions import home_dock_actions

        dock = home_dock_actions(frame, {"layer": "page"})
        if dock:
            from screen_dialog_close import home_light_dialog
            from screen_home_actions import inspect_home_apps

            overlay = home_light_dialog(frame)
            if overlay is None:
                apps, home_grid = inspect_home_apps(frame, None, font_index)
                if apps:
                    height, width = frame.shape[:2]
                    layer = "page"
                    scoped = (0, 0, width, height), apps + dock
                    photo_home = proposal_frame is not None
    dark_form, radio_form = None, None
    if overlay is None and scoped is None and font_index is not None:
        dark_form = _dark_dialog_form(frame, font_index)
        if dark_form is not None:
            overlay = BlockingOverlay("dialog", dark_form.panel)
        elif (radio_form := _dark_radio_scope(frame, font_index)) is not None:
            overlay = BlockingOverlay("dialog", radio_form.panel)
        elif (scoped := _dark_prompt_scope(frame, font_index)) is not None:
            layer = "bottom_sheet"
    search_results = []
    if scoped is None and overlay is None and font_index is not None:
        from screen_search_result_rows import search_result_row_actions

        search_results = search_result_row_actions(frame, overlay, font_index)
        if (sum(item["role"] == "open_result" for item in search_results) >= 2
                and {"back", "switch_tab"} <= {item["role"] for item in search_results}):
            height, width = frame.shape[:2]
            layer = "page"
            scoped = (0, 0, width, height), search_results
    if overlay is not None:
        scoped = None
        if overlay.kind == "dialog" and overlay.close_button is None:
            from screen_native_notification_choices import native_notification_choices

            if (choices := native_notification_choices(frame, overlay)) is not None:
                layer = "dialog"
                scoped = choices.panel, list(choices.actions)
        if scoped is None and overlay.kind == "dialog" and overlay.close_button is None and font_index is not None:
            from screen_compact_profile_form import complete_compact_profile

            if (entries := complete_compact_profile(frame, overlay.panel, font_index)) is not None:
                layer = "dialog"
                scoped = overlay.panel, entries
            elif entries := _native_numeric_controls(frame, {"layer": "dialog", "regions": overlay.regions}, font_index):
                layer = "dialog"
                scoped = overlay.panel, entries
    amount_controls = []
    if overlay is None and scoped is None and font_index is not None:
        from screen_amount_entry_keypad import amount_entry_keypad_actions

        amount_controls = amount_entry_keypad_actions(frame, {"layer": "page"}, font_index)
        if {"switch_tab", "close"} <= {item["role"] for item in amount_controls}:
            height, width = frame.shape[:2]
            layer = "page"
            scoped = (0, 0, width, height), amount_controls
    if scoped is not None:
        panel, entries = scoped
        height, width = frame.shape[:2]
        regions = {"layer": layer, "image_size": [width, height], "coordinate_space": "original_image_px",
                   "origin": "image_top_left", "regions": [] if layer == "page" else [list(panel)],
                   "candidates": [{"id": i, "box": item["box"], "center": item["click_point"],
                                   "visibility": "page" if layer == "page" else "foreground",
                                   **({"review_only": True} if item.get("review_only") else {})}
                                  for i, item in enumerate(entries, 1)]}
        if photo_home:
            from screen_home_actions import refine_home_app_box

            regions["proposal_input_bounds"] = (0, 0, width, height)
            rows = regions["candidates"]
            for proposal in detector.detect(proposal_frame, pixel_budget=384):
                box = list(_box(proposal, width, height))
                left, top, right, bottom = box
                if right <= left or bottom <= top:
                    continue
                center = [(left + right) / 2, (top + bottom) / 2]
                if center[1] <= .06 * height:
                    continue
                if any(row["box"][0] <= center[0] < row["box"][2] and row["box"][1] <= center[1] < row["box"][3]
                       or _box_overlap(box, row["box"]) >= .40 for row in rows):
                    continue
                if proposal_visibility is not None and proposal_visibility(box, center):
                    face = refine_home_app_box(proposal_frame, home_grid, box)
                    if face is not None:
                        x0, y0, x1, y1 = face
                        face_center = [(x0 + x1) / 2, (y0 + y1) / 2]
                        if proposal_visibility(face, face_center):
                            overlap = max((_box_overlap(face, row["box"]) for row in rows), default=0.)
                            if overlap >= .70:
                                continue
                            if overlap < .40:
                                box, center = face, face_center
                rows.append({"id": len(rows) + 1, "box": box, "center": center,
                             "score": proposal["score"], "visibility": "page"})
        elapsed = round((perf_counter() - start) * 1000, 3)
        if layer == "layer_uncertain":
            return BoxInspection(regions, [], entries, [], [],
                                 {"base": elapsed, "labels": 0., "entries": 0., "total": elapsed})
        return BoxInspection(regions, entries, [], [], entries,
                             {"base": elapsed, "labels": 0., "entries": 0., "total": elapsed})
    if overlay is None:
        from screen_front_surface import detect_painted_front_surface

        overlay = detect_painted_front_surface(frame, working_view=working_view)
    if overlay is None and proposal_frame is not None:
        from screen_front_surface import detect_photo_sheet_with_partial_top

        overlay = detect_photo_sheet_with_partial_top(frame, working_view=working_view)
    if overlay is None and font_index is not None:
        from screen_dark_dialog_dismissal import dark_dialog_literal_dismissal

        literal = dark_dialog_literal_dismissal(frame, font_index, working_view=working_view)
        if literal is not None:
            overlay = replace(literal, close_style="literal")
    if (proposal_frame is not None and overlay is not None and overlay.kind == "dialog"
            and overlay.close_button is not None and overlay.attached_close is None and overlay.close_style is None):
        from screen_foreground import _centered_cross
        from screen_pixel_modal import verified_dialog_face
        from screen_rim_attached_dismiss import rim_attached_dismiss

        left, top, right, bottom = overlay.close_button
        # Recheck detached ornaments at the rectified resolution before retaining a separate clickable region.
        if (not _centered_cross(frame[top:bottom, left:right]) and verified_dialog_face(frame, overlay.panel) is not None
                and (close := rim_attached_dismiss(frame, overlay.panel)) is not None
                and (proposal_visibility is None or proposal_visibility(close["box"], close["click_point"]))):
            overlay = replace(overlay, close_button=None)
    regions = extract_boxes(frame, detector, label_index=font_index, _overlay=overlay,
                            _crop_dialog=dark_form is not None or radio_form is not None,
                            _crop_comment=regions_only,
                            _view=working_view, proposal_frame=proposal_frame)
    raised_sheet = None
    if font_index is not None and regions["layer"] == "page":
        from screen_scoped_action_rows import verified_raised_sheet

        raised_sheet = verified_raised_sheet(frame, regions, font_index)
    if raised_sheet is not None:
        regions, sheet_actions = raised_sheet
        overlay = BlockingOverlay("bottom_sheet", tuple(regions["regions"][0]))
    glyphs = CandidateGlyphs(frame)
    active, review = suggest_box_actions(frame, regions, glyphs=glyphs)
    if raised_sheet is not None:
        width = frame.shape[1]
        active = sheet_actions + _unseen_controls(active, sheet_actions, width)
        review = _unseen_controls(review, active, width)
    detected = perf_counter()
    if regions["layer"] in {"wide_panel", "ime_sheet", "voice_recording", "voice_stop_panel"}:
        entries = []
        if font_index is not None and regions["layer"] == "wide_panel":
            from screen_equipment_plan_control import equipment_plan_actions
            from screen_wide_panel_actions import wide_panel_actions
            from screen_wide_panel_checks import wide_panel_check_actions
            from screen_wide_panel_equipment import wide_panel_equipment_actions

            for recognize in (wide_panel_actions, wide_panel_check_actions, wide_panel_equipment_actions,
                              equipment_plan_actions):
                entries.extend(recognize(frame, regions, font_index))
            regions["candidates"].extend({"id": i, "box": item["box"], "center": item["click_point"],
                                          "visibility": "foreground"} for i, item in enumerate(entries, 2))
        elif font_index is not None and regions["layer"] == "ime_sheet":
            entries = ime_search_key_actions(frame, regions, font_index)
        finished = perf_counter()
        timings = {"base": round((detected - start) * 1000, 3), "labels": 0.,
                   "entries": round((finished - detected) * 1000, 3), "total": round((finished - start) * 1000, 3)}
        return BoxInspection(regions, active + entries, review, [], entries, timings)
    if regions["layer"] in {"page", "layer_uncertain"} and frame.shape[1] >= 1.42 * frame.shape[0]:
        from screen_game_filters import game_filter_actions
        from screen_game_section_panel import game_section_panel_actions
        from screen_landscape_choice_dialog import landscape_choice_dialog_actions
        from screen_landscape_side_panel import landscape_side_panel_actions

        visual_candidates = []
        layer = "choice_dialog"
        scoped = landscape_choice_dialog_actions(frame, regions, font_index)
        if scoped is None:
            layer = "side_settings"
            scoped = landscape_side_panel_actions(frame, regions, font_index)
        if scoped is None and font_index is not None and regions["layer"] == "page":
            layer = "filter_panel"
            filtered = game_filter_actions(frame, regions, font_index, include_visual_candidates=True,
                                           camera=proposal_frame is not None)
            if filtered is not None:
                panel, entries, visual_candidates = filtered
                if proposal_visibility is not None:
                    visual_candidates = [item for item in visual_candidates
                                         if proposal_visibility(item["box"], item["click_point"])]
                scoped = (panel, entries)
        if scoped is None:
            layer = "wide_panel"
            scoped = game_section_panel_actions(frame, regions, font_index)
        if scoped is not None:
            panel, entries = scoped
            if layer == "side_settings" and font_index is not None:
                from screen_landscape_panel_switches import landscape_panel_switch_actions

                entries.extend(landscape_panel_switch_actions(frame, panel, font_index))
            regions.update(layer=layer, regions=[list(panel)], candidates=[
                {"id": i, "box": item["box"], "center": item["click_point"], "visibility": "foreground",
                 **{key: item[key] for key in ("fixed_center", "review_only", "context_box") if key in item}}
                for i, item in enumerate([*entries, *visual_candidates], 1)])
            finished = perf_counter()
            timings = {"base": round((detected - start) * 1000, 3), "labels": 0.,
                       "entries": round((finished - detected) * 1000, 3), "total": round((finished - start) * 1000, 3)}
            return BoxInspection(regions, entries, [], [], entries, timings)
    if regions["layer"] in {"page", "dialog", "layer_uncertain"} and frame.shape[0] >= 1.31 * frame.shape[1]:
        from screen_modal_footer_choices import modal_footer_choices
        from screen_pixel_modal import pixel_dialog_controls
        from screen_stacked_modal_cards import stacked_modal_card_actions

        scoped = stacked_modal_card_actions(frame, regions, font_index)
        if scoped is None:
            scoped = modal_footer_choices(frame, regions, font_index)
            if scoped is not None:
                panel, valid = scoped
                dismiss = [item for item in active if item["role"] == "close"
                           and panel[0] <= item["click_point"][0] < panel[2] and panel[1] <= item["click_point"][1] < panel[3]]
                valid.extend(dismiss or pixel_dialog_controls(frame, {"layer": "dialog", "regions": [panel]}, None, valid))
        if scoped is not None:
            panel, entries = scoped
            regions.update(layer="dialog", regions=[list(panel)], candidates=[
                {"id": i, "box": item["box"], "center": item["click_point"], "visibility": "foreground"}
                for i, item in enumerate(entries, 1)])
            finished = perf_counter()
            timings = {"base": round((detected - start) * 1000, 3), "labels": 0.,
                       "entries": round((finished - detected) * 1000, 3), "total": round((finished - start) * 1000, 3)}
            return BoxInspection(regions, entries, [], [], entries, timings)
    labels = []
    if font_index is not None:
        from screen_label_ownership import supplement_command_cells

        regions = supplement_command_cells(frame, regions)
        if not regions_only:
            from screen_short_labels import resolve_short_labels

            active, additions, labels, _ = resolve_short_labels(frame, regions, font_index, active)
            active.extend(additions)
    captions = perf_counter()
    floating = landscape_pip_actions(frame, regions)
    if floating is not None:
        pane, _ = floating
        regions["floating_controls"] = [list(pane)]
    entries = keyboard_search_actions(frame, regions) + product_card_actions(frame, regions, font_index=font_index)
    for item in vertical_tile_card_actions(frame, regions):
        left, top, right, bottom = item["box"]
        if not any(prior["role"] == "open_product" and left <= prior["click_point"][0] < right
                   and top <= prior["click_point"][1] < bottom for prior in active + entries):
            entries.append(item)
    entries.extend(sheet_search_actions(frame, regions))
    if not any(item["role"] == "back" for item in active):
        if (back := suggest_back_action(frame, regions, working_view=working_view)) is not None:
            entries.append(back)
    width = frame.shape[1]
    for recognize in (landscape_header_actions, landscape_media_actions, landscape_app_bar_actions,
                      landscape_split_bar_actions, landscape_camera_toolbar_actions):
        entries.extend(_unseen_controls(recognize(frame, regions), active + entries, width))
    entries.extend(_unseen_controls(landscape_media_navigation_actions(frame, regions, font_index, active + entries),
                                    active + entries, width))
    if overlay is None:
        entries.extend(_unseen_controls(landscape_hub_header_actions(frame, regions, font_index), active + entries, width))
    rail, rail_review = video_control_actions(frame, regions, glyphs=glyphs)
    entries.extend(_unseen_controls(rail, active + entries, width))
    review.extend(_unseen_controls(rail_review, active + review, width))
    if regions["layer"] == "page":
        from screen_owned_corner_close import owned_corner_close_actions
        from screen_transient_close import transient_close_action

        entries.extend(_unseen_controls(owned_corner_close_actions(frame, regions), active + entries, width))
        if (dismissal := transient_close_action(frame, regions)) is not None:
            entries.extend(_unseen_controls([dismissal], active + entries, width))
    from screen_player_comment_actions import player_comment_actions

    entries.extend(_unseen_controls(player_comment_actions(frame, regions), active + entries, width))
    native_tabs = []
    if font_index is not None:
        from screen_commerce_actions import commerce_button_actions
        from screen_compact_post_footer import compact_post_comments
        from screen_counted_post_actions import counted_post_actions
        from screen_drawer_invites import drawer_invitation_actions
        from screen_game_collection_cards import game_collection_card_actions
        from screen_game_friend_messages import game_friend_message_actions
        from screen_game_merchandise import game_merchandise_actions
        from screen_game_offer_cards import game_offer_card_actions
        from screen_game_primary import game_primary_actions
        from screen_game_shop_tabs import aligned_catalog_tab_actions, shop_tab_actions
        from screen_grouped_shop_navigation import grouped_shop_navigation_actions
        from screen_horizontal_post_actions import horizontal_post_actions
        from screen_inline_post_comments import inline_post_comments
        from screen_label_ownership import toolbar_search_actions
        from screen_landscape_diamond_actions import landscape_side_actions
        from screen_landscape_edit_actions import landscape_edit_actions
        from screen_landscape_footer_tabs import landscape_footer_tab_actions
        from screen_landscape_icon_tabs import landscape_icon_tab_actions
        from screen_landscape_library_toolbar import landscape_library_toolbar_actions
        from screen_landscape_mode_rail import landscape_mode_rail_actions
        from screen_landscape_radio_rows import landscape_radio_actions
        from screen_landscape_search_field import landscape_search_field_actions
        from screen_landscape_settings_list import landscape_settings_list_actions
        from screen_landscape_system_settings import landscape_system_settings_actions
        from screen_mixed_footer import mixed_footer_actions
        from screen_native_tab_bar import native_tab_actions
        from screen_paired_game_cards import paired_game_card_actions
        from screen_related_footer import related_footer_comments
        from screen_repeated_counters import repeated_counter_comments
        from screen_search_owners import search_owner_actions
        from screen_sheet_actions import comment_input_actions, sheet_dismiss_actions
        from screen_single_product_card import single_product_card_actions
        from screen_social_footer import social_footer_actions
        from screen_split_search_field import split_search_field_actions
        from screen_video_cards import video_card_actions

        entries.extend(commerce_button_actions(frame, regions, font_index, active + entries))
        entries.extend(single_product_card_actions(frame, regions, active + entries))
        entries.extend(sheet_dismiss_actions(frame, regions, font_index, active + entries))
        entries.extend(comment_input_actions(frame, regions, font_index, working_view=working_view))
        search_submits = toolbar_search_actions(frame, regions, font_index, active + entries)
        entries.extend(search_submits)
        entries.extend(split_search_field_actions(frame, regions, active + entries, search_submits))
        if not any(item["role"] == "search" for item in active + entries):
            from screen_owned_search_fields import owned_search
            from screen_wide_search_fields import wide_search_fields

            entries.extend(search_owner_actions(frame, regions, font_index)
                           or wide_search_fields(frame, regions, font_index)
                           or owned_search(frame, regions, font_index))
        invitations = drawer_invitation_actions(frame, regions, font_index)
        entries.extend(invitations)
        entries.extend(_unseen_controls(game_friend_message_actions(frame, regions, font_index, invitations=invitations),
                                        active + entries, width))
        entries.extend(game_primary_actions(frame, regions, font_index, active + entries))
        if overlay is None:
            entries.extend(_unseen_controls(landscape_mode_rail_actions(frame, regions, font_index),
                                            active + entries, width))
            entries.extend(_unseen_controls(landscape_side_actions(frame, regions, font_index),
                                            active + entries, width))
        entries.extend(_unseen_controls(landscape_edit_actions(frame, regions, font_index), active + entries, width))
        for recognize in (landscape_library_toolbar_actions, landscape_radio_actions, landscape_settings_list_actions,
                          landscape_system_settings_actions, landscape_search_field_actions):
            entries.extend(_unseen_controls(recognize(frame, regions, font_index), active + entries, width))
        if overlay is None and width >= 1.42 * frame.shape[0] and regions["layer"] == "page":
            from screen_landscape_proposal_recovery import recovered_outlined, recovered_tabs
            from screen_landscape_stacked_tabs import outlined_control_actions, stacked_setting_tabs

            for recognize, recover in ((stacked_setting_tabs, recovered_tabs),
                                       (outlined_control_actions, recovered_outlined)):
                controls = recognize(frame, regions, font_index) or recover(frame, regions, font_index)
                entries.extend(_unseen_controls(controls, active + entries, width))
        native_tabs = _unseen_controls(native_tab_actions(frame, regions, font_index), active + entries, width)
        entries.extend(native_tabs)
        for recognize_tab in (mixed_footer_actions, partial(social_footer_actions, glyphs=glyphs),
                              landscape_footer_tab_actions, landscape_icon_tab_actions):
            entries.extend(_unseen_controls(recognize_tab(frame, regions, font_index), active + entries, width))
        for recognize_post in (partial(horizontal_post_actions, glyphs=glyphs), counted_post_actions):
            entries.extend(_unseen_controls(recognize_post(frame, regions, font_index), active + entries, width))
        entries.extend(_unseen_controls(inline_post_comments(frame, regions), active + entries, width))
        entries.extend(_unseen_controls(compact_post_comments(frame, regions, font_index), active + entries, width))
        entries.extend(_unseen_controls(related_footer_comments(frame, regions, font_index), active + entries, width))
        entries.extend(_unseen_controls(repeated_counter_comments(frame, regions, font_index), active + entries, width))
        products = [item["box"] for item in entries if item["role"] == "open_product"]
        entries.extend(item for item in video_card_actions(frame, regions, font_index)
                       if not any(_box_overlap(item["box"], product) >= .57 for product in products))
        entries.extend(_unseen_controls(game_offer_card_actions(frame, regions, font_index), active + entries, width))
        for recognize_cards in (game_collection_card_actions, paired_game_card_actions, game_merchandise_actions,
                                shop_tab_actions, grouped_shop_navigation_actions):
            entries.extend(_unseen_controls(recognize_cards(frame, regions, font_index), active + entries, width))
        if overlay is None and width >= 1.45 * frame.shape[0] and regions["layer"] == "page":
            from screen_carousel_shelf_cards import shelf_card_actions
            from screen_priced_catalog_cards import priced_catalog_card_actions

            entries.extend(_unseen_controls(aligned_catalog_tab_actions(frame, regions, font_index),
                                            active + entries, width))
            product_roles = {"open_product", "game_offer_card", "game_equipment_item"}
            owned_points = [row["click_point"] for row in active + entries if row["role"] in product_roles]
            for recognize_catalog in (priced_catalog_card_actions, shelf_card_actions):
                for item in recognize_catalog(frame, regions, font_index):
                    left, top, right, bottom = item["box"]
                    if not any(left <= x < right and top <= y < bottom for x, y in owned_points):
                        entries.append(item)
                        owned_points.append(item["click_point"])
    entries.extend(_unseen_controls(icon_create_actions(frame, regions), active + entries, width))
    entries.extend(toolbar_settings_actions(frame, regions, active + entries))
    if overlay is None:
        entries.extend(toolbar_mail_actions(frame, regions, active + entries))
    if regions["layer"] == "page":
        direct = _unseen_controls(_focused_keyboard_controls(frame, regions, font_index=font_index),
                                  active + entries, width)
        direct.extend(_unseen_controls(_editor_surface_actions(frame, regions), active + entries + direct, width))
        direct.extend(_unseen_controls(_keyboard_type_actions(frame, regions, font_index),
                                       active + entries + direct, width))
        direct.extend(_unseen_controls(amount_controls, active + entries + direct, width))
        direct.extend(_unseen_controls(dock, active + entries + direct, width))
        direct.extend(_unseen_controls(search_results, active + entries + direct, width))
        if font_index is not None:
            from screen_dark_barrage_input import dark_barrage_input_actions
            from screen_dark_reply_editor import dark_reply_editor_actions
            from screen_comment_thread import comment_thread_actions
            from screen_compact_comment_rows import compact_comment_row_actions
            from screen_game_paper_notebook import game_paper_notebook_actions
            from screen_header_search_plate import header_search_plate_actions
            from screen_landscape_footer_commands import landscape_footer_command_actions
            from screen_landscape_live_viewer import landscape_live_viewer_actions
            from screen_landscape_paired_switches import landscape_paired_switch_actions
            from screen_landscape_segmented_choices import landscape_segmented_choices
            from screen_landscape_settings_rail import landscape_settings_rail_actions
            from screen_light_filter_peers import light_filter_peer_actions
            from screen_live_chat_input import live_chat_input_actions
            from screen_live_producer_preview import live_producer_preview_actions
            from screen_live_room_header import live_room_header_actions
            from screen_mail_compose_controls import mail_compose_controls
            from screen_mail_compose_fields import mail_compose_fields, mail_compose_layout
            from screen_native_switch_list import native_switch_list_actions
            from screen_native_notification_page import notification_page_actions
            from screen_native_tab_bar import apply_header_tab_context
            from screen_outlined_editor_form import outlined_editor_actions
            from screen_portrait_peer_actions import portrait_peer_actions
            from screen_reaction_icon_column import reaction_icon_column
            from screen_section_entry_actions import section_entry_actions

            header_context = None
            for recognize in (outlined_editor_actions, dark_reply_editor_actions, header_search_plate_actions,
                              live_chat_input_actions, live_producer_preview_actions, dark_barrage_input_actions,
                              portrait_peer_actions, game_paper_notebook_actions, native_switch_list_actions,
                              comment_thread_actions, compact_comment_row_actions):
                if recognize is header_search_plate_actions:
                    recognized = recognize(frame, regions, font_index, verify_menu=not regions_only)
                    if native_tabs:
                        header_context = apply_header_tab_context(native_tabs, recognized)
                else:
                    recognized = recognize(frame, regions, font_index)
                direct.extend(_unseen_controls(recognized, active + entries + direct, width))
            filters = light_filter_peer_actions(frame, regions, font_index,
                                                below=header_context[1] if header_context else None)
            if header_context:
                noun, band_bottom = header_context
                for item in filters:
                    _, top, _, bottom = item["box"]
                    depth = bottom - top
                    label = item["preview_text"]
                    if .10 * depth <= top - band_bottom <= 1.02 * depth and label.isalpha() and not label.isascii():
                        title = label if label.endswith(noun) else label + noun
                        item.update(role="select_filter", function=f"筛选{title}",
                                    setting_label=label, setting_display_label=title)
            covered = [item["click_point"] for item in active + entries + direct]
            for item in filters:
                left, top, right, bottom = item["box"]
                if not any(left <= x < right and top <= y < bottom for x, y in covered):
                    direct.append(item)
            known = {item["role"] for item in active + entries + direct}
            missing = {"comments", "share"} - known
            if missing and "comment_input" not in known:
                direct.extend(item for item in reaction_icon_column(frame, regions, font_index) if item["role"] in missing)
            direct.extend(_unseen_controls(notification_page_actions(frame, overlay), active + entries + direct, width))
            for recognize in (landscape_segmented_choices, landscape_settings_rail_actions, section_entry_actions,
                              live_room_header_actions, landscape_live_viewer_actions, landscape_footer_command_actions):
                direct.extend(_unseen_controls(recognize(frame, overlay, font_index), active + entries + direct, width))
            selected_options = []
            switches = landscape_paired_switch_actions(frame, regions, font_index, selected_boxes=selected_options)
            if selected_options:
                active = [item for item in active if item["role"] != "labeled_control" or not any(
                    _covered_by_pane(item["box"], item["click_point"], box) for box in selected_options)]
            direct.extend(_unseen_controls(switches, active + entries + direct, width))
            if layout := mail_compose_layout(frame):
                composition = (mail_compose_fields(frame, regions, font_index, layout=layout)
                               + mail_compose_controls(frame, regions, layout))
                direct.extend(_unseen_controls(composition, active + entries + direct, width))
        entries.extend(direct)
        if direct:
            candidates = regions["candidates"]
            next_id = max((item["id"] for item in candidates), default=0) + 1
            candidates.extend({"id": next_id + i, "box": item["box"], "center": item["click_point"], "visibility": "page"}
                              for i, item in enumerate(direct)
                              if all(_box_overlap(item["box"], prior["box"]) < .52 for prior in candidates))
    if regions["layer"] == "dialog":
        from screen_native_property_dialog import native_property_dialog_actions
        from screen_pixel_modal import pixel_dialog_controls

        direct = pixel_dialog_controls(frame, regions, font_index, active + entries)
        property_actions = native_property_dialog_actions(frame, regions, font_index)
        direct.extend(_unseen_controls(property_actions, active + entries + direct, width))
        owned = active + entries + direct
        direct.extend(_unseen_controls(_dialog_search_actions(frame, regions, font_index, owned), owned, width))
        if dark_form is not None:
            from screen_native_dialog_selectors import native_dialog_selector_actions

            accessories = _dark_dialog_accessories(frame, regions, font_index, dark_form)
            labeled_checks = {tuple(item["click_point"]) for item in accessories if item["role"] == "checkbox_setting"}
            if labeled_checks:
                active = [item for item in active if item["role"] != "caption_control"
                          or tuple(item["click_point"]) not in labeled_checks]
                entries = [item for item in entries if item["role"] != "caption_control"
                           or tuple(item["click_point"]) not in labeled_checks]
            fields = [*dark_form.actions, *native_dialog_selector_actions(frame, regions, font_index, dark_form.panel),
                      *accessories]
            direct.extend(_unseen_controls(fields, active + entries + direct, width))
        if radio_form is not None:
            def outside_radio_list(item):
                return item["role"] not in {"radio_setting", "caption_control"} or not _covered_by_pane(
                    item["box"], item["click_point"], radio_form.options_box)

            active = [item for item in active if outside_radio_list(item)]
            review = [item for item in review if outside_radio_list(item)]
            entries = [item for item in entries if outside_radio_list(item)]
            direct = [item for item in direct if outside_radio_list(item)]
            direct.extend(_unseen_controls(radio_form.actions, active + entries + direct, width))
        next_id = max((item["id"] for item in regions["candidates"]), default=0) + 1
        regions["candidates"].extend({"id": next_id + i, "box": item["box"], "center": item["click_point"],
                                      "visibility": "foreground"} for i, item in enumerate(direct))
        entries.extend(direct)
    if regions["layer"] == "bottom_sheet":
        from screen_attachment_comment_send import attachment_comment_send_actions
        from screen_sheet_paired_tabs import sheet_paired_tab_actions

        direct = _unseen_controls(_timeline_sheet_controls(frame, regions, font_index), active + entries, width)
        direct.extend(_unseen_controls(sheet_paired_tab_actions(frame, regions, font_index,
                                                               search_actions=active + entries),
                                        active + entries + direct, width))
        direct.extend(_unseen_controls(attachment_comment_send_actions(frame, regions, font_index),
                                        active + entries + direct, width))
        comments = _unseen_controls(_comment_control_candidates(frame, regions, font_index, working_view,
                                                                proposal_visibility), active + entries + direct, width)
        candidates = regions["candidates"]
        next_id = max((item["id"] for item in candidates), default=0) + 1
        if comments:
            candidates[:] = [row for row in candidates if row.get("disabled") or row.get("review_only")
                             or not any(_box_overlap(row["box"], item["box"]) >= .36 for item in comments)]
        candidates.extend({"id": next_id + i, "box": item["box"], "center": item["click_point"],
                           "visibility": "foreground", **({"fixed_center": True} if i >= len(direct) else {})}
                          for i, item in enumerate(direct + comments))
        entries.extend(direct + comments)
    if font_index is not None and floating is None and regions["layer"] in {"page", "dialog"}:
        from screen_setting_ranges import setting_range_actions

        panels = [(0, 0, width, frame.shape[0])] if regions["layer"] == "page" else regions["regions"]
        ranged = [item for panel in panels for item in setting_range_actions(frame, panel, font_index)]
        if regions["layer"] == "page":
            from screen_equalizer_curve import equalizer_curve_actions
            from screen_range_card import range_card_actions

            ranged.extend(equalizer_curve_actions(frame, regions, font_index))
            ranged.extend(_unseen_controls(range_card_actions(frame, overlay, font_index), ranged, width))
        if ranged:
            covered = tuple(item["box"] for item in ranged)

            def outside_range(item):
                x, y = item["click_point"]
                return not any(left <= x < right and top <= y < bottom for left, top, right, bottom in covered)

            active = [item for item in active if outside_range(item)]
            entries = [item for item in entries if outside_range(item)] + ranged
            review = [item for item in review if outside_range(item)]
            candidates = regions["candidates"]
            next_id = max((item["id"] for item in candidates), default=0) + 1
            candidates.extend({"id": next_id + i, "box": item["box"], "center": item["click_point"],
                               "visibility": "page" if regions["layer"] == "page" else "foreground"}
                              for i, item in enumerate(ranged))
    if overlay is None:
        from screen_landscape_heading_help import landscape_heading_help_actions

        entries.extend(landscape_heading_help_actions(frame, regions, active + entries))
    if font_index is not None:
        from screen_painted_commands import painted_command_actions
        from screen_scoped_action_rows import scoped_row_actions

        entries.extend(painted_command_actions(frame, regions, font_index, active + entries))
        if labels:
            from screen_faint_commands import faint_caption_actions

            entries.extend(faint_caption_actions(frame, regions, labels, active + entries))
        if raised_sheet is None:
            entries.extend(scoped_row_actions(frame, regions, font_index, active + entries))
    if floating is not None:
        pane, controls = floating
        active = [item for item in active if not _covered_by_pane(item["box"], item["click_point"], pane)]
        review = [item for item in review if not _covered_by_pane(item["box"], item["click_point"], pane)]
        entries = [item for item in entries if not _covered_by_pane(item["box"], item["click_point"], pane)]
        entries.extend(controls)
        candidates = regions["candidates"]
        next_id = max((item["id"] for item in candidates), default=0) + 1
        regions["candidates"] = [item for item in candidates
                                 if not _covered_by_pane(item["box"], item["center"], pane)]
        regions["candidates"].extend({"id": next_id + i, "box": item["box"], "center": item["click_point"],
                                      "visibility": "foreground", "floating_control": item["role"]}
                                     for i, item in enumerate(controls))
    if duplicate_captions := _duplicate_tab_captions(active + entries):
        active = [item for item in active if id(item) not in duplicate_captions]
        entries = [item for item in entries if id(item) not in duplicate_captions]
    finished = perf_counter()
    times = {name: round((last - first) * 1000, 3) for name, first, last in (
        ("base", start, detected), ("labels", detected, captions), ("entries", captions, finished),
        ("total", start, finished))}
    return BoxInspection(regions, active + entries, review, labels, entries, times)
