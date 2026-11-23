"""Check whether a proposed point lies on trusted, visible screenshot geometry."""

from math import isfinite
from sys import float_info


def _number(value):
    return type(value) is int or (type(value) is float and isfinite(value))


def _point(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(_number(axis) for axis in value)


def _contains(box, point):
    left, top, right, bottom = box
    x, y = point
    return left <= x < right and top <= y < bottom


def _within(box, outer):
    return outer[0] <= box[0] < box[2] <= outer[2] and outer[1] <= box[1] < box[3] <= outer[3]


def _box(value, image):
    return (isinstance(value, (list, tuple)) and len(value) == 4 and all(_number(axis) for axis in value)
            and _within(value, image))


def _overlap(first, second):
    return (max(first[0], second[0]) < min(first[2], second[2])
            and max(first[1], second[1]) < min(first[3], second[3]))


def _cross(a, b, point):
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def _edges(polygon):
    return zip(polygon, (*polygon[1:], polygon[0]), strict=True)


def _polygon(value, outer):
    if (not isinstance(value, (list, tuple)) or len(value) < 3 or not all(_point(point) for point in value)
            or len({tuple(point) for point in value}) != len(value)
            or any(not (outer[0] <= x <= outer[2] and outer[1] <= y <= outer[3]) for x, y in value)):
        return False
    signs = set()
    for start, end in _edges(value):
        for point in value:
            turn = _cross(start, end, point)
            if turn:
                signs.add(turn > 0)
            if len(signs) > 1:
                return False
    return len(signs) == 1


def _in_polygon(polygon, point):
    turns = [_cross(start, end, point) for start, end in _edges(polygon)]
    return min(turns) >= 0 or max(turns) <= 0


def _boxes(value, image):
    if not isinstance(value, (list, tuple)) or not all(_box(box, image) for box in value):
        raise ValueError("页面范围必须是位于原图中的非空像素框。")
    return value


def on_selection_marker(element, point):
    marker = element.get("selection_marker")
    return bool(marker and _contains(marker["box"], point)
                and ("polygon" not in marker or _in_polygon(marker["polygon"], point)))


def on_disabled_control(element, point):
    if not element.get("disabled") or not _contains(element["box"], point):
        return False
    outline = element.get("polygon", element.get("photo_outline_px"))
    return outline is None or _in_polygon(outline, point)


def _geometry(page):
    if (not isinstance(page, dict) or page.get("schema") != "screen-page/1"
            or page.get("coordinate_space") != "original_image_px" or page.get("origin") != "image_top_left"):
        raise ValueError("需要使用原图左上角为原点的页面分析。")
    size = page.get("image_size")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or not all(type(axis) is int and axis > 0 for axis in size)):
        raise ValueError("页面分析的图片尺寸无效。")
    image = (0, 0, *size)
    layer = page.get("layer")
    if not isinstance(layer, dict) or not isinstance(kind := layer.get("kind"), str) or not kind:
        raise ValueError("页面分析缺少有效的前景层。")
    panels = _boxes(layer.get("boxes"), image)
    floating = _boxes(layer.get("floating_boxes", ()), image)
    corners = page.get("screen_corners")
    if "screen_corners" in page and not _polygon(corners, image):
        raise ValueError("屏幕边界不在原图中或形状无效。")
    calibration = page.get("calibration")
    if isinstance(calibration, dict) and "rectified_size_px" in calibration:
        from screen_photo_geometry import display_size

        rectified = calibration["rectified_size_px"]
        physical = calibration.get("reference_size_mm")
        if (calibration.get("space") != "screen_mm" or calibration.get("registration") != "photo_screen_corners"
                or corners is None or len(corners) != 4
                or not _point(physical) or not all(0 < axis <= float_info.max for axis in physical)
                or not _point(rectified) or not all(type(axis) is int for axis in rectified)
                or tuple(rectified) != display_size(corners, physical)):
            raise ValueError("实拍画面的校正尺寸与屏幕标定不一致。")

    items = page.get("elements")
    if not isinstance(items, (list, tuple)):
        raise ValueError("页面分析缺少组件列表。")
    elements = {}
    for item in items:
        if (not isinstance(item, dict) or not isinstance(identity := item.get("id"), str)
                or not identity or identity in elements or not _box(box := item.get("box"), image)):
            raise ValueError("页面分析中有重复组件或无效的组件框。")
        if "polygon" in item and not _polygon(item["polygon"], box):
            raise ValueError("组件的多边形边界无效。")
        if "disabled" in item and (item["disabled"] is not True or item.get("type") != "control_candidate"):
            raise ValueError("控件禁用状态无效。")
        if "fixed_center" in item and (item["fixed_center"] is not True or item.get("type") != "control_candidate"
                                       or not _point(item.get("center")) or not _contains(box, item["center"])):
            raise ValueError("固定点击点必须在对应的控件框内。")
        if "control_scope" in item and (item.get("type") != "control_candidate"
                                        or item["control_scope"] not in ("media_picker", "composer", "photo_picker")):
            raise ValueError("控件所属区域无效。")
        if "media_thumbnail" in item and (item["media_thumbnail"] is not True
                                          or item.get("control_scope") not in ("media_picker", "photo_picker")
                                          or not item.get("fixed_center")):
            raise ValueError("媒体缩略图需要有效的范围和点击点。")
        if "selection_marker" in item:
            marker = item["selection_marker"]
            if (not item.get("media_thumbnail") or item.get("control_scope") != "photo_picker"
                    or not isinstance(marker, dict) or set(marker) - {"box", "state", "polygon"}
                    or marker.get("state") not in ("selected", "unselected")
                    or not _box(marker_box := marker.get("box"), box) or tuple(marker_box) == tuple(box)
                    or ("polygon" in marker and not _polygon(marker["polygon"], marker_box))
                    or on_selection_marker(item, item["center"])):
                raise ValueError("照片的选中标记或范围无效。")
        if "component_box" in item and (item.get("type") != "control_candidate"
                                         or not _box(item["component_box"], image)
                                         or not _within(box, item["component_box"])):
            raise ValueError("组件的整体范围无效。")
        if "component_evidence" in item:
            evidence = item["component_evidence"]
            if (item.get("type") != "control_candidate" or not isinstance(evidence, dict)
                    or set(evidence) != {"artwork", "caption"}
                    or any(not _box(part, box) or any(type(axis) is not int for axis in part)
                           for part in evidence.values())):
                raise ValueError("组件的主图或标题范围无效。")
        if "context_box" in item and (item.get("type") != "control_candidate"
                                       or not _box(item["context_box"], image)
                                       or any(type(axis) is not int for axis in item["context_box"])
                                       or not _within(box, item["context_box"])):
            raise ValueError("用于辨认控件的局部范围必须包含该控件。")
        if "reviewed_sample" in item:
            sample = item["reviewed_sample"]
            if (not isinstance(sample, dict) or item.get("type") != "control_candidate"
                    or sample.get("match") not in ("exact", "near", "relocated")
                    or sample.get("origin", "manual") not in ("manual", "ai", "detector")):
                raise ValueError("组件的学习样本无效。")
            purpose = sample.get("purpose")
            fields = set(sample) - {"origin"}
            if purpose is None:
                if (fields != {"purpose", "match", "box"} or not _box(sample["box"], image)
                        or not _overlap(sample["box"], box)):
                    raise ValueError("组件的排除范围无效。")
            elif (fields != {"purpose", "match"} or not isinstance(purpose, str)
                  or not 1 <= len(purpose.strip()) <= 50 or not purpose.isprintable()):
                raise ValueError("组件的样本用途无效。")
        if "floating_owner" in item:
            owner = item["floating_owner"]
            if (item.get("type") != "control_candidate" or type(owner) is not int
                    or not 0 <= owner < len(floating) or not _overlap(box, floating[owner])):
                raise ValueError("组件没有有效的悬浮层归属。")
        elements[identity] = item
    for identity, item in elements.items():
        if "parent_id" in item:
            owner = item["parent_id"]
            if (not isinstance(owner, str) or owner == identity or (parent := elements.get(owner)) is None
                    or item.get("type") != "control_candidate" or parent.get("type") != "control_candidate"
                    or "parent_id" in parent
                    or not _within(item["box"], frame_box := parent.get("component_box", parent["box"]))
                    or tuple(item["box"]) == tuple(frame_box)):
                raise ValueError("控件没有有效的上级组件。")
    return image, kind, panels, floating, corners, elements


def validate_page_geometry(page):
    """Raise ValueError when the page bundle contains invalid geometry."""
    _geometry(page)


def visible_point(page, element, point):
    """Validate an original-pixel preview point, optionally tied to a trusted component."""
    if not _point(point):
        return False
    try:
        image, kind, panels, floating, corners, elements = _geometry(page)
    except ValueError:
        return False
    if (not _contains(image, point) or (corners is not None and not _in_polygon(corners, point))
            or (panels and not any(_contains(panel, point) for panel in panels))
            or (kind != "page" and not panels) or kind == "voice_recording"):
        return False
    if any(_contains(sample["box"], point) for item in elements.values()
           if (sample := item.get("reviewed_sample")) is not None and sample["purpose"] is None):
        return False
    if any(on_disabled_control(item, point) for item in elements.values()):
        return False

    covering = [index for index, box in enumerate(floating) if _contains(box, point)]
    if len(covering) > 1:
        return False
    if element is None:
        return (not covering and kind != "voice_stop_panel"
                and not any(on_selection_marker(item, point) for item in elements.values()))
    if (not isinstance(element, dict) or not isinstance(identity := element.get("id"), str)
            or (target := elements.get(identity)) is None or target.get("type") != "control_candidate"
            or target.get("disabled")
            or not _contains(target["box"], point)):
        return False
    if "polygon" in target and not _in_polygon(target["polygon"], point):
        return False
    if on_selection_marker(target, point):
        return False
    if any(child.get("parent_id") == identity and _contains(child["box"], point)
           and ("polygon" not in child or _in_polygon(child["polygon"], point)) for child in elements.values()):
        return False
    if (owner := target.get("floating_owner")) is not None:
        return covering == [owner]
    # A legacy page bundle omits ownership, but it only retains ordinary
    # background proposals with limited overlap with the floating pane.
    return not covering or (kind == "page" and _within(target["box"], floating[covering[0]]))
