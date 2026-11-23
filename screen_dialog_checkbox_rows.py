"""Read owned checkboxes in the body of a verified dialog."""

from screen_caption_transcript import read_caption
from screen_choice_row_geometry import choice_row_controls
from screen_landscape_search_field import _available


def dialog_checkbox_actions(frame, regions, index, panel, after, existing=(), *, join_letters=False):
    if index is None or regions["layer"] != "dialog":
        return []
    left, _, right, bottom = panel
    cache, actions = {}, []
    for row in choice_row_controls(frame, regions):
        a, b, c, d = row.box
        if (row.kind != "checkbox" or row.disabled or row.state not in {"on", "off"}
                or not left <= a < c <= right or not after < b < d <= bottom
                or not _available(row.box, regions)):
            continue
        if any(other["role"] in {"caption_control", "checkbox_setting"}
               and a <= other["click_point"][0] < c and b <= other["click_point"][1] < d for other in existing):
            continue
        label = read_caption(frame, row.caption_area, index, cache=cache, choice=True, sentence_case=True,
                             join_letters=join_letters)
        if label is None or len(label["text"]) < 2:
            continue
        name = label["text"]
        verb = "取消勾选" if row.state == "on" else "勾选"
        actions.append({"role": "checkbox_setting", "function": f"{verb}「{name}」",
                        "setting_label": name, "setting_state": row.state, "preview_text": name,
                        "box": list(row.box), "click_point": list(row.click_point)})
    return actions
