"""Find a single named control in the inspector's confirmed action inventory."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import re
import unicodedata

from offline_target import _COMPOUND, _INTENTS, _LOCATION, _NEGATIVE, _QUOTES, _WRAPPER, _at_location


_ACTION_PREFIX = (
    r"点击屏幕上的|点击|点一下|点开|点选|打开|进入|选择|轻点|查看|切换到|"
    r"click(?:\s+on)?\b|tap\b|press\b|open\b|select\b|choose\b|view\b"
)
_OPERATOR = re.compile(r"^(?:" + _ACTION_PREFIX + r"|点)\s*")
_FUNCTION_PREFIX = re.compile(r"^(?:" + _ACTION_PREFIX + r")\s*")
_LOOKUP = re.compile(
    r"^(?:帮我找一下|找一下|找一找|定位一下|定位|查找|找到|找|在哪(?:里|儿)?(?:可以)?|"
    r"where is\b|where can i (?:click|tap|find)\b|locate\b|find\b|show me\b)\s*"
)
_LOOKUP_END = re.compile(r"\s*(?:在哪(?:里|儿)?|在哪里可以点|怎么点)$")
_MORE_NEGATIVE = re.compile(
    r"请勿|不许|(?:不|别|勿)\s*(?:点|轻点|打开|进入|选择|切换|查看|搜索|返回|关闭|分享|按|碰|帮我|去)|"
    r"\b(?:not|cannot|can't|avoid|skip|except|without)\b"
)
_MORE_COMPOUND = re.compile(
    r"(?:\band\s+(?:open|press|select|choose|switch|go|return|view|find|search)\b|然后|接下来|以及|\s[&+/]\s)"
)
_CONTROL_NOUN = re.compile(r"(?:按钮|图标|入口|选项|键|\s+(?:button|icon|entry|tab|option))$")
_SHARE_COMMANDS = {"分享", "分享菜单", "分享当前页面", "分享这件商品", "share", "share menu",
                   "share this page", "share this product"}
_VIDEO_SHARE_COMMANDS = {"分享视频", "分享这条视频", "分享当前视频", "视频分享", "share video",
                         "share the video", "share this video", "share this reel"}
_ICON_ACTION_NAMES = {
    "repost": ("转发这条视频", "repost", "repost this video"),
    "video_more": ("视频的更多操作", "more video options"),
    "video_audio": ("打开视频音乐", "open the audio page"),
    "nav_home": ("回首页", "home"),
    "nav_messages": ("私信", "direct messages", "dm"),
    "nav_following": ("关注页", "打开关注页", "following"),
    "nav_reels": ("reels", "open reels"),
    "nav_profile": ("打开我的主页", "my profile"),
    "keyboard_done": ("键盘完成键", "键盘的完成键", "keyboard done"),
    "keyboard_enter": ("键盘回车键", "键盘换行键", "换行键", "keyboard return"),
    "keyboard_hide": ("隐藏键盘", "折叠键盘", "hide keyboard"),
}
_FOLDER_NAME = re.compile(r"^(.+?)(?:文件夹|\s+folder)$")
_NUMBER = r"(?:[1-9][0-9]?|[一二三四五六七八九]?十[一二三四五六七八九]?|[一二三四五六七八九])"
_GRID_KINDS = {"照片": {"open_photo"}, "图片": {"open_photo"}, "媒体项目": {"open_media"}, "播客封面": {"open_media"},
               "相册": {"open_album"}, "书": {"open_book"}, "书籍": {"open_book"}, "漫画": {"open_book"},
               "缩略图": {"open_photo", "open_video"}, "媒体缩略图": {"open_photo", "open_video"}}
_GRID_POSITION = re.compile(rf"^第\s*({_NUMBER})\s*[排行]\s*的?\s*第\s*({_NUMBER})\s*[张个本]\s*"
                            rf"({'|'.join(_GRID_KINDS)})$")
_DATE_REQUEST = re.compile(rf"^(?:(\d{{4}})\s*年\s*)?({_NUMBER})\s*月\s*({_NUMBER})\s*[日号]\s*(?:的\s*)?(.*)$")
_CALENDAR_DAY_ROLES = {"open_day", "select_day"}
_DAY_DISPLAY = re.compile(r"^(?:查看(?:\d{4}年)?\d{1,2}月\d{1,2}日的日程|选中(?:\d{4}年)?\d{1,2}月\d{1,2}日并查看日程)"
                          r"(?:\(可见:.+\))?$")
_COLLECTION_NAMES = {"打开相册": ("相册", "album"), "打开书籍": ("书籍", "漫画", "book", "manga")}
_DIGITS = {char: value for value, char in enumerate("零一二三四五六七八九")}
_FILTER_QUERY = re.compile(r"^(?:(?:清除|移除|取消)\s*(.+?)\s*(?:筛选|过滤)|(?:clear|remove)\s+(?:the\s+)?(.+?)\s+filter)$")
_CURRENT_VALUE_DROPDOWN = re.compile(
    r"^(?:展开\s*)?当前值\s*(?:为|是)\s*(.+?)\s*的?\s*下拉(?:框|菜单)$"
    r"|^(?:the\s+)?dropdown\s+(?:currently\s+set\s+to|showing)\s+(.+)$"
)
_NAMED_DROPDOWN = re.compile(r"^(?:展开\s*)?(?:the\s+)?(.+?)(?:下拉框|下拉菜单|\s+dropdown)$")
_DROPDOWN_VERTICAL = re.compile(r"^(?:展开\s*)?(?:the\s+)?(?:(upper|lower)\s+|(上方|下方|上面|下面)的?\s*)")
_SETTING_VERBS = {
    "取消勾选": "uncheck", "uncheck": "uncheck",
    "勾选": "check", "check": "check",
    "切换到": "switch_to", "switch to": "switch_to",
    "切换": "toggle", "toggle": "toggle",
    "开启": "on", "启用": "on", "enable": "on", "turn on": "on",
    "关闭": "off", "关掉": "off", "禁用": "off", "disable": "off", "turn off": "off",
    "打开": "open", "进入": "open", "open": "open",
    "选择": "choose", "choose": "choose", "select": "choose",
    "增加": "increment", "增大": "increment", "increment": "increment", "increase": "increment",
    "减小": "decrement", "减少": "decrement", "decrement": "decrement", "decrease": "decrement",
    "拖动": "adjust", "调节": "adjust", "drag": "adjust", "adjust": "adjust",
}
_CHECKBOX = "checkbox_setting"
_RADIO = "radio_setting"
_SLIDER = "adjust_slider"
_TAB = "switch_tab"
_CLOSE_TAB = "close_tab"
_SEGMENT = "select_segment"
_PANEL_CHOICES = {_TAB, _SEGMENT}
_RANGE_INTENTS = {"increment", "decrement", "adjust"}
_NAMED_SETTINGS = {_CHECKBOX, _RADIO, _SLIDER} | _PANEL_CHOICES
_RANGE_ROLES = {_SLIDER, "increment_setting", "decrement_setting"}
_RANGE_NAMES = {
    "brightness": {"亮度"},
    "contrast": {"对比度"},
    "volume": {"音量"},
    "level (db)": {"电平"},
    "font size": {"字号", "字体大小"},
    "width": {"宽度"},
    "blur (%)": {"模糊"},
    "opacity (%)": {"不透明度"},
    "hue": {"色调"},
    "chroma": {"色度"},
    "lightness": {"明度"},
}
_SETTING_ROLES = {
    "check": {_CHECKBOX}, "uncheck": {_CHECKBOX},
    "on": {"toggle_setting", _CHECKBOX}, "off": {"toggle_setting", _CHECKBOX},
    "toggle": {"toggle_setting", _CHECKBOX} | _PANEL_CHOICES,
    "switch_to": _PANEL_CHOICES,
    "open": {"open_setting", "toggle_setting", "choose_setting", _TAB},
    "choose": {"choose_setting", _RADIO} | _PANEL_CHOICES,
    "increment": {"increment_setting"},
    "decrement": {"decrement_setting"},
    "adjust": {_SLIDER},
}
_SLIDER_SUFFIX = re.compile(r"(?:滑块|\s+slider)$")
_PANEL_SUFFIXES = {
    _TAB: re.compile(r"(?:标签页|选项卡|\s+tab)$"),
    _SEGMENT: re.compile(r"(?:分段按钮|分段|\s+segment)$"),
}
_MENU_ROLES = {"select_setting_option", "invoke_dropdown_command"}
_MENU_COMMAND_OPERATOR = re.compile(r"^(?:执行|运行|invoke\b|run\b)\s*")
_SETTING_REQUEST = re.compile(r"^(" + "|".join(
    verb + (r"\s+" if verb.isascii() else r"\s*") for verb in _SETTING_VERBS
) + r")(.+)$")
_OPTION_ASSIGNMENT = re.compile(
    r"^(?:(?:把|将)\s*)?(.+?)\s*(?:设置为|设为|改为|改成|切换到)\s*(.+)$"
    r"|^(?:set|change)\s+(?:the\s+)?(.+?)\s+to\s+(.+)$"
)
_OPTION_QUALIFIED = re.compile(r"^(?:选择|点选|点击|select\s+|choose\s+)?\s*(.+?)\s*(?:中的|下的|的)\s*(.+)$")
_ENGLISH_LOCATION_NAMES = {
    "top left": "左上角", "top right": "右上角", "bottom left": "左下角", "bottom right": "右下角",
    "left side": "左边", "right side": "右边", "left": "左边", "right": "右边",
    "top half": "上半屏", "bottom half": "下半屏",
}
_CORNER_SIDES = {"左上角": (False, False), "右上角": (True, False),
                 "左下角": (False, True), "右下角": (True, True)}
_ENGLISH_LOCATION = r"(top[ -]left|top[ -]right|bottom[ -]left|bottom[ -]right|left side|right side|top half|bottom half|left|right)"
_ENGLISH_PREFIX = re.compile(r"^(?:the\s+)?" + _ENGLISH_LOCATION + r"(?:\s+corner)?\s+")
_ENGLISH_SUFFIX = re.compile(r"\s+(?:on|at|in)\s+(?:the\s+)?" + _ENGLISH_LOCATION + r"(?:\s+corner)?$")
_CHINESE_SUFFIX = re.compile(r"\s*(?:位于|在)(左上角|右上角|左下角|右下角|左侧|左边|右侧|右边|上半屏|下半屏|底栏|底部栏)$")
_END_MARKS = "。！!？?."
_ADDITIONAL_NAMES = {
    "back": {"返回上一页", "返回上一级", "回到上一页", "previous page"},
    "comments": {"视频评论", "视频的评论", "打开视频评论", "查看视频评论", "评论气泡", "视频评论气泡"},
    "搜索": {"搜索书库", "search library"},
}
_CLOSE_NAMES = {"关闭", "关闭弹窗", "关闭对话框", "close", "close dialog", "close the dialog",
                "close popup", "close the popup", "dismiss dialog", "dismiss the dialog"}
_SEARCH_INPUT_NAMES = {"搜索框", "搜索输入框", "编辑搜索内容", "输入搜索内容", "聚焦搜索框",
                       "search field", "search input", "search box", "edit search text"}
_CLOSE_TAB_PREFIX = re.compile(r"^关闭\s*(?:标签页|选项卡)\s*(.+)$")
_CLOSE_TAB_SUFFIX = (
    re.compile(r"^关闭\s*(.+?)\s*(?:标签页|选项卡)$"),
    re.compile(r"^close\s+(?:the\s+)?(.+?)\s+tab$"),
)
_CLOSE_TAB_CHAIN = re.compile(r"(?:然后|接着|紧接着|同时|并且|或者|再|并)\s*关闭"
                              r"|\b(?:and(?:\s+then)?|then|or)\s+close\s+")
_CURRENT_TAB_NAMES = {"当前", "当前的", "活动", "活动的", "当前选中", "当前选中的", "current", "active"}
_TAB_TITLE_QUOTES = dict(zip("\"'“‘「『《", "\"'”’」』》"))


def _text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("’", "'").strip()
    return " ".join(value.strip(_END_MARKS + _QUOTES + " ").split())


def _named_keys(function: str) -> set[str]:
    full = _text(function)
    keys = {full, _text(_FUNCTION_PREFIX.sub("", full, count=1))} - {""}
    operation, _, title = full.partition(":")
    title = _text(title)
    if title and operation in _COLLECTION_NAMES:
        for kind in _COLLECTION_NAMES[operation]:
            spacing = (" ",) if kind.isascii() else ("", " ")
            keys.update(alias for space in spacing for alias in (f"{title}{space}{kind}", f"{kind}{space}{title}"))
    return keys


def _location_keys(command: str, *, normalize=_text) -> tuple[str, set[str]]:
    locations = set()
    chinese = _LOCATION.match(command.removeprefix("在"))
    if chinese:
        locations.add(chinese[1])
        command = command.removeprefix("在")[chinese.end():]
    elif (english := _ENGLISH_PREFIX.match(command)) is not None:
        locations.add(_ENGLISH_LOCATION_NAMES[english[1].replace("-", " ")])
        command = command[english.end():]
    if (chinese := _CHINESE_SUFFIX.search(command)) is not None:
        locations.add(chinese[1])
        command = command[:chinese.start()]
    elif (english := _ENGLISH_SUFFIX.search(command)) is not None:
        locations.add(_ENGLISH_LOCATION_NAMES[english[1].replace("-", " ")])
        command = command[:english.start()]
    return normalize(command), locations


def _decision(rows: list[Mapping]) -> dict:
    if len(rows) == 1:
        return {"status": "matched", "target_id": rows[0]["id"]}
    if rows:
        return {"status": "ambiguous", "choices": [row["id"] for row in rows]}
    return {"status": "not_located"}


def _screen_position(position):
    if (isinstance(position, (list, tuple)) and len(position) == 2
            and all(type(value) in (int, float) and 0 <= value <= 1 for value in position)):
        return tuple(position)
    return None


def _located_decision(rows: list[Mapping], location: str | None) -> dict:
    if location is None:
        return _decision(rows)
    if location in {"底栏", "底部栏"}:
        return _decision([row for row in rows if row.get("role") in {"dock", "dock_icon"}
                          or row.get("action_context") == "bottom_navigation"])
    located, unknown = [], []
    for row in rows:
        if _screen_position(row.get("screen_fraction")) is None:
            unknown.append(row)
        elif _at_location(row, location):
            located.append(row)
    if not located and not unknown and len(rows) == 1 and location in _CORNER_SIDES:
        x, y = rows[0]["screen_fraction"]
        right, lower = _CORNER_SIDES[location]
        side, end = (1 - x if right else x), (1 - y if lower else y)
        if side <= .16 and end <= 1 / 3:
            located = rows
    return _decision(located + unknown if located else [])


def _tab_title(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _tab_owner(row):
    box = row.get("tab_owner_box")
    if (isinstance(box, (tuple, list)) and len(box) == 4
            and all(type(value) in (int, float) for value in box)
            and box[0] < box[2] and box[1] < box[3]):
        return tuple(box)
    return None


def _current_tab_close(visible, tabs, location):
    selected = [row for row, _ in visible if row.get("role") == _TAB and row.get("setting_state") == "on"]
    decision = _located_decision(selected, location)
    if decision["status"] != "matched":
        return {"status": "not_located"}
    current = next(row for row in selected if row["id"] == decision["target_id"])
    label = current.get("setting_label")
    if not isinstance(label, str) or not label.strip():
        return {"status": "not_located"}
    title = _tab_title(label)
    closes = [row for row, caption in tabs if row.get("setting_state") == "on" and caption == title]
    owner = _tab_owner(current)
    if owner is not None:
        return _decision([row for row in closes if _tab_owner(row) == owner])
    if (current.get("tab_owner_box") is not None
            or any(row is not current and isinstance(peer := row.get("setting_label"), str)
                   and _tab_title(peer) == title for row in selected)):
        return {"status": "not_located"}
    return _decision([row for row in closes if row.get("tab_owner_box") is None])


def _close_tab_request(query, visible):
    command = _tab_title(query)
    while (wrapper := _WRAPPER.match(command)) is not None:
        command = command[wrapper.end():]
    rest, outside = _location_keys(command, normalize=_tab_title)
    readings = [(command, set()), (rest, outside)] if outside else [(command, set())]
    typed = False
    tabs = [(row, _tab_title(row["setting_label"])) for row, _ in visible
            if row.get("role") == _CLOSE_TAB and isinstance(row.get("setting_label"), str)
            and row["setting_label"].strip()]
    for stem, places in readings:
        captures = [(match[1], True) for pattern in _CLOSE_TAB_SUFFIX
                    if (match := pattern.fullmatch(stem.rstrip(_END_MARKS).rstrip())) is not None]
        if (prefix := _CLOSE_TAB_PREFIX.fullmatch(stem)) is not None:
            captures.insert(0, (prefix[1], False))
        for title, allows_current in captures:
            typed = True
            title = title.strip()
            if not title:
                continue
            quoted = len(title) > 1 and _TAB_TITLE_QUOTES.get(title[0]) == title[-1]
            if quoted:
                names = [(title[1:-1].strip(), places)]
            elif _CLOSE_TAB_CHAIN.search(title):
                continue
            else:
                names = [(title, places)]
                inner_title, inner = _location_keys(title, normalize=_tab_title)
                if inner:
                    names.append((inner_title, places | inner))
            for name, locations in names:
                if len(locations) > 1:
                    return {"status": "unsupported_command"}
                location = next(iter(locations), None)
                if allows_current and not quoted and name in _CURRENT_TAB_NAMES:
                    return _current_tab_close(visible, tabs, location)
                matches = [row for row, caption in tabs if caption == name]
                if matches:
                    return _located_decision(matches, location)
    if typed:
        if (_NEGATIVE.search(command) or _MORE_NEGATIVE.search(command) or _CLOSE_TAB_CHAIN.search(command)
                or _COMPOUND.search(command) or _MORE_COMPOUND.search(command)):
            return {"status": "unsupported_command"}
        return {"status": "not_located"}
    return None


def _ordinal(number: str) -> int:
    if number.isdecimal():
        return int(number)
    if "十" in number:
        before, after = number.split("十")
        return 10 * _DIGITS.get(before, 1) + _DIGITS.get(after, 0)
    return _DIGITS[number]


def _event_key(value):
    return "".join(_text(value).split())


def _setting_keys(row, *, include_category_suffix=False):
    title = row.get("setting_label")
    if not isinstance(title, str):
        return set()
    qualifier = row.get("setting_qualifier")
    names = {title}
    if row.get("role") in {"choose_setting", _TAB} and isinstance(observed := row.get("setting_display_label"), str):
        names.add(observed)
    if row.get("role") in _RANGE_ROLES:
        names.update(_RANGE_NAMES.get(_text(title), ()))
    if include_category_suffix:
        suffix = unicodedata.normalize("NFKC", title).partition(":")[2].strip()
        if suffix:
            names.add(suffix)
    if isinstance(qualifier, str) and qualifier:
        names.update((f"{title} {qualifier}", f"{title}（{qualifier}）"))
        if row.get("role") in _PANEL_CHOICES:
            names.update((f"{qualifier}中的{title}", f"{qualifier}的{title}"))
    return {_event_key(name) for name in names}


def _other_dropdown_positions(rows, location):
    if location in {"底栏", "底部栏"}:
        return set()
    positions = set()
    for row in rows:
        peers = row.get("setting_value_other_points", ())
        if not isinstance(peers, (list, tuple)):
            positions.add(None)
            continue
        for peer in peers:
            point = _screen_position(peer)
            if point is None or location is None or _at_location({"screen_fraction": point}, location):
                positions.add(point)
    return positions


def _vertical_dropdown_decision(matches, other_positions, location, vertical):
    if None in other_positions or len(matches) + len(other_positions) < 2:
        return {"status": "not_located"}
    ranked = [(point[1], None) for point in other_positions]
    for row in matches:
        xy = _screen_position(row.get("screen_fraction"))
        if xy is None:
            return _decision(matches) if len(matches) > 1 else {"status": "not_located"}
        if location is None or _at_location(row, location):
            ranked.append((xy[1], row))
    if len(ranked) < 2:
        return {"status": "not_located"}
    ranked.sort(key=lambda result: result[0], reverse=vertical in {"lower", "下方", "下面"})
    first_y = ranked[0][0]
    first = [row for y, row in ranked if abs(y - first_y) <= .01]
    named = [row for row in first if row is not None]
    if None in first and named:
        return {"status": "ambiguous", "choices": [row["id"] for row in named]}
    return _decision(named)


def _dropdown_request(stem, visible, location):
    relative = _DROPDOWN_VERTICAL.match(stem)
    readings = [(stem, None)]
    if relative is not None:
        readings.append((stem[relative.end():], relative[1] or relative[2]))
    selectors = [row for row, _ in visible if row.get("role") == "choose_setting"]
    typed = False
    for request, vertical in readings:
        other_positions = set()
        if (field := _CURRENT_VALUE_DROPDOWN.fullmatch(request)) is not None:
            value = _event_key(field[1] or field[2])
            matches = [row for row in selectors if isinstance(row.get("setting_value"), str)
                       and _event_key(row["setting_value"]) == value]
            other_positions = _other_dropdown_positions(matches, location)
        elif (field := _NAMED_DROPDOWN.fullmatch(request)) is not None:
            label = _event_key(field[1])
            matches = [row for row in selectors if label in _setting_keys(row)]
        else:
            continue
        typed = True
        if not matches:
            continue
        if vertical is None:
            decision = _located_decision(matches, location)
            if other_positions and decision["status"] == "matched":
                return {"status": "ambiguous", "choices": [decision["target_id"]]}
            return decision
        return _vertical_dropdown_decision(matches, other_positions, location, vertical)
    return {"status": "not_located"} if typed else None


def _action_keys(row):
    keys = _named_keys(row["function"])
    role = row.get("role")
    keys.update(_ICON_ACTION_NAMES.get(role, ()))
    if role == "nav_following":
        # The following tab and following a creator are separate operations.
        keys.discard("关注")
    if role == "like" and row.get("action_context") == "video":
        if row["function"] == "点赞":
            keys.update(("点个赞", "给这条视频点赞", "like this video"))
        elif row["function"] == "取消点赞":
            keys.update(("撤回点赞", "取消喜欢", "unlike", "unlike this video"))
    if role == "favorite" and row.get("action_context") == "video":
        keys.update(("收藏视频", "收藏这条视频", "save this video"))
    if role == "save":
        keys.add("save")
    if role == "comment_input":
        keys.update(("写评论", "添加评论", "评论框", "输入一条评论", "add a comment", "write a comment"))
    if role == "open_product":
        keys.update(("商品", "商品详情"))
        if isinstance(title := row.get("product_title"), str):
            keys.add(_text(title))
    if role == "select_filter" and isinstance(label := row.get("setting_label"), str):
        keys.update((_text(label), f"筛选{_text(label)}"))
        if isinstance(title := row.get("setting_display_label"), str):
            keys.add(_text(title))
    if (role in _NAMED_SETTINGS or role == "choose_setting") and isinstance(row.get("setting_label"), str):
        label = _text(row["setting_label"])
        keys.add(label)
        if role == "choose_setting" and isinstance(observed := row.get("setting_display_label"), str):
            keys.add(_text(observed))
        if row["role"] == _SLIDER:
            for title in {label, *_RANGE_NAMES.get(label, ())}:
                keys.update((title, f"{title}滑块", f"{title} 滑块", f"{title} slider"))
        elif row["role"] == _TAB:
            keys.update(f"{label}{suffix}" for suffix in ("标签页", " 标签页", "选项卡", " 选项卡", " tab"))
        elif row["role"] == _SEGMENT:
            keys.update(f"{label}{suffix}" for suffix in ("分段按钮", " 分段按钮", "分段", " 分段", " segment"))
    return keys


def _literal_setting_lookup(command, visible):
    while (wrapper := _WRAPPER.match(command)) is not None:
        command = command[wrapper.end():]
    name = _LOOKUP.sub("", _LOOKUP_END.sub("", command), count=1)
    name = _text(_OPERATOR.sub("", name, count=1).removeprefix("the "))
    for caption in dict.fromkeys((command, name)):
        matches = [row for row, keys in visible if caption in keys]
        if any(row.get("role") in _NAMED_SETTINGS or row.get("role") == "choose_setting" for row in matches):
            return _decision(matches)
    stem, locations = _location_keys(name)
    if len(locations) <= 1:
        dropdown = _dropdown_request(stem, visible, next(iter(locations), None))
        if dropdown is not None and dropdown["status"] in {"matched", "ambiguous"}:
            return dropdown
    return {"status": "unsupported_command"}


def _setting_request(command, visible):
    while (wrapper := _WRAPPER.match(command)) is not None:
        command = command[wrapper.end():]
    request = _SETTING_REQUEST.fullmatch(command)
    if request is None:
        return None
    intent = _SETTING_VERBS[request[1].strip()]
    title, locations = _location_keys(request[2])
    title = title.removeprefix("the ")
    names = {_event_key(title)}
    if intent == "adjust":
        names.add(_event_key(_SLIDER_SUFFIX.sub("", title)))
    panel_names = {role: _event_key(pattern.sub("", title)) for role, pattern in _PANEL_SUFFIXES.items()
                   if pattern.search(title)}
    named = []
    for row, _ in visible:
        keys = _setting_keys(row)
        if (panel_names.get(row.get("role")) in keys if panel_names else names & keys):
            named.append(row)
    matches = [row for row in named if row.get("role") in _SETTING_ROLES[intent]]
    if not matches:
        # A switch-to request may still describe an existing launcher action.
        if intent == "switch_to":
            return {"status": "not_located"} if panel_names else None
        if intent in {"check", "uncheck"} | _RANGE_INTENTS or any(row.get("role") in _NAMED_SETTINGS for row in named):
            return {"status": "not_located"}
        return None
    if len(locations) > 1:
        return {"status": "unsupported_command"}
    decision = _located_decision(matches, next(iter(locations), None))
    if decision["status"] != "matched":
        return decision
    row = next(match for match in matches if match["id"] == decision["target_id"])
    role = row["role"]
    state = row.get("setting_state")
    if role in _PANEL_CHOICES or role == _RADIO:
        if state == "on":
            return {"status": "not_located", "already_selected": True}
        return decision if state == "off" else {"status": "not_located"}
    if role not in {"toggle_setting", _CHECKBOX} or intent == "toggle":
        return decision
    desired = "off" if intent in {"off", "uncheck"} else "on"
    if state in {"on", "off"} and state != desired:
        return decision
    refusal = {"status": "not_located"}
    if role == _CHECKBOX and state == desired:
        refusal["already_selected"] = True
    return refusal


def _menu_title(row):
    return row.get("setting_option") if row["role"] == "select_setting_option" else row.get("menu_command_label")


def _option_request(command, entries):
    while (wrapper := _WRAPPER.match(command)) is not None:
        command = command[wrapper.end():]
    exact = [row for row, keys in entries if command in keys]
    if exact:
        return _decision(exact)
    assignment = _OPTION_ASSIGNMENT.fullmatch(command)
    if assignment is not None:
        title, name = assignment[1] or assignment[3], assignment[2] or assignment[4]
        entries = [(row, keys) for row, keys in entries if row["role"] == "select_setting_option"]
    else:
        title = None
        name = _LOOKUP.sub("", _LOOKUP_END.sub("", command), count=1)
        name = _OPERATOR.sub("", name, count=1)
        if (invocation := _MENU_COMMAND_OPERATOR.match(name)) is not None:
            name = name[invocation.end():]
            entries = [(row, keys) for row, keys in entries if row["role"] == "invoke_dropdown_command"]
        if (qualified := _OPTION_QUALIFIED.fullmatch(name)) is not None:
            if not any(_event_key(name) == _event_key(label) for row, _ in entries
                       if isinstance((label := _menu_title(row)), str)):
                title, name = qualified.groups()
    if title is not None:
        title_key = _event_key(title)
        entries = [(row, keys) for row, keys in entries
                   if title_key in _setting_keys(row, include_category_suffix=True)]
    name, locations = _location_keys(name.removeprefix("the "))
    if len(locations) > 1:
        return {"status": "unsupported_command"}
    choices = [row for row, _ in entries if isinstance((label := _menu_title(row)), str)
               and _event_key(label) == _event_key(name)]
    decision = _located_decision(choices, next(iter(locations), None))
    if assignment is not None and decision["status"] == "matched" and any(
            row.get("setting_option_selected") is True and row["id"] == decision["target_id"] for row in choices):
        return {"status": "not_located", "already_selected": True}
    return decision


def _visible_event(target, event):
    names = target.get("calendar_events")
    return isinstance(names, (list, tuple)) and any(_event_key(name) == event for name in names if isinstance(name, str))


def _calendar_date_target(request, targets, location):
    year = int(request[1]) if request[1] is not None else None
    month, day = _ordinal(request[2]), _ordinal(request[3])
    title = _event_key(request[4])
    choices = []
    for target in targets:
        if target.get("role") not in _CALENDAR_DAY_ROLES:
            continue
        date = target.get("calendar_date")
        if (not isinstance(date, (list, tuple)) or len(date) != 3 or tuple(date[1:]) != (month, day)
                or (year is not None and date[0] != year)):
            continue
        if title in {"", "日程", "的日程"} or _visible_event(target, title):
            choices.append(target)
    return _located_decision(choices, location)


def _grid_target(position: re.Match, targets: list[Mapping], location: str | None) -> dict:
    row_number, item_number = map(_ordinal, position.groups()[:2])
    roles = _GRID_KINDS[position[3]]
    entries = []
    for target in targets:
        if target.get("role") not in roles:
            continue
        box = target.get("screen_box_fraction")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or not all(type(value) in (int, float) and 0 <= value <= 1 for value in box)
                or not (box[0] < box[2] and box[1] < box[3])):
            return {"status": "not_located"}
        entries.append((box, target))

    grid = []
    for box, target in sorted(entries, key=lambda entry: (entry[0][1], entry[0][0])):
        left, top, right, bottom = box
        if grid:
            previous = grid[-1]
            first = previous[0][0]
            overlap = min(bottom, first[3]) - max(top, first[1])
            smaller_height = min(bottom - top, first[3] - first[1])
            if abs(top - first[1]) <= .18 * smaller_height and overlap >= .76 * smaller_height:
                if any(min(right, prior[2]) - max(left, prior[0]) > .05 * min(right - left, prior[2] - prior[0])
                       for prior, _ in previous):
                    return {"status": "not_located"}
                previous.append((box, target))
                continue
            if any(prior[3] - top > .13 * min(bottom - top, prior[3] - prior[1]) for prior, _ in previous):
                return {"status": "not_located"}
        grid.append([(box, target)])

    if row_number > len(grid) or item_number > len(grid[row_number - 1]):
        return {"status": "not_located"}
    selected = sorted(grid[row_number - 1], key=lambda entry: entry[0][0])[item_number - 1][1]
    return _located_decision([selected], location)


def find_named_action(query: str, targets: Iterable[Mapping]) -> dict:
    """Resolve an exact visible name or a single known intent to an inventory ID."""
    if not isinstance(query, str) or not query.strip():
        return {"status": "not_located"}
    command = _text(query)
    if len(query) > 300 or len(query.strip().splitlines()) > 1:
        return {"status": "unsupported_command"}

    rows = [row for row in targets if isinstance(row, Mapping) and isinstance(row.get("id"), str) and row["id"]]
    ids = Counter(row["id"] for row in rows)
    visible = [(row, _action_keys(row)) for row in rows
               if ids[row["id"]] == 1 and row.get("type") == "control_candidate"
               and row.get("interactivity") not in {"disabled", "noninteractive", "unknown"}
               and isinstance(row.get("function"), str)]
    menu_entries = [(row, keys) for row, keys in visible if row.get("role") in _MENU_ROLES]
    close_tab = _close_tab_request(query, [] if menu_entries else visible)
    if close_tab is not None:
        return close_tab
    if menu_entries:
        return _option_request(command, menu_entries)
    visible = [(row, keys) for row, keys in visible if row.get("role") != _CLOSE_TAB]
    # The exact displayed title may itself contain command punctuation.
    setting = _setting_request(command, visible)
    if setting is not None:
        return setting
    if _DAY_DISPLAY.fullmatch(command):
        exact = [row for row, _ in visible if row.get("role") in _CALENDAR_DAY_ROLES and _text(row["function"]) == command]
        if exact:
            return _decision(exact)
    if (_NEGATIVE.search(command) or _MORE_NEGATIVE.search(command)
            or _COMPOUND.search(command) or _MORE_COMPOUND.search(command)):
        return _literal_setting_lookup(command, visible)
    while (prefix := _WRAPPER.match(command)) is not None:
        command = command[prefix.end():]

    requested = _LOOKUP_END.sub("", command)
    stem = _LOOKUP.sub("", requested, count=1)
    operator = _OPERATOR.match(stem)
    location_text = stem[operator.end():] if operator is not None else stem
    stem, locations = _location_keys(location_text)
    if len(locations) > 1:
        return {"status": "unsupported_command"}
    location = next(iter(locations), None)
    if operator is None and location is not None:
        operator = _OPERATOR.match(stem)
        if operator is not None:
            stem = _text(stem[operator.end():])

    dropdown = _dropdown_request(stem, visible, location)
    if dropdown is not None:
        return dropdown

    # A named tab or segment must not fall through to a same-named ordinary control.
    for role, pattern in _PANEL_SUFFIXES.items():
        suffix = pattern.search(stem)
        if suffix is None:
            continue
        name = _text(stem[:suffix.start()])
        for title in dict.fromkeys((_event_key(name), _event_key(name.removeprefix("the ")))):
            matches = [row for row, keys in visible if (
                row.get("role") == role and title in _setting_keys(row)
                or role == _TAB and row.get("role") == "tab" and any(title == _event_key(key) for key in keys)
            )]
            if matches:
                return _located_decision(matches, location)
        return {"status": "not_located"}

    if (position := _GRID_POSITION.fullmatch(stem)) is not None:
        return _grid_target(position, [row for row, _ in visible], location)
    if (date := _DATE_REQUEST.fullmatch(stem)) is not None:
        return _calendar_date_target(date, [row for row, _ in visible], location)
    if (specified_filter := _FILTER_QUERY.fullmatch(stem)) is not None:
        label = specified_filter[1] or specified_filter[2]
        name = "".join(f"清除{label}筛选".split())
        matches = [row for row, _ in visible if row.get("role") == "clear_filter"
                   and "".join(_text(row["function"]).split()) == name]
        if matches:
            return _located_decision(matches, location)

    alternatives = [command]
    if stem:
        alternatives.append(stem)
    if stem.startswith("the "):
        alternatives.append(_text(stem[4:]))
    alternatives.append(_text(_CONTROL_NOUN.sub("", alternatives[-1])))
    if _SEARCH_INPUT_NAMES.intersection(alternatives):
        inputs = [row for row, _ in visible if row.get("role") == "search_input"]
        if inputs:
            return _located_decision(inputs, location)
    if _VIDEO_SHARE_COMMANDS.intersection(alternatives):
        return _located_decision([row for row, _ in visible
                                  if row.get("role") == "share" and row.get("action_context") == "video"], location)
    if _SHARE_COMMANDS.intersection(alternatives):
        return _located_decision([row for row, _ in visible if row.get("role") == "share"], location)

    semantic_command = (operator[0] if operator else "") + stem
    semantic_names = set()
    for index, (pattern, aliases) in enumerate(_INTENTS):
        names = {_text(alias) for alias in aliases}
        for key in names & _ADDITIONAL_NAMES.keys():
            names |= _ADDITIONAL_NAMES[key]
        # Only the named Search control is eligible. A freeform "find X" is
        # a lookup for X, even when a Search button happens to be on screen.
        if names.intersection(alternatives) or (index != 0 and pattern.fullmatch(semantic_command)):
            semantic_names.update(names)
    if _CLOSE_NAMES.intersection(alternatives) or semantic_command in _CLOSE_NAMES:
        semantic_names.update(_CLOSE_NAMES)

    for name in dict.fromkeys(alternatives):
        matches = [row for row, keys in visible if name in keys]
        if matches:
            ids = {row["id"] for row in matches}
            return _located_decision([row for row, keys in visible if row["id"] in ids or semantic_names & keys],
                                     location)
    folder = _FOLDER_NAME.fullmatch(stem.removeprefix("the "))
    if folder is not None:
        name = _text(folder[1])
        matches = [row for row, keys in visible if row.get("role") == "open_list_item" and name in keys]
        if matches:
            return _located_decision(matches, location)
    matches = [row for row, keys in visible if semantic_names & keys]
    if matches:
        return _located_decision(matches, location)
    if {"搜索", "search"}.intersection(alternatives):
        entrances = [row for row, keys in visible if row.get("role") == "search" and "搜索入口" in keys]
        return _located_decision(entrances, location)
    event = _event_key(stem)
    if event and (operator is None or operator[0].strip() in {"查看", "view"}):
        matches = [row for row, _ in visible if row.get("role") in _CALENDAR_DAY_ROLES and _visible_event(row, event)]
        if matches:
            return _located_decision(matches, location)
    return {"status": "not_located"}
