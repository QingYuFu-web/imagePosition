"""Resolve a single visible control from a narrow local language vocabulary."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


_CONTROL_TYPES = {"control_candidate", "icon", "dock_icon", "widget", "search"}
_WRAPPER = re.compile(
    r"^(?:能不能\s*(?:帮我)?|可以\s*(?:帮我)?|麻烦(?:你)?|请(?:你)?|帮我|我想(?:要)?|我要|我需要|先|"
    r"please\b|could you\b|can you\b|i want to\b|i would like to\b)\s*"
)
_DIRECT = re.compile(r"^(?:点击屏幕上的|点击|点一下|点开|点选|打开|进入|选择|轻点|点|tap\s+|click\s+|open\s+|select\s+)(.+)$")
_NEGATIVE = re.compile(r"不要|别(?:点|打开|选|搜|买)|不用|禁止|不想|\b(?:don't|do not|never)\b")
_COMPOUND = re.compile(
    r"[，,、；;。！？!?\n\r]|然后|接着|紧接着|同时|并且|或者|(?:再|并)(?:点|打开|选|搜|买|输入|发|返回|暂停|查看|添加|结算|支付)|"
    r"\b(?:and then|then|and click|and tap|or)\b"
)
_LOCATION = re.compile(r"^(左上角|右上角|左下角|右下角|左侧|左边|右侧|右边|上半屏|下半屏|底栏|底部栏)(?:的|那个)?\s*")
_QUOTES = '\"\'“”‘’「」『』《》'


# Intent labels must appear on actual visible controls. Add-to-cart and opening
# the cart deliberately use different labels, as do starting and pausing.
_INTENTS = (
    (re.compile(r"^(?:搜(?:索|一下|一搜)?|查找|search(?:\s+for)?\b|find\b).*$"),
     ("搜索", "搜索框", "搜索栏", "查找", "搜一搜", "输入关键词", "search", "search box", "find")),
    (re.compile(r"^(?:找(?:一下)?联系人|(?:看(?:一下|看)?|查看|打开|进入)通讯录|open (?:my )?contacts|view contacts)$"),
     ("通讯录", "联系人", "联系人列表", "contacts", "address book")),
    (re.compile(r"^(?:(?:看(?:一下|看)?|查看|打开|进入|去|回到)(?:我的)?购物车|(?:open|view|go to) (?:my |the )?(?:shopping )?cart)$"),
     ("购物车", "我的购物车", "cart", "my cart", "shopping cart")),
    (re.compile(r"^(?:(?:把|将)(?:这个|这件|当前)(?:商品)?(?:加|放)入购物车|加入购物车|加购|add(?: this)? to (?:the )?cart)$"),
     ("加入购物车", "加入购物袋", "加购", "add to cart", "add to bag")),
    (re.compile(r"^(?:(?:把|将)?(?:这段|这个|当前)?(?:视频|游戏|播放)?(?:暂停|停一下)(?:一下|游戏|视频|播放)?|pause(?: (?:the |this )?(?:game|video|playback))?)$"),
     ("暂停", "暂停游戏", "暂停视频", "暂停播放", "pause")),
    (re.compile(r"^(?:返回(?:上一页)?|回到上一页|后退|go back|previous page)$"),
     ("返回", "上一页", "后退", "back", "go back")),
    (re.compile(r"^(?:(?:回到|返回|去|打开)(?:首页|主页)|(?:go to|open) (?:the )?(?:home|home page))$"),
     ("首页", "主页", "home", "home page")),
    (re.compile(r"^(?:(?:看(?:看|一下)?|查看|打开)(?:视频的)?评论(?:区)?|(?:view|open) (?:the )?comments)$"),
     ("评论", "评论区", "查看评论", "comments")),
    (re.compile(r"^(?:开始游戏|进入游戏|start (?:the )?game|play (?:the )?game)$"),
     ("开始游戏", "进入游戏", "start game", "play game")),
    (re.compile(r"^(?:看(?:一下|看)?|打开|进入|查看)(?:订单|我的订单)$"),
     ("订单", "我的订单", "全部订单", "orders", "my orders")),
)


def _name(label: str) -> str:
    label = unicodedata.normalize("NFKC", label).casefold().strip().strip(_QUOTES)
    return re.sub(r"\s+", "", label)


def _target_names(point: dict) -> set[str]:
    names = {_name(point["label"])}
    aliases = point.get("aliases", ())
    if isinstance(aliases, (list, tuple)):
        names.update(_name(alias) for alias in aliases if isinstance(alias, str))
    if point["type"] in {"icon", "dock_icon"}:
        names.update(_name(point["label"] + suffix) for suffix in ("图标", "应用"))
    return names - {"", "未命名控件", "组合区域", "状态栏指示"}


def _evidenced_control(point: dict) -> bool:
    if (point.get("type") not in _CONTROL_TYPES or point.get("interactivity") in {"unknown", "disabled", "noninteractive"}
            or not isinstance(point.get("id"), str) or not point["id"]
            or not isinstance(point.get("label"), str) or not point["label"].strip()):
        return False
    if point["type"] != "control_candidate":
        return True
    score = point.get("confidence")
    if isinstance(score, (int, float)) and not .34 <= score <= 1:
        return False
    evidence = point.get("text_evidence")
    if evidence is None:
        return point.get("source") not in {"yolo", "visual_ocr"}
    if not isinstance(evidence, (list, tuple)) or len(evidence) != 1:
        return False
    item = evidence[0]
    return (isinstance(item, dict) and isinstance(item.get("text"), str)
            and _name(item["text"]) == _name(point["label"])
            and isinstance(item.get("confidence"), (int, float)) and .72 <= item["confidence"] <= 1)


def _at_location(point: dict, location: str | None) -> bool:
    if location is None:
        return True
    if location in {"底栏", "底部栏"}:
        return point["type"] == "dock_icon"
    position = point.get("screen_fraction")
    if (not isinstance(position, (list, tuple)) or len(position) != 2
            or not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in position)):
        return False
    x, y = position
    return {
        "左上角": x < .30 and y < .28, "右上角": x > .70 and y < .28,
        "左下角": x < .30 and y > .72, "右下角": x > .70 and y > .72,
        "左侧": x < .40, "左边": x < .40, "右侧": x > .60, "右边": x > .60,
        "上半屏": y < .50, "下半屏": y > .50,
    }[location]


def _decision(matches: list[dict]) -> dict:
    if len(matches) == 1:
        point = matches[0]
        return {"status": "matched", "target_id": point["id"], "target_label": point["label"],
                "explanation": f"当前画面中匹配到「{point['label']}」。"}
    return {"status": "ambiguous", "choices": [{"id": point["id"], "label": point["label"]} for point in matches],
            "explanation": "当前画面有多个符合描述的控件，请指明位置或更完整的名称。"}


def resolve_local_target(goal: str, observation: dict) -> dict:
    """Suggest a visible next target when its label supports the request."""
    if not isinstance(goal, str) or len(goal) > 300:
        return {"status": "not_located", "explanation": "请用 1 到 300 个字符描述你要办的事。"}
    query = unicodedata.normalize("NFKC", goal).casefold().strip().rstrip("。！!?？").strip()
    if not query:
        return {"status": "not_located", "explanation": "请描述你要办的事。"}
    if _NEGATIVE.search(query) or _COMPOUND.search(query):
        return {"status": "not_located", "explanation": "本地定位需要一个明确的正向操作。"}
    if observation.get("status", "ok") not in {"ok", "partial"}:
        return {"status": "not_located", "explanation": "当前画面还没有可用的识别结果。"}
    while (prefix := _WRAPPER.match(query)) is not None:
        query = query[prefix.end():]

    found = [point for point in observation.get("keypoints", ()) if isinstance(point, dict) and _evidenced_control(point)]
    ids = Counter(point["id"] for point in found)
    controls = [point for point in found if ids[point["id"]] == 1]
    named = _DIRECT.match(query)
    location = None
    semantic_query = query
    named_hint = None
    if named:
        query_name = named[1].strip()
        qualifier = _LOCATION.match(query_name)
        if qualifier is not None:
            location = qualifier[1]
            query_name = query_name[qualifier.end():]
            semantic_query = query[:named.start(1)] + query_name
        exact = _name(query_name)
        named_hint = exact
        matches = [point for point in controls if exact in _target_names(point) and _at_location(point, location)]
        if matches:
            return _decision(matches)
        base = re.sub(r"(?:按钮|图标|入口|选项)$", "", exact)
        if base and base != exact:
            named_hint = base
            matches = [point for point in controls if base in _target_names(point) and _at_location(point, location)]
            if matches:
                return _decision(matches)

    # A request with only a task verb should match labels in the appropriate
    # group. Descriptive searches must not accidentally select iOS Spotlight.
    for index, (pattern, aliases) in enumerate(_INTENTS):
        names = {_name(alias) for alias in aliases}
        if named_hint in names or pattern.fullmatch(semantic_query):
            exact_search_label = (named_hint or _name(query)) in names
            matches = [point for point in controls if names & _target_names(point) and _at_location(point, location)
                       and not (index == 0 and not exact_search_label and point["type"] == "search")]
            if matches:
                return _decision(matches)

    return {"status": "not_located", "explanation": "当前画面没有可确认的对应控件。可以换一张当前页面的图片，或说出控件上的完整文字。"}
