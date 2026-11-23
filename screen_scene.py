"""Describe the evidence visible in a single locally recognized screen."""

from __future__ import annotations

import re
import unicodedata


_CONTROL_TYPES = {"control_candidate", "icon", "dock_icon", "widget", "search"}
_ROLE_LABELS = {
    "search": ("搜索", "搜索框", "搜索栏", "查找", "搜一搜", "search"),
    "message_input": ("输入消息", "请输入消息", "说点什么", "输入文字", "message"),
    "send": ("发送", "send"),
    "contacts": ("通讯录", "联系人", "contacts"),
    "cart": ("购物车", "我的购物车", "cart"),
    "add_to_cart": ("加入购物车", "加入购物袋", "加购", "add to cart"),
    "buy": ("立即购买", "马上购买", "单独购买", "发起拼单", "参与拼单", "buy now"),
    "checkout": ("去结算", "结算", "提交订单", "立即支付", "去支付", "checkout"),
    "comments": ("评论", "评论区", "查看评论", "comments"),
    "share": ("分享", "share"),
    "like": ("点赞", "喜欢", "like"),
    "favorite": ("收藏", "favorite", "bookmark", "save to favorites"),
    "save": ("保存", "save"),
    "play": ("播放", "播放视频", "play"),
    "pause": ("暂停", "暂停播放", "暂停游戏", "pause"),
    "back": ("返回", "上一页", "后退", "back"),
    "close": ("关闭", "close"),
    "cancel": ("取消", "cancel"),
    "confirm": ("确认", "确定", "继续", "confirm", "ok"),
    "settings": ("设置", "系统设置", "settings"),
    "orders": ("我的订单", "全部订单", "订单列表", "my orders", "order history"),
    "pending_payment": ("待付款", "待支付", "unpaid", "awaiting payment"),
    "pending_shipment": ("待发货", "待配送", "to ship", "awaiting shipment"),
    "pending_receipt": ("待收货", "待签收", "to receive", "awaiting delivery"),
    "pending_review": ("待评价", "待评论", "to review"),
    "pending_group": ("待拼单", "待成团"),
    "after_sales": ("退款售后", "退款/售后", "退款与售后", "退换售后", "售后服务", "售后", "refunds", "returns"),
    "coupons": ("优惠券", "我的优惠券", "卡券", "coupons"),
    "history": ("历史浏览", "浏览记录", "浏览历史", "足迹", "我的足迹", "browsing history"),
    "saved_items": ("商品收藏", "我的收藏", "收藏夹", "saved items", "wishlist"),
    "support": ("官方客服", "联系客服", "联系客服中心", "客服", "帮助中心", "support", "customer service"),
    "addresses": ("收货地址", "地址管理", "我的地址", "addresses"),
    "account_center": ("个人中心", "账户中心", "账号中心", "我的账户", "my account", "account"),
}
_ROLE_NAMES = {
    "app_icon": "应用图标", "dock_icon": "底栏图标", "widget": "小组件", "search": "搜索入口",
    "message_input": "消息输入", "send": "发送", "contacts": "联系人", "cart": "购物车",
    "add_to_cart": "加入购物车", "buy": "购买", "checkout": "结算或支付", "comments": "评论",
    "share": "分享", "like": "点赞", "favorite": "收藏", "save": "保存", "play": "播放", "pause": "暂停",
    "back": "返回", "close": "关闭", "cancel": "取消", "confirm": "确认", "settings": "设置",
    "orders": "订单入口", "all_orders": "查看全部订单", "pending_payment": "待付款订单",
    "pending_shipment": "待发货订单", "pending_receipt": "待收货订单", "pending_review": "待评价订单",
    "pending_group": "待成团订单", "after_sales": "退款或售后", "coupons": "优惠券入口",
    "history": "浏览记录", "saved_items": "收藏入口", "support": "客服入口", "addresses": "收货地址",
    "account_center": "个人中心", "navigation_tab": "底部导航",
    "unknown": "用途未确认",
}
_ORDER_STATUS_ROLES = {"pending_payment", "pending_shipment", "pending_receipt", "pending_review", "pending_group"}
_SERVICE_ROLES = {"after_sales", "coupons", "history", "saved_items", "support", "addresses"}
_REGION_NAMES = {
    "upper_left": "左上", "upper_center": "上方中央", "upper_right": "右上",
    "middle_left": "左侧", "middle_center": "中间", "middle_right": "右侧",
    "lower_left": "左下", "lower_center": "下方中央", "lower_right": "右下", "unknown": "位置未确认",
}


def _normal(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


_ROLES = {_normal(text): role for role, labels in _ROLE_LABELS.items() for text in labels}
_NAV_LABELS = {_normal(text) for text in (
    "首页", "发现", "视频", "分类", "聊天", "消息", "购物车", "个人中心", "我的", "我", "通讯录", "朋友",
    "home", "discover", "video", "categories", "chat", "messages", "cart", "account", "profile", "contacts", "friends",
)}
_PAGE_TITLE_LABELS = {_normal(text) for text in (
    "搜索", "搜索结果", "联系人", "通讯录", "聊天", "消息", "个人中心", "账户中心", "账号中心", "我的订单",
    "全部订单", "订单列表", "订单详情", "设置", "系统设置", "购物车", "商品详情", "确认订单", "提交订单", "选择规格",
    "收货地址", "地址管理", "优惠券", "历史浏览", "浏览记录", "商品收藏", "收藏夹", "退款售后", "退款/售后",
    "官方客服", "帮助中心", "分享到", "分享至", "游戏设置", "战绩", "对局记录",
    "search", "search results", "contacts", "chat", "messages", "my account", "my orders", "order details", "settings",
    "cart", "product details", "checkout", "addresses", "support", "coupons", "game settings",
)}


def _position(point):
    xy = point.get("screen_fraction")
    if (not isinstance(xy, (list, tuple)) or len(xy) != 2
            or not all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in xy)):
        return None, "unknown"
    x, y = xy
    column = "left" if x < .34 else "right" if x > .66 else "center"
    row = "upper" if y < .23 else "lower" if y > .75 else "middle"
    return list(xy), f"{row}_{column}"


def _supported_label(point):
    if point.get("interactivity") in {"disabled", "noninteractive", "unknown"}:
        return ""
    evidence = point.get("text_evidence")
    if evidence is None:
        return "" if point.get("source") in {"yolo", "visual_ocr"} else _normal(point["label"])
    if not isinstance(evidence, (list, tuple)):
        return ""
    words = set()
    for entry in evidence:
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            return ""
        word = _normal(entry["text"]).rstrip(">›»→❯")
        if not word or re.fullmatch(r"\d{1,3}\+?|[¥$€£]+", word):
            continue
        if not _strong(entry.get("confidence"), .73):
            return ""
        words.add(word)
    return next(iter(words)) if len(words) == 1 else ""


def _role(point, supported_label):
    known = {"icon": "app_icon", "dock_icon": "dock_icon", "widget": "widget", "search": "search"}
    return known.get(point.get("type"), _ROLES.get(supported_label, "unknown"))


def _strong(score, threshold):
    return score is None or isinstance(score, (int, float)) and threshold <= score <= 1


def _visible_text(points):
    items = []
    seen = set()
    for point in points:
        if point.get("type") == "text":
            evidence = [{"text": point["label"], "confidence": point.get("confidence")}]
        else:
            evidence = point.get("text_evidence") or []
        if not isinstance(evidence, (list, tuple)):
            continue
        xy, zone = _position(point)
        for fragment in evidence:
            if not isinstance(fragment, dict):
                continue
            label = fragment.get("text")
            if not isinstance(label, str) or not label.strip():
                continue
            identity = (point["id"], _normal(label))
            if identity in seen:
                continue
            seen.add(identity)
            item = {"id": point["id"], "text": label, "zone": zone, "zone_label": _REGION_NAMES[zone],
                    "screen_fraction": xy, "confidence": fragment.get("confidence"), "source_type": point.get("type")}
            if point.get("type") != "text":
                item["position_basis"] = "associated_control"
            if "relation" in fragment:
                item["relation"] = fragment["relation"]
            items.append(item)
    return items


def _title_clues(texts):
    hits = []
    for item in texts:
        position = item["screen_fraction"]
        label = item["text"].strip()
        if (item["source_type"] != "text" or position is None or not _strong(item["confidence"], .82)
                or not (.26 <= position[0] <= .74 and .045 <= position[1] <= .15)
                or _normal(label) not in _PAGE_TITLE_LABELS):
            continue
        hits.append(item)
    hits.sort(key=lambda item: (abs(item["screen_fraction"][0] - .5), item["screen_fraction"][1]))
    return [{"id": item["id"], "text": item["text"], "zone": item["zone"]} for item in hits[:3]]


def _page_indicators(controls, texts):
    clues = []
    reliable = [item for item in texts if item["source_type"] != "status_indicator" and _strong(item["confidence"], .76)]

    def roles(*wanted):
        return [item for item in controls if item["role"] in wanted]

    def mentions(*labels):
        tokens = tuple(_normal(label) for label in labels)
        return [item for item in reliable if any(token in _normal(item["text"]) for token in tokens)]

    def add(kind, label, items):
        unique = {}
        for item in items:
            unique.setdefault(item["id"], {"id": item["id"], "text": item.get("text", item.get("label"))})
        if len(unique) >= 2:
            clues.append({"kind": kind, "label": label, "evidence": list(unique.values())[:6]})

    search = roles("search") + [item for item in reliable if _normal(item["text"]) in {"搜索", "搜索框", "search"}]
    search_context = mentions("搜索结果", "热搜", "搜索历史", "最近搜索", "搜索发现", "相关搜索", "猜你想搜", "热门搜索")
    if search and search_context:
        add("search", "有搜索界面线索", search + search_context)

    messages = roles("message_input") + mentions("聊天记录", "输入消息", "请输入消息")
    send = roles("send")
    if send and messages:
        add("chat", "有聊天界面线索", send + messages)

    media = roles("play", "pause")
    reactions = roles("comments", "share", "like", "favorite")
    if (media and reactions) or len({item["role"] for item in reactions}) >= 3:
        add("media", "有内容播放或浏览线索", media + reactions)

    product_actions = roles("add_to_cart", "buy") + mentions("加入购物车", "立即购买", "单独购买", "发起拼单", "参与拼单")
    product_details = mentions("商品详情", "商品评价", "运费", "商品规格", "选择规格", "配送", "型号")
    if product_actions and (product_details or len({item.get("role") for item in product_actions} - {None}) == 2):
        add("product", "有商品详情线索", product_actions + product_details)
    elif len({item["id"] for item in product_details}) >= 2 and mentions("优惠", "商品金额", "商品价格"):
        add("product", "有商品规格和交易信息线索", product_details + mentions("优惠", "商品金额", "商品价格"))

    checkout = roles("checkout") + mentions("确认订单", "提交订单")
    checkout_details = mentions("收货地址", "支付方式", "商品金额", "应付金额", "实付款")
    if checkout and checkout_details:
        add("checkout", "有结算或订单确认线索", checkout + checkout_details)

    orders = mentions("我的订单", "全部订单", "个人中心", "账户中心", "账号中心")
    order_sections = mentions("待付款", "待收货", "待发货", "退款售后", "退款/售后", "订单记录", "收货地址")
    if orders and len({item["id"] for item in order_sections}) >= 2:
        add("account_orders", "有账户中心或订单入口线索", orders + order_sections)

    share_title = mentions("分享到", "分享至")
    share_context = roles("cancel") + mentions("好友", "朋友圈", "复制链接", "取消")
    if share_title and share_context:
        add("share", "有分享面板线索", share_title + share_context)

    keypad_groups = {"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}
    keys = [item for item in controls if item["screen_fraction"] is not None and item["screen_fraction"][1] > .54
            and (re.fullmatch(r"[a-z0-9]", _normal(item["label"])) or _normal(item["label"]) in keypad_groups)]
    if (len({_normal(item["label"]) for item in keys}) >= 6
            and max(item["screen_fraction"][1] for item in keys) - min(item["screen_fraction"][1] for item in keys) >= .05
            and max(item["screen_fraction"][0] for item in keys) - min(item["screen_fraction"][0] for item in keys) >= .25):
        add("keyboard", "下方有多行键盘按键线索", keys)

    confirms = roles("confirm")
    cancellation = roles("cancel")
    modal_title = [item for item in mentions("提示", "确认退出", "是否删除", "是否退出")
                   if item["source_type"] == "text" and item["screen_fraction"] is not None]
    for left in cancellation:
        a = left["screen_fraction"]
        if a is None or not .33 < a[1] < .86:
            continue
        aligned = [item for item in confirms if item["screen_fraction"] is not None
                   and abs(item["screen_fraction"][1] - a[1]) < .035
                   and .11 < abs(item["screen_fraction"][0] - a[0]) < .52]
        above = [item for item in modal_title if .035 < a[1] - item["screen_fraction"][1] < .23]
        if aligned and above:
            add("modal", "有确认弹窗线索", [left, aligned[0], above[0]])
            break

    apps = roles("app_icon")
    dock = roles("dock_icon")
    if len(apps) >= 6 and len(dock) >= 3:
        add("home", "有手机主屏幕图标和底栏线索", dock + apps)
    return clues


def _row_controls(items, tolerance, *, min_count, min_spread):
    """Pick a horizontal row supported by distinct columns and labels."""
    best = []
    for anchor in items:
        center = anchor["screen_fraction"][1]
        nearby = sorted((item for item in items if abs(item["screen_fraction"][1] - center) <= tolerance),
                        key=lambda item: item["screen_fraction"][0])
        row = []
        for item in nearby:
            if any(item["label"] == known["label"] or abs(item["screen_fraction"][0] - known["screen_fraction"][0]) < .055
                   for known in row):
                continue
            row.append(item)
        if (len(row) >= min_count and row[-1]["screen_fraction"][0] - row[0]["screen_fraction"][0] >= min_spread
                and len(row) > len(best)):
            best = row
    return best


def _section(key, label, items):
    return {"id": key, "label": label, "control_ids": [item["id"] for item in items]}


def _page_structure(controls, labels, indicators, titles):
    located = [item for item in controls if item["screen_fraction"] is not None]
    sections = []

    anchors = sorted((item for item in located if item["role"] == "orders"),
                     key=lambda item: item["screen_fraction"][1])
    for anchor in anchors:
        x, y = anchor["screen_fraction"]
        statuses = [item for item in located if item["role"] in _ORDER_STATUS_ROLES
                    and .018 <= item["screen_fraction"][1] - y <= .16]
        row = _row_controls(statuses, .023, min_count=2, min_spread=.12)
        if not row:
            continue
        see_all = [item for item in located if labels[item["id"]] in {"全部", "查看全部", "all", "viewall"}
                   and abs(item["screen_fraction"][1] - y) <= .025 and item["screen_fraction"][0] - x >= .20]
        if len(see_all) == 1:
            see_all[0].update(role="all_orders", role_label=_ROLE_NAMES["all_orders"])
        else:
            see_all = []
        sections.append(_section("orders", "订单", [anchor, *see_all, *row]))
        break

    service_items = [item for item in located if item["role"] in _SERVICE_ROLES and .23 <= item["screen_fraction"][1] <= .82]
    service_row = _row_controls(service_items, .032, min_count=3, min_spread=.25)
    if service_row:
        sections.append(_section("services", "个人服务", service_row))

    navigation = _row_controls(
        [item for item in located if item["screen_fraction"][1] >= .87 and labels[item["id"]] in _NAV_LABELS],
        .024, min_count=3, min_spread=.42,
    )
    if navigation:
        for item in navigation:
            if item["role"] == "unknown":
                item.update(role="navigation_tab", role_label=_ROLE_NAMES["navigation_tab"])
        sections.append(_section("navigation", "底部导航", navigation))

    page_name = None
    if titles and len({_normal(item["text"]) for item in titles}) == 1:
        title = titles[0]
        page_name = {"label": title["text"], "status": "visible_title",
                     "evidence": [{"id": title["id"], "text": title["text"]}]}
    account = next((clue for clue in indicators if clue["kind"] == "account_orders"), None)
    account_tab = next((item for item in navigation if item["role"] == "account_center"), None)
    if (page_name is None and not titles and account_tab is not None and account is not None
            and any(s["id"] == "orders" for s in sections)):
        page_name = {"label": account_tab["label"], "status": "inferred",
                     "evidence": [{"id": account_tab["id"], "text": account_tab["label"]},
                                  *[item for item in account["evidence"] if item["id"] != account_tab["id"]]][:5]}
    active_tab = None
    if page_name is not None:
        matching_tab = [item for item in navigation if labels[item["id"]] == _normal(page_name["label"])]
        if len(matching_tab) == 1:
            item = matching_tab[0]
            active_tab = {"id": item["id"], "label": item["label"], "status": "candidate", "basis": "page_context"}
    return page_name, active_tab, sections


def analyze_screen(observation: dict) -> dict:
    """Summarize current observable evidence without guessing the app or page."""
    points = [point for point in observation.get("keypoints", ()) if isinstance(point, dict)
              and isinstance(point.get("id"), str) and isinstance(point.get("label"), str)]
    texts = _visible_text(points)
    controls = []
    labels = {}
    for point in points:
        if point.get("type") not in _CONTROL_TYPES:
            continue
        xy, zone = _position(point)
        label = _supported_label(point)
        labels[point["id"]] = label
        role = _role(point, label)
        controls.append({"id": point["id"], "label": point["label"], "type": point["type"], "role": role,
                         "role_label": _ROLE_NAMES[role], "zone": zone, "zone_label": _REGION_NAMES[zone],
                         "screen_fraction": xy, "geometry_method": (point.get("geometry") or {}).get("method")})
    indicators, titles = _page_indicators(controls, texts), _title_clues(texts)
    page_name, active_tab, sections = _page_structure(controls, labels, indicators, titles)
    return {"status": observation.get("status", "ok" if points else "no_verified_targets"),
            "page_indicators": indicators, "title_clues": titles,
            "page_name": page_name, "active_tab": active_tab, "sections": sections,
            "visible_text": texts, "controls": controls, "text_count": len(texts), "control_count": len(controls)}


def format_scene(scene: dict) -> str:
    """Format the local scene report for the screenshot workbench."""
    if scene.get("recording_gesture"):
        return "正在录音\n\n当前前景是按住、滑动或松开的手势区。"
    page = scene.get("page_name")
    lines = ["当前页面"]
    if page:
        qualifier = "（根据可见信息推断）" if page["status"] == "inferred" else ""
        lines.append(f"{page['label']}{qualifier}")
    else:
        lines.append("页面名称未确认。")
    active = scene.get("active_tab")
    if active:
        lines.append(f"可能的当前标签：[{active['id']}] {active['label']}")
    lines.extend(["", "页面线索"])
    for indicator in scene["page_indicators"]:
        evidence = "、".join(item["text"] for item in indicator["evidence"] if item["text"])
        lines.append(f"{indicator['label']}（{evidence}）")
    if not scene["page_indicators"]:
        lines.append("未能确认页面类型。")
    if scene["title_clues"] and page is None:
        lines.extend(["", "页面标题候选"])
        lines.extend(item["text"] for item in scene["title_clues"])
    controls = {item["id"]: item for item in scene["controls"]}
    grouped = set()
    for section in scene.get("sections", ()):
        lines.extend(["", section["label"]])
        for control_id in section["control_ids"]:
            item = controls[control_id]
            grouped.add(control_id)
            lines.append(f"[{item['id']}] {item['label']} · {item['zone_label']}")
    other = [item for item in scene["controls"] if item["id"] not in grouped and item["role"] != "unknown"]
    if other:
        lines.extend(["", "其他已识别入口"])
        lines.extend(f"[{item['id']}] {item['label']} · {item['zone_label']}" for item in other)
    if not scene["controls"]:
        lines.extend(["", "未识别到控件。"])
    else:
        unknown = sum(item["role"] == "unknown" for item in scene["controls"])
        lines.extend(["", f"共 {scene['control_count']} 项控件候选，{unknown} 项用途未确认。"])
    lines.append(f"{scene['text_count']} 处识别文字，完整结果见 JSON。")
    return "\n".join(lines)
