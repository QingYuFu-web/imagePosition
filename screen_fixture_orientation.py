"""Decide whether a calibrated portrait raster contains upright sideways controls."""

from functools import lru_cache

import cv2
import numpy as np
from PIL import ImageFont


_CAPTIONS = tuple(dict.fromkeys((
    "设置 画面 图像 显示 音效 音乐 声音 操作 操控 控制 按键 键位 灵敏度 基础 系统 常用 通用 辅助 功能 "
    "确定 确认 取消 返回 退出 关闭 继续 完成 开启 关闭 保存 重置 恢复 默认 选择 查看 搜索 刷新 更多 "
    "模式 游戏 开始 开始游戏 进入 战斗 观战 对战 排位 排行 榜单 活动 奖励 组队 好友 好友列表 地图 "
    "消息 私信 聊天 评论 分享 发送 点赞 关注 个人 主页 账号 资源 下载 装备 技能 商店 背包 购买 出售 "
    "商城 任务 领取 前往 选择 载入 推荐 自动 手动 完成 已开启 已关闭 本地 全部 自定义 设置帮助 "
    "布局 画质 帧率 高清 超清 极高 最低 标准 普通 流畅 均衡 关闭 阴影 特效 抗锯齿 垂直同步 "
    "界面 镜头 弹幕 字幕 截图 录像 录制 清晰度 实名认证 未开启 立即加入 返回大厅 我的队伍 好友申请 "
    "邀请好友 不感兴趣 友情手账 恢复默认 声音设置 基础设置 画面设置 操作设置 载具设置 灵敏度设置 "
    "拾取设置 效果设置 录像设置 自动拾取手枪 自动拾取瞄准镜 自动拾取三级包 自动装配 瞄准镜 "
    "练习 排位赛 排行榜 积分 赛季 训练 英雄 皮肤 铭文 战绩 免费 时间 拾取 自动拾取 助手 小地图 "
    "移动 方向 按钮 单人 双人 多人 输入 编辑 退出游戏 音频 视频 语言 隐私 帮助 反馈 高级 显示设置 "
    "Settings Controls Graphics Display Camera Audio Sound Music Play Pause Start Game Continue Resume Back "
    "Close Confirm Cancel Accept Save Reset Default Status Profile Search Select Help Friends Shop Map Exit "
    "Edit Done Account Apply Quality Normal High Low Photos Inventory Chat Send Inbox"
).split()))


def _foreground(diff, floor):
    mask = np.uint8(diff >= floor)
    _, owners, parts, _ = cv2.connectedComponentsWithStats(mask)
    _, _, w, h, count = parts.T
    good = (w <= 36) & (h <= 36) & (w >= 2) & (h >= 2) & (count >= 4) & (count >= .14 * w * h)
    good[0] = False
    return np.uint8(good[owners])


def _lines(mask):
    result = set()
    for length in (5, 9):
        linked = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, length), np.uint8))
        _, _, parts, _ = cv2.connectedComponentsWithStats(linked)
        _, _, w, h, count = parts.T
        eligible = (w >= 14) & (w <= 180) & (h >= 7) & (h <= 30) & (w >= 1.40 * h) & (w <= 10 * h)
        eligible &= (count >= .34 * w * h) & (count <= .98 * w * h)
        for x, y, w, h, _ in parts[eligible]:
            observed = mask[y:y + h, x:x + w]
            a, b, c, d = cv2.boundingRect(observed)
            if d >= .85 * h and c >= .81 * w and .15 <= observed.mean() <= .78:
                result.add((int(x + a), int(y + b), c, d))
    return result


@lru_cache(maxsize=4)
def _font_words(cjk, latin):
    answers = []
    for path in dict.fromkeys((*cjk, *latin)):
        try:
            font = ImageFont.truetype(str(path), 42)
        except OSError:
            continue
        for word in _CAPTIONS:
            if word.isascii() == (path in cjk):
                continue
            im = font.getmask(word, mode="L")
            pixels = np.asarray(im, np.uint8).reshape(im.size[1], im.size[0])
            x, y, w, h = cv2.boundingRect(np.uint8(pixels >= 40))
            if w and h:
                glyph = pixels[y:y + h, x:x + w]
                answers.append((word, w / h, glyph))
    return answers


@lru_cache(maxsize=52)
def _font_bank(cjk, latin, ratio_bin):
    words, penalty, masks = [], [], []
    for word, aspect, pixels in _font_words(cjk, latin):
        if not .77 <= aspect / ratio_bin <= 1.28:
            continue
        mask = cv2.resize(pixels, (38, 20), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        mask = cv2.GaussianBlur(mask, (5, 3), .73).ravel()
        words.append(word)
        penalty.append(.10 * abs(np.log(aspect / ratio_bin)))
        masks.append(mask / np.linalg.norm(mask))
    return words, np.asarray(penalty, np.float32), np.asarray(masks, np.float32)


def _score(mask, cjk, latin):
    h, w = mask.shape
    words, penalty, bank = _font_bank(cjk, latin, round(8 * w / h) / 8)
    if len(words) < 2:
        return ()
    sample = cv2.resize(mask.astype(np.float32), (38, 20), interpolation=cv2.INTER_AREA)
    sample = cv2.GaussianBlur(sample, (3, 3), .48).ravel()
    norm = max(1e-5, np.linalg.norm(sample))
    values = (bank @ np.column_stack((sample, sample[::-1]))) / norm - penalty[:, None]
    output = []
    for column in values.T:
        best = int(column.argmax())
        second = max((value for text, value in zip(words, column, strict=True) if text != words[best]), default=0)
        output.append((words[best], float(column[best]), float(column[best] - second)))
    return output


def _orientation_votes(masks, cjk, latin):
    candidates = []
    for pass_no, (mask, lines) in enumerate(masks):
        for x, y, w, h in sorted(lines, key=lambda r: (-r[3], r))[:77]:
            patch = mask[y:y + h, x:x + w]
            outcomes = _score(patch, cjk, latin)
            if len(outcomes) != 2:
                continue
            direction = int(outcomes[1][1] > outcomes[0][1])
            word, confidence, margin = outcomes[direction]
            opposition = outcomes[1 - direction][1]
            if confidence >= .81 and margin >= .040 and confidence - opposition >= .064:
                candidates.append((confidence, direction, word, x + w / 2, y + h / 2, h, pass_no))
    result = [[], []]
    for confidence, direction, word, x, y, h, _ in sorted(candidates, reverse=True):
        if all(abs(x - xx) > 1.4 * max(h, hh) or abs(y - yy) > .70 * max(h, hh)
               for _, _, xx, yy, hh, _ in result[direction]):
            support = {pass_no for _, turn, text, xx, yy, hh, pass_no in candidates
                       if turn == direction and word == text and abs(x - xx) <= .75 * max(h, hh)
                       and abs(y - yy) <= .50 * max(h, hh)}
            result[direction].append((word, confidence, x, y, h, len(support)))
    return result


def _fonts(index):
    if index is None:
        from screen_font_glyphs import FontGlyphIndex

        index = FontGlyphIndex()
    return (tuple(path for path in index.cjk_sources if path.name.lower() in {"msyh.ttc", "msyhbd.ttc", "simhei.ttf"}),
            tuple(path for path in index.choice_latin_sources if path.name.lower() in {"arial.ttf", "segoeui.ttf"}))


def _prepared(frame):
    chip = cv2.resize(frame, (416, 896), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY), (3, 3), .42)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
        delta = cv2.morphologyEx(gray, operation, kernel)
        for floor in (17, 29):
            yield _foreground(delta, floor)


def _focused_caption(frame, witnesses, direction, cjk, latin):
    """Confirm the same caption at the same position under separate resampling."""
    strong = [row for row in witnesses if len(row[0]) >= 2 and row[1] >= .875]
    if not strong:
        return False
    chip = cv2.resize(frame, (416, 896), interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY), (3, 3), .50)
    gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for word, _, x, y, height, _ in strong:
        left, top = max(0, round(x - 5 * height)), max(0, round(y - 2.5 * height))
        right, bottom = round(x + 5 * height), round(y + 2.5 * height)
        crop = gray[top:bottom, left:right]
        checked, agreements = set(), 0
        for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
            delta = cv2.morphologyEx(crop, operation, kernel)
            for floor in range(13, 35, 3):
                mask = _foreground(delta, floor)
                for xx, yy, w, h in _lines(mask):
                    scale = max(height, h)
                    if abs(left + xx + w / 2 - x) > .60 * scale or abs(top + yy + h / 2 - y) > .45 * scale:
                        continue
                    snippet = mask[yy:yy + h, xx:xx + w]
                    key = snippet.shape, snippet.tobytes()
                    if key in checked:
                        continue
                    checked.add(key)
                    scores = _score(snippet, cjk, latin)
                    if len(scores) != 2:
                        continue
                    caption, confidence, margin = scores[direction]
                    if (caption == word and confidence >= .865 and margin >= .040
                            and confidence - scores[1 - direction][1] >= .064):
                        agreements += 1
                        if agreements == 2:
                            return True
    return False


def _structural_direction(frame, blocked, directions):
    from screen_fixture_gallery_orientation import gallery_orientation_cue
    from screen_fixture_settings_orientation import settings_orientation_cue

    matches = []
    for direction in directions:
        view = cv2.rotate(frame, direction)
        covered = cv2.rotate(blocked, direction) if blocked is not None else None
        if gallery_orientation_cue(view, blocked=covered) or settings_orientation_cue(view, blocked=covered):
            matches.append(direction)
    return matches[0] if len(matches) == 1 else None


def fixture_orientation(frame, overlay=None, *, index=None, blocked=None):
    """Return a cv2 quarter-turn when captions or a marked rail verify its direction."""
    height, width = frame.shape[:2]
    if overlay is not None or width < 300 or not 1.76 <= height / width <= 2.45:
        return None
    native, sideways = [], []
    for pass_no, mask in enumerate(_prepared(frame)):
        rotated = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        native.append((mask, _lines(mask)))
        sideways.append((rotated, _lines(rotated)))
        if pass_no == 0 and len(native[0][1]) >= max(12, 2.65 * len(sideways[0][1])):
            return None
    native_count = sum(len(lines) for _, lines in native)
    side_count = sum(len(lines) for _, lines in sideways)
    directions = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
    upright_caption = None

    def has_upright_caption():
        nonlocal upright_caption
        if upright_caption is None:
            upright_caption = any(_orientation_votes(native, *_fonts(index)))
        return upright_caption

    def structural(turns=None):
        candidate = _structural_direction(frame, blocked, directions)
        if candidate is None:
            return None
        if turns is not None:
            opposing = turns[1 - directions.index(candidate)]
            if len(opposing) >= 2 or any(score >= .90 and support >= 2 for _, score, _, _, _, support in opposing):
                return None
        if has_upright_caption():
            return None
        return candidate

    if side_count < 18 or side_count < 1.30 * native_count:
        return structural()
    cjk, latin = _fonts(index)
    turns = _orientation_votes(sideways, cjk, latin)
    winner = max(range(2), key=lambda i: (len(turns[i]), sum(item[1] for item in turns[i])))
    candidates = turns[winner]
    opposite = turns[1 - winner]
    distinct = len({item[0] for item in candidates})
    repeated = any(len(word) >= 2 and confidence >= .865 and passes >= 2
                   for word, confidence, _, _, _, passes in candidates)
    sideways_layout = side_count >= max(45, 2.30 * native_count)
    one_caption = not opposite and repeated and sideways_layout
    if not (one_caption or (distinct >= 2 and len(candidates) >= len(opposite) + 2)):
        if not sideways_layout:
            return structural(turns)
        from screen_landscape_mode_rail import marked_mode_rail_actions

        matches = [i for i, direction in enumerate(directions) if marked_mode_rail_actions(cv2.rotate(frame, direction))]
        if len(matches) == 1 and not turns[1 - matches[0]]:
            winner, = matches
        elif not matches and not opposite and not has_upright_caption():
            if _focused_caption(frame, candidates, winner, cjk, latin):
                return directions[winner]
            # A bounded filter panel and checked glyphs can establish direction
            # when its small captions cannot provide a reliable transcript.
            from screen_game_filters import game_filter_layout

            supported = [direction for i, direction in enumerate(directions)
                         if not turns[1 - i] and game_filter_layout(cv2.rotate(frame, direction), camera=True) is not None]
            if supported:
                return supported[0] if len(supported) == 1 else None
            return structural(turns)
        else:
            return structural(turns) if not matches else None
    if has_upright_caption():
        return None
    return directions[winner]
