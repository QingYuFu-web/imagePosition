"""Read the same short printed CTA across separately owned, opposite-polarity buttons."""

from itertools import combinations
from statistics import median

import cv2
import numpy as np

from screen_caption_transcript import _cjk_parts


# Paired Simplified/Traditional spellings are equivalent for output naming.
# This is deliberately a narrow inventory of conventional two-character UI commands.
_VOCABULARY = (
    "确认 確認 确定 確定 取消 返回 关闭 關閉 打开 打開 继续 繼續 重试 重試 保存 放弃 放棄 应用 應用 重置 编辑 編輯 删除 刪除 "
    "修改 启动 啟動 开始 開始 完成 停止 暂停 暫停 搜索 搜尋 查询 查詢 筛选 篩選 排序 刷新 选择 選擇 登录 登錄 注册 註冊 "
    "分享 发送 發送 提交 提取 投递 投遞 交付 交货 交貨 交卷 交单 交單 交费 交費 交款 交纳 交納 交账 交賬 交钱 交錢 交租 "
    "支付 付款 贷款 貸款 借贷 借貸 收款 收货 收貨 发货 發貨 送货 送貨 退货 退貨 补货 補貨 换货 換貨 进货 進貨 卖货 賣貨 "
    "售货 售貨 供货 供貨 回收 兑换 兌換 领取 領取 领奖 領獎 购买 購買 出售 换取 換取 升级 升級 解锁 解鎖 开启 開啟 装备 裝備 "
    "入库 入庫 出库 出庫 配送 接单 接單 退还 退還 排队 排隊 招募 退出 进入 進入 确收 確收 预约 預約 打印 下载 下載 上传 上傳"
).split()
_SIMPLE = str.maketrans("確關開繼續試棄應編輯刪啟動暫詢篩選擇錄註冊發遞貨單費納賬錢貸補換進賣兌領獎購級鎖裝庫還隊預載傳",
                        "确关开继续试弃应编辑删启动暂询筛选择录注册发递货单费纳账钱贷补换进卖兑领奖购级锁装库还队预载传")


def _glyph_samples(frame, box):
    x, y, right, bottom = map(round, box)
    h, w = bottom - y, right - x
    height, width = frame.shape[:2]
    if not (0 <= x < right <= width and 0 <= y < bottom <= height and min(h, w) >= 28):
        return None
    dx, dy = round(.18 * w), round(.22 * h)
    patch = frame[y + dy:bottom - dy, x + dx:right - dx]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    low, base, high = np.percentile(gray, (2.5, 50, 97.5))
    if max(base - low, high - base) < 43:
        return None
    bright = high - base > base - low
    views = (gray, patch.min(axis=2)) if bright else (255 - gray, 255 - patch.max(axis=2))
    records = {}
    for view in views:
        ground, paint = np.percentile(view, (45, 98))
        if paint - ground < 46:
            continue
        for cut in (.23, .32, .42, .54):
            mask = np.uint8(view >= ground + cut * (paint - ground))
            left, top, w_ink, h_ink = cv2.boundingRect(mask)
            if (not .185 * h <= h_ink <= .47 * h or not .22 * w <= w_ink <= .54 * w
                    or abs(left + w_ink / 2 - patch.shape[1] / 2) > .075 * w
                    or abs(top + h_ink / 2 - patch.shape[0] / 2) > .095 * h):
                continue
            parts = _cjk_parts(mask[top:top + h_ink, left:left + w_ink], h_ink)
            if len(parts) == 2:
                glyphs = tuple(part[-1] for part in parts)
                records[tuple((g.shape, g.tobytes()) for g in glyphs)] = glyphs
    return (bright, tuple(records.values())) if len(records) >= 2 else None


def _matching(samples, other):
    def normalize(records):
        return [tuple(cv2.resize(g.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA) >= .50 for g in glyphs)
                for glyphs in records]

    for a in normalize(samples):
        for b in normalize(other):
            if all((left & right).sum() >= .72 * (left | right).sum() for left, right in zip(a, b, strict=True)):
                return True
    return False


def read_lowres_action_group(frame, boxes, index):
    """Return a shared label only from consistent glyphs with visibly different polarity."""
    if index is None or len(boxes) < 3 or any(
            min(r, right) > max(l, left) and min(b, bottom) > max(t, top)
            for (l, t, r, b), (left, top, right, bottom) in combinations(boxes, 2)):
        return None
    reads = [_glyph_samples(frame, box) for box in boxes]
    if any(record is None for record in reads):
        return None
    light = [samples for bright, samples in reads if bright]
    dark = [samples for bright, samples in reads if not bright]
    if (len(light) < 2 or len(dark) != 1 or not all(_matching(dark[0], group) for group in light)
            or not all(_matching(light[0], other) for other in light[1:])):
        return None

    cache = {}
    by_control = []
    for bright, samples in reads:
        words = {}
        for glyphs in samples:
            ranks = []
            for glyph in glyphs:
                key = glyph.shape, glyph.tobytes()
                if key not in cache:
                    cache[key] = {choice["text"]: choice["distance"] for choice in index.rank(glyph, alphabet="cjk", limit=44)}
                ranks.append(cache[key])
            if any(not row for row in ranks):
                continue
            for word in _VOCABULARY:
                scores = [row.get(ch, 1.) for ch, row in zip(word, ranks, strict=True)]
                if (max(scores) <= (.281 if bright else .251)
                        and any(score - min(row.values()) <= .016 for score, row in zip(scores, ranks, strict=True))):
                    key = word.translate(_SIMPLE)
                    words[key] = min(words.get(key, 1.), sum(scores) / 2)
        by_control.append(words)
    evidence = set.intersection(*(set(words) for words in by_control))
    if not evidence:
        return None
    winner = min(evidence, key=lambda name: median(words[name] for words in by_control))
    # All three copies must independently favor the same term. A rival may
    # fit one raster, but needs support from both foreground polarities.
    margins = [min((score for name, score in words.items() if name != winner), default=1.) - words[winner]
               for words in by_control]
    if min(margins) < .008 or median(margins) < .017:
        return None
    return winner
