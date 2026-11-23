"""Verify an empty multi-line body below a writing toolbar and above option rows."""

from dataclasses import dataclass

import cv2
import numpy as np

from screen_colored_settings_panel import _printed
from screen_component_captions import _bands


_TITLES = ("发表文字", "發表文字", "撰写帖子", "編輯貼文", "新建帖子", "新建文章", "编辑文章", "撰写文章",
           "发布动态", "發佈動態", "分享动态", "编辑帖子", "New Post", "Create Post", "Write a Post", "Edit Post",
           "New Article", "Write an Article", "Edit Draft")
_OTHER_TITLES = ("发现", "朋友圈", "照片", "新消息", "详情", "评论", "设置", "个人资料", "位置",
                 "Discover", "Gallery", "New Message", "Settings", "Profile", "Comments", "Details")
_PROMPTS = ("这一刻的想法...", "這一刻的想法...", "写点什么...", "寫點什麼...", "写下你的想法...",
            "分享你的想法...", "说点什么...", "输入正文", "請輸入正文", "请输入正文", "添加正文", "编辑内容",
            "Write something...", "Share your thoughts...", "What's on your mind?", "Write your post...",
            "Type your text...", "Enter body text", "Add body text")
_OTHER_PROMPTS = ("请输入密码", "分享文章", "请填写理由", "输入评论", "添加标题", "添加留言", "搜索",
                  "今天心情", "保存草稿", "Write a comment", "Add a title", "Type a message", "Location")
_EQUIVALENT_TITLES = (("发表文字", "發表文字"), ("发布动态", "發佈動態"), ("编辑帖子", "編輯貼文"))
_EQUIVALENT_PROMPTS = (("这一刻的想法...", "這一刻的想法..."), ("写点什么...", "寫點什麼..."),
                       ("请输入正文", "請輸入正文", "输入正文"))


@dataclass(frozen=True, slots=True)
class EmptyDraftField:
    placeholder: str
    placeholder_box: tuple[int, int, int, int]
    body_box: tuple[int, int, int, int]
    toolbar_box: tuple[int, int, int, int]
    header_title: str
    title_box: tuple[int, int, int, int]


def _seam(gray, paper, after):
    width = gray.shape[1]
    start, stop = after + round(.27 * width), min(len(gray) - 4, after + round(.85 * width))
    stripes = gray[start - 3:stop + 3, round(.075 * width):round(.93 * width):3]
    medians = np.median(stripes, axis=1)
    choices = np.flatnonzero((medians[3:-3] <= paper - 3) & (medians[:-6] - medians[3:-3] >= 3)
                             & (medians[6:] - medians[3:-3] >= 3))
    if not len(choices):
        return None
    return start + int(choices[0])


def _paired_rows(gray, seam):
    width = gray.shape[1]
    start, stop = seam + round(.012 * width), seam + round(.42 * width)
    icon = gray[start:stop, round(.045 * width):round(.16 * width)] < 205
    caption = gray[start:stop, round(.16 * width):round(.55 * width)] < 205
    runs = _bands(caption.sum(axis=1) >= max(3, round(.017 * width)))
    centers = []
    for first, last in runs:
        length = last - first
        if not .027 * width <= length <= .071 * width:
            continue
        x, _, span, _ = cv2.boundingRect(np.uint8(caption[first:last]))
        if x > .083 * width or not .082 * width <= span <= .385 * width:
            continue
        glyph = icon[max(0, first - length // 3):last + length // 3]
        if cv2.countNonZero(np.uint8(glyph)) >= max(4, round(.23 * length**2)):
            centers.append(start + (first + last) / 2)
    return (len(centers) >= 2 and .034 * width <= centers[0] - seam <= .124 * width
            and .084 * width <= centers[1] - centers[0] <= .207 * width)


def empty_draft_field(frame, index):
    """Return visible field ownership only when the writing body is visibly empty."""
    height, width = frame.shape[:2]
    if index is None or width < 224 or not 1.47 * width <= height <= 2.92 * width:
        return None
    # Full-width blank paper is a cheap screen-wide veto before loading a glyph bank.
    sample = frame[round(.34 * width):round(.48 * width):3, round(.074 * width):round(.932 * width):4]
    if (sample.min(axis=2) >= 235).mean() < .987:
        return None
    stop = min(height, round(1.70 * width))
    gray = cv2.cvtColor(frame[:stop], cv2.COLOR_BGR2GRAY)
    toolbar = (round(.232 * width), round(.081 * width), round(.768 * width), round(.204 * width))
    title = _printed(gray, toolbar, index, vocabulary=(*_TITLES, *_OTHER_TITLES), light=False, with_box=True,
                     minimum_score=.81, minimum_margin=.060, equivalent=_EQUIVALENT_TITLES)
    if title is None or title[0] not in _TITLES:
        return None
    write_top = title[1][3] + round(.029 * width)
    area = (round(.024 * width), write_top, round(.65 * width), write_top + round(.136 * width))
    prompt = _printed(gray, area, index, vocabulary=(*_PROMPTS, *_OTHER_PROMPTS), light=False, with_box=True,
                      minimum_score=.80, minimum_margin=.061, equivalent=_EQUIVALENT_PROMPTS)
    if prompt is None or prompt[0] not in _PROMPTS:
        return None
    text, (left, top, right, bottom) = prompt
    if not .028 * width <= left <= .165 * width or top - title[1][3] > .129 * width:
        return None
    paper = int(np.median(sample))
    ink = gray[top:bottom, left:right]
    # A gray literal prompt is a placeholder here only when its paper remains empty.
    level = float(np.percentile(ink, 10))
    if not 36 <= paper - level <= 101 or np.min(ink) < paper - 138:
        return None
    seam = _seam(gray, paper, bottom)
    if seam is None or not _paired_rows(gray, seam):
        return None
    x0, x1 = round(.028 * width), round(.945 * width)
    body = frame[write_top:seam - 2, x0:x1]
    clear = body.min(axis=2) >= paper - 14
    pad = max(2, round(.009 * width))
    clear[max(0, top - pad - write_top):bottom + pad - write_top,
          max(0, left - pad - x0):right + pad - x0] = True
    if not clear.all():
        # A few lossy-image pixels can differ. Any connected visible ink is withheld.
        if clear.mean() < .998:
            return None
        count, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(~clear))
        if count > 1 and stats[1:, cv2.CC_STAT_AREA].max() > max(3, round(.000032 * width**2)):
            return None
    return EmptyDraftField(text, (left, top, right, bottom), (x0, write_top, x1, seam), toolbar, *title)
