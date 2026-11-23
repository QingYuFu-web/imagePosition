"""Plan an optional focused online locate request from strong existing evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import re
from typing import Literal
import unicodedata

from offline_target import _COMPOUND, _DIRECT, _LOCATION, _NEGATIVE, _WRAPPER, _name


_INDIVIDUAL_LAYOUT = "individual-crops/1"
_CANDIDATE_ID = re.compile(r"E[1-9][0-9]*")
_REQUEST_ID = re.compile(r"[0-9a-f]{24}")
_CONTROL_SUFFIX = re.compile(r"(?:按钮|图标|入口|选项)$")
_INPUT_COMMAND = re.compile(r"^(?:找到|查找|寻找|定位|点击屏幕上的|点击|点一下|点选|选择|轻点|点)(.+)$")
_BODY_INPUT = re.compile(r"^(?P<context>[\u3400-\u9fff]{0,10}?)(?:的)?(?:文字输入框|文本输入框|正文输入框|多行文本框|多行输入框|编辑正文的输入框)$")
_INPUT_QUALIFIERS = re.compile(r"密码|口令|搜索|查找|邮箱|邮件地址|账号|帐号|用户名|验证码|手机号|电话号码|金额|标题|收件人|第[一二三四五六七八九十]|上面|下面|左边|右边|左侧|右侧")
_ENGLISH_INPUT = re.compile(r"^(?:find|locate|tap|click|select)\s+(?:the\s+)?(?:text\s+(?:input|field)|multiline\s+(?:editor|field)|body\s+field)$")


@dataclass(frozen=True)
class LocateShortlist:
    """A single attested target and its individual image in an unchanged packet."""

    element_id: str
    attachments: tuple[str, ...]
    evidence: Literal["visible_ocr", "manual_review", "verified_text_input"]


def _positive_query(task):
    if not isinstance(task, str) or not 1 <= len(task) <= 300:
        return None
    query = unicodedata.normalize("NFKC", task).casefold().strip().rstrip("。！!?？").strip()
    if not query or _NEGATIVE.search(query) or _COMPOUND.search(query):
        return None
    while (prefix := _WRAPPER.match(query)) is not None:
        query = query[prefix.end():]
    return query or None


def _literal_names(query):
    direct = _DIRECT.fullmatch(query)
    if not direct:
        return set()
    label = direct[1].strip()
    if _LOCATION.match(label):
        return set()
    exact = _name(label)
    return {exact, _CONTROL_SUFFIX.sub("", exact)} - {""}


def _explicit_text_input(query):
    if _ENGLISH_INPUT.fullmatch(query):
        return True
    command = _INPUT_COMMAND.fullmatch(query)
    if not command:
        return False
    target = command[1].strip()
    if _LOCATION.match(target) or _INPUT_QUALIFIERS.search(target):
        return False
    return target in {"输入框", "当前输入框", "当前的输入框"} or bool(_BODY_INPUT.fullmatch(target))


def _box_contains(outer, inner):
    for box in (outer, inner):
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or not all(type(value) in (int, float) and isfinite(value) for value in box)
                or not (box[0] < box[2] and box[1] < box[3])):
            return False
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def _manual_match(candidate, query, names):
    sample = candidate.get("reviewed_sample")
    if not isinstance(sample, dict) or sample.get("origin", "manual") != "manual" or sample.get("match") != "exact":
        return False
    purpose = _positive_query(sample.get("purpose"))
    if purpose is None:
        return False
    return _name(purpose) == _name(query) or bool(_literal_names(purpose) & names)


def plan_locate_shortlist(request, *, ocr_only=False) -> LocateShortlist | None:
    """Find a unique exact label or manual match, without modifying the full request.

    The caller must validate a response for the original request and fall back to the
    full packet whenever the focused request does not select this exact target.
    """
    if (not isinstance(request, dict) or request.get("mode") != "locate"
            or request.get("sheet_layout") != _INDIVIDUAL_LAYOUT
            or "follow_up_of" in request or "context" in request
            or not isinstance(scope := request.get("selection"), dict)
            or scope.get("element_ids") is not None or scope.get("region") is not None
            or not isinstance(identity := request.get("request_id"), str) or not _REQUEST_ID.fullmatch(identity)
            or not isinstance(attachments := request.get("attachments"), list) or len(attachments) < 2
            or not isinstance(candidates := request.get("candidates"), list)
            or not isinstance(words := request.get("visible_text", []), list)
            or not all(isinstance(row, dict) for row in candidates + words)
            or not (query := _positive_query(request.get("task")))):
        return None

    names = _literal_names(query)
    label_words = {word["id"]: word["box"] for word in words
                   if word.get("type") == "text" and isinstance(word.get("id"), str)
                   and isinstance(word.get("label"), str) and _name(word["label"]) in names and "box" in word}
    matches = []
    input_candidates = [row for row in candidates if row.get("control_role") == "text_input"]
    proven_input = (input_candidates[0] if not ocr_only and _explicit_text_input(query) and len(input_candidates) == 1
                    and input_candidates[0].get("input_state") == "empty" else None)
    for candidate in candidates:
        direct = candidate.get("text_ids", ())
        text_evidence = {key for key in direct if isinstance(key, str) and key in label_words
                         and _box_contains(candidate.get("box"), label_words[key])} if isinstance(direct, (list, tuple)) else set()
        manual = not ocr_only and _manual_match(candidate, query, names)
        if text_evidence or manual or candidate is proven_input:
            basis = "manual_review" if manual else "visible_ocr" if text_evidence else "verified_text_input"
            matches.append((candidate, text_evidence, basis))
    if len(matches) != 1:
        return None
    candidate, text_evidence, basis = matches[0]
    # Identical labels in distinct visible regions can describe more than one owner.
    if set(label_words) - text_evidence:
        return None
    if (any(candidate.get(flag) for flag in ("disabled", "review_only", "partial_crop", "child_controls",
                                             "media_thumbnail", "selection_marker"))
            or not isinstance(identity_point := candidate.get("id"), str) or not _CANDIDATE_ID.fullmatch(identity_point)
            or not isinstance(crop := candidate.get("crop"), dict)):
        return None
    attachment = f"screen-{identity}.ai-{identity_point}.jpg"
    if crop.get("file") != attachment or attachment not in attachments:
        return None
    return LocateShortlist(identity_point, (attachment,), basis)
