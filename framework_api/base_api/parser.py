"""Base 模型输出解析。"""

from __future__ import annotations

import json
import re
from typing import Any

from .schema import ActionCandidate, BaseActResponse


def parse_act_response(raw_text: str, n_candidates: int) -> BaseActResponse:
    """把模型文本解析成候选动作；失败时返回 wait 保底动作。"""
    try:
        data = _load_json_object(raw_text)
        candidates = _normalize_candidates(data, n_candidates)
    except (TypeError, ValueError, json.JSONDecodeError):
        candidates = [_fallback_candidate(raw_text)]
    return BaseActResponse(candidates=candidates)


def _load_json_object(raw_text: str) -> dict[str, Any]:
    """优先解析完整 JSON，否则提取首个 JSON 对象。"""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("模型输出不是 JSON 对象。")
    return data


def _normalize_candidates(data: dict[str, Any], n_candidates: int) -> list[ActionCandidate]:
    """校验并裁剪候选动作列表。"""
    items = data.get("candidates")
    if not isinstance(items, list) or not items:
        raise ValueError("模型输出缺少 candidates。")
    candidates: list[ActionCandidate] = []
    for item in items[:n_candidates]:
        if not isinstance(item, dict):
            continue
        action = _normalize_action(item.get("action"))
        thought = item.get("thought") or item.get("reason") or ""
        candidates.append(ActionCandidate(action=action, thought=str(thought)))
    if not candidates:
        raise ValueError("没有有效候选动作。")
    return candidates


def _normalize_action(action: Any) -> dict[str, Any]:
    """把旧式 type 字段规范成 AndroidWorld 的 action_type。"""
    if not isinstance(action, dict):
        raise ValueError("候选动作不是对象。")
    normalized = {key: value for key, value in action.items() if value is not None}
    if "action_type" not in normalized and "type" in normalized:
        normalized["action_type"] = normalized.pop("type")
    if "action_type" not in normalized:
        raise ValueError("候选动作缺少 action_type。")
    return normalized


def _fallback_candidate(raw_text: str) -> ActionCandidate:
    """解析失败时保持评测循环可继续运行。"""
    snippet = " ".join(raw_text.strip().split())[:120]
    thought = f"模型输出解析失败，执行 wait。原始片段：{snippet}"
    return ActionCandidate(action={"action_type": "wait"}, thought=thought)
