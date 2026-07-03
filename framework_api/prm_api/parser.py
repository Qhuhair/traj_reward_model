"""PRM 模型输出解析。"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_score_response(raw_text: str, candidate_count: int) -> tuple[list[float], int, str]:
    """解析 PRM 输出；失败时返回保守的 0 分列表。"""
    try:
        data = _load_json_object(raw_text)
        scores = _normalize_scores(data.get("scores"), candidate_count)
        best_index = _normalize_best_index(data.get("best_index"), scores)
        reason = str(data.get("reason") or "")
        return scores, best_index, reason
    except (TypeError, ValueError, json.JSONDecodeError):
        return [0.0 for _ in range(candidate_count)], 0, "PRM 输出解析失败，返回保守评分。"


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
        raise ValueError("PRM 输出不是 JSON 对象。")
    return data


def _normalize_scores(values: Any, candidate_count: int) -> list[float]:
    """把模型分数裁剪到候选动作数量和 0-1 区间。"""
    if not isinstance(values, list):
        raise ValueError("PRM 输出缺少 scores。")
    scores = [min(1.0, max(0.0, float(value))) for value in values[:candidate_count]]
    while len(scores) < candidate_count:
        scores.append(0.0)
    return scores


def _normalize_best_index(value: Any, scores: list[float]) -> int:
    """使用模型 best_index；无效时退化为最高分索引。"""
    if not scores:
        return 0
    try:
        index = int(value)
    except (TypeError, ValueError):
        return max(range(len(scores)), key=lambda item: scores[item])
    if 0 <= index < len(scores):
        return index
    return max(range(len(scores)), key=lambda item: scores[item])
