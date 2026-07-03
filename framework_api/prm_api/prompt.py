"""PRM 评分 Prompt 构造。"""

from __future__ import annotations

import json

from .schema import PRMMode, PRMScoreRequest


def build_score_prompt(request: PRMScoreRequest, history_steps: list[dict]) -> str:
    """根据评分模式构造文本输入。"""
    payload = {
        "goal": request.goal,
        "step_index": request.step_index,
        "screen_size": request.screen_size,
        "mode": request.mode.value,
        "ui_elements": request.ui_elements,
        "history": _history_for_mode(request.mode, history_steps),
        "candidates": [candidate.model_dump() for candidate in request.candidates],
    }
    return "\n".join(
        [
            "你是 AndroidWorld 轨迹的过程奖励模型，只负责评价候选动作质量。",
            "请结合任务目标、当前屏幕元素、历史步骤和候选动作，为每个候选动作给出 0 到 1 的分数。",
            "评分标准：1.0 表示明确推进或完成任务；0.8 表示大概率正确；0.6 表示部分相关但不稳定；0.3 表示弱相关或可能走偏；0.0 表示明显错误。",
            "只返回严格 JSON，不要 Markdown，不要额外解释。",
            "返回格式必须为：",
            '{"scores":[0.8,0.3],"best_index":0,"reason":"选择第 0 个动作的原因"}',
            "当前评分输入如下：",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


def include_image(mode: PRMMode) -> bool:
    """判断当前模式是否需要把截图传给模型。"""
    return mode in {PRMMode.MULTIMODAL_STEP, PRMMode.MULTIMODAL_WINDOW}


def _history_for_mode(mode: PRMMode, history_steps: list[dict]) -> list[dict]:
    """单步模式不读取窗口历史，窗口模式保留归档历史。"""
    if mode in {PRMMode.TEXT_STEP, PRMMode.MULTIMODAL_STEP}:
        return []
    return history_steps
