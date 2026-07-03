"""Base 策略模型的 Prompt 构造。"""

from __future__ import annotations

import json

from .schema import BaseActRequest


VALID_ACTIONS = (
    "click, double_tap, scroll, swipe, input_text, navigate_home, "
    "navigate_back, keyboard_enter, open_app, status, wait, long_press, "
    "answer, unknown"
)


def build_action_prompt(request: BaseActRequest) -> str:
    """把 AndroidWorld 观测压缩成只要求生成动作的文本。"""
    observation = {
        "goal": request.goal,
        "step_index": request.step_index,
        "screen_size": request.screen_size,
        "ui_elements": request.ui_elements,
        "history": request.history,
        "n_candidates": request.n_candidates,
    }
    return "\n".join(
        [
            "你是 AndroidWorld 手机任务的策略模型，只负责提出下一步候选动作。",
            "不要执行动作，不要评价动作，只返回严格 JSON。",
            f"合法 action_type 包括：{VALID_ACTIONS}。",
            "优先使用 AndroidWorld JSONAction 字段，例如：",
            '{"action_type":"click","index":0}',
            '{"action_type":"input_text","index":1,"text":"Alice"}',
            '{"action_type":"navigate_back"}',
            '{"action_type":"status","goal_status":"task_complete"}',
            "返回格式必须为：",
            '{"candidates":[{"action":{"action_type":"wait"},"thought":"原因"}]}',
            "候选数量不要超过 n_candidates。",
            "当前观测如下：",
            json.dumps(observation, ensure_ascii=False, indent=2),
        ]
    )
