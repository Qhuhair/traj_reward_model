"""AndroidWorld 到 crossAPP 窗口 Prompt 的适配。"""

from __future__ import annotations

import json


def build_crossapp_kwargs(request, history_steps, index, candidate) -> dict:
    """把 AndroidWorld observation 适配成 LLMCaller Prompt 参数。"""
    return {
        "task_desc": request.goal,
        "window_context": format_window(history_steps),
        "step_index": request.step_index,
        "screen_size": request.screen_size,
        "ui_elements": json.dumps(request.ui_elements, ensure_ascii=False, indent=2),
        "candidate_index": index,
        "candidate_action": json.dumps(candidate.action, ensure_ascii=False) if candidate else "",
        "candidate_thought": candidate.thought or "" if candidate else "",
    }


def render_prompt(template: str, request, history_steps) -> str:
    """保存用于人工复查的候选动作 Prompt 集合。"""
    prompts = []
    for index, candidate in enumerate(request.candidates):
        kwargs = build_crossapp_kwargs(request, history_steps, index, candidate)
        prompts.append(template.format(**kwargs))
    return "\n\n--- candidate boundary ---\n\n".join(prompts)


def format_window(history_steps: list[dict]) -> str:
    """把已保存步骤压缩成 crossAPP 风格的窗口文本。"""
    if not history_steps:
        return "（轨迹起点，无历史步骤）"
    return "\n".join(json.dumps(_compact_step(item), ensure_ascii=False) for item in history_steps)


def _compact_step(item: dict) -> dict:
    """只保留窗口判断需要的历史字段。"""
    return {
        "step_index": item.get("step_index"),
        "goal": item.get("goal"),
        "history": item.get("history", [])[-2:],
        "candidates": item.get("candidates", []),
    }
