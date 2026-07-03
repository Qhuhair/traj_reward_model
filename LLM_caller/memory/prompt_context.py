from .recent_window import format_recent_steps
from .schema import MemoryContext


def build_prompt_values(std_data: dict, context: MemoryContext) -> dict:
    """把 MemoryContext 转换为分层记忆 Prompt 占位符。"""
    current = _current_step(context)
    return {
        "task_desc": std_data.get("task", ""),
        "curr_subgoal": current.subgoal,
        "history_summary": context.history.text or "无历史步骤摘要。",
        "recent_steps_detail": format_recent_steps(context.recent_steps),
        "step_idx": current.step_idx,
        "curr_action": current.action,
        "curr_element_id": _element_id(std_data, current.step_idx),
        "curr_state_before": current.before,
        "curr_state_after": current.after,
        "image_before": context.current_image_before,
        "image_after": context.current_image_after,
    }


def _current_step(context: MemoryContext):
    for step in context.recent_steps:
        if step.is_current:
            return step
    return context.recent_steps[-1]


def _element_id(std_data: dict, step_idx: int) -> str:
    for step in std_data.get("steps", []):
        if step.get("step_idx") == step_idx:
            return step.get("element_id", "")
    return ""
