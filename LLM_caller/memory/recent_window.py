from .schema import MemoryConfig, RecentStep


def select_recent_steps(steps: list[dict], current_index: int,
                        scores: list[dict] | None,
                        config: MemoryConfig) -> list[RecentStep]:
    """选择当前步及之前窗口步骤，禁止包含未来步骤。"""
    start = max(0, current_index - config.recent_window_size + 1)
    selected = []
    for idx in range(start, current_index + 1):
        step = steps[idx]
        score = _score_at(scores, idx) if config.include_previous_scores else None
        selected.append(RecentStep(
            step_idx=step.get("step_idx", idx + 1),
            action=step.get("action_desc") or step.get("action", ""),
            subgoal=step.get("subgoal_text", ""),
            before=_trim(step.get("state_desc_before", ""), config.recent_step_max_chars),
            after=_trim(step.get("state_desc_after", ""), config.recent_step_max_chars),
            score=score.get("score") if score else None,
            critique=_trim(score.get("critique", ""), 240) if score else "",
            is_current=(idx == current_index),
        ))
    return selected


def format_recent_steps(recent_steps: list[RecentStep]) -> str:
    """将最近窗口转换为 Prompt 文本。"""
    if not recent_steps:
        return "无最近步骤。"
    lines = []
    for step in recent_steps:
        marker = "（当前步骤）" if step.is_current else ""
        lines.extend([
            f"- 步骤{step.step_idx}{marker}",
            f"  子目标：{step.subgoal or '无'}",
            f"  动作：{step.action or '无'}",
            f"  执行前：{step.before or '无'}",
            f"  执行后：{step.after or '无'}",
        ])
        if step.score is not None:
            lines.append(f"  已有评分：{step.score}")
        if step.critique:
            lines.append(f"  已有评价：{step.critique}")
    return "\n".join(lines)


def _score_at(scores: list[dict] | None, index: int) -> dict | None:
    if not scores or index >= len(scores):
        return None
    return scores[index]


def _trim(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "..."
