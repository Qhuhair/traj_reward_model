import os
import sys

from .schema import HistorySummary, MemoryConfig

CALLER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CALLER_DIR not in sys.path:
    sys.path.insert(0, CALLER_DIR)

from caller import LLMCaller


class HistorySummarizer:
    """历史步骤摘要器，只基于文本生成摘要，不读取图片。"""

    def __init__(self, config: MemoryConfig, caller: LLMCaller | None = None):
        self.config = config
        self.caller = caller
        self._cache: dict[tuple[int, int], HistorySummary] = {}

    def summarize(self, task: str, steps: list[dict], current_index: int,
                  scores: list[dict] | None = None) -> HistorySummary:
        """摘要 steps[0 : current_index - window + 1]，不包含当前窗口和未来步骤。"""
        end_exclusive = current_index - self.config.recent_window_size + 1
        if end_exclusive <= 0:
            return HistorySummary("", None, None, [], self.config.summarizer)

        source_steps = steps[:end_exclusive]
        source_indices = [s.get("step_idx", i + 1) for i, s in enumerate(source_steps)]
        cache_key = (source_indices[0], source_indices[-1])
        if cache_key in self._cache:
            return self._cache[cache_key]

        history_text = format_history_steps(source_steps, scores, self.config)
        summary_text = self._summarize_with_llm(task, history_text)
        summary = HistorySummary(
            text=_trim(summary_text, self.config.history_summary_max_chars),
            start_step_idx=source_indices[0],
            end_step_idx=source_indices[-1],
            source_step_indices=source_indices,
            summarizer=self.config.summarizer,
        )
        self._cache[cache_key] = summary
        return summary

    def _summarize_with_llm(self, task: str, history_text: str) -> str:
        if not history_text.strip():
            return ""
        if self.config.summarizer != "llm":
            raise ValueError(f"Unsupported summarizer: {self.config.summarizer}")
        caller = self.caller or LLMCaller(
            model=self.config.summarizer_model,
            prompt=self.config.summarizer_prompt,
        )
        result = caller.call(
            parse_response=False,
            task_desc=task,
            history_steps_text=history_text,
            history_summary_max_chars=self.config.history_summary_max_chars,
            image_before="",
            image_after="",
        )
        return _summary_text(result)


def format_history_steps(steps: list[dict], scores: list[dict] | None,
                         config: MemoryConfig) -> str:
    """将历史步骤格式化为摘要模型输入，严格不包含图片路径。"""
    lines = []
    for i, step in enumerate(steps):
        idx = step.get("step_idx", i + 1)
        lines.extend([
            f"步骤{idx}:",
            f"- 子目标：{step.get('subgoal_text', '')}",
            f"- 动作：{step.get('action_desc') or step.get('action', '')}",
            f"- 目标页面/结果：{step.get('to_node_desc', '')}",
            f"- 是否回退：{bool(step.get('is_backtrack', False))}",
        ])
        score = scores[i] if scores and i < len(scores) else None
        if score and config.include_previous_scores:
            lines.append(f"- 已有评分：{score.get('score', 'N/A')}")
            critique = _trim(score.get("critique", ""), 240)
            if critique:
                lines.append(f"- 已有评价：{critique}")
    return "\n".join(lines)


def _summary_text(result) -> str:
    if isinstance(result, dict):
        parts = [
            result.get("think", ""),
            result.get("critique", ""),
        ]
        text = "\n".join(p for p in parts if p and p != "N/A")
        return text or str(result)
    return str(result)


def _trim(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "..."
