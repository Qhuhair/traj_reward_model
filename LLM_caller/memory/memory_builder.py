from .history_summarizer import HistorySummarizer
from .prompt_context import build_prompt_values
from .recent_window import select_recent_steps
from .schema import MemoryConfig, MemoryContext


class MemoryBuilder:
    """构建单步分层记忆上下文。"""

    def __init__(self, config: MemoryConfig | dict | None = None,
                 summarizer: HistorySummarizer | None = None):
        self.config = config if isinstance(config, MemoryConfig) else MemoryConfig.from_dict(config)
        self.summarizer = summarizer or HistorySummarizer(self.config)

    def build_context(self, std_data: dict, current_index: int,
                      scores: list[dict] | None = None) -> MemoryContext:
        """构建当前步上下文，current_index 为 0 基索引。"""
        steps = std_data.get("steps", [])
        if current_index < 0 or current_index >= len(steps):
            raise IndexError(f"current_index out of range: {current_index}")

        current = steps[current_index]
        history = self.summarizer.summarize(
            std_data.get("task", ""),
            steps,
            current_index,
            scores,
        )
        recent_steps = select_recent_steps(steps, current_index, scores, self.config)
        return MemoryContext(
            step_idx=current.get("step_idx", current_index + 1),
            history=history,
            recent_steps=recent_steps,
            current_image_before=_current_image(current, "before", self.config),
            current_image_after=_current_image(current, "after", self.config),
        )

    def build_prompt_values(self, std_data: dict, current_index: int,
                            scores: list[dict] | None = None) -> dict:
        """直接返回可传给 LLMCaller.call 的参数。"""
        context = self.build_context(std_data, current_index, scores)
        return build_prompt_values(std_data, context)

    def context_to_prompt_values(self, std_data: dict, context: MemoryContext) -> dict:
        """将已构建的上下文转换为 Prompt 参数，避免重复生成摘要。"""
        return build_prompt_values(std_data, context)


def _current_image(step: dict, side: str, config: MemoryConfig) -> str:
    if not config.include_current_images:
        return ""
    annotated_key = f"image_{side}_annotated"
    raw_key = f"image_{side}"
    return step.get(annotated_key) or step.get(raw_key, "")
