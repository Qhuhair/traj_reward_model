from dataclasses import asdict, dataclass, field


@dataclass
class MemoryConfig:
    """分层记忆配置，默认关闭，避免改变现有实验口径。"""

    enabled: bool = False
    recent_window_size: int = 3
    history_summary_max_chars: int = 1200
    recent_step_max_chars: int = 600
    include_previous_scores: bool = True
    include_current_images: bool = True
    summarizer: str = "llm"
    summarizer_model: str = "codex_mm_baseline"
    summarizer_prompt: str = "RRM_HistorySummary"

    @classmethod
    def from_dict(cls, data: dict | None):
        values = data or {}
        known = {k: values[k] for k in cls.__dataclass_fields__ if k in values}
        return cls(**known)


@dataclass
class HistorySummary:
    """历史摘要及其覆盖范围。"""

    text: str
    start_step_idx: int | None
    end_step_idx: int | None
    source_step_indices: list[int] = field(default_factory=list)
    summarizer: str = "llm"

    def to_dict(self):
        return asdict(self)


@dataclass
class RecentStep:
    """最近窗口中的单步文本信息，不包含图片。"""

    step_idx: int
    action: str
    subgoal: str
    before: str
    after: str
    score: float | None = None
    critique: str = ""
    is_current: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class MemoryContext:
    """单个当前步骤的完整记忆上下文。"""

    step_idx: int
    history: HistorySummary
    recent_steps: list[RecentStep]
    current_image_before: str = ""
    current_image_after: str = ""

    def to_dict(self):
        return {
            "step_idx": self.step_idx,
            "history": self.history.to_dict(),
            "recent_steps": [s.to_dict() for s in self.recent_steps],
            "current_image_before": self.current_image_before,
            "current_image_after": self.current_image_after,
        }
