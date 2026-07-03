"""PRM API 的请求和响应结构。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PRMMode(str, Enum):
    """PRM 支持的评分输入方式。"""

    TEXT_STEP = "text_step"
    TEXT_WINDOW = "text_window"
    MULTIMODAL_STEP = "multimodal_step"
    MULTIMODAL_WINDOW = "multimodal_window"
    CROSSAPP_MULTIMODAL_WINDOW = "crossapp_multimodal_window"


class CandidateAction(BaseModel):
    """待评分的候选动作。"""

    action: dict[str, Any]
    thought: str | None = None


class PRMScoreRequest(BaseModel):
    """AndroidWorld 单步 PRM 评分输入。"""

    model_config = ConfigDict(extra="allow")

    goal: str
    step_index: int = 0
    screen_size: tuple[int, int] | None = None
    screenshot: str | None = None
    screenshot_b64: str | None = None
    ui_elements: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[CandidateAction] = Field(default_factory=list)
    model: str | None = None
    mode: PRMMode = PRMMode.MULTIMODAL_WINDOW
    task_category: str | None = None
    task_name: str | None = None
    episode_id: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize_aliases(self) -> "PRMScoreRequest":
        """兼容 AndroidWorld orchestrator 和手工请求的字段差异。"""
        if self.screenshot is None and self.screenshot_b64:
            self.screenshot = self.screenshot_b64
        if self.task_category is None:
            self.task_category = self.extras.get("task_category") or self.extras.get("suite")
        if self.task_name is None:
            self.task_name = self.extras.get("task_name") or self.extras.get("task")
        if self.episode_id is None:
            self.episode_id = self.extras.get("episode_id") or self.extras.get("run_id")
        return self


class PRMScoreResponse(BaseModel):
    """PRM 返回给 orchestrator 的评分结果。"""

    scores: list[float]
    best_index: int
    reason: str = ""
    mode: PRMMode
    archive_dir: str
