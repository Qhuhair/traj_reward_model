"""Base API 的请求和响应结构。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseActRequest(BaseModel):
    """AndroidWorld 单步观测输入。"""

    model_config = ConfigDict(extra="allow")

    goal: str
    step_index: int = 0
    screen_size: tuple[int, int] | None = None
    screenshot: str | None = None
    screenshot_b64: str | None = None
    ui_elements: list[dict[str, Any]] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    n_candidates: int = Field(default=1, ge=1, le=16)
    model: str | None = None

    @model_validator(mode="after")
    def fill_screenshot_alias(self) -> "BaseActRequest":
        """兼容 orchestrator 的 screenshot_b64 字段和文档中的 screenshot 字段。"""
        if self.screenshot is None and self.screenshot_b64:
            self.screenshot = self.screenshot_b64
        return self


class ActionCandidate(BaseModel):
    """Base 模型提出的候选动作。"""

    action: dict[str, Any]
    thought: str = ""


class BaseActResponse(BaseModel):
    """Base API 返回给 orchestrator 的候选动作列表。"""

    candidates: list[ActionCandidate]
