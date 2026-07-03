"""对外 API 的稳定数据结构。"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskType(str, Enum):
    """外部调用方请求的任务类型。"""

    EVALUATE_TRAJECTORY = "evaluate_trajectory"
    SCORE_STEP = "score_step"
    FILTER_DATASET = "filter_dataset"
    RUN_PIPELINE = "run_pipeline"
    COMPARE_RESULTS = "compare_results"


class InputMode(str, Enum):
    """外部输入形态，用于把不同来源统一映射到内部处理器。"""

    PATH = "path"
    JSON = "json"
    TEXT = "text"
    MULTIMODAL = "multimodal"
    BATCH = "batch"


@dataclass(frozen=True)
class APIRequest:
    """一次框架级调用请求；payload 的具体 schema 由 handler 解释。"""

    task_type: TaskType
    input_mode: InputMode
    payload: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None


@dataclass(frozen=True)
class APIResponse:
    """框架级统一响应；成功和失败都保持同一外形。"""

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    request_id: str | None = None


def normalize_path(value: str | Path) -> str:
    """把路径输入规范成字符串，避免对外暴露 Path 实现细节。"""
    return str(Path(value).expanduser())
