"""PRM 轨迹归档路径工具。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from framework_api.logging_utils import LOG_ROOT


TRAJ_ROOT = LOG_ROOT / "prm" / "trajs"
ACTIVE_SESSIONS: dict[str, Path] = {}


def resolve_session_dir(
    *,
    task_category: str | None,
    task_name: str | None,
    episode_id: str | None,
    goal: str,
    step_index: int,
) -> Path:
    """按 AndroidWorld 任务类别/任务名/episode 生成轨迹目录。"""
    category = safe_name(task_category or "android_world")
    task = safe_name(task_name or task_from_goal(goal))
    session_key = f"{category}/{task}/{episode_id or 'active'}"
    if step_index == 0 or session_key not in ACTIVE_SESSIONS:
        episode = safe_name(episode_id or datetime.now().strftime("episode_%Y%m%d_%H%M%S"))
        ACTIVE_SESSIONS[session_key] = TRAJ_ROOT / category / task / episode
    episode_dir = ACTIVE_SESSIONS[session_key]
    episode_dir.mkdir(parents=True, exist_ok=True)
    return episode_dir


def clear_session(episode_dir: Path) -> None:
    """轨迹完成后清除内存中的 active session。"""
    for key, value in list(ACTIVE_SESSIONS.items()):
        if value == episode_dir:
            ACTIVE_SESSIONS.pop(key, None)


def task_from_goal(goal: str) -> str:
    """缺少任务名时用 goal 生成可读目录名。"""
    words = re.sub(r"\s+", "_", goal.strip())[:60]
    return words or "unknown_task"


def safe_name(value: str) -> str:
    """把任务类别、任务名、episode id 转成安全路径片段。"""
    cleaned = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"
