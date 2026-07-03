"""PRM 评分轨迹归档。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .images import save_screenshot
from .paths import clear_session, resolve_session_dir
from .schema import PRMScoreRequest


def resolve_episode_dir(request: PRMScoreRequest) -> Path:
    """按 AndroidWorld 任务类别/任务名/episode 保存当前轨迹。"""
    return resolve_session_dir(
        task_category=request.task_category,
        task_name=request.task_name,
        episode_id=request.episode_id,
        goal=request.goal,
        step_index=request.step_index,
    )


def load_window_steps(episode_dir: Path, window_size: int) -> list[dict[str, Any]]:
    """读取最近窗口步骤，作为文本窗口或多模态窗口上下文。"""
    steps_dir = episode_dir / "steps"
    if not steps_dir.exists():
        return []
    step_files = sorted(steps_dir.glob("step_*.json"))
    history: list[dict[str, Any]] = []
    for step_file in step_files[-window_size:]:
        history.append(json.loads(step_file.read_text(encoding="utf-8")))
    return history


def save_score_input(episode_dir: Path, request: PRMScoreRequest, prompt: str) -> Path:
    """保存每一步的完整评分输入，图片另存为 png。"""
    step_dir = episode_dir / "steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    _save_prompt(episode_dir, request.step_index, prompt)
    image_path = save_screenshot(episode_dir, request)
    payload = request.model_dump(mode="json")
    payload["prompt"] = prompt
    payload["saved_screenshot"] = str(image_path) if image_path else None
    path = step_dir / f"step_{request.step_index:04d}.json"
    _append_jsonl(episode_dir / "score_inputs.jsonl", payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _save_prompt(episode_dir: Path, step_index: int, prompt: str) -> None:
    """单独保存模型评分 Prompt，便于人工复查输入。"""
    prompt_dir = episode_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / f"step_{step_index:04d}.txt").write_text(prompt, encoding="utf-8")


def save_score_output(episode_dir: Path, step_index: int, output: dict[str, Any]) -> None:
    """保存 PRM 原始输出和解析后的评分结果。"""
    path = episode_dir / "score_outputs.jsonl"
    record = {"step_index": step_index, **output}
    _append_jsonl(path, record)


def finish_episode_if_done(episode_dir: Path, request: PRMScoreRequest, best_index: int) -> None:
    """任务完成时写入完成标记，并结束当前 active session。"""
    if not _is_done_action(request, best_index):
        return
    payload = {"finished_at": datetime.now().isoformat(), "step_index": request.step_index}
    (episode_dir / "finished.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    clear_session(episode_dir)

def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """追加 JSONL，保留完整调用历史。"""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_done_action(request: PRMScoreRequest, best_index: int) -> bool:
    """根据最佳候选动作判断任务是否结束。"""
    if best_index >= len(request.candidates):
        return False
    action = request.candidates[best_index].action
    action_type = action.get("action_type") or action.get("type")
    return action_type == "status" and action.get("goal_status") == "task_complete"
