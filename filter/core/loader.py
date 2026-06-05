import os
import json
from dataclasses import dataclass, field


@dataclass
class StepDetail:
    step_idx: int
    action: str
    q: float
    progress: float
    gae: float
    qa_accepted: bool
    meta_score: float
    density_index: float
    conflict_index: float


@dataclass
class TrajData:
    traj_id: str
    app: str
    task: str
    n_steps: int
    avg_q: float
    avg_progress: float
    avg_gae: float
    qa_pass: int
    qa_total: int
    steps: list = field(default_factory=list)


def safe_load(path: str, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def load_traj_data(traj_dir: str) -> TrajData | None:
    """读取一条轨迹的报告产物，构建 TrajData"""
    summary = safe_load(os.path.join(traj_dir, "summary.json"))
    if not summary:
        return None

    qa = safe_load(os.path.join(traj_dir, "qa_reports.json"), [])

    steps_raw = summary.get("steps", [])
    step_details = []
    for i, s in enumerate(steps_raw):
        q_info = qa[i] if i < len(qa) else {}
        step_details.append(StepDetail(
            step_idx=s["step_idx"],
            action=s.get("action", ""),
            q=s.get("q", 0),
            progress=s.get("progress", 0),
            gae=s.get("gae", 0),
            qa_accepted=s.get("qa_accepted", True),
            meta_score=q_info.get("meta_score", 0),
            density_index=q_info.get("metrics", {}).get("density_index", 0),
            conflict_index=q_info.get("metrics", {}).get("conflict_index", 0),
        ))

    return TrajData(
        traj_id=summary["trajectory_id"],
        app=summary.get("app", ""),
        task=summary.get("task", ""),
        n_steps=summary["n_steps"],
        avg_q=summary["avg_q"],
        avg_progress=summary["avg_progress"],
        avg_gae=summary["avg_gae"],
        qa_pass=summary["qa_pass"],
        qa_total=summary["qa_total"],
        steps=step_details,
    )


def load_run_dir(run_dir: str) -> list:
    """扫描 output/<timestamp>/ 下所有 traj_* 子目录"""
    results = []
    if not os.path.isdir(run_dir):
        return results
    for name in sorted(os.listdir(run_dir)):
        sub = os.path.join(run_dir, name)
        if not os.path.isdir(sub) or not name.startswith("traj_"):
            continue
        data = load_traj_data(sub)
        if data:
            results.append(data)
    return results
