import os
import json
from dataclasses import dataclass, field


@dataclass
class StepRow:
    step_idx: int
    action: str
    q: float
    progress: float
    gae: float
    qa_accepted: bool


@dataclass
class TrajReport:
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


def load_traj_output(out_dir: str) -> TrajReport | None:
    std = safe_load(os.path.join(out_dir, "standardized.json"))
    llm = safe_load(os.path.join(out_dir, "llm_scores.json"), [])
    prm = safe_load(os.path.join(out_dir, "prm_scores.json"), {})
    qa = safe_load(os.path.join(out_dir, "qa_reports.json"), [])

    if not std or not std.get("steps"):
        return None

    steps_raw = std["steps"]
    n = len(steps_raw)
    q_seq = [llm[i]["score"] if i < len(llm) else 0.0 for i in range(n)]
    gae_seq = prm.get("advantages", [0.0] * n)
    prog_seq = prm.get("progress", [0.0] * n)
    qa_accept = [qa[i].get("is_accepted", True) if i < len(qa) else True for i in range(n)]

    step_rows = []
    for i, step in enumerate(steps_raw):
        action = step.get("action_desc") or step.get("action", "")
        step_rows.append(StepRow(
            step_idx=step["step_idx"],
            action=action,
            q=q_seq[i],
            progress=prog_seq[i] if i < len(prog_seq) else 0.0,
            gae=gae_seq[i] if i < len(gae_seq) else 0.0,
            qa_accepted=qa_accept[i],
        ))

    return TrajReport(
        traj_id=std.get("trajectory_id", os.path.basename(out_dir)),
        app=std.get("app", "?"),
        task=(std.get("task") or "")[:120],
        n_steps=n,
        avg_q=sum(q_seq) / n,
        avg_progress=sum(prog_seq) / n if prog_seq else 0.0,
        avg_gae=sum(gae_seq) / n if gae_seq else 0.0,
        qa_pass=sum(1 for a in qa_accept if a),
        qa_total=n,
        steps=step_rows,
    )


def load_all_outputs(output_root: str) -> list:
    reports = []
    if not os.path.isdir(output_root):
        return reports
    for name in sorted(os.listdir(output_root)):
        sub = os.path.join(output_root, name)
        if not os.path.isdir(sub) or not name.startswith("traj_"):
            continue
        r = load_traj_output(sub)
        if r:
            reports.append(r)
    return reports
