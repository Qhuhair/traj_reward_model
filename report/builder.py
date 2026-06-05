from report.loader import TrajReport


def build_traj_summary(r: TrajReport) -> dict:
    """构建单轨迹汇总 dict (用于 summary.json)"""
    return {
        "trajectory_id": r.traj_id,
        "app": r.app,
        "task": r.task,
        "n_steps": r.n_steps,
        "avg_q": round(r.avg_q, 4),
        "avg_progress": round(r.avg_progress, 4),
        "avg_gae": round(r.avg_gae, 4),
        "qa_pass": r.qa_pass,
        "qa_total": r.qa_total,
        "steps": [
            {
                "step_idx": s.step_idx,
                "action": s.action,
                "q": round(s.q, 4),
                "progress": round(s.progress, 4),
                "gae": round(s.gae, 4),
                "qa_accepted": s.qa_accepted,
            }
            for s in r.steps
        ],
    }


def build_master_summary(reports: list, run_dir: str, ts: str) -> dict:
    """构建总报告 dict (用于 master_summary.json)"""
    trajs = []
    total_steps = 0
    total_q = 0.0
    total_progress = 0.0
    total_gae = 0.0
    total_qa_pass = 0
    total_qa_total = 0

    for r in reports:
        trajs.append({
            "trajectory_id": r.traj_id,
            "app": r.app,
            "n_steps": r.n_steps,
            "avg_q": round(r.avg_q, 4),
            "avg_progress": round(r.avg_progress, 4),
            "avg_gae": round(r.avg_gae, 4),
            "qa_pass": r.qa_pass,
            "qa_total": r.qa_total,
        })
        total_steps += r.n_steps
        total_q += r.avg_q * r.n_steps
        total_progress += r.avg_progress * r.n_steps
        total_gae += r.avg_gae * r.n_steps
        total_qa_pass += r.qa_pass
        total_qa_total += r.qa_total

    return {
        "generated_at": ts,
        "run_dir": run_dir,
        "trajectories": trajs,
        "overall": {
            "total_trajectories": len(reports),
            "total_steps": total_steps,
            "avg_q": round(total_q / total_steps, 4) if total_steps else 0,
            "avg_progress": round(total_progress / total_steps, 4) if total_steps else 0,
            "avg_gae": round(total_gae / total_steps, 4) if total_steps else 0,
            "qa_total_pass": total_qa_pass,
            "qa_total_steps": total_qa_total,
        },
    }
