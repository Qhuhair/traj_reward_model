import os
import json
from datetime import datetime

from report.loader import TrajReport, load_all_outputs
from report.builder import build_traj_summary, build_master_summary


def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_traj_table(r: TrajReport):
    print(f"\n--- {r.traj_id} ---")
    print(f"  App:     {r.app}")
    print(f"  Task:    {r.task}")
    print(f"  Steps:   {r.n_steps}")
    print(f"  QA:      {r.qa_pass}/{r.qa_total} accepted")
    print(f"  Avg-Q:   {r.avg_q:.4f}")
    print(f"  Avg-Delta: {r.avg_progress:+.4f}")
    print(f"  Avg-GAE: {r.avg_gae:+.4f}")
    print()

    header = f"  {'Step':<5} {'Action':<32} {'Q':<8} {'Delta':<10} {'GAE':<10} {'QA':<4}"
    print(header)
    print("-" * len(header))

    for s in r.steps:
        action = s.action[:30] + "..." if len(s.action) > 30 else s.action
        qa_tag = "V" if s.qa_accepted else "X"
        print(
            f"  {s.step_idx:<5} {action:<32} "
            f"{s.q:<8.4f} {s.progress:<+10.4f} "
            f"{s.gae:<+10.4f} {qa_tag:<4}"
        )


def print_master_table(reports: list):
    print(f"\n{'='*80}")
    print(f"  Pipeline Summary Report")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    header = f"  {'Trajectory':<14} {'Steps':<6} {'Avg-Q':<8} {'Avg-Delta':<10} {'Avg-GAE':<10} {'QA':<8} {'App'}"
    print()
    print(header)
    print("-" * len(header))

    for r in reports:
        print(
            f"  {r.traj_id:<14} {r.n_steps:<6} {r.avg_q:<8.4f} "
            f"{r.avg_progress:<+10.4f} {r.avg_gae:<+10.4f} "
            f"{r.qa_pass}/{r.qa_total:<5} {r.app}"
        )

    print(f"{'='*80}")


def generate(run_dir: str):
    """
    总报告生成入口。
    扫描 run_dir 下所有 traj_* 子目录，构建并保存报告。
    """
    reports = load_all_outputs(run_dir)
    if not reports:
        print("[REPORT] No trajectory outputs found")
        return

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # 每条轨迹写 summary.json
    for r in reports:
        traj_out = os.path.join(run_dir, r.traj_id)
        save_json(os.path.join(traj_out, "summary.json"), build_traj_summary(r))

    # 总报告
    master = build_master_summary(reports, os.path.basename(run_dir), ts)
    save_json(os.path.join(run_dir, "master_summary.json"), master)

    # 控制台输出
    print_master_table(reports)
    for r in reports:
        print_traj_table(r)

    print(f"\nReports saved to: {run_dir}")
