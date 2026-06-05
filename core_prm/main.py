import os
import json
import sys
from prm_orchestrator import PRM_Orchestrator


def _build_step_list(steps: list) -> list:
    """标准化 steps 列表：确保 action_desc 字段存在"""
    result = []
    for s in steps:
        desc = s.get("action_desc", "")
        if not desc and isinstance(s.get("action"), dict):
            desc = s["action"].get("description", "")
        result.append({
            "step_idx": s.get("step_idx", 0),
            "action": desc[:30],
        })
    return result


def run_on_trajectory(data: dict) -> dict:
    """
    统一入口：接受标准化轨迹数据（含 score 字段）或 mock 格式。
    返回 PRM 计算结果 dict。
    """
    engine = PRM_Orchestrator()
    output = engine.run(data)

    steps = _build_step_list(data.get("steps", []))
    output["step_list"] = steps

    return output


def _fmt_row(step_idx, action, q, r, progress, td, advantage):
    def v(x): return f"{x:+.4f}" if x is not None else "  N/A  "
    return (
        f"  {step_idx:<5} {action:<30} "
        f"Q={q:<8.4f} R={r:<8.4f} "
        f"Δ={v(progress)} "
        f"δ={v(td)} "
        f"GAE={v(advantage)}"
    )


def _print_report(output: dict):
    print(f"\n估计器: {output['estimator']}")
    if output.get("warnings"):
        for w in output["warnings"]:
            print(f"  ! {w}")
    print()

    header = f"  {'Step':<5} {'Action':<30} {'Q':<9} {'R':<9} {'Progress':<9} {'TD-Err':<9} {'GAE':<9}"
    print(header)
    print("-" * len(header))

    for i, step in enumerate(output.get("step_list", [])):
        action = step["action"][:28]
        q = output["q_sequence"][i]
        r = output["rewards"][i]
        progress = output["progress"][i] if output["progress"] else None
        td = output["td_errors"][i] if output["td_errors"] else None
        gae = output["advantages"][i] if output["advantages"] else None
        print(_fmt_row(step["step_idx"], action, q, r, progress, td, gae))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py <input.json> [output.json]")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    task = data.get("task") or data.get("task_description", "Unknown")
    print(f"任务: {task}")

    output = run_on_trajectory(data)

    if len(sys.argv) >= 3:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    else:
        _print_report(output)
