import os
import json
import sys
from engine import QA_Orchestrator


def run_on_trajectory(data: dict, llm_outputs: list = None,
                      indices: set = None) -> list:
    """
    统一入口：接受标准化轨迹数据 + LLM 输出列表。
    indices: 只审查指定 step_idx 的步骤 (None=全部)
    """
    engine = QA_Orchestrator()
    task_desc = data.get("task") or data.get("task_description", "")
    steps = data.get("steps", [])
    if llm_outputs is None:
        llm_outputs = [{} for _ in steps]

    reports = []
    prev_score = None
    prev_state = None

    for i, step in enumerate(steps):
        idx = step.get("step_idx", i + 1)
        curr_state = _resolve_state(step)

        llm_out = llm_outputs[i] if i < len(llm_outputs) else {}

        # 只处理指定 indices
        if indices is not None and idx not in indices:
            reports.append(None)  # placeholder
            prev_score = None
            prev_state = None
            continue

        lg = step.get("logic_augmentation", {})
        think = llm_out.get("think") or lg.get("thought", "")
        critique = llm_out.get("critique") or lg.get("critique", "")
        score = llm_out.get("score") or lg.get("promise_score", 0.0)

        try:
            score = float(score)
        except (ValueError, TypeError):
            score = 0.0

        eval_data = {
            "context": {"task": task_desc, "step": idx},
            "llm_output": {"think": think, "critique": critique, "score": score},
            "state": curr_state,
            "prev_state": prev_state,
            "prev_score": prev_score,
        }

        report = engine.evaluate(eval_data)
        report["step_idx"] = idx
        report["action_desc"] = _resolve_action_desc(step)
        reports.append(report)

        prev_score = score
        prev_state = curr_state

    return reports


def _resolve_state(step: dict):
    """优先取 env_parser 格式的 state_desc，否则取 mock state dict"""
    if "state_desc" in step:
        return step["state_desc"]
    return step.get("state", {})


def _resolve_action_desc(step: dict) -> str:
    act = step.get("action_desc", "")
    if act:
        return act
    action = step.get("action", "")
    if isinstance(action, dict):
        return action.get("description", "")
    return str(action)


def _print_reports(reports: list):
    for r in reports:
        status = "V" if r["is_accepted"] else "X"
        print(
            f"{status} Step {r.get('step_idx', '?')} | "
            f"动作: {r.get('action_desc', '?')[:40]}"
        )
        print(f"   Meta-Score: {r['meta_score']:.4f} | 采纳: {r['is_accepted']}")
        print(f"   指标: {r['metrics']}")
        if r["failed_reasons"]:
            for reason in r["failed_reasons"]:
                print(f"   FAIL: {reason}")
        print()


def _parse_indices() -> set | None:
    for arg in sys.argv:
        if arg.startswith("--indices="):
            return set(int(x.strip()) for x in arg.split("=")[1].split(",") if x.strip())
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py <traj_json> [llm_output.json] [output.json] [--indices=1,3]")
        sys.exit(1)

    indices = _parse_indices()

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    llm_outputs = None
    if len(sys.argv) >= 3:
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            llm_outputs = json.load(f)

    # 读取已有 QA 结果 (用于 --indices 模式下的部分更新)
    existing = None
    out_path = sys.argv[3] if len(sys.argv) >= 4 else None
    if out_path and os.path.exists(out_path) and indices:
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    reports = run_on_trajectory(data, llm_outputs, indices=indices)

    # 合并
    if existing and indices:
        merged = []
        new_idx = 0
        for i, old in enumerate(existing):
            step_idx = old.get("step_idx", i + 1)
            if step_idx in indices and new_idx < len(reports):
                r = reports[new_idx]
                if r is not None:
                    merged.append(r)
                    new_idx += 1
                    continue
            merged.append(old)
        reports = merged

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
    else:
        _print_reports(reports)
