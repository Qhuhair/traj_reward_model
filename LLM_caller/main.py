import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from caller import LLMCaller

MOCK_KEYS = ("task_description",)
STD_KEYS = ("trajectory_id", "app", "task", "subgoals")
MAX_STATE_CHARS = 400


def _is_standard_format(data: dict) -> bool:
    """自动检测输入是否为 env_parser 标准化格式"""
    return all(k in data for k in STD_KEYS)


def _extract_step(step: dict, is_standard: bool) -> dict:
    """从单个步骤中提取所有 prompt 占位符"""
    if is_standard:
        # 优先使用标注后的图片路径（多模态模式），fallback 到原始路径
        img_before = step.get("image_before_annotated") or step.get("image_before", "")
        img_after = step.get("image_after_annotated") or step.get("image_after", "")
        return {
            "task_desc": step.get("task", ""),
            "state_desc_before": step.get("state_desc_before", "")[:MAX_STATE_CHARS],
            "state_desc_after": step.get("state_desc_after", "")[:MAX_STATE_CHARS],
            "image_before": img_before,
            "image_after": img_after,
            "action_desc": step.get("action_desc", ""),
            "element_id": step.get("element_id", ""),
            "subgoal": step.get("subgoal_text", ""),
            "step_idx": step.get("step_idx", 0),
            "action_raw": step.get("action", ""),
        }

    state = step.get("state", {})
    parts = [f"页面: {state.get('current_page', '')}",
             f"焦点: {state.get('ui_focus', '')}"]
    for el in state.get("key_elements", []):
        parts.append(f"[{el.get('id', '')}] {el.get('text', '')} ({el.get('type', '')})")
    state_str = "\n".join(parts)

    act = step.get("action", {})
    if isinstance(act, dict):
        action_desc = act.get("description", "")
    else:
        action_desc = str(act)

    return {
        "task_desc": "",
        "state_desc_before": state_str,
        "state_desc_after": "",
        "action_desc": action_desc,
        "element_id": "",
        "subgoal": "",
        "step_idx": step.get("step_idx", 0),
        "action_raw": str(act),
    }


def _extract_task(data: dict, is_standard: bool) -> str:
    if is_standard:
        return data.get("task", "")
    return data.get("task_description", "")


def _eval_one_step(step_info: dict, full_task: str, caller: LLMCaller) -> dict:
    """单个步骤的 LLM 评估（供并发调用）"""
    step_info["task_desc"] = full_task
    result = caller.call(
        task_desc=step_info["task_desc"],
        subgoal=step_info["subgoal"],
        step_idx=step_info["step_idx"],
        action_desc=step_info["action_desc"],
        element_id=step_info["element_id"],
        state_desc_before=step_info["state_desc_before"],
        state_desc_after=step_info["state_desc_after"],
        image_before=step_info["image_before"],
        image_after=step_info["image_after"],
    )
    return {
        "step_idx": step_info["step_idx"],
        "action": step_info["action_raw"],
        "think": result["think"],
        "critique": result["critique"],
        "score": result["score"],
    }


def run_on_trajectory(data: dict, caller: LLMCaller = None,
                      indices: set = None, max_workers: int = 4,
                      model: str = None, prompt: str = None) -> list:
    """
    统一入口：接受 env_parser 标准化格式 或 旧 mock 格式。
    indices: 只处理指定 step_idx 的步骤 (None=全部)。
    使用 ThreadPoolExecutor 并发调用 LLM。
    """
    if caller is None:
        caller = LLMCaller(model=model, prompt=prompt)

    is_std = _is_standard_format(data)
    full_task = _extract_task(data, is_std)
    steps = data.get("steps", [])

    # 准备所有步骤信息
    step_infos = []
    for step in steps:
        info = _extract_step(step, is_std)
        if indices is not None and info["step_idx"] not in indices:
            step_infos.append(None)
        else:
            step_infos.append(info)

    # 并发提交
    results = [None] * len(steps)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for i, info in enumerate(step_infos):
            if info is None:
                continue
            futures[pool.submit(_eval_one_step, info, full_task, caller)] = i

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results


def _parse_indices() -> set | None:
    """解析 CLI --indices=1,3,7 参数"""
    for arg in sys.argv:
        if arg.startswith("--indices="):
            return set(int(x.strip()) for x in arg.split("=")[1].split(",") if x.strip())
    return None


def _parse_flag(flag: str) -> str | None:
    """解析 CLI --flag=value 参数"""
    for arg in sys.argv:
        if arg.startswith(f"--{flag}="):
            return arg.split("=", 1)[1]
    return None


def save_results(output_dir: str, traj_id: str, results: list):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{traj_id}_llm_scores.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"LLM 结果已写入: {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python main.py <input.json> [output.json] [--indices=1,3,7] [--model=xxx] [--prompt=xxx]")
        sys.exit(1)

    indices = _parse_indices()
    model_override = _parse_flag("model")
    prompt_override = _parse_flag("prompt")

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    # 读取已有结果 (用于 --indices 模式下的部分更新)
    existing = None
    out_path = sys.argv[2] if len(sys.argv) >= 3 else None
    if out_path and os.path.exists(out_path) and indices:
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # 多模态模式下减少并发数以防 Ollama 过载
    mm_max_workers = 2 if model_override and model_override.endswith("_mm") else 4
    results = run_on_trajectory(data, indices=indices,
                                model=model_override, prompt=prompt_override,
                                max_workers=mm_max_workers)

    # 合并: 用新结果替换已有结果中对应索引的条目
    if existing and indices:
        # existing 和 results 都是按 step 顺序排列
        merged = []
        new_idx = 0
        for i, old in enumerate(existing):
            step_idx = old.get("step_idx", i + 1)
            if step_idx in indices and new_idx < len(results):
                r = results[new_idx]
                if r is not None:
                    merged.append(r)
                    new_idx += 1
                    continue
            merged.append(old)
        results = merged

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        for r in results:
            if r:
                print(f"[Step {r['step_idx']}] score={r['score']:.3f} | {r['think'][:60]}...")
