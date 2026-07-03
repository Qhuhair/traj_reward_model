"""分层记忆多模态流水线：env_parser → annotation → memory LLM → QA → PRM。"""
import argparse
import json
import os
import subprocess
import sys

from pipelines.common_cli import (
    build_parser,
    iter_targets,
    resolve_all_root,
    resolve_target_dir,
    run_quietly_if_needed,
)
from pipelines.paths import PROJECT_ROOT, PYTHON

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output", "hierarchical-memory")
TRAJS_ROOT = os.path.join(PROJECT_ROOT, "trajs")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "LLM_caller", "utils"))

from env_parser.core.pipeline import EnvParserPipeline
from env_parser.utils.path_utils import list_traj_dirs_in_set
from report.writer import save_json
from image_annotator import ImageAnnotator
from annotate_traj_images import annotate_standardized


def run_cmd(name, cmd_list, timeout=3600):
    """运行阶段子进程，保留失败时的关键信息。"""
    print(f"\n[{name}]")
    result = subprocess.run(cmd_list, capture_output=True, text=True,
                            timeout=timeout, cwd=PROJECT_ROOT)
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            print(f"  {line.strip()}")
    if result.stderr.strip():
        for line in result.stderr.strip().split("\n")[:8]:
            if line.strip() and "INFO" not in line:
                print(f"  [stderr] {line.strip()[:180]}")
    return result.returncode == 0


def process_one_traj(set_name, traj_id, run_dir, args):
    """处理单条轨迹，按参数决定是否启用分层记忆。"""
    out_dir = os.path.join(run_dir, traj_id)
    os.makedirs(out_dir, exist_ok=True)
    traj_dir = os.path.join(TRAJS_ROOT, set_name, traj_id)

    std_path = os.path.join(out_dir, "standardized.json")
    llm_path = os.path.join(out_dir, "llm_scores.json")
    qa_path = os.path.join(out_dir, "qa_reports.json")
    prm_path = os.path.join(out_dir, "prm_scores.json")
    memory_path = os.path.join(out_dir, "memory_contexts.json")

    print(f"\n{'=' * 60}")
    print(f"Hierarchical memory pipeline: {set_name}/{traj_id}")
    print(f"{'=' * 60}")

    pipeline = EnvParserPipeline()
    std_data = pipeline.run(traj_dir)
    save_json(std_path, std_data)
    print(f"  Steps: {len(std_data['steps'])}")

    annotator = ImageAnnotator()
    annotate_standardized(std_data, os.path.join(out_dir, "annotated_images"), annotator)
    save_json(std_path, std_data)
    print("  Annotations saved")

    if args.use_memory:
        results, contexts = _run_memory_llm(std_data, args)
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(contexts, f, ensure_ascii=False, indent=2)
        print(f"  -> {memory_path}")
    else:
        results = _run_plain_mm_llm(std_data, args)

    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  -> {llm_path} ({len(results)} steps)")

    if not run_cmd("logic_QA", [PYTHON, os.path.join(PROJECT_ROOT, "logic_QA", "main.py"),
                                std_path, llm_path, qa_path]):
        return False

    with open(llm_path, "r", encoding="utf-8") as f:
        llm_results = json.load(f)
    for step, llm in zip(std_data["steps"], llm_results):
        step["score"] = llm["score"]
    save_json(std_path, std_data)

    if not run_cmd("core_prm", [PYTHON, os.path.join(PROJECT_ROOT, "core_prm", "main.py"),
                                std_path, prm_path]):
        return False

    print(f"  [OK] Done: {traj_id}")
    return True


def _run_memory_llm(std_data, args):
    from hierarchical_memory_evaluator import evaluate_hierarchical_memory
    config = {
        "enabled": True,
        "recent_window_size": args.memory_window_size,
        "history_summary_max_chars": args.memory_summary_max_chars,
        "recent_step_max_chars": args.memory_recent_step_max_chars,
        "include_previous_scores": True,
        "include_current_images": True,
        "summarizer": "llm",
        "summarizer_model": args.memory_summarizer_model,
        "summarizer_prompt": args.memory_summarizer_prompt,
    }
    return evaluate_hierarchical_memory(std_data, args.llm_model, args.llm_prompt, config)


def _run_plain_mm_llm(std_data, args):
    from main import run_on_trajectory
    return run_on_trajectory(
        std_data,
        model=args.llm_model,
        prompt=args.no_memory_prompt,
        max_workers=args.max_workers,
        text_only=False,
    )


def process_set(set_name, run_dir, args):
    """处理一个轨迹集，并生成集合报告和筛选结果。"""
    os.makedirs(run_dir, exist_ok=True)
    trajs = list_traj_dirs_in_set(TRAJS_ROOT, set_name)
    print(f"Processing set: {set_name} ({len(trajs)} trajectories)")

    for tid in trajs:
        if not process_one_traj(set_name, tid, run_dir, args):
            print(f"  [WARN] {tid} had errors, continuing...")

    print(f"\n{'=' * 50}")
    print("Generating report and filter...")
    from report.writer import generate as generate_report
    from filter.core.filter_engine import FilterEngine
    generate_report(run_dir)
    report = FilterEngine().run(run_dir)
    print(f"  Filter: {report.get('summary', {}).get('total_trajectories', 0)} trajectories")


def parse_args():
    """解析分层记忆流水线参数。"""
    parser = build_parser("运行分层记忆多模态轨迹评估流水线。")
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--use-memory", action="store_true",
                        help="启用分层记忆。")
    memory.add_argument("--no-memory", action="store_true",
                        help="关闭分层记忆，退化为普通多模态评分。")
    parser.add_argument("--llm-model", default="qwen_vllm_mm",
                        help="评分模型配置名。默认 qwen_vllm_mm。")
    parser.add_argument("--llm-prompt", default="RRM_Qwen_HierMemory",
                        help="启用记忆时的评分 Prompt。")
    parser.add_argument("--no-memory-prompt", default="RRM_Qwen_MM",
                        help="关闭记忆时的普通多模态 Prompt。")
    parser.add_argument("--memory-window-size", type=int, default=3,
                        help="最近窗口大小，默认 3。")
    parser.add_argument("--memory-summary-max-chars", type=int, default=1200,
                        help="历史摘要最大长度。")
    parser.add_argument("--memory-recent-step-max-chars", type=int, default=600,
                        help="最近窗口单步状态文本最大长度。")
    parser.add_argument("--memory-summarizer-model", default="qwen_vllm_text",
                        help="历史摘要模型配置名，默认 qwen_vllm_text。")
    parser.add_argument("--memory-summarizer-prompt", default="RRM_HistorySummary",
                        help="历史摘要 Prompt。")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="关闭记忆时普通多模态并发数。")
    args = parser.parse_args()
    if args.no_memory:
        args.use_memory = False
    return args


def main():
    args = parse_args()

    def runner():
        all_root = resolve_all_root(args, OUTPUT_BASE)
        for set_name in iter_targets(args):
            run_dir = (
                resolve_target_dir(set_name, args, OUTPUT_BASE)
                if args.target else os.path.join(all_root, set_name)
            )
            process_set(set_name, run_dir, args)

    run_quietly_if_needed(args, OUTPUT_BASE, runner)


if __name__ == "__main__":
    main()
