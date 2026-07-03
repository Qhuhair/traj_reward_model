"""纯文本滑动窗口流水线：env_parser → text_slide_LLM → logic_QA → core_prm。"""
import os, sys, json, subprocess

from pipelines.paths import PROJECT_ROOT, PYTHON
from pipelines.common_cli import (
    build_parser,
    iter_targets,
    resolve_all_root,
    resolve_target_dir,
    run_quietly_if_needed,
)

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output", "qwen-text-window")
TRAJS_ROOT = os.path.join(PROJECT_ROOT, "trajs")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "LLM_caller", "utils"))
from env_parser.core.pipeline import EnvParserPipeline
from env_parser.utils.path_utils import list_traj_dirs_in_set
from report.writer import save_json


def run_cmd(name, cmd_list, timeout=3600):
    """运行阶段子进程，并过滤过长或低价值的日志。"""
    print(f"\n[{name}]")
    result = subprocess.run(cmd_list, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT)
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            print(f"  {line.strip()}")
    if result.stderr.strip():
        for line in result.stderr.strip().split("\n")[:5]:
            if line.strip() and "INFO" not in line:
                print(f"  [stderr] {line.strip()[:150]}")
    return result.returncode == 0


def process_one_traj(set_name, traj_id, run_dir):
    """处理单条轨迹：纯文本窗口评分、QA、PRM。"""
    out_dir = os.path.join(run_dir, traj_id)
    os.makedirs(out_dir, exist_ok=True)
    traj_dir = os.path.join(TRAJS_ROOT, set_name, traj_id)

    std_path = os.path.join(out_dir, "standardized.json")
    llm_path = os.path.join(out_dir, "llm_scores.json")
    qa_path = os.path.join(out_dir, "qa_reports.json")
    prm_path = os.path.join(out_dir, "prm_scores.json")

    # Stage 1：解析原始轨迹为统一 State-Action JSON。
    pipeline = EnvParserPipeline()
    std_data = pipeline.run(traj_dir)
    save_json(std_path, std_data)
    print(f"  Steps: {len(std_data['steps'])}")

    # Stage 2：用前后步骤文本上下文评估当前步骤。
    print(f"\n[Text Slide LLM]")
    from text_slide_evaluator import TextSlideEvaluator
    from caller import LLMCaller
    caller = LLMCaller(model="qwen_vllm_text", prompt="RRM_Qwen_TextSlide")
    evaluator = TextSlideEvaluator(caller)
    step_results = evaluator.evaluate(std_data)

    # Stage 2.5：追加轨迹级总结，便于后续人工查看整体完成度。
    print(f"\n[Trajectory Summary]")
    from traj_summarizer import TrajectorySummarizer
    summary_caller = LLMCaller(model="qwen_vllm_text", prompt="RRM_TrajSummary")
    summarizer = TrajectorySummarizer(summary_caller)
    traj_summary = summarizer.summarize(std_data, step_results)
    print(f"  Trajectory score: {traj_summary['score']:.2f}")

    # llm_scores.json 中前 N 条是步骤分数，最后一条是轨迹级 summary。
    all_results = step_results + [traj_summary]
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  -> {llm_path} ({len(step_results)} steps + 1 summary)")

    # Stage 3：QA 只评估步骤条目，summary 会被下游按 step_idx 忽略。
    cmd = [PYTHON, os.path.join(PROJECT_ROOT, "logic_QA", "main.py"),
           std_path, llm_path, qa_path]
    if not run_cmd("logic_QA", cmd):
        print(f"  [WARN] QA failed, continuing...")

    # 将步骤分数写回 standardized.json，供 core_prm 使用。
    with open(llm_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    for step, score_data in zip(std_data["steps"], saved):
        if score_data.get("step_idx", 0) > 0:
            step["score"] = score_data.get("score", 0.0)
    save_json(std_path, std_data)

    # Stage 4：计算过程奖励。
    cmd = [PYTHON, os.path.join(PROJECT_ROOT, "core_prm", "main.py"),
           std_path, prm_path]
    if not run_cmd("core_prm", cmd):
        print(f"  [WARN] core_prm failed, continuing...")

    print(f"  [OK] Done: {traj_id}")
    return True


def process_set(set_name, run_dir):
    """处理一个轨迹集，并生成报告和筛选结果。"""
    os.makedirs(run_dir, exist_ok=True)

    trajs = list_traj_dirs_in_set(TRAJS_ROOT, set_name)
    print(f"Processing set: {set_name} ({len(trajs)} trajectories)")

    for tid in trajs:
        success = process_one_traj(set_name, tid, run_dir)
        if not success:
            print(f"  [WARN] {tid} had errors, continuing...")

    # 所有轨迹处理完成后统一生成集合报告和 A/B/C/D 分类。
    print(f"\n{'='*50}")
    from report.writer import generate as generate_report
    from filter.core.filter_engine import FilterEngine
    try:
        generate_report(run_dir)
        print("  Report generated")
    except Exception as e:
        print(f"  Report error: {e}")
    try:
        engine = FilterEngine()
        report = engine.run(run_dir)
        s = report.get("summary", {})
        print(f"  Filter: {s.get('total_trajectories', 0)} trajectories")
        for cid in ["A", "B", "C", "D"]:
            cnt = s.get("category_counts", {}).get(cid, 0)
            print(f"    {cid}: {cnt}")
    except Exception as e:
        print(f"  Filter error: {e}")


def parse_args():
    """解析统一 CLI 参数。"""
    return build_parser("运行纯文本滑动窗口轨迹评估流水线。").parse_args()


def main():
    args = parse_args()

    def runner():
        all_root = resolve_all_root(args, OUTPUT_BASE)
        for set_name in iter_targets(args):
            if args.target:
                run_dir = resolve_target_dir(set_name, args, OUTPUT_BASE)
            else:
                run_dir = os.path.join(all_root, set_name)
            process_set(set_name, run_dir)

    run_quietly_if_needed(args, OUTPUT_BASE, runner)


if __name__ == "__main__":
    main()
