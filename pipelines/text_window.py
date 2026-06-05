"""
纯文本滑动窗口流水线：env_parser → text_slide_LLM → logic_QA → core_prm → traj_summary
所有步骤评分完成后，追加轨迹级总结。
输出到 output/qwen-text-window/
"""
import os, sys, json, subprocess

from pipelines.paths import PROJECT_ROOT, PYTHON

SET_NAME = sys.argv[1] if len(sys.argv) > 1 else "20250113_21442_test"
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output", "qwen-text-window")
TRAJS_ROOT = os.path.join(PROJECT_ROOT, "trajs")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "LLM_caller", "utils"))
from env_parser.core.pipeline import EnvParserPipeline
from env_parser.utils.path_utils import list_traj_dirs_in_set
from report.writer import save_json


def run_cmd(name, cmd_list, timeout=3600):
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


def process_one_traj(traj_id, run_dir):
    out_dir = os.path.join(run_dir, traj_id)
    os.makedirs(out_dir, exist_ok=True)
    traj_dir = os.path.join(TRAJS_ROOT, SET_NAME, traj_id)

    std_path = os.path.join(out_dir, "standardized.json")
    llm_path = os.path.join(out_dir, "llm_scores.json")
    qa_path = os.path.join(out_dir, "qa_reports.json")
    prm_path = os.path.join(out_dir, "prm_scores.json")

    # Stage 1: env_parser
    pipeline = EnvParserPipeline()
    std_data = pipeline.run(traj_dir)
    save_json(std_path, std_data)
    print(f"  Steps: {len(std_data['steps'])}")

    # Stage 2: Text Sliding Window LLM
    print(f"\n[Text Slide LLM]")
    from text_slide_evaluator import TextSlideEvaluator
    from caller import LLMCaller
    caller = LLMCaller(model="qwen_local", prompt="RRM_Qwen_TextSlide")
    evaluator = TextSlideEvaluator(caller)
    step_results = evaluator.evaluate(std_data)

    # Stage 2.5: Trajectory Summary
    print(f"\n[Trajectory Summary]")
    from traj_summarizer import TrajectorySummarizer
    summary_caller = LLMCaller(model="qwen_local", prompt="RRM_TrajSummary")
    summarizer = TrajectorySummarizer(summary_caller)
    traj_summary = summarizer.summarize(std_data, step_results)
    print(f"  Trajectory score: {traj_summary['score']:.2f}")

    # Save: step results + trajectory summary
    all_results = step_results + [traj_summary]
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"  -> {llm_path} ({len(step_results)} steps + 1 summary)")

    # Stage 3: QA
    cmd = [PYTHON, os.path.join(PROJECT_ROOT, "logic_QA", "main.py"),
           std_path, llm_path, qa_path]
    if not run_cmd("logic_QA", cmd):
        print(f"  [WARN] QA failed, continuing...")

    # Inject scores
    with open(llm_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    for step, score_data in zip(std_data["steps"], saved):
        if score_data.get("step_idx", 0) > 0:
            step["score"] = score_data.get("score", 0.0)
    save_json(std_path, std_data)

    # Stage 4: core_prm
    cmd = [PYTHON, os.path.join(PROJECT_ROOT, "core_prm", "main.py"),
           std_path, prm_path]
    if not run_cmd("core_prm", cmd):
        print(f"  [WARN] core_prm failed, continuing...")

    print(f"  [OK] Done: {traj_id}")
    return True


def main():
    run_dir = os.path.join(OUTPUT_BASE, SET_NAME)
    os.makedirs(run_dir, exist_ok=True)

    trajs = list_traj_dirs_in_set(TRAJS_ROOT, SET_NAME)
    print(f"Processing set: {SET_NAME} ({len(trajs)} trajectories)")

    for tid in trajs:
        success = process_one_traj(tid, run_dir)
        if not success:
            print(f"  [WARN] {tid} had errors, continuing...")

    # Report + Filter
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


if __name__ == "__main__":
    main()
