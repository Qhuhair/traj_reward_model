"""
GUI-Shepherd 两步流水线：env_parser → annotation → GS_Step1(视觉) → GS_Step2(评分) → core_prm

两步分离设计：
  Step 1: 仅截图 → 视觉识别（元素定位 + 匹配判断），不评分
  Step 2: 纯文本 → 基于 Step 1 结论 + 状态描述 → 最终得分

目的：消除多模态模型在同时处理视觉和逻辑时的上下文冲突。
"""
import os, sys, json, subprocess

from pipelines.paths import PROJECT_ROOT, PYTHON

SET_NAME = sys.argv[1] if len(sys.argv) > 1 else "20250113_21442_test"
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output", "qwen3.5-4b-text-image-gs")
TRAJS_ROOT = os.path.join(PROJECT_ROOT, "trajs")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "LLM_caller", "utils"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "LLM_caller"))
from env_parser.core.pipeline import EnvParserPipeline
from env_parser.utils.path_utils import list_traj_dirs_in_set
from report.writer import save_json
from image_annotator import ImageAnnotator
from annotate_traj_images import annotate_standardized
from two_step_gs_evaluator import TwoStepGSEvaluator


def process_one_traj(traj_id, run_dir):
    out_dir = os.path.join(run_dir, traj_id)
    os.makedirs(out_dir, exist_ok=True)
    traj_dir = os.path.join(TRAJS_ROOT, SET_NAME, traj_id)

    std_path = os.path.join(out_dir, "standardized.json")
    llm_path = os.path.join(out_dir, "llm_scores.json")
    prm_path = os.path.join(out_dir, "prm_scores.json")

    # Stage 1: env_parser
    pipeline = EnvParserPipeline()
    std_data = pipeline.run(traj_dir)
    save_json(std_path, std_data)
    n_steps = len(std_data["steps"])
    print(f"  Steps: {n_steps}")

    # Stage 1.5: annotation
    annotator = ImageAnnotator()
    annotate_standardized(std_data, os.path.join(out_dir, "annotated_images"), annotator)
    save_json(std_path, std_data)
    print(f"  Annotations saved")

    # Stage 2: GS two-step evaluation (Step 1 视觉 → Step 2 评分)
    print(f"\n[GS Two-Step LLM]")
    evaluator = TwoStepGSEvaluator()
    try:
        results = evaluator.evaluate_trajectory(std_data)
    except Exception as e:
        print(f"  [FAIL] TwoStepGSEvaluator error: {e}")
        return

    # Save
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  -> {llm_path}")

    # Inject scores into standardized
    for step, r in zip(std_data["steps"], results):
        step["score"] = r["score"]
    save_json(std_path, std_data)

    # Stage 3: core_prm
    print(f"\n[core_prm]")
    result = subprocess.run(
        [PYTHON, os.path.join(PROJECT_ROOT, "core_prm", "main.py"), std_path, prm_path],
        capture_output=True, text=True, timeout=60
    )
    if result.stdout.strip():
        print(f"  {result.stdout.strip()[:100]}")

    print(f"  [OK] Done: {traj_id}")
    return True


def main():
    run_dir = os.path.join(OUTPUT_BASE, SET_NAME)
    os.makedirs(run_dir, exist_ok=True)

    trajs = list_traj_dirs_in_set(TRAJS_ROOT, SET_NAME)
    print(f"Processing set: {SET_NAME} ({len(trajs)} trajectories)")

    for tid in trajs:
        process_one_traj(tid, run_dir)

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
            print(f"    {cid}: {s.get('category_counts', {}).get(cid, 0)}")
    except Exception as e:
        print(f"  Filter error: {e}")


if __name__ == "__main__":
    main()
