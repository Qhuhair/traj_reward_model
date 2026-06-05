"""
全流水线集成脚本。各阶段通过 JSON 文件传递数据，松耦合。

用法:
    python run_pipeline.py 20260113_214142_subgoal   # 处理指定轨迹集
    python run_pipeline.py                            # 处理所有轨迹集 + 生成总报告 + 筛选

流水线: env_parser -> LLM_caller -> core_prm -> logic_QA -> report -> filter
"""
import os
import sys
import json
import subprocess

from pipelines.paths import PROJECT_ROOT, PYTHON
sys.path.insert(0, PROJECT_ROOT)

from env_parser.core.pipeline import EnvParserPipeline
from env_parser.utils.path_utils import list_traj_sets, list_traj_dirs_in_set
from report.writer import save_json, generate as generate_report
from filter.core.filter_engine import FilterEngine

TRAJS_ROOT = os.path.join(PROJECT_ROOT, "trajs")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")


def _sub_run(module: str, *args, timeout: int = 2400) -> int:
    workdir = os.path.join(PROJECT_ROOT, module)
    cmd = [PYTHON, "main.py"] + list(args)
    try:
        result = subprocess.run(cmd, cwd=workdir, capture_output=True,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {module} exceeded {timeout}s")
        return 1

    for line in result.stdout.strip().split("\n"):
        cleaned = line.strip()
        if cleaned:
            print(f"  {cleaned}")
    if result.stderr.strip():
        first_err = result.stderr.strip().split("\n")[0]
        print(f"  [stderr] {first_err[:120]}")
    return result.returncode


def run_single(traj_id: str, run_dir: str, set_name: str = None):
    if set_name:
        traj_dir = os.path.join(TRAJS_ROOT, set_name, traj_id)
    else:
        traj_dir = os.path.join(TRAJS_ROOT, traj_id)

    if not os.path.isdir(traj_dir):
        print(f"[SKIP] Directory not found: {traj_dir}")
        return

    out_dir = os.path.join(run_dir, traj_id)
    os.makedirs(out_dir, exist_ok=True)

    std_path = os.path.join(out_dir, "standardized.json")
    llm_path = os.path.join(out_dir, "llm_scores.json")
    prm_path = os.path.join(out_dir, "prm_scores.json")
    qa_path  = os.path.join(out_dir, "qa_reports.json")

    print(f"\n{'='*60}")
    print(f"Pipeline: {traj_id}")
    print(f"{'='*60}")

    # Stage 1: env_parser
    print(f"\n[1/4] env_parser")
    pipeline = EnvParserPipeline()
    std_data = pipeline.run(traj_dir)
    save_json(std_path, std_data)
    print(f"  -> {std_path} ({len(std_data['steps'])} steps)")

    # Stage 1.5: 图片标注 (生成带标注的截图)
    print(f"\n[1.5/4] Image Annotation")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "LLM_caller", "utils"))
    from image_annotator import ImageAnnotator
    from annotate_traj_images import annotate_standardized
    annotated_dir = os.path.join(out_dir, "annotated_images")
    annotator = ImageAnnotator()
    annotate_standardized(std_data, annotated_dir, annotator)
    # 更新 std_path 引用: 写入标注版 JSON 供 LLM_caller 使用
    std_annotated = std_data
    save_json(std_path, std_annotated)
    print(f"  -> annotated images saved to {annotated_dir}")

    # Stage 2: LLM_caller (首次全部调用)
    if os.path.exists(llm_path):
        print(f"\n[2/4] LLM_caller  (skipped, output exists)")
    else:
        print(f"\n[2/4] LLM_caller")
        if _sub_run("LLM_caller", std_path, llm_path, timeout=2400) != 0:
            print("[FAIL] LLM_caller failed, stopping")
            return

    # Stage 2.5: QA + Retry loop (最多3次)
    print(f"\n[QA] logic_QA review + retry")
    max_retries = 3
    for attempt in range(max_retries + 1):
        if _sub_run("logic_QA", std_path, llm_path, qa_path) != 0:
            print("[FAIL] logic_QA failed, stopping")
            return

        qa = json.load(open(qa_path, "r", encoding="utf-8"))
        failed = [r["step_idx"] for r in qa if not r.get("is_accepted")]

        if not failed or attempt == max_retries:
            if attempt > 0:
                print(f"  Retry done: {len(failed)} steps still failed after {attempt} retries")
            break

        print(f"  [Retry {attempt+1}/{max_retries}] {len(failed)} failed steps: {failed}")

        # 使用上下文感知的滑动窗口重试
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "LLM_caller", "utils"))
        from qa_retry_with_context import retry_failed_steps

        with open(std_path, "r", encoding="utf-8") as f:
            std_ref = json.load(f)
        with open(llm_path, "r", encoding="utf-8") as f:
            llm_ref = json.load(f)

        updated_scores = retry_failed_steps(std_ref, llm_ref, qa,
                                            model="qwen_local_mm",
                                            prompt="RRM_Qwen_Sliding",
                                            max_retries=1)  # 每次 QA 循环内只做 1 次上下文重试

        # 写回更新后的评分
        with open(llm_path, "w", encoding="utf-8") as f:
            json.dump(updated_scores, f, ensure_ascii=False, indent=2)

    # Inject scores to standardized.json
    if os.path.exists(llm_path):
        with open(llm_path, "r", encoding="utf-8") as f:
            llm_results = json.load(f)
        for step, llm in zip(std_data["steps"], llm_results):
            step["score"] = llm["score"]
        save_json(std_path, std_data)

    # Stage 3: core_prm
    print(f"\n[3/4] core_prm")
    if _sub_run("core_prm", std_path, prm_path) != 0:
        print("[FAIL] core_prm failed, stopping")
        return

    print(f"\nDone. Output: {out_dir}")


def process_set(set_name: str):
    run_dir = os.path.join(OUTPUT_ROOT, set_name)
    os.makedirs(run_dir, exist_ok=True)

    trajs = list_traj_dirs_in_set(TRAJS_ROOT, set_name)
    if not trajs:
        print(f"[SKIP] No trajectories found in set: {set_name}")
        return

    print(f"\n{'#'*60}")
    print(f"  Set: {set_name}  ({len(trajs)} trajectories)")
    print(f"{'#'*60}")

    for tid in trajs:
        run_single(tid, run_dir, set_name)

    generate_report(run_dir)

    print(f"\n[Filter] 轨迹筛选...")
    engine = FilterEngine()
    report = engine.run(run_dir)
    s = report.get("summary", {})
    print(f"  Total: {s.get('total_trajectories', 0)} trajectories")
    counts = s.get("category_counts", {})
    labels = {"A": "优质轨迹", "B": "高分低质", "C": "低分高质", "D": "劣质轨迹"}
    for cid in ["A", "B", "C", "D"]:
        cnt = counts.get(cid, 0)
        print(f"  {cid}-{labels.get(cid, '?')}: {cnt}")
    out = os.path.join(run_dir, "filtered.json")
    print(f"  Saved: {out}")


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # 如果参数是已知的轨迹集 → 只处理该集
        sets = list_traj_sets(TRAJS_ROOT)
        if arg in sets:
            process_set(arg)
        else:
            # 尝试作为旧格式的单轨迹处理
            run_dir = os.path.join(OUTPUT_ROOT, arg)
            os.makedirs(run_dir, exist_ok=True)
            run_single(arg, run_dir)
            generate_report(run_dir)
            FilterEngine().run(run_dir)
    else:
        for s in list_traj_sets(TRAJS_ROOT):
            process_set(s)


if __name__ == "__main__":
    main()
