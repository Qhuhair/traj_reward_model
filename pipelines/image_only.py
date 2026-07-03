"""纯图片模式流水线：env_parser → annotation → LLM_caller → logic_QA → core_prm。"""
import os, sys, json, subprocess

from pipelines.paths import PROJECT_ROOT, PYTHON
from pipelines.common_cli import (
    build_parser,
    iter_targets,
    resolve_all_root,
    resolve_target_dir,
    run_quietly_if_needed,
)

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "output", "qwen3.5-image")
TRAJS_ROOT = os.path.join(PROJECT_ROOT, "trajs")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "LLM_caller", "utils"))
from env_parser.core.pipeline import EnvParserPipeline
from env_parser.utils.path_utils import list_traj_dirs_in_set
from report.writer import save_json
from image_annotator import ImageAnnotator
from annotate_traj_images import annotate_standardized


def run_cmd(name, cmd_list, timeout=3600):
    """运行阶段子进程，并过滤低价值日志。"""
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
    """处理单条轨迹；该模式已废弃，仅保留兼容能力。"""
    out_dir = os.path.join(run_dir, traj_id)
    os.makedirs(out_dir, exist_ok=True)
    traj_dir = os.path.join(TRAJS_ROOT, set_name, traj_id)

    std_path = os.path.join(out_dir, "standardized.json")
    llm_path = os.path.join(out_dir, "llm_scores.json")
    qa_path = os.path.join(out_dir, "qa_reports.json")
    prm_path = os.path.join(out_dir, "prm_scores.json")

    # Stage 1：解析原始轨迹。
    pipeline = EnvParserPipeline()
    std_data = pipeline.run(traj_dir)
    save_json(std_path, std_data)
    print(f"  Steps: {len(std_data['steps'])}")

    # Stage 1.5：生成标注图片。
    annotator = ImageAnnotator()
    annotate_standardized(std_data, os.path.join(out_dir, "annotated_images"), annotator)
    save_json(std_path, std_data)
    print(f"  Annotations saved")

    # Stage 2：使用 vLLM 多模态接口执行纯图片 prompt。
    cmd = [PYTHON, os.path.join(PROJECT_ROOT, "LLM_caller", "main.py"),
           std_path, llm_path,
           "--model=qwen_vllm_mm", "--prompt=RRM_Qwen_ImageOnly"]
    if not run_cmd("LLM_caller", cmd):
        return False

    # Stage 2.5：QA 审查。
    cmd = [PYTHON, os.path.join(PROJECT_ROOT, "logic_QA", "main.py"),
           std_path, llm_path, qa_path]
    if not run_cmd("logic_QA", cmd):
        return False

    # 将 LLM 分数写回 standardized.json，供 PRM 读取。
    with open(llm_path, "r", encoding="utf-8") as f:
        llm_results = json.load(f)
    for step, llm in zip(std_data["steps"], llm_results):
        step["score"] = llm["score"]
    save_json(std_path, std_data)

    # Stage 3：计算过程奖励。
    cmd = [PYTHON, os.path.join(PROJECT_ROOT, "core_prm", "main.py"),
           std_path, prm_path]
    if not run_cmd("core_prm", cmd):
        return False

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

    # 所有轨迹完成后统一生成集合报告和筛选结果。
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
    return build_parser("运行纯图片兼容轨迹评估流水线。").parse_args()


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
