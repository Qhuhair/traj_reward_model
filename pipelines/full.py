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
import argparse
import contextlib

from pipelines.paths import PROJECT_ROOT, PYTHON
sys.path.insert(0, PROJECT_ROOT)

from env_parser.core.pipeline import EnvParserPipeline
from env_parser.utils.path_utils import list_traj_sets, list_traj_dirs_in_set
from report.writer import save_json, generate as generate_report
from filter.core.filter_engine import FilterEngine

TRAJS_ROOT = os.path.join(PROJECT_ROOT, "trajs")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
# 兼容历史命名：内部标准文件仍是 llm_scores.json，同时额外同步 LLMscore.json。
LLM_ALIAS_NAME = "LLMscore.json"


def _sub_run(module: str, *args, timeout: int = 2400) -> int:
    """以子进程方式运行阶段模块，统一收集 stdout/stderr 和超时状态。"""
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
        # 失败时保留更多 stderr，方便定位模型接口、依赖或格式错误。
        err_lines = result.stderr.strip().split("\n")
        max_lines = 20 if result.returncode else 1
        for line in err_lines[:max_lines]:
            print(f"  [stderr] {line[:180]}")
    return result.returncode


def _sync_llm_alias(llm_path: str) -> None:
    """将标准 LLM 评分文件同步为兼容文件名，便于旧脚本或人工查找。"""
    if not os.path.exists(llm_path):
        return
    alias_path = os.path.join(os.path.dirname(llm_path), LLM_ALIAS_NAME)
    with open(llm_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    save_json(alias_path, data)


def _run_llm(std_path: str, llm_path: str, args, indices: list = None) -> int:
    """运行 LLM_caller；indices 不为空时只重跑 QA 未通过的步骤。"""
    cmd_args = [std_path, llm_path, f"--model={args.llm_model}", f"--prompt={args.llm_prompt}"]
    if args.input_mode == "text":
        # 纯文本模式强制切断图片输入，避免 adapter 自动读取标准化 JSON 中的截图路径。
        cmd_args.append("--text-only")
    if indices:
        cmd_args.append("--indices=" + ",".join(str(i) for i in indices))
    return _sub_run("LLM_caller", *cmd_args, timeout=args.llm_timeout)


def _annotate_images(std_data: dict, out_dir: str) -> dict:
    """多模态模式下生成点击位置标注图，并把标注路径写回标准化数据。"""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "LLM_caller", "utils"))
    from image_annotator import ImageAnnotator
    from annotate_traj_images import annotate_standardized

    annotated_dir = os.path.join(out_dir, "annotated_images")
    annotate_standardized(std_data, annotated_dir, ImageAnnotator())
    print(f"  -> annotated images saved to {annotated_dir}")
    return std_data


def run_single(traj_id: str, run_dir: str, set_name: str = None, args=None):
    """处理单条轨迹：解析、LLM 评分、QA 重试、PRM 计算。"""
    if set_name:
        traj_dir = os.path.join(TRAJS_ROOT, set_name, traj_id)
    else:
        traj_dir = os.path.join(TRAJS_ROOT, traj_id)

    if not os.path.isdir(traj_dir):
        print(f"[SKIP] Directory not found: {traj_dir}")
        return False

    out_dir = os.path.join(run_dir, traj_id)
    os.makedirs(out_dir, exist_ok=True)

    std_path = os.path.join(out_dir, "standardized.json")
    llm_path = os.path.join(out_dir, "llm_scores.json")
    prm_path = os.path.join(out_dir, "prm_scores.json")
    qa_path  = os.path.join(out_dir, "qa_reports.json")

    # 每条轨迹都独立落盘，失败时不污染其他轨迹的输出目录。
    print(f"\n{'='*60}")
    print(f"Pipeline: {traj_id}")
    print(f"{'='*60}")

    # Stage 1：把原始轨迹解析成统一的 State-Action JSON。
    print(f"\n[1/4] env_parser")
    pipeline = EnvParserPipeline()
    std_data = pipeline.run(traj_dir)
    save_json(std_path, std_data)
    print(f"  -> {std_path} ({len(std_data['steps'])} steps)")

    if args.input_mode == "multimodal":
        # 只有显式指定多模态时才生成图片标注；默认纯文本流程不产生 annotated_images。
        print(f"\n[1.5/4] Image Annotation")
        std_data = _annotate_images(std_data, out_dir)
        save_json(std_path, std_data)

    # Stage 2：首次运行会评分全部步骤；已有结果时直接复用，避免重复调用模型。
    if os.path.exists(llm_path):
        print(f"\n[2/4] LLM_caller  (skipped, output exists)")
        _sync_llm_alias(llm_path)
    else:
        print(f"\n[2/4] LLM_caller")
        if _run_llm(std_path, llm_path, args) != 0:
            print("[FAIL] LLM_caller failed, stopping")
            return False
        _sync_llm_alias(llm_path)

    # Stage 2.5：QA 只用于发现格式或内容质量问题，并触发有限次数的局部重试。
    print(f"\n[QA] logic_QA review + retry")
    max_retries = 3
    for attempt in range(max_retries + 1):
        if _sub_run("logic_QA", std_path, llm_path, qa_path) != 0:
            print("[FAIL] logic_QA failed, stopping")
            return False

        qa = json.load(open(qa_path, "r", encoding="utf-8"))
        failed = [r["step_idx"] for r in qa if not r.get("is_accepted")]

        # 没有失败步骤，或达到最大重试次数时结束 QA 循环。
        if not failed or attempt == max_retries:
            if attempt > 0:
                print(f"  Retry done: {len(failed)} steps still failed after {attempt} retries")
            break

        print(f"  [Retry {attempt+1}/{max_retries}] {len(failed)} failed steps: {failed}")
        # 重试只覆盖失败 step，成功 step 保持原结果，减少模型调用和结果波动。
        if _run_llm(std_path, llm_path, args, failed) != 0:
            print("[FAIL] LLM_caller retry failed, stopping")
            return False
        _sync_llm_alias(llm_path)

    # 将 LLM 分数写回 standardized.json，供 core_prm 读取统一 schema。
    if os.path.exists(llm_path):
        with open(llm_path, "r", encoding="utf-8") as f:
            llm_results = json.load(f)
        for step, llm in zip(std_data["steps"], llm_results):
            step["score"] = llm["score"]
        save_json(std_path, std_data)
        _sync_llm_alias(llm_path)

    # Stage 3：基于步骤分数计算 Progress、TD-Error 和 GAE。
    print(f"\n[3/4] core_prm")
    if _sub_run("core_prm", std_path, prm_path) != 0:
        print("[FAIL] core_prm failed, stopping")
        return False

    print(f"\nDone. Output: {out_dir}")
    return True


def process_set(set_name: str, run_dir: str = None, args=None):
    """处理一个轨迹集，并在集合级别生成报告和筛选结果。"""
    run_dir = run_dir or os.path.join(OUTPUT_ROOT, set_name)
    os.makedirs(run_dir, exist_ok=True)

    trajs = list_traj_dirs_in_set(TRAJS_ROOT, set_name)
    if not trajs:
        print(f"[SKIP] No trajectories found in set: {set_name}")
        return

    print(f"\n{'#'*60}")
    print(f"  Set: {set_name}  ({len(trajs)} trajectories)")
    print(f"{'#'*60}")

    success_count = 0
    for tid in trajs:
        if run_single(tid, run_dir, set_name, args):
            success_count += 1

    if success_count == 0:
        print("[FAIL] No successful trajectories; skip report and filter.")
        return False

    # 集合内所有轨迹处理完成后，再统一生成 summary/master_summary。
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
    return True


def parse_args():
    """解析命令行参数，默认走纯文本 vLLM 评分流程。"""
    parser = argparse.ArgumentParser(description="运行完整轨迹评估流水线。")
    parser.add_argument("target", nargs="?", help="轨迹集名称；省略时处理所有轨迹集。")
    parser.add_argument("--output-name", help="指定 output/ 下的输出子目录名。")
    parser.add_argument("--output-root", help="指定完整输出目录路径。")
    parser.add_argument("--input-mode", choices=("text", "multimodal"), default="text",
                        help="LLM 输入模式。默认 text，只传文本，不做图片标注。")
    parser.add_argument("--llm-model", default="qwen_vllm_text",
                        help="LLM_caller 使用的模型配置名。默认 qwen_vllm_text。")
    parser.add_argument("--llm-prompt", default="RRM_Qwen",
                        help="LLM_caller 使用的 prompt 名。默认 RRM_Qwen。")
    parser.add_argument("--llm-timeout", type=int, default=2400,
                        help="单次 LLM_caller 子进程超时时间，单位秒。")
    parser.add_argument("--quiet", action="store_true",
                        help="不向控制台打印流水线日志，改写入 pipeline.log。")
    return parser.parse_args()


def resolve_run_dir(target: str, args) -> str:
    """解析单个 target 的输出目录，优先级：output_root > output_name > target。"""
    if args.output_root:
        return os.path.abspath(args.output_root)
    if args.output_name:
        return os.path.join(OUTPUT_ROOT, args.output_name)
    return os.path.join(OUTPUT_ROOT, target)


def resolve_all_sets_root(args) -> str:
    """解析全量运行的总输出目录；每个轨迹集会写入其子目录。"""
    if args.output_root:
        return os.path.abspath(args.output_root)
    if args.output_name:
        return os.path.join(OUTPUT_ROOT, args.output_name)
    return OUTPUT_ROOT


def _log_root(args) -> str:
    """quiet 模式下 pipeline.log 的根目录。"""
    if args.target:
        return resolve_run_dir(args.target, args)
    return resolve_all_sets_root(args)


def _run(args) -> None:
    """根据是否指定 target，选择单轨迹集/单轨迹或全量运行。"""
    if args.target:
        arg = args.target
        # 如果参数是已知的轨迹集 → 只处理该集
        sets = list_traj_sets(TRAJS_ROOT)
        if arg in sets:
            process_set(arg, resolve_run_dir(arg, args), args)
        else:
            # 尝试作为旧格式的单轨迹处理
            run_dir = resolve_run_dir(arg, args)
            os.makedirs(run_dir, exist_ok=True)
            if run_single(arg, run_dir, args=args):
                generate_report(run_dir)
                FilterEngine().run(run_dir)
            else:
                print("[FAIL] Single trajectory failed; skip report and filter.")
    else:
        # 未指定 target 时遍历 trajs/ 下所有轨迹集，并保留轨迹集名作为子目录。
        all_sets_root = resolve_all_sets_root(args)
        for s in list_traj_sets(TRAJS_ROOT):
            process_set(s, os.path.join(all_sets_root, s), args)


def main():
    args = parse_args()
    if not args.quiet:
        _run(args)
        return

    # quiet 模式不丢日志，只把控制台输出重定向到 pipeline.log。
    log_root = _log_root(args)
    os.makedirs(log_root, exist_ok=True)
    log_path = os.path.join(log_root, "pipeline.log")
    with open(log_path, "a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            _run(args)


if __name__ == "__main__":
    main()
