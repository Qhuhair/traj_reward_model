"""流水线入口共享 CLI 工具。"""

import argparse
import contextlib
import os

from env_parser.utils.path_utils import list_traj_sets
from pipelines.paths import PROJECT_ROOT


TRAJS_ROOT = os.path.join(PROJECT_ROOT, "trajs")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")


def build_parser(description: str) -> argparse.ArgumentParser:
    """创建统一命令行解析器，保持各流水线入口参数一致。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("target", nargs="?", help="轨迹集名称；省略时处理所有轨迹集。")
    parser.add_argument("--output-name", help="指定 output/ 下的输出子目录名。")
    parser.add_argument("--output-root", help="指定完整输出目录路径。")
    parser.add_argument("--quiet", action="store_true",
                        help="不向控制台打印流水线日志，改写入 pipeline.log。")
    return parser


def resolve_target_dir(target: str, args, default_base: str) -> str:
    """解析单个轨迹集输出目录，优先级：output_root > output_name > 默认目录。"""
    if args.output_root:
        return os.path.abspath(args.output_root)
    if args.output_name:
        return os.path.join(OUTPUT_ROOT, args.output_name)
    return os.path.join(default_base, target)


def resolve_all_root(args, default_base: str) -> str:
    """解析全量运行的根输出目录；每个轨迹集会写入一个子目录。"""
    if args.output_root:
        return os.path.abspath(args.output_root)
    if args.output_name:
        return os.path.join(OUTPUT_ROOT, args.output_name)
    return default_base


def iter_targets(args) -> list[str]:
    """返回本次需要处理的轨迹集列表。"""
    if args.target:
        return [args.target]
    return list_traj_sets(TRAJS_ROOT)


def log_root(args, default_base: str) -> str:
    """quiet 模式下 pipeline.log 的写入目录。"""
    if args.target:
        return resolve_target_dir(args.target, args, default_base)
    return resolve_all_root(args, default_base)


def run_quietly_if_needed(args, default_base: str, runner) -> None:
    """按需把标准输出和错误输出重定向到 pipeline.log。"""
    if not args.quiet:
        runner()
        return

    root = log_root(args, default_base)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "pipeline.log"), "a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            runner()
