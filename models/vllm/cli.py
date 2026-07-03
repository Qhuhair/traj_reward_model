"""CLI entrypoint for starting the vLLM service."""

import argparse
import os

from .config import VLLMConfig, dated_log_file
from .process import start_background, start_foreground


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start one base model with optional vLLM LoRA adapters.")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--served-model-name", default="Qwen3.5-4B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS", "4"))
    parser.add_argument("--log-file", default=str(dated_log_file()))
    parser.add_argument("--enable-flashinfer-sampler", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--lora-path", default=os.environ.get("LORA_PATH"))
    parser.add_argument("--lora-name", default=os.environ.get("LORA_NAME", "qwen35_lora"))
    parser.add_argument("--lora-module", action="append", default=[])
    parser.add_argument("--foreground", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> VLLMConfig:
    if not args.model_path:
        raise SystemExit("MODEL_PATH is not set. Pass --model-path or export it.")
    return VLLMConfig(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_ids=args.gpu_ids,
        log_file=dated_log_file().parent / os.path.basename(args.log_file),
        disable_flashinfer_sampler=not args.enable_flashinfer_sampler,
        enable_thinking=args.enable_thinking,
        lora_modules=_lora_modules(args),
    )


def _lora_modules(args: argparse.Namespace) -> tuple[str, ...]:
    modules = list(args.lora_module)
    if args.lora_path:
        modules.append(f"{args.lora_name}={args.lora_path}")
    return tuple(modules)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    _log_console(config, "Starting vLLM service")
    _log_console(config, f"Log file: {config.log_file}")
    _log_console(config, f"GPU ids: {config.gpu_ids}")
    _log_console(config, f"Model: {config.model_path}")
    _log_model_names(config)
    if args.foreground:
        raise SystemExit(start_foreground(config))
    process = start_background(config)
    _log_console(config, f"Started vLLM pid={process.pid}")


def _log_model_names(config: VLLMConfig) -> None:
    """记录对外模型名；请求中的 model 字段用它区分 base 和 LoRA。"""
    _log_console(config, f"Base request model: {config.served_model_name}")
    for module in config.lora_modules:
        name = module.split("=", 1)[0]
        _log_console(config, f"LoRA request model: {name}")


def _log_console(config: VLLMConfig, message: str) -> None:
    print(message)
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    with config.log_file.open("a", encoding="utf-8") as f:
        f.write(f"{message}\n")


if __name__ == "__main__":
    main()
