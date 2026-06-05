"""CLI entrypoint for starting the vLLM service."""

import argparse
import os

from .config import LOG_DIR, VLLMConfig
from .process import start_background


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Qwen3.5-4B with vLLM.")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--served-model-name", default="Qwen3.5-4B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS", "3"))
    parser.add_argument("--log-file", default=str(LOG_DIR / "vllm_qwen35_4b.log"))
    parser.add_argument("--enable-flashinfer-sampler", action="store_true")
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
        gpu_ids=args.gpu_ids,
        log_file=LOG_DIR / os.path.basename(args.log_file),
        disable_flashinfer_sampler=not args.enable_flashinfer_sampler,
    )


def main() -> None:
    config = build_config(parse_args())
    process = start_background(config)
    print(f"Started vLLM pid={process.pid}")
    print(f"Log file: {config.log_file}")
    print(f"GPU ids: {config.gpu_ids}")
    print(f"Model: {config.model_path}")


if __name__ == "__main__":
    main()

