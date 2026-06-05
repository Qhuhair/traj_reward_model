"""Download a Hugging Face model file through a mirror endpoint."""

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one model file.")
    parser.add_argument("--repo-id", default="unsloth/Qwen3.5-4B-GGUF")
    parser.add_argument("--filename", default="Qwen3.5-4B-Q8_0.gguf")
    parser.add_argument("--local-dir", default="models/Qwen3.5-GGUF")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    return parser.parse_args()


def configure_endpoint(endpoint: str) -> None:
    os.environ["HF_ENDPOINT"] = endpoint


def ensure_dir(path: str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def download_file(args: argparse.Namespace) -> str:
    configure_endpoint(args.endpoint)
    target = ensure_dir(args.local_dir)
    return hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def main() -> None:
    path = download_file(parse_args())
    print(f"Downloaded: {path}")


if __name__ == "__main__":
    main()
