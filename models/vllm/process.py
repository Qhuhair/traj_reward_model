"""Process helpers for running vLLM in the background."""

import subprocess

from .command import build_command, build_env
from .config import VLLMConfig


def ensure_log_dir(config: VLLMConfig) -> None:
    config.log_file.parent.mkdir(parents=True, exist_ok=True)


def start_background(config: VLLMConfig) -> subprocess.Popen:
    ensure_log_dir(config)
    log_handle = config.log_file.open("a", encoding="utf-8")
    return subprocess.Popen(
        build_command(config),
        env=build_env(config),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

