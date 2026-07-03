"""Process helpers for running vLLM in the background."""

import shlex
import subprocess

from .command import build_command, build_env
from .config import VLLMConfig


def ensure_log_dir(config: VLLMConfig) -> None:
    config.log_file.parent.mkdir(parents=True, exist_ok=True)


def start_background(config: VLLMConfig) -> subprocess.Popen:
    ensure_log_dir(config)
    command = build_command(config)
    log_handle = config.log_file.open("a", encoding="utf-8")
    log_handle.write(f"Command: {shlex.join(command)}\n")
    log_handle.flush()
    return subprocess.Popen(
        command,
        env=build_env(config),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def start_foreground(config: VLLMConfig) -> int:
    ensure_log_dir(config)
    command = build_command(config)
    with config.log_file.open("a", encoding="utf-8") as f:
        f.write(f"Command: {shlex.join(command)}\n")
        f.flush()
        process = subprocess.Popen(
            command,
            env=build_env(config),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout or []:
            print(line, end="")
            f.write(line)
            f.flush()
        return process.wait()
