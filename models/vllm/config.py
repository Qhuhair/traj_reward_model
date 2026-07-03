"""Configuration helpers for the vLLM service."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
LOG_DIR = PACKAGE_DIR / "log"


def dated_log_file(prefix: str = "vllm_qwen35_4b") -> Path:
    return LOG_DIR / f"{prefix}_{date.today():%Y%m%d}.log"


@dataclass(frozen=True)
class VLLMConfig:
    model_path: str
    served_model_name: str = "Qwen3.5-4B"
    host: str = "0.0.0.0"
    port: int = 8002
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.65
    max_model_len: int = 8192
    max_num_seqs: int = 128
    gpu_ids: str = "5"
    log_file: Path = dated_log_file()
    trust_remote_code: bool = True
    disable_flashinfer_sampler: bool = True
    enable_thinking: bool = False
    lora_modules: tuple[str, ...] = ()
