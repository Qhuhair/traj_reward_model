"""Configuration helpers for the vLLM service."""

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
LOG_DIR = PACKAGE_DIR / "log"


@dataclass(frozen=True)
class VLLMConfig:
    model_path: str
    served_model_name: str = "Qwen3.5-4B"
    host: str = "0.0.0.0"
    port: int = 8000
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.55
    max_model_len: int = 8192
    gpu_ids: str = "5"
    log_file: Path = LOG_DIR / "vllm_qwen35_4b.log"
    trust_remote_code: bool = True
    disable_flashinfer_sampler: bool = True

