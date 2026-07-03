"""Build vLLM command lines and process environments."""

import os

from .config import VLLMConfig


def build_command(config: VLLMConfig) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        config.model_path,
        "--served-model-name",
        config.served_model_name,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--dtype",
        config.dtype,
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--default-chat-template-kwargs",
        f'{{"enable_thinking": {str(config.enable_thinking).lower()}}}',
    ]
    if config.lora_modules:
        cmd.extend(["--enable-lora", "--lora-modules", *config.lora_modules])
    if config.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def build_env(config: VLLMConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    if config.disable_flashinfer_sampler:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return env
