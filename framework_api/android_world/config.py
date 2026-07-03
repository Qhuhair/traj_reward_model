"""AndroidWorld 评测配置结构。"""

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_ANDROID_WORLD_ROOT = Path.home() / "code/android_mobile_world_bench/android_world"


@dataclass(frozen=True)
class AndroidWorldEvalConfig:
    """描述一次 AndroidWorld 评测，不直接绑定具体 agent 实现。"""

    android_world_root: Path = DEFAULT_ANDROID_WORLD_ROOT
    suite_family: str = "android_world"
    agent_name: str = "prm_api_agent"
    base_url: str = "http://127.0.0.1:8002/v1"
    policy_model: str = "Qwen3.5-4B"
    prm_model: str = "crossapp_kto"
    output_path: Path = Path("output/android_world_eval")
    tasks: tuple[str, ...] = ()
    n_task_combinations: int = 1
    task_random_seed: int = 30
    console_port: int = 5554
    adb_path: str | None = None
    perform_emulator_setup: bool = False
    fixed_task_seed: bool = False
    extra_flags: dict[str, str | int | float | bool] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict, options: dict | None = None) -> "AndroidWorldEvalConfig":
        """从 APIRequest 的 payload/options 构造配置。"""
        merged = {**payload, **(options or {})}
        tasks = merged.get("tasks") or ()
        if isinstance(tasks, str):
            tasks = tuple(x.strip() for x in tasks.split(",") if x.strip())
        return cls(
            android_world_root=Path(merged.get("android_world_root", DEFAULT_ANDROID_WORLD_ROOT)),
            suite_family=merged.get("suite_family", "android_world"),
            agent_name=merged.get("agent_name", "prm_api_agent"),
            base_url=merged.get("base_url", "http://127.0.0.1:8002/v1"),
            policy_model=merged.get("policy_model", "Qwen3.5-4B"),
            prm_model=merged.get("prm_model", "crossapp_kto"),
            output_path=Path(merged.get("output_path", "output/android_world_eval")),
            tasks=tuple(tasks),
            n_task_combinations=int(merged.get("n_task_combinations", 1)),
            task_random_seed=int(merged.get("task_random_seed", 30)),
            console_port=int(merged.get("console_port", 5554)),
            adb_path=merged.get("adb_path"),
            perform_emulator_setup=bool(merged.get("perform_emulator_setup", False)),
            fixed_task_seed=bool(merged.get("fixed_task_seed", False)),
            extra_flags=dict(merged.get("extra_flags", {})),
        )
