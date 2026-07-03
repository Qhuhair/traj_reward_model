"""AndroidWorld 评测命令构建。"""

import shlex

from .config import AndroidWorldEvalConfig


def build_android_world_command(config: AndroidWorldEvalConfig) -> list[str]:
    """构建 AndroidWorld 评测命令；当前先复用 run.py 的命令行入口。"""
    cmd = [
        "python",
        "run.py",
        f"--suite_family={config.suite_family}",
        f"--agent_name={config.agent_name}",
        f"--n_task_combinations={config.n_task_combinations}",
        f"--task_random_seed={config.task_random_seed}",
        f"--console_port={config.console_port}",
        f"--output_path={config.output_path}",
    ]
    if config.tasks:
        cmd.append("--tasks=" + ",".join(config.tasks))
    if config.adb_path:
        cmd.append(f"--adb_path={config.adb_path}")
    if config.perform_emulator_setup:
        cmd.append("--perform_emulator_setup")
    if config.fixed_task_seed:
        cmd.append("--fixed_task_seed")
    for key, value in sorted(config.extra_flags.items()):
        cmd.append(f"--{key}" if value is True else f"--{key}={value}")
    return cmd


def command_preview(config: AndroidWorldEvalConfig) -> str:
    """返回可复制执行的命令字符串，便于人工检查。"""
    return shlex.join(build_android_world_command(config))


def model_env(config: AndroidWorldEvalConfig) -> dict[str, str]:
    """定义策略模型和 PRM 的 API 标识，供后续自定义 agent 读取。"""
    return {
        "PRM_API_BASE_URL": config.base_url,
        "PRM_POLICY_MODEL": config.policy_model,
        "PRM_REWARD_MODEL": config.prm_model,
    }
