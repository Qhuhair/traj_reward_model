"""AndroidWorld checkpoint 结果摘要。"""

import gzip
import pickle
from pathlib import Path
from typing import Any


def summarize_checkpoint_dir(checkpoint_dir: str | Path) -> dict[str, Any]:
    """读取 AndroidWorld checkpoint 目录并汇总成功率。"""
    episodes = _load_episodes(Path(checkpoint_dir))
    total = len(episodes)
    successful = sum(float(ep.get("is_successful", 0.0) or 0.0) for ep in episodes)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "total_episodes": total,
        "successful_episodes": successful,
        "success_rate": successful / total if total else 0.0,
        "mean_episode_length": _mean(ep.get("episode_length") for ep in episodes),
    }


def _load_episodes(checkpoint_dir: Path) -> list[dict[str, Any]]:
    episodes = []
    for path in sorted(checkpoint_dir.glob("*.pkl.gz")):
        with gzip.open(path, "rb") as f:
            episodes.extend(pickle.load(f))
    return episodes


def _mean(values) -> float:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else 0.0
