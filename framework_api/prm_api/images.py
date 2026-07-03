"""PRM 图片保存工具。"""

from __future__ import annotations

import base64
from pathlib import Path

from .schema import PRMScoreRequest


def save_screenshot(episode_dir: Path, request: PRMScoreRequest) -> Path | None:
    """把当前步骤截图保存为图片，供多模态 scorer 复用。"""
    if not request.screenshot:
        return None
    image_dir = episode_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"step_{request.step_index:04d}.png"
    path.write_bytes(base64.b64decode(request.screenshot))
    return path
