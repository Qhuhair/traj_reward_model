"""复用 LLM_caller 的导入桥接。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLM_ROOT = PROJECT_ROOT / "LLM_caller"


def load_llm_caller():
    """按 LLM_caller 的历史导入方式加载 LLMCaller。"""
    for path in (str(LLM_ROOT), str(PROJECT_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from caller import LLMCaller

    return LLMCaller
