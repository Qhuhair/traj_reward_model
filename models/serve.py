"""Compatibility wrapper for starting the vLLM model service."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.vllm.cli import main


if __name__ == "__main__":
    main()
