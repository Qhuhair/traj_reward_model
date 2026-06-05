"""Shared paths for pipeline entrypoints."""

import os
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
PYTHON = os.sys.executable
