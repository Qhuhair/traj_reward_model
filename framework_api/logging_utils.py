"""framework_api 模块的日志工具。"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


LOG_ROOT = Path(__file__).resolve().parent / "logs"


class TeeStream:
    """同时写入原始终端和日志文件的输出流。"""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, data: str) -> int:
        """保持终端可见，同时把所有输出追加进日志。"""
        self.terminal.write(data)
        self.log_file.write(data)
        self.flush()
        return len(data)

    def flush(self) -> None:
        """同步刷新终端和文件，避免服务异常退出时丢日志。"""
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        """保留 uvicorn 对终端能力的判断。"""
        return self.terminal.isatty()


def dated_log_path(component: str, prefix: str) -> Path:
    """生成 framework_api/logs/<component>/<prefix>_YYYYMMDD.log。"""
    log_dir = LOG_ROOT / component
    log_dir.mkdir(parents=True, exist_ok=True)
    date_suffix = datetime.now().strftime("%Y%m%d")
    return log_dir / f"{prefix}_{date_suffix}.log"


def configure_file_logger(name: str, component: str, prefix: str) -> logging.Logger:
    """配置追加写入的模块日志器。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_path = dated_log_path(component, prefix)
    if not _has_file_handler(logger, log_path):
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def redirect_std_streams(component: str, prefix: str) -> Path:
    """把 stdout/stderr 追加复制到同一个日期日志文件。"""
    log_path = dated_log_path(component, prefix)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    return log_path


def ensure_component_log_dirs() -> None:
    """预创建 base/prm 日志目录，PRM API 后续接入时复用。"""
    (LOG_ROOT / "base").mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / "prm").mkdir(parents=True, exist_ok=True)


def _has_file_handler(logger: logging.Logger, log_path: Path) -> bool:
    """避免同一进程重复添加相同文件 handler。"""
    target = str(log_path)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == target:
            return True
    return False
