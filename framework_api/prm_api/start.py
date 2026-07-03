"""PRM API 本地启动入口。"""

from __future__ import annotations

import argparse
import os

import uvicorn

from framework_api.logging_utils import ensure_component_log_dirs, redirect_std_streams


def parse_args() -> argparse.Namespace:
    """解析 PRM API 启动参数。"""
    parser = argparse.ArgumentParser(description="启动 AndroidWorld PRM API 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8102, help="监听端口")
    parser.add_argument("--base-url", default=None, help="vLLM/OpenAI 兼容服务地址")
    parser.add_argument("--model", default=None, help="PRM 模型名称，例如 LoRA 标识 crossapp_kto")
    parser.add_argument("--mode", default=None, help="默认评分模式：text_step/text_window/multimodal_step/multimodal_window/crossapp_multimodal_window")
    parser.add_argument("--window-size", type=int, default=None, help="窗口模式保留的最近步骤数量")
    return parser.parse_args()


def main() -> None:
    """设置环境变量后启动 PRM FastAPI 应用。"""
    args = parse_args()
    ensure_component_log_dirs()
    log_path = redirect_std_streams("prm", "prm_api")
    print(f"PRM API 日志追加写入：{log_path}")
    if args.base_url:
        os.environ["PRM_API_OPENAI_BASE_URL"] = args.base_url
    if args.model:
        os.environ["PRM_API_MODEL"] = args.model
    if args.mode:
        os.environ["PRM_API_MODE"] = args.mode
    if args.window_size is not None:
        os.environ["PRM_API_WINDOW_SIZE"] = str(args.window_size)
    uvicorn.run("framework_api.prm_api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
