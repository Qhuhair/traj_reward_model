"""Base API 本地启动入口。"""

from __future__ import annotations

import argparse
import os

import uvicorn

from framework_api.logging_utils import ensure_component_log_dirs, redirect_std_streams


def parse_args() -> argparse.Namespace:
    """解析启动参数，并允许覆盖默认模型服务配置。"""
    parser = argparse.ArgumentParser(description="启动 AndroidWorld Base API 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8101, help="监听端口")
    parser.add_argument("--base-url", default=None, help="vLLM/OpenAI 兼容服务地址")
    parser.add_argument("--model", default=None, help="base 策略模型名称")
    parser.add_argument("--include-image", action="store_true", help="把 screenshot 作为图片输入传给模型")
    return parser.parse_args()


def main() -> None:
    """设置环境变量后交给 uvicorn 托管 FastAPI 应用。"""
    args = parse_args()
    ensure_component_log_dirs()
    log_path = redirect_std_streams("base", "base_api")
    print(f"Base API 日志追加写入：{log_path}")
    if args.base_url:
        os.environ["BASE_API_OPENAI_BASE_URL"] = args.base_url
    if args.model:
        os.environ["BASE_API_MODEL"] = args.model
    if args.include_image:
        os.environ["BASE_API_INCLUDE_IMAGE"] = "1"
    uvicorn.run("framework_api.base_api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
