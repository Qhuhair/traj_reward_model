"""Base API 的 FastAPI 入口。"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, HTTPException

from framework_api.logging_utils import configure_file_logger

from .client import ChatCompletionClient
from .parser import parse_act_response
from .prompt import build_action_prompt
from .schema import BaseActRequest, BaseActResponse

app = FastAPI(title="Trajectory Base Action API")
LOGGER = configure_file_logger("framework_api.base_api.app", "base", "base_api")


@app.get("/health")
def health() -> dict[str, str]:
    """轻量健康检查，不触发模型调用。"""
    LOGGER.info("收到健康检查请求")
    return {"status": "ok", "service": "base_api"}


@app.post("/v1/act", response_model=BaseActResponse)
def act(request: BaseActRequest) -> BaseActResponse:
    """生成 AndroidWorld 下一步候选动作。"""
    model = request.model or os.getenv("BASE_API_MODEL") or "Qwen3.5-4B"
    prompt = build_action_prompt(request)
    client = ChatCompletionClient()
    started_at = time.time()
    LOGGER.info(
        "收到 /v1/act 调用 model=%s step_index=%s n_candidates=%s ui_elements=%s history=%s has_screenshot=%s",
        model,
        request.step_index,
        request.n_candidates,
        len(request.ui_elements),
        len(request.history),
        bool(request.screenshot),
    )
    try:
        raw_text = client.complete(
            model=model,
            prompt=prompt,
            screenshot_b64=request.screenshot,
        )
    except Exception as exc:
        # HTTP 层只暴露调用失败原因，不把内部堆栈返回给评测框架。
        LOGGER.exception("base 模型调用失败 model=%s step_index=%s", model, request.step_index)
        raise HTTPException(status_code=502, detail=f"base 模型调用失败：{exc}") from exc
    response = parse_act_response(raw_text, request.n_candidates)
    LOGGER.info(
        "完成 /v1/act 调用 model=%s step_index=%s duration=%.3fs candidates=%s",
        model,
        request.step_index,
        time.time() - started_at,
        len(response.candidates),
    )
    return response
