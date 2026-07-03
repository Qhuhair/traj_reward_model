"""PRM API 的 FastAPI 入口。"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, HTTPException

from framework_api.logging_utils import configure_file_logger

from .archive import finish_episode_if_done, load_window_steps, resolve_episode_dir, save_score_input, save_score_output
from .scorers import select_scorer
from .schema import PRMMode, PRMScoreRequest, PRMScoreResponse


app = FastAPI(title="Trajectory PRM Score API")
LOGGER = configure_file_logger("framework_api.prm_api.app", "prm", "prm_api")


@app.get("/health")
def health() -> dict[str, str]:
    """轻量健康检查，不触发模型调用。"""
    LOGGER.info("收到 PRM 健康检查请求")
    return {"status": "ok", "service": "prm_api"}


@app.post("/v1/score", response_model=PRMScoreResponse)
def score(request: PRMScoreRequest) -> PRMScoreResponse:
    """对 base 提出的候选动作进行 PRM 评分。"""
    if not request.candidates:
        raise HTTPException(status_code=400, detail="PRM 请求缺少 candidates。")
    request = _apply_runtime_mode(request)
    started_at = time.time()
    model = request.model or os.getenv("PRM_API_MODEL") or "crossapp_kto"
    window_size = int(os.getenv("PRM_API_WINDOW_SIZE", "3"))
    episode_dir = resolve_episode_dir(request)
    history_steps = load_window_steps(episode_dir, window_size)
    scorer = select_scorer(request.mode)
    LOGGER.info("收到 /v1/score 调用 model=%s mode=%s step_index=%s candidates=%s", model, request.mode.value, request.step_index, len(request.candidates))
    try:
        result = scorer.score(request, history_steps, episode_dir)
    except Exception as exc:
        LOGGER.exception("PRM 模型调用失败 model=%s step_index=%s", model, request.step_index)
        raise HTTPException(status_code=502, detail=f"PRM 模型调用失败：{exc}") from exc
    input_path = save_score_input(episode_dir, request, result.prompt)
    LOGGER.info("PRM 评分输入已保存 input=%s", input_path)
    output = {"raw_response": result.raw_response, "scores": result.scores, "best_index": result.best_index, "reason": result.reason, "mode": request.mode.value, "scorer": result.scorer}
    save_score_output(episode_dir, request.step_index, output)
    finish_episode_if_done(episode_dir, request, result.best_index)
    LOGGER.info("完成 /v1/score 调用 model=%s mode=%s duration=%.3fs best_index=%s scores=%s", model, request.mode.value, time.time() - started_at, result.best_index, result.scores)
    return PRMScoreResponse(scores=result.scores, best_index=result.best_index, reason=result.reason, mode=request.mode, archive_dir=str(episode_dir))


def _apply_runtime_mode(request: PRMScoreRequest) -> PRMScoreRequest:
    """请求未显式指定 mode 时，使用服务启动时配置的默认模式。"""
    if "mode" in request.model_fields_set:
        return request
    mode_value = os.getenv("PRM_API_MODE")
    if not mode_value:
        return request
    try:
        return request.model_copy(update={"mode": PRMMode(mode_value)})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"无效 PRM_API_MODE：{mode_value}") from exc
