"""PRM 评分策略分发。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .client import PRMChatClient
from .crossapp_adapter import build_crossapp_kwargs, render_prompt
from .images import save_screenshot
from .llm_bridge import load_llm_caller
from .parser import parse_score_response
from .prompt import build_score_prompt, include_image
from .schema import PRMMode, PRMScoreRequest


@dataclass(frozen=True)
class PRMScoreResult:
    """PRM scorer 的统一返回结构。"""

    scores: list[float]
    best_index: int
    reason: str
    raw_response: str
    prompt: str
    scorer: str


class DirectPromptScorer:
    """当前轻量 PRM：直接构造 JSON 评分 Prompt。"""

    def score(self, request: PRMScoreRequest, history_steps: list[dict], episode_dir: Path) -> PRMScoreResult:
        prompt = build_score_prompt(request, history_steps)
        model = request.model or os.getenv("PRM_API_MODEL") or "crossapp_kto"
        raw = PRMChatClient().complete(
            model=model,
            prompt=prompt,
            screenshot_b64=request.screenshot if include_image(request.mode) else None,
        )
        scores, best_index, reason = parse_score_response(raw, len(request.candidates))
        return PRMScoreResult(scores, best_index, reason, raw, prompt, "direct_prompt")


class CrossAppSlidingWindowScorer:
    """复用 LLMCaller 多模态调用和 crossAPP 窗口评分范式。"""

    def score(self, request: PRMScoreRequest, history_steps: list[dict], episode_dir: Path) -> PRMScoreResult:
        LLMCaller = load_llm_caller()
        image_path = save_screenshot(episode_dir, request)
        prompt_name = os.getenv("PRM_API_LLM_PROMPT", "RRM_Qwen_Sliding_Android_PRM")
        caller = LLMCaller(
            model=os.getenv("PRM_API_LLM_MODEL", "qwen_vllm_mm"),
            prompt=prompt_name,
        )
        scores: list[float] = []
        reasons: list[str] = []
        raw_items: list[dict] = []
        for index, candidate in enumerate(request.candidates):
            kwargs = build_crossapp_kwargs(request, history_steps, index, candidate)
            kwargs["image_before"] = str(image_path) if image_path else None
            result = caller.call(**kwargs)
            scores.append(float(result.get("score", 0.0)))
            reasons.append(str(result.get("critique", "")))
            raw_items.append({"candidate_index": index, **result})
        best_index = max(range(len(scores)), key=lambda item: scores[item])
        return PRMScoreResult(
            scores=scores,
            best_index=best_index,
            reason=reasons[best_index],
            raw_response=json.dumps(raw_items, ensure_ascii=False),
            prompt=render_prompt(caller.prompt_template, request, history_steps),
            scorer="crossapp_sliding_window",
        )


def select_scorer(mode: PRMMode):
    """按 mode 选择评分实现。"""
    if mode == PRMMode.CROSSAPP_MULTIMODAL_WINDOW:
        return CrossAppSlidingWindowScorer()
    return DirectPromptScorer()
