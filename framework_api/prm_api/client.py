"""PRM 使用的 OpenAI 兼容模型客户端。"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from framework_api.logging_utils import configure_file_logger


LOGGER = configure_file_logger("framework_api.prm_api.client", "prm", "prm_api")


class PRMChatClient:
    """调用 vLLM/OpenAI 兼容接口获取 PRM 评分。"""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("PRM_API_OPENAI_BASE_URL") or "http://127.0.0.1:8002/v1").rstrip("/")
        self.api_key = api_key or os.getenv("PRM_API_KEY") or "EMPTY"
        self.timeout = int(os.getenv("PRM_API_TIMEOUT", "120"))

    def complete(self, *, model: str, prompt: str, screenshot_b64: str | None) -> str:
        """发送评分请求并返回模型原始文本。"""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": self._content(prompt, screenshot_b64)}],
            "temperature": float(os.getenv("PRM_API_TEMPERATURE", "0.0")),
            "max_tokens": int(os.getenv("PRM_API_MAX_TOKENS", "1024")),
        }
        started_at = time.time()
        LOGGER.info("调用 PRM 模型开始 model=%s base_url=%s prompt_chars=%s has_image=%s", model, self.base_url, len(prompt), bool(screenshot_b64))
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = _read_message_content(response.json())
        LOGGER.info("调用 PRM 模型完成 model=%s status_code=%s duration=%.3fs response_chars=%s", model, response.status_code, time.time() - started_at, len(content))
        return content

    @staticmethod
    def _content(prompt: str, screenshot_b64: str | None) -> str | list[dict[str, Any]]:
        """有截图时采用 OpenAI 多模态 content；无截图时退化为纯文本。"""
        if not screenshot_b64:
            return prompt
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
        ]


def _read_message_content(data: dict[str, Any]) -> str:
    """从 OpenAI 兼容响应中取出 assistant 文本。"""
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("PRM 响应中没有 choices。")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    raise ValueError("PRM 响应中没有可解析的 message.content。")
