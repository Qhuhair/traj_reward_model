"""OpenAI 兼容模型服务客户端。"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from framework_api.logging_utils import configure_file_logger


LOGGER = configure_file_logger("framework_api.base_api.client", "base", "base_api")


class ChatCompletionClient:
    """调用 vLLM/OpenAI 兼容的 chat completions 接口。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("BASE_API_OPENAI_BASE_URL") or "http://127.0.0.1:8002/v1").rstrip("/")
        self.api_key = api_key or os.getenv("BASE_API_KEY") or "EMPTY"
        self.timeout = timeout or int(os.getenv("BASE_API_TIMEOUT", "120"))

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        screenshot_b64: str | None = None,
    ) -> str:
        """发送一次模型请求并返回原始文本。"""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": self._content(prompt, screenshot_b64)}],
            "temperature": float(os.getenv("BASE_API_TEMPERATURE", "0.2")),
            "max_tokens": int(os.getenv("BASE_API_MAX_TOKENS", "1024")),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        started_at = time.time()
        LOGGER.info(
            "调用模型开始 model=%s base_url=%s include_image=%s prompt_chars=%s",
            model,
            self.base_url,
            bool(screenshot_b64 and os.getenv("BASE_API_INCLUDE_IMAGE", "0") == "1"),
            len(prompt),
        )
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = _read_message_content(response.json())
        LOGGER.info(
            "调用模型完成 model=%s status_code=%s duration=%.3fs response_chars=%s",
            model,
            response.status_code,
            time.time() - started_at,
            len(content),
        )
        return content

    @staticmethod
    def _content(prompt: str, screenshot_b64: str | None) -> str | list[dict[str, Any]]:
        """默认纯文本；显式开启时才把截图传给多模态模型。"""
        if not screenshot_b64 or os.getenv("BASE_API_INCLUDE_IMAGE", "0") != "1":
            return prompt
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            },
        ]


def _read_message_content(data: dict[str, Any]) -> str:
    """从 OpenAI 兼容响应中取出 assistant 文本。"""
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("模型响应中没有 choices。")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    raise ValueError("模型响应中没有可解析的 message.content。")
