import base64
import os
import requests
from .base import BaseAdapter


class QwenVLLMAdapter(BaseAdapter):
    """Qwen 多模态适配器 — 支持传入两张图片 (before/after) + 文本 prompt"""

    def __init__(self, config):
        self.config = config

    def invoke(self, prompt, image_before=None, image_after=None):
        base_url = os.environ.get("QWEN_BASE_URL", self.config.get("base_url", ""))
        model_name = os.environ.get("QWEN_MODEL_NAME", self.config.get("model_name", ""))
        api_key = os.environ.get("QWEN_API_KEY", self.config.get("api_key", "EMPTY"))
        max_tokens = int(os.environ.get("QWEN_MAX_TOKENS", self.config.get("max_tokens", 500)))
        temperature = float(os.environ.get("QWEN_TEMPERATURE", self.config.get("temperature", 0.0)))

        content = []

        if image_before and self._file_exists(image_before):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_before)}"}
            })
        if image_after and self._file_exists(image_after):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_after)}"}
            })

        if content:
            content.append({"type": "text", "text": prompt})
            user_content = content
        else:
            user_content = prompt

        headers = {"Authorization": f"Bearer {api_key}"}
        messages = []
        system_message = self.config.get("system_message", "")
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload, headers=headers
        )
        return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _file_exists(path):
        return bool(path) and os.path.exists(path)


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
