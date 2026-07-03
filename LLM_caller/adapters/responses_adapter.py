import base64
import os

import requests

from .base import BaseAdapter


class ResponsesAdapter(BaseAdapter):
    """Responses API 适配器，支持 before/after 两张截图和文本 prompt。"""

    def __init__(self, config):
        self.config = config

    def invoke(self, prompt, image_before=None, image_after=None):
        payload = self.build_payload(prompt, image_before, image_after)
        response = requests.post(
            self._endpoint(),
            json=payload,
            headers=self._headers(),
            timeout=self.config.get("timeout", 300),
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Responses API error {response.status_code}: {response.text[:800]}")
        return _extract_text(response.json())

    def build_payload(self, prompt, image_before=None, image_after=None):
        """构造 payload，便于离线测试请求结构。"""
        messages = []
        system_message = self.config.get("system_message", "")
        if system_message:
            messages.append(_message("system", [{"type": "input_text", "text": system_message}]))

        content = []
        for image_path in (image_before, image_after):
            if image_path and os.path.exists(image_path):
                content.append({"type": "input_image", "image_url": _data_url(image_path)})
        content.append({"type": "input_text", "text": prompt})
        messages.append(_message("user", content))

        payload = {"model": self.config["model_name"], "input": messages}
        if self.config.get("reasoning_effort"):
            payload["reasoning"] = {"effort": self.config["reasoning_effort"]}
        if self.config.get("max_output_tokens"):
            payload["max_output_tokens"] = self.config["max_output_tokens"]
        if self.config.get("temperature") is not None:
            payload["temperature"] = self.config["temperature"]
        return payload

    def _endpoint(self):
        return f"{self.config['base_url'].rstrip('/')}/responses"

    def _headers(self):
        env_key = self.config.get("env_key", "OPENAI_API_KEY")
        api_key = os.environ.get(env_key) or self.config.get("api_key")
        if not api_key:
            raise RuntimeError(f"Missing API key: set environment variable {env_key}")
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _message(role, content):
    return {"role": role, "content": content}


def _data_url(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _extract_text(data):
    if data.get("output_text"):
        return data["output_text"]
    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str) and content:
            return content
    chunks = []
    for item in data.get("output", []):
        for part in item.get("content", []):
            if part.get("type") in {"output_text", "text"} and part.get("text"):
                chunks.append(part["text"])
    if chunks:
        return "\n".join(chunks)
    raise RuntimeError("Responses API returned no text output")
