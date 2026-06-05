import requests
import json
from .base import BaseAdapter


class RemoteQwenAdapter(BaseAdapter):
    """远程 Qwen API 适配器 — 调用另一台服务器的 /generate 接口"""

    def __init__(self, config):
        self.config = config

    def invoke(self, prompt, image_before=None, image_after=None):
        url = self.config.get("base_url", "http://154.8.219.40:9090/generate")
        max_tokens = self.config.get("max_tokens", 256)
        temperature = self.config.get("temperature", 0.7)

        # 构建请求体，system_message 前置
        sys_msg = self.config.get("system_message", "")
        if sys_msg:
            full_prompt = f"{sys_msg}\n\n{prompt}"
        else:
            full_prompt = prompt

        payload = {
            "prompt": full_prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = requests.post(url, json=payload, timeout=120)
        result = resp.json()

        # 解析响应：API 返回 {"text": "..."}
        text = result.get("text", "")
        if not text:
            raise RuntimeError(f"Remote API returned empty: {json.dumps(result, ensure_ascii=False)[:300]}")
        return text
