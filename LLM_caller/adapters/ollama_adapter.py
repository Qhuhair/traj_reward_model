import base64
import json as json_mod
import os
import requests
from .base import BaseAdapter


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OllamaAdapter(BaseAdapter):
    """Ollama 原生 API 适配器 — 支持单模态文本和多模态图像+文本"""

    def __init__(self, config):
        self.config = config

    def invoke(self, prompt, image_before=None, image_after=None):
        messages = []
        sys_msg = self.config.get("system_message", "")
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})

        user_msg = {"role": "user", "content": prompt}

        # ── 多模态：传入标注后的图片 ──
        images = []
        if image_before and os.path.exists(image_before):
            images.append(_encode_image(image_before))
        if image_after and os.path.exists(image_after):
            images.append(_encode_image(image_after))
        if images:
            user_msg["images"] = images

        messages.append(user_msg)

        payload = {
            "model": self.config["model_name"],
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self.config.get("max_tokens", 8192),
                "temperature": self.config.get("temperature", 0.1),
                "think": self.config.get("think", True),
                "seed": self.config.get("seed", 42),
                "repeat_penalty": self.config.get("repeat_penalty", 1.0),
            },
        }

        resp = requests.post(
            f"{self.config['base_url']}/api/chat", json=payload
        ).json()
        if "message" not in resp:
            raise RuntimeError(f"Ollama API error: {json_mod.dumps(resp, ensure_ascii=False)[:500]}")
        msg = resp["message"]
        # 兼容 think:false 模式：content 为空时取 thinking 字段
        content = msg.get("content", "") or msg.get("thinking", "")
        if not content:
            raise RuntimeError(f"Ollama returned empty content: {json_mod.dumps(resp, ensure_ascii=False)[:300]}")
        return content
