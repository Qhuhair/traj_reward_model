import base64
import requests
from .base import BaseAdapter


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OpenAIStyleAdapter(BaseAdapter):
    def __init__(self, config):
        self.config = config

    def invoke(self, prompt, image_before=None, image_after=None):
        headers = {"Authorization": f"Bearer {self.config['api_key']}"}
        system_msg = self.config.get("system_message", "")
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})

        # 多模态：拼接图片到 user message（OpenAI Vision 兼容格式）
        user_content = []
        if image_before and __import__("os").path.exists(image_before):
            b64 = _encode_image(image_before)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        if image_after and __import__("os").path.exists(image_after):
            b64 = _encode_image(image_after)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content if len(user_content) > 1 else prompt})

        payload = {
            "model": self.config['model_name'],
            "messages": messages,
            "temperature": self.config.get("temperature", 0.3),
        }
        if self.config.get("max_tokens"):
            payload["max_tokens"] = self.config["max_tokens"]
        response = requests.post(
            f"{self.config['base_url']}/chat/completions",
            json=payload, headers=headers
        )
        return response.json()['choices'][0]['message']['content']

# 这里可以添加更多适配器，如 LocalModelAdapter 等