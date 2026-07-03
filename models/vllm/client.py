"""Small OpenAI-compatible client for vLLM smoke tests."""

import argparse

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a vLLM chat endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--lora-model", help="Optional LoRA adapter model name to test after the base model.")
    parser.add_argument("--prompt", default="你好，请只回复：连接成功")
    return parser.parse_args()


def request_completion(args: argparse.Namespace, model_name: str):
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    return client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0.1,
        max_tokens=64,
    )


def main() -> None:
    args = parse_args()
    _print_response(args, args.model)
    if args.lora_model:
        _print_response(args, args.lora_model)


def _print_response(args: argparse.Namespace, model_name: str) -> None:
    """打印指定模型名的响应，model_name 是区分 base / LoRA 的标识。"""
    response = request_completion(args, model_name)
    print(f"[{model_name}] {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
