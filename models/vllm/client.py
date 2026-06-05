"""Small OpenAI-compatible client for vLLM smoke tests."""

import argparse

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a vLLM chat endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--prompt", default="你好，请只回复：连接成功")
    return parser.parse_args()


def request_completion(args: argparse.Namespace):
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    return client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0.1,
        max_tokens=64,
    )


def main() -> None:
    response = request_completion(parse_args())
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()

