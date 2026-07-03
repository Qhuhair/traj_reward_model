import argparse
import json
import os
import sys

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

from adapters.responses_adapter import ResponsesAdapter
from caller import LLMCaller


DEFAULT_STD_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "crossapp_qwen35_4b_vllm_text",
    "20250113_21442_test",
    "traj_007",
    "standardized.json",
)


def _load_config(model_name):
    with open(os.path.join(CURRENT_DIR, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["models"][model_name]


def _load_step(std_path, step_idx):
    with open(std_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    step = data["steps"][step_idx - 1]
    return data, step


def _prompt_args(data, step):
    return {
        "task_desc": data.get("task", ""),
        "subgoal": step.get("subgoal_text", ""),
        "step_idx": step.get("step_idx", 0),
        "action_desc": step.get("action_desc", ""),
        "element_id": step.get("element_id", ""),
        "state_desc_before": step.get("state_desc_before", "")[:400],
        "state_desc_after": step.get("state_desc_after", "")[:400],
        "image_before": step.get("image_before_annotated") or step.get("image_before", ""),
        "image_after": step.get("image_after_annotated") or step.get("image_after", ""),
    }


def _format_prompt(prompt_name, values):
    prompt_path = os.path.join(CURRENT_DIR, "prompts", f"{prompt_name}.yaml")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)["template"]
    return template.format(**values)


def _run_payload_check(model, prompt, std_path, step_idx):
    data, step = _load_step(std_path, step_idx)
    values = _prompt_args(data, step)
    adapter = ResponsesAdapter(_load_config(model))
    payload = adapter.build_payload(
        _format_prompt(prompt, values),
        values["image_before"],
        values["image_after"],
    )

    user_content = payload["input"][-1]["content"]
    image_count = sum(1 for item in user_content if item["type"] == "input_image")
    text_count = sum(1 for item in user_content if item["type"] == "input_text")
    print(f"model={payload['model']}")
    print(f"messages={len(payload['input'])}")
    print(f"user_images={image_count}")
    print(f"user_text_blocks={text_count}")
    print(f"has_reasoning={'reasoning' in payload}")
    print(f"step_idx={step_idx}")
    print(f"image_before_exists={os.path.exists(values['image_before'])}")
    print(f"image_after_exists={os.path.exists(values['image_after'])}")


def _run_online(model, prompt, std_path, step_idx):
    data, step = _load_step(std_path, step_idx)
    values = _prompt_args(data, step)
    caller = LLMCaller(model=model, prompt=prompt)
    result = caller.call(**values)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="测试 Responses API 多模态适配器。")
    parser.add_argument("--model", default="codex_mm_baseline")
    parser.add_argument("--prompt", default="RRM_Qwen_MM")
    parser.add_argument("--std-path", default=DEFAULT_STD_PATH)
    parser.add_argument("--step-idx", type=int, default=1)
    parser.add_argument("--online", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.online:
        _run_online(args.model, args.prompt, args.std_path, args.step_idx)
    else:
        _run_payload_check(args.model, args.prompt, args.std_path, args.step_idx)
