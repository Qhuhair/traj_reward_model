"""
在标准化 JSON 每步的截图上绘制点击位置/元素框，输出新 JSON。
供流水线中 env_parser 和 LLM_caller 之间使用。

用法:
    python annotate.py input.json output.json --output-dir annotated_images/
"""
import json
import os
import sys

from image_annotator import ImageAnnotator


def annotate_standardized(data: dict, output_dir: str, annotator: ImageAnnotator = None) -> dict:
    """在每步的 image_before / image_after 上标注，返回更新后的 data"""
    if annotator is None:
        annotator = ImageAnnotator()

    os.makedirs(output_dir, exist_ok=True)

    for step in data.get("steps", []):
        action_x = step.get("action_x")
        action_y = step.get("action_y")
        element_bbox = step.get("element_bbox")
        traj_id = data.get("trajectory_id", "unk")

        for tag, key in [("before", "image_before"), ("after", "image_after")]:
            img_path = step.get(key)
            if not img_path or not os.path.exists(img_path):
                continue
            sub_dir = os.path.join(output_dir, traj_id)
            annotated = annotator.annotate_step(
                image_path=img_path,
                action_x=action_x,
                action_y=action_y,
                element_bbox=element_bbox,
                output_dir=sub_dir,
                tag=tag,
            )
            if annotated:
                step[f"{key}_annotated"] = annotated

    return data


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python annotate.py <input.json> <output.json> [--output-dir DIR]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    output_dir = "annotated_images"
    for i, arg in enumerate(sys.argv):
        if arg == "--output-dir" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = annotate_standardized(data, output_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Annotated JSON written to {output_path}")
