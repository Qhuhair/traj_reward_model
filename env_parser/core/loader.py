import os
import json
from dataclasses import dataclass, field

from ..utils.path_utils import discover_step_files


@dataclass
class AlignedStep:
    """对齐后的单步数据，整合 metadata + history + perception 三方信息"""
    index: int
    from_state: str
    to_state: str
    action: str
    action_desc: str
    to_node_desc: str
    subgoal_idx: int
    subgoal_text: str
    element_id: str | None
    is_backtrack: bool
    backtrack_desc: str
    perception_before: list = field(default_factory=list)
    perception_after: list = field(default_factory=list)
    image_before: str = ""
    image_after: str = ""
    image_file: str = ""


class TrajectoryLoader:
    """
    轨迹加载器。
    只读取原始 trajs/ 目录，不修改任何原始文件。
    """

    def load(self, traj_dir: str) -> dict:
        """返回原始 metadata dict (仅读取，不修改)"""
        metadata_path = os.path.join(traj_dir, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_history(self, traj_dir: str) -> list:
        """返回 history.json 列表"""
        path = os.path.join(traj_dir, "history.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_step_perception(self, traj_dir: str, filename: str) -> list:
        """读取单个 step JSON 的 perception 数组"""
        path = os.path.join(traj_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("perception", [])

    def align(self, traj_dir: str) -> list:
        """
        将 metadata.images[] + history[] + per-step perception JSON 对齐。
        step_files[0] 是 00_start (初始状态), step_files[i>0] 是第 i 步执行后的屏幕。
        每个步骤: before = step_files[idx], after = step_files[idx+1]。
        返回 list[AlignedStep]（不修改原始数据）
        """
        metadata = self.load(traj_dir)
        history = self.load_history(traj_dir)
        step_files = discover_step_files(traj_dir)

        images = metadata.get("images", [])
        result = []

        for idx, img in enumerate(images):
            step_idx = img["index"]
            action = img.get("action", "")
            action_desc = img.get("desc", "")
            to_node_desc = img.get("to_node_desc", "")
            subgoal_idx = img.get("subgoal_index", 0)
            subgoal_text = img.get("subgoal_text", "")

            hist = history[idx] if idx < len(history) else {}
            element_id = hist.get("element_id")
            is_backtrack = hist.get("backtrack", False)
            backtrack_desc = hist.get("description", "")

            perception_before = []
            perception_after = []
            image_before = ""
            image_after = ""
            image_file = img.get("image", "")

            # before = step_files[idx] (对第一步而言即 00_start)
            if idx < len(step_files):
                perception_before = self.load_step_perception(
                    traj_dir, step_files[idx][0]
                )
                image_before = os.path.join(
                    traj_dir, step_files[idx][0].replace(".json", ".jpg")
                )
            # after = step_files[idx+1]
            if idx + 1 < len(step_files):
                perception_after = self.load_step_perception(
                    traj_dir, step_files[idx + 1][0]
                )
                image_after = os.path.join(
                    traj_dir, step_files[idx + 1][0].replace(".json", ".jpg")
                )

            result.append(AlignedStep(
                index=step_idx,
                from_state=img.get("from", hist.get("from", "")),
                to_state=img.get("to", hist.get("to", "")),
                action=action,
                action_desc=action_desc,
                to_node_desc=to_node_desc,
                subgoal_idx=subgoal_idx,
                subgoal_text=subgoal_text,
                element_id=element_id,
                is_backtrack=is_backtrack,
                backtrack_desc=backtrack_desc,
                perception_before=perception_before,
                perception_after=perception_after,
                image_before=image_before,
                image_after=image_after,
                image_file=image_file,
            ))

        return result
