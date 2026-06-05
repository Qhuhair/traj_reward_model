import re

from .loader import AlignedStep


class TrajectoryFormatter:
    """将 AlignedStep + state_desc 组合为标准化的 S-A 对"""

    @staticmethod
    def _parse_action_coords(action: str) -> tuple:
        """从 'Tap (538, 2324)' 解析出 (x, y)，失败返回 None"""
        m = re.search(r"\((\d+),\s*(\d+)\)", action)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None, None

    @staticmethod
    def _find_element_bbox(perception_list: list, x: int, y: int,
                           max_dist: int = 120) -> list | None:
        """在 perception 中找距离点击坐标最近的元素 bbox"""
        if not perception_list or x is None:
            return None
        best_bbox = None
        best_dist = float("inf")
        for el in perception_list:
            cx, cy = el.get("center", el.get("coordinates", [None, None]))
            if cx is None:
                continue
            dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_bbox = el.get("bbox")
        if best_dist <= max_dist:
            return best_bbox
        return None

    def format_step(self, step: AlignedStep,
                    state_before: str, state_after: str,
                    task_desc: str) -> dict:
        action_x, action_y = self._parse_action_coords(step.action)
        # 从 perception_after 中匹配元素 bbox
        element_bbox = self._find_element_bbox(
            step.perception_after, action_x, action_y
        )
        return {
            "step_idx": step.index,
            "from_state": step.from_state,
            "to_state": step.to_state,
            "action": step.action,
            "action_desc": step.action_desc,
            "action_x": action_x,
            "action_y": action_y,
            "element_bbox": element_bbox or [],
            "state_desc_before": state_before,
            "state_desc_after": state_after,
            "image_before": step.image_before,
            "image_after": step.image_after,
            "to_node_desc": step.to_node_desc,
            "subgoal_idx": step.subgoal_idx,
            "subgoal_text": step.subgoal_text,
            "element_id": step.element_id or "",
            "is_backtrack": step.is_backtrack,
            "image_file": step.image_file,
        }

    def format_trajectory(self, steps: list,
                          state_before_list: list, state_after_list: list,
                          task_desc: str) -> list:
        return [
            self.format_step(step, bf, af, task_desc)
            for step, bf, af in zip(steps, state_before_list, state_after_list)
        ]

    def format_output(self, traj_id: str, metadata: dict,
                      standardized_steps: list) -> dict:
        return {
            "trajectory_id": traj_id,
            "app": metadata.get("app_id", ""),
            "task": metadata.get("overall_task", metadata.get("instruction", "")),
            "subgoals": [
                sg.get("goal", "") for sg in metadata.get("subgoals", [])
            ],
            "steps": standardized_steps,
        }
