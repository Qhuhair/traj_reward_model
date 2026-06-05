"""
纯文本滑动窗口评估器 — 3步滑动窗口，评估中间步。
不修改现有模块，仅作为新增工具。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from caller import LLMCaller

MAX_STATE_CHARS = 400


class TextSlideEvaluator:
    """
    纯文本滑动窗口评估器：
    - 每步取前后共3步的文本描述
    - 用 qwen_local 纯文本模型评估
    """

    def __init__(self, caller: LLMCaller = None):
        self.caller = caller or LLMCaller(
            model="qwen_local",
            prompt="RRM_Qwen_TextSlide"
        )

    def evaluate(self, std_data: dict) -> list:
        steps = std_data.get("steps", [])
        task = std_data.get("task", "")
        results = [None] * len(steps)

        for i in range(len(steps)):
            step = steps[i]

            # 上一步（只传文本，不传评分）
            if i > 0:
                prev_step = steps[i - 1]
                prev_action = prev_step.get("action_desc", "")
                prev_subgoal = prev_step.get("subgoal_text", "")
                prev_state_before = prev_step.get("state_desc_before", "")[:MAX_STATE_CHARS]
                prev_state_after = prev_step.get("state_desc_after", "")[:MAX_STATE_CHARS]
            else:
                prev_action = "（轨迹起点）"
                prev_subgoal = ""
                prev_state_before = ""
                prev_state_after = ""

            # 下一步（仅文本）
            if i + 1 < len(steps):
                next_step = steps[i + 1]
                next_action = next_step.get("action_desc", "")
                next_subgoal = next_step.get("subgoal_text", "")
            else:
                next_action = "（轨迹终点）"
                next_subgoal = ""

            # 当前步
            curr_action = step.get("action_desc", "")
            curr_element = step.get("element_id", "")
            curr_subgoal = step.get("subgoal_text", "")
            curr_state_before = step.get("state_desc_before", "")[:MAX_STATE_CHARS]
            curr_state_after = step.get("state_desc_after", "")[:MAX_STATE_CHARS]

            try:
                result = self.caller.call(
                    task_desc=task,
                    prev_action=prev_action,
                    prev_subgoal=prev_subgoal,
                    prev_state_before=prev_state_before,
                    prev_state_after=prev_state_after,
                    curr_subgoal=curr_subgoal,
                    curr_action=curr_action,
                    curr_element_id=curr_element,
                    curr_state_before=curr_state_before,
                    curr_state_after=curr_state_after,
                    next_subgoal=next_subgoal,
                    next_action=next_action,
                )
            except Exception as e:
                print(f"  [ERROR] Step {i+1}: {e}")
                result = {"think": "N/A", "critique": f"Error: {e}", "score": 0.0}

            results[i] = {
                "step_idx": step.get("step_idx", i + 1),
                "action": step.get("action", ""),
                "think": result.get("think", "N/A"),
                "critique": result.get("critique", ""),
                "score": result.get("score", 0.0),
            }

            print(f"  Step {i+1}/{len(steps)}: score={results[i]['score']:.2f}")

        return results
