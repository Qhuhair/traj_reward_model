"""
滑动窗口评估器 — 以当前步的图片+前后步的文本上下文进行多模态评估。

用法:
    from sliding_window_evaluator import SlidingWindowEvaluator
    evaluator = SlidingWindowEvaluator()
    results = evaluator.evaluate(std_data, llm_caller)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from caller import LLMCaller


MAX_STATE_CHARS = 400


class SlidingWindowEvaluator:
    """
    滑动窗口评估器：
    - 对每步 N，提供步 N-1 的文本+评分 + 步 N 的图片+文本 + 步 N+1 的文本
    - 让模型在完整上下文流中评分，降低 <think> 缺失概率
    """

    def __init__(self, caller: LLMCaller = None):
        self.caller = caller or LLMCaller(
            model="qwen_local_mm",
            prompt="RRM_Qwen_Sliding"
        )

    def evaluate(self, std_data: dict, max_workers: int = 1) -> list:
        """
        滑动窗口评估整条轨迹。

        Returns:
            list[dict]: 每步的 {step_idx, action, think, critique, score}
        """
        steps = std_data.get("steps", [])
        task = std_data.get("task", "")
        results = [None] * len(steps)

        for i in range(len(steps)):
            step = steps[i]

            # ── 上一步上下文 ──
            if i > 0 and results[i - 1] is not None:
                prev = results[i - 1]
                prev_action = prev.get("action", "")
                prev_subgoal = steps[i - 1].get("subgoal_text", "")
                prev_score = prev.get("score", "N/A")
                prev_critique = prev.get("critique", "")[:200]
            else:
                prev_action = "（轨迹起点）"
                prev_subgoal = ""
                prev_score = "N/A"
                prev_critique = ""

            # ── 下一步上下文（仅文本，不暴露图片也不暴露评分）──
            if i + 1 < len(steps):
                next_step = steps[i + 1]
                next_action = next_step.get("action_desc", "")
                next_subgoal = next_step.get("subgoal_text", "")
            else:
                next_action = "（轨迹终点）"
                next_subgoal = ""

            # ── 当前步信息 ──
            curr_action = step.get("action_desc", "")
            curr_element = step.get("element_id", "")
            curr_subgoal = step.get("subgoal_text", "")
            curr_state_before = step.get("state_desc_before", "")[:MAX_STATE_CHARS]
            curr_state_after = step.get("state_desc_after", "")[:MAX_STATE_CHARS]

            # 优先用标注后的图片
            img_before = step.get("image_before_annotated") or step.get("image_before", "")
            img_after = step.get("image_after_annotated") or step.get("image_after", "")

            # ── 调用 LLM ──
            try:
                result = self.caller.call(
                    task_desc=task,
                    # 上一步
                    prev_action=prev_action,
                    prev_subgoal=prev_subgoal,
                    prev_score=str(prev_score),
                    prev_critique=prev_critique,
                    # 当前步
                    curr_subgoal=curr_subgoal,
                    curr_action=curr_action,
                    curr_element_id=curr_element,
                    curr_state_before=curr_state_before,
                    curr_state_after=curr_state_after,
                    # 下一步
                    next_subgoal=next_subgoal,
                    next_action=next_action,
                    # 图片
                    image_before=img_before,
                    image_after=img_after,
                )
            except Exception as e:
                print(f"  [ERROR] Step {i+1}: {e}")
                result = {
                    "think": "N/A",
                    "critique": f"Evaluation failed: {e}",
                    "score": 0.0,
                }

            results[i] = {
                "step_idx": step.get("step_idx", i + 1),
                "action": step.get("action", ""),
                "think": result.get("think", "N/A"),
                "critique": result.get("critique", ""),
                "score": result.get("score", 0.0),
            }

            print(f"  Step {i+1}/{len(steps)}: score={results[i]['score']:.2f} | think_len={len(results[i]['think'])}")

        return results


def evaluate_sliding(std_data: dict, model: str = None, prompt: str = None) -> list:
    """便捷入口"""
    caller = LLMCaller(model=model or "qwen_local_mm", prompt=prompt or "RRM_Qwen_Sliding")
    evaluator = SlidingWindowEvaluator(caller)
    return evaluator.evaluate(std_data)
