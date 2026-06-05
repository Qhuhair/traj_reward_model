"""
轨迹总结模块 — 评估整条轨迹是否达成总目标。
在所有步骤评分完成后调用，追加到评分末尾。
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from caller import LLMCaller


class TrajectorySummarizer:
    """
    轨迹级总结：
    - 输入：所有步骤的文本描述 + 评分 + 总目标 + 子目标
    - 输出：{think, critique, score} 评估轨迹是否完成目标
    - 结果以 step_idx=0 追加到评分列表末尾
    """

    def __init__(self, caller: LLMCaller = None):
        self.caller = caller or LLMCaller(
            model="qwen_local",
            prompt="RRM_TrajSummary"
        )

    def summarize(self, std_data: dict, step_scores: list) -> dict:
        task = std_data.get("task", "")
        subgoals = std_data.get("subgoals", [])

        # 格式化子目标列表
        subgoals_text = "\n".join(f"  {i+1}. {sg}" for i, sg in enumerate(subgoals))

        # 格式化每步摘要
        lines = []
        for s in step_scores:
            act = s.get("action", "")[:50]
            score = s.get("score", "N/A")
            lines.append(f"  步{s['step_idx']}: [{score}] {act}")
        steps_summary = "\n".join(lines)

        try:
            result = self.caller.call(
                task_desc=task,
                subgoals_text=subgoals_text,
                steps_summary=steps_summary,
            )
        except Exception as e:
            print(f"  [ERROR] Trajectory summary: {e}")
            result = {"think": "N/A", "critique": f"Error: {e}", "score": 0.0}

        return {
            "step_idx": 0,
            "action": "[TRAJECTORY SUMMARY]",
            "think": result.get("think", "N/A"),
            "critique": result.get("critique", ""),
            "score": result.get("score", 0.0),
        }


def summarize_trajectory(std_data: dict, step_scores: list) -> dict:
    """便捷入口"""
    summarizer = TrajectorySummarizer()
    return summarizer.summarize(std_data, step_scores)
