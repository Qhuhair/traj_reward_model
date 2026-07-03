import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CALLER_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(CALLER_DIR)
for path in (PROJECT_ROOT, CALLER_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from caller import LLMCaller
from LLM_caller.memory import MemoryBuilder, MemoryConfig


class HierarchicalMemoryEvaluator:
    """分层记忆评估器：历史摘要 + 最近窗口文本 + 当前步图片。"""

    def __init__(self, caller: LLMCaller, memory_config: MemoryConfig | dict):
        self.caller = caller
        self.memory_config = (
            memory_config
            if isinstance(memory_config, MemoryConfig)
            else MemoryConfig.from_dict(memory_config)
        )
        self.builder = MemoryBuilder(self.memory_config)
        self.contexts = []

    def evaluate(self, std_data: dict) -> tuple[list[dict], list[dict]]:
        """逐步评分，返回 LLM 结果和每步记忆上下文。"""
        steps = std_data.get("steps", [])
        results = []
        self.contexts = []

        for idx, step in enumerate(steps):
            context = self.builder.build_context(std_data, idx, results)
            prompt_values = self.builder.context_to_prompt_values(std_data, context)
            self.contexts.append(context.to_dict())

            try:
                result = self.caller.call(**prompt_values)
            except Exception as exc:
                print(f"  [ERROR] Step {idx + 1}: {exc}")
                result = {
                    "think": "N/A",
                    "critique": f"Evaluation failed: {exc}",
                    "score": 0.0,
                }

            record = {
                "step_idx": step.get("step_idx", idx + 1),
                "action": step.get("action", ""),
                "think": result.get("think", "N/A"),
                "critique": result.get("critique", ""),
                "score": result.get("score", 0.0),
            }
            results.append(record)
            print(
                f"  Step {idx + 1}/{len(steps)}: "
                f"score={record['score']:.2f} | "
                f"history_steps={context.history.source_step_indices}"
            )

        return results, self.contexts


def evaluate_hierarchical_memory(std_data: dict, model: str,
                                 prompt: str,
                                 memory_config: MemoryConfig | dict):
    """便捷入口，供流水线调用。"""
    caller = LLMCaller(model=model, prompt=prompt)
    evaluator = HierarchicalMemoryEvaluator(caller, memory_config)
    return evaluator.evaluate(std_data)
