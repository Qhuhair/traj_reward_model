import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from LLM_caller.memory.memory_builder import MemoryBuilder
from LLM_caller.memory.schema import HistorySummary, MemoryConfig


DEFAULT_STD_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "crossapp_qwen35_4b_vllm_text",
    "20250113_21442_test",
    "traj_007",
    "standardized.json",
)


class FakeSummarizer:
    def __init__(self):
        self.calls = []

    def summarize(self, task, steps, current_index, scores=None):
        end_exclusive = current_index - 2
        source = steps[:max(0, end_exclusive)]
        indices = [s.get("step_idx", i + 1) for i, s in enumerate(source)]
        self.calls.append(indices)
        return HistorySummary(
            text="历史摘要: " + ",".join(str(i) for i in indices),
            start_step_idx=indices[0] if indices else None,
            end_step_idx=indices[-1] if indices else None,
            source_step_indices=indices,
            summarizer="fake",
        )


def main():
    with open(DEFAULT_STD_PATH, "r", encoding="utf-8") as f:
        std_data = json.load(f)

    summarizer = FakeSummarizer()
    config = MemoryConfig(enabled=True, recent_window_size=3)
    builder = MemoryBuilder(config, summarizer)
    context = builder.build_context(std_data, current_index=3)
    values = builder.build_prompt_values(std_data, current_index=3)

    assert context.history.source_step_indices == [1]
    assert [s.step_idx for s in context.recent_steps] == [2, 3, 4]
    assert context.recent_steps[-1].is_current
    assert context.current_image_before
    assert context.current_image_after
    assert "步骤5" not in values["recent_steps_detail"]
    print("memory builder ok")
    print(json.dumps(context.to_dict(), ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
