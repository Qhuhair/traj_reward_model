import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from LLM_caller.memory.memory_builder import MemoryBuilder
from LLM_caller.memory.schema import HistorySummary, MemoryConfig


class FakeSummarizer:
    """离线摘要器：只记录历史范围，不调用大模型。"""

    def summarize(self, task, steps, current_index, scores=None):
        end_exclusive = current_index - 2
        source = steps[:max(0, end_exclusive)]
        indices = [s["step_idx"] for s in source]
        return HistorySummary(
            text="历史摘要步骤: " + ",".join(str(i) for i in indices),
            start_step_idx=indices[0] if indices else None,
            end_step_idx=indices[-1] if indices else None,
            source_step_indices=indices,
            summarizer="fake",
        )


def build_fake_data():
    """构造 5 步假轨迹，用于验证第 4 步的记忆边界。"""
    steps = []
    for i in range(1, 6):
        steps.append({
            "step_idx": i,
            "action": f"Tap fake {i}",
            "action_desc": f"执行第{i}步动作",
            "subgoal_text": f"子目标{i}",
            "element_id": f"elem_{i}",
            "state_desc_before": f"第{i}步执行前页面",
            "state_desc_after": f"第{i}步执行后页面",
            "image_before": f"/tmp/before_{i}.jpg",
            "image_after": f"/tmp/after_{i}.jpg",
            "to_node_desc": f"第{i}步目标页面",
            "is_backtrack": False,
        })
    return {"task": "测试分层记忆模块", "steps": steps}


def main():
    config = MemoryConfig(enabled=True, recent_window_size=3)
    builder = MemoryBuilder(config=config, summarizer=FakeSummarizer())
    data = build_fake_data()

    # 当前测试第 4 步：历史应只有第 1 步，最近窗口应为第 2、3、4 步。
    context = builder.build_context(data, current_index=3)
    values = builder.build_prompt_values(data, current_index=3)

    assert context.history.source_step_indices == [1]
    assert [s.step_idx for s in context.recent_steps] == [2, 3, 4]
    assert context.recent_steps[-1].is_current is True
    assert context.current_image_before == "/tmp/before_4.jpg"
    assert context.current_image_after == "/tmp/after_4.jpg"
    assert "步骤5" not in values["recent_steps_detail"]
    assert values["history_summary"] == "历史摘要步骤: 1"

    print("minimal memory test ok")
    print("history:", context.history.source_step_indices)
    print("recent:", [s.step_idx for s in context.recent_steps])
    print("current_images:", context.current_image_before, context.current_image_after)


if __name__ == "__main__":
    main()
