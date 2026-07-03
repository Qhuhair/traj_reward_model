"""
QA 重试模块 — 为审查失败的步骤提供滑动窗口上下文重试。

用法:
    from qa_retry_with_context import retry_failed_steps
    updated_scores = retry_failed_steps(std_data, llm_scores, qa_reports)

流程:
    1. 读取 QA 报告，找出 is_accepted=False 的步骤
    2. 对每个失败步骤，构建上下文窗口：
       - 当前步的截图对
       - 前后步的文本状态描述
       - 前后步已通过审查的评分/评价
    3. 调用滑动窗口 LLM 重新评估
    4. 更新 llm_scores 中对应步骤的结果
"""
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from caller import LLMCaller

MAX_STATE_CHARS = 400


def retry_failed_steps(
    std_data: dict,
    llm_scores: list,
    qa_reports: list,
    model: str = "qwen_vllm_mm",
    prompt: str = "RRM_Qwen_Sliding",
    max_retries: int = 3,
) -> list:
    """
    对 QA 未通过的步骤进行上下文感知的重试。

    Args:
        std_data: 标准化轨迹数据
        llm_scores: 当前的 LLM 评分列表
        qa_reports: QA 审查报告列表
        model: LLM 模型名
        prompt: Prompt 模板名
        max_retries: 最大重试次数（默认 3）

    Returns:
        更新后的 llm_scores 列表
    """
    steps = std_data.get("steps", [])
    task = std_data.get("task", "")
    failed_indices = set()

    for qa in qa_reports:
        if not qa.get("is_accepted", False):
            failed_indices.add(qa["step_idx"])

    if not failed_indices:
        return llm_scores

    print(f"  [QA Retry] {len(failed_indices)} steps need retry: {sorted(failed_indices)}")

    caller = LLMCaller(model=model, prompt=prompt)
    scores = {s.get("step_idx", i + 1): s for i, s in enumerate(llm_scores)}

    for attempt in range(max_retries):
        still_failed = []

        for step_idx in sorted(failed_indices):
            i = step_idx - 1  # 0-indexed
            if i < 0 or i >= len(steps):
                continue

            step = steps[i]
            curr_score = scores.get(step_idx, {})

            # ── 构建上下文 ──
            # 上一步（仅文本+已通过标签）
            prev_action = ""
            prev_subgoal = ""
            prev_score = "N/A"
            prev_critique = ""
            if i > 0:
                prev_step = steps[i - 1]
                prev_action = prev_step.get("action_desc", "")
                prev_subgoal = prev_step.get("subgoal_text", "")
                # 如果上一步有评分且 QA 通过，带上标签
                prev_s = scores.get(step_idx - 1, {})
                prev_score = prev_s.get("score", "N/A")
                prev_critique = prev_s.get("critique", "")[:200]

            # 下一步（仅文本）
            next_action = ""
            next_subgoal = ""
            if i + 1 < len(steps):
                next_step = steps[i + 1]
                next_action = next_step.get("action_desc", "")
                next_subgoal = next_step.get("subgoal_text", "")
            else:
                next_action = "（轨迹终点）"

            # 当前步
            curr_action = step.get("action_desc", "")
            curr_element = step.get("element_id", "")
            curr_subgoal = step.get("subgoal_text", "")
            curr_state_before = step.get("state_desc_before", "")[:MAX_STATE_CHARS]
            curr_state_after = step.get("state_desc_after", "")[:MAX_STATE_CHARS]
            img_before = step.get("image_before_annotated") or step.get("image_before", "")
            img_after = step.get("image_after_annotated") or step.get("image_after", "")

            # ── 调用 LLM ──
            try:
                result = caller.call(
                    task_desc=task,
                    prev_action=prev_action,
                    prev_subgoal=prev_subgoal,
                    prev_score=str(prev_score),
                    prev_critique=prev_critique,
                    curr_subgoal=curr_subgoal,
                    curr_action=curr_action,
                    curr_element_id=curr_element,
                    curr_state_before=curr_state_before,
                    curr_state_after=curr_state_after,
                    next_subgoal=next_subgoal,
                    next_action=next_action,
                    image_before=img_before,
                    image_after=img_after,
                )

                # 检查是否生成了有效输出
                think = result.get("think", "")
                if think not in ("N/A", "") and isinstance(result.get("score"), (int, float)):
                    # 更新评分
                    scores[step_idx] = {
                        "step_idx": step_idx,
                        "action": step.get("action", ""),
                        "think": think,
                        "critique": result.get("critique", ""),
                        "score": result["score"],
                    }
                    print(f"    Retry step {step_idx}: score={result['score']:.2f} [OK]")
                else:
                    still_failed.append(step_idx)
                    print(f"    Retry step {step_idx}: still invalid output [X]")

            except Exception as e:
                still_failed.append(step_idx)
                print(f"    Retry step {step_idx}: error {e} [X]")

        # 判断是否继续重试
        failed_indices = set(still_failed)
        if not failed_indices:
            print(f"  [QA Retry] All retries successful after {attempt + 1} attempt(s)")
            break
        elif attempt < max_retries - 1:
            print(f"  [QA Retry] {len(failed_indices)} still failed, retry round {attempt + 2}/{max_retries}")
        else:
            print(f"  [QA Retry] {len(failed_indices)} steps still failed after {max_retries} retries, giving up")

    # 按原始顺序返回结果
    updated = []
    for i, s in enumerate(llm_scores):
        step_idx = s.get("step_idx", i + 1)
        updated.append(scores.get(step_idx, s))

    return updated
