"""
两步 GUI-Shepherd 评估器：
Step 1: 视觉识别（仅截图）→ 识别 UI 元素 + 判断与动作是否一致
Step 2: 逻辑评分（纯文本）→ 基于 Step 1 结论 + 状态描述 → 最终得分

分离目的：消除多模态模型在同时处理视觉和逻辑时的上下文冲突。
"""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from caller import LLMCaller

MAX_STATE_CHARS = 400


class TwoStepGSEvaluator:
    """
    两步评估（分离视觉与逻辑）：

    Step 1 (视觉层): 只传截图，模型填空识别 UI 元素 + 匹配判断。
        输出: "1. 图中标注的点击位置对应的UI元素是：XXX\n2. 该元素与动作描述是否一致：一致/不一致"

    Step 2 (逻辑层): 按 Step 1 结论分支：
        一致 → 传截图 + 状态文本，模型结合视觉和文本验证评分
        不一致 → 纯文本，仅根据状态描述评分（不传截图避免混淆）
        输出: 【最终得分: 0】或【最终得分: 1】
    """

    def __init__(self, model="qwen_gs"):
        self.caller_step1 = LLMCaller(model=model, prompt="RRM_Qwen_GS")
        # step2 caller 在 evaluate_step 中按分支动态创建

    @staticmethod
    def _parse_step1_output(text: str) -> dict:
        """从模型冗长思考中提取元素名和匹配结论。

        策略：从文本末尾反向扫描，取最后一个实质性的「一致/不一致」作为结论。
        模型的 CoT 思考期间会反复提及这些词，但最终结论总是在末尾附近。
        """
        element = "未知"
        matched = "不一致"

        # 1. 提取 UI 元素：搜索常见的答句模式
        for pat in [
            r"UI元素[是为]：?\s*(.+?)(?:\n|$)",
            r"元素[是为]：?\s*(.+?)(?:\n|$)",
            r"答[：:]\s*(.+?)(?:\n|$)",
            r"对应.*?(?:是|为)\s*(.+?)(?:\n|$|，|。)",
        ]:
            m1 = re.search(pat, text)
            if m1:
                candidate = m1.group(1).strip().rstrip("。，,;；\"'")
                if len(candidate) > 1 and candidate not in ("一致", "不一致", "____", "N/A"):
                    element = candidate
                    break

        # 2. 匹配结论：从文本末尾反向扫描，找第一个实质性的 一致/不一致
        lines = text.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            # 跳过纯格式/元分析行
            if re.match(r"^[\*#\-\d\.\s]+$", line):
                continue
            if any(kw in line for kw in ["问题", "任务", "格式", "输出", "约束", "评分规则", "____"]):
                continue
            # 找"一致"且前面不是"不"
            if re.search(r"(?<!不)一致", line) and "不一致" not in line:
                matched = "一致"
                break
            if "不一致" in line:
                matched = "不一致"
                break

        return {"element": element, "matched": matched}

    @staticmethod
    def _parse_step2_score(text: str) -> float:
        """从 Step 2 输出提取最终得分（兼容多种格式）"""
        # 优先匹配结构化标签
        m = re.search(r"【最终得分[】:：]\s*(\d+)", text)
        if m:
            return 1.0 if int(m.group(1)) >= 1 else 0.0

        # 回退：搜索"最终得分: X" 或 "得分: X"
        m = re.search(r"(?:最终)?得分[：:]\s*(\d+)", text)
        if m:
            return 1.0 if int(m.group(1)) >= 1 else 0.0

        # 最后回退：取文本末尾的孤立的 0 或 1
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for l in reversed(lines):
            if re.fullmatch(r"[01]", l):
                return float(l)

        return 0.0

    def evaluate_step(self, step: dict, task: str) -> dict:
        """评估单步，返回 {think, critique, score}"""

        img_before = step.get("image_before_annotated") or step.get("image_before", "")
        img_after  = step.get("image_after_annotated") or step.get("image_after", "")

        # ── Step 1: 仅视觉 — 截图 → 元素识别 + 匹配判断 ──
        step1_result = self.caller_step1.call(
            task_desc=task,
            subgoal=step.get("subgoal_text", ""),
            action_desc=step.get("action_desc", ""),
            image_before=img_before,
            image_after=img_after,
        )
        step1_text = step1_result.get("think", "").strip()
        s1 = self._parse_step1_output(step1_text)

        # 质量检测：如果 step1 输出过长（>500 chars），模型在循环/混乱，保守标记为不一致
        if len(step1_text) > 500:
            s1["matched"] = "不一致"

        # ── Step 2: 分支评分 ──
        step1_summary = f"UI元素: {s1['element']}, 与动作描述是否一致: {s1['matched']}"
        step2_kwargs = dict(
            task_desc=task,
            subgoal=step.get("subgoal_text", ""),
            action_desc=step.get("action_desc", ""),
            state_desc_before=step.get("state_desc_before", "")[:MAX_STATE_CHARS],
            state_desc_after=step.get("state_desc_after", "")[:MAX_STATE_CHARS],
            step1_output=step1_summary,
        )

        if s1["matched"] == "一致":
            # 视觉匹配一致 → 传截图让模型验证
            step2_caller = LLMCaller(model="qwen_gs", prompt="RRM_Qwen_GS2")
            step2_result = step2_caller.call(
                **step2_kwargs,
                image_before=img_before,
                image_after=img_after,
            )
        else:
            # 视觉匹配不一致 → 纯文本，不传截图
            step2_caller = LLMCaller(model="qwen_gs", prompt="RRM_Qwen_GS2_text")
            step2_result = step2_caller.call(**step2_kwargs)
        # 优先用 caller 解析出的 score（_parse_gs 已正确提取【最终得分】）
        score = float(step2_result.get("score", 0) or 0)
        if score == 0.0:
            # 回退：从 think 文本中重新提取（兼容 _parse_gs 把 think 设为 N/A 的情况）
            step2_text = step2_result.get("think", "").strip()
            score = self._parse_step2_score(step2_text)
        else:
            step2_text = step2_result.get("think", "").strip()

        # ── 组装 think / critique ──
        think = (
            f"[视觉识别] 元素: {s1['element']} | 匹配: {s1['matched']}\n"
            f"[逻辑评分] {step2_text}"
        )
        critique = f"视觉匹配={s1['matched']}; 最终得分={score:.0f}"

        return {"think": think, "critique": critique, "score": score}

    def evaluate_trajectory(self, std_data: dict) -> list:
        """评估整条轨迹，逐步骤执行两步评估"""
        steps = std_data.get("steps", [])
        task = std_data.get("task", "")
        results = []

        for i, step in enumerate(steps):
            result = self.evaluate_step(step, task)
            results.append({
                "step_idx": step.get("step_idx", i + 1),
                "action": step.get("action", ""),
                "think": result["think"],
                "critique": result["critique"],
                "score": result["score"],
            })
            print(f"  Step {i+1}/{len(steps)}: score={result['score']:.0f} | {result['critique']}")

        return results
