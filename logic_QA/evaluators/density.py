from .base import BaseEvaluator
from utils.text_processor import (
    extract_entities,
    extract_ui_entities,
    count_template_phrases,
)

class DensityEvaluator(BaseEvaluator):
    def __init__(self, config: dict):
        self.config = config
        self.min_think_length = config.get("thresholds", {}).get("min_think_length", 20)

    def evaluate(self, data: dict) -> dict:
        llm = data.get("llm_output", {})
        state = data.get("state", {})
        think = llm.get("think", "")

        reasons = []

        # 1. 长度校验
        if len(think.strip()) < self.min_think_length:
            return {"score": 0.0, "reason": f"Think length ({len(think)}) below minimum ({self.min_think_length})"}

        # 2. 实体对齐检查
        think_entities = extract_entities(think)
        ui_entities = extract_ui_entities(state)
        if ui_entities:
            overlap = think_entities & ui_entities
            if len(overlap) == 0:
                reasons.append("No UI entity references found in think")
            # 对齐分: 匹配数 / UI实体数，上限 1.0
            alignment_score = min(len(overlap) / len(ui_entities), 1.0)
        else:
            alignment_score = 1.0

        # 3. 模板化检测
        template_count = count_template_phrases(think)
        if template_count > 0:
            reasons.append(f"Detected {template_count} template phrase(s)")
        # 模板化惩罚: 每多一个模板短语降 0.2, 下限 0.3
        template_penalty = min(template_count * 0.2, 0.7)
        template_score = 1.0 - template_penalty

        # 综合信息密度评分
        density_score = alignment_score * 0.5 + template_score * 0.5

        return {
            "score": round(density_score, 4),
            "reason": "; ".join(reasons) if reasons else "OK",
        }
