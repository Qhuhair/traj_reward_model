from .base import BaseEvaluator
from utils.text_processor import count_negative_keywords, has_state_changed

class ConflictEvaluator(BaseEvaluator):
    def __init__(self, config: dict):
        self.config = config
        self.delta_limit = config.get("thresholds", {}).get("conflict_delta_limit", -0.2)
        self.score_threshold = config.get("thresholds", {}).get("semantic_score_threshold", 0.8)

    def evaluate(self, data: dict) -> dict:
        llm = data.get("llm_output", {})
        critique = llm.get("critique", "")
        score = llm.get("score", 0.0)
        prev_score = data.get("prev_score", None)

        reasons = []
        conflict_score = 1.0
        penalty = 0.0

        # 1. 进展一致性校验: δ = Q_t - Q_{t-1}
        if prev_score is not None:
            delta = score - prev_score
            if delta < self.delta_limit:
                has_negative = count_negative_keywords(critique) > 0
                if not has_negative:
                    reasons.append(
                        f"Score dropped by {delta:.3f} but critique lacks negative feedback"
                    )
                    penalty += 0.4

        # 2. 语义-分值对齐: 负面关键词 + 高分 → 冲突
        negative_count = count_negative_keywords(critique)
        if negative_count > 0 and score >= self.score_threshold:
            reasons.append(
                f"Critique contains {negative_count} negative keyword(s) but score is {score} (>= {self.score_threshold})"
            )
            penalty += 0.5

        # 3. 状态变化监测
        prev_state = data.get("prev_state", None)
        curr_state = data.get("state", None)
        if prev_state and curr_state:
            if not has_state_changed(prev_state, curr_state) and score > 0.5:
                reasons.append("No UI state change detected but score > 0.5")
                penalty += 0.3

        conflict_score = max(0.0, 1.0 - penalty)

        return {
            "score": round(conflict_score, 4),
            "reason": "; ".join(reasons) if reasons else "OK",
        }
