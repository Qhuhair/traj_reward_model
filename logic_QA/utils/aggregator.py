class ScoreAggregator:
    def __init__(self, weights: dict, accept_threshold: float = 0.5):
        self.weights = weights
        self.accept_threshold = accept_threshold

    def aggregate(self, evaluator_results: list) -> dict:
        """
        evaluator_results: [{ 'name': str, 'score': float, 'reason': str }, ...]
        返回 QualityReport dict
        """
        if not evaluator_results:
            return self._empty_report("No evaluators ran")

        meta_score = 0.0
        total_weight = 0.0
        failed_reasons = []
        metrics = {}

        for r in evaluator_results:
            name = r["name"].lower()
            score = r["score"]
            reason = r.get("reason", "")

            key = name.replace("evaluator", "") + "_index"
            metrics[key] = score

            w = self.weights.get(name, 1.0 / len(evaluator_results))
            total_weight += w
            meta_score += score * w

            if score < self.accept_threshold and reason:
                failed_reasons.append(f"[{r['name']}] {reason}")

        if total_weight > 0:
            meta_score /= total_weight

        is_accepted = meta_score >= self.accept_threshold and len(failed_reasons) == 0

        return {
            "is_accepted": is_accepted,
            "meta_score": round(meta_score, 4),
            "failed_reasons": failed_reasons,
            "metrics": metrics,
        }

    @staticmethod
    def _empty_report(reason: str) -> dict:
        return {
            "is_accepted": False,
            "meta_score": 0.0,
            "failed_reasons": [reason],
            "metrics": {},
        }
