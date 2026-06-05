import os
import yaml
from evaluators.density import DensityEvaluator
from evaluators.conflict import ConflictEvaluator
from utils.aggregator import ScoreAggregator


class QA_Orchestrator:
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "config", "qa_settings.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.evaluators = self._load_evaluators()
        self.aggregator = ScoreAggregator(
            weights=self.config.get("weights", {}),
            accept_threshold=self.config.get("accept_threshold", 0.5),
        )

    def _load_evaluators(self) -> list:
        enabled = self.config.get("enabled_evaluators", [])
        registry = {
            "DensityEvaluator": DensityEvaluator,
            "ConflictEvaluator": ConflictEvaluator,
        }
        evaluators = []
        for name in enabled:
            cls = registry.get(name)
            if cls:
                evaluators.append(cls(self.config))
        return evaluators

    def evaluate(self, data: dict) -> dict:
        """执行完整评判流水线"""
        results = []
        for ev in self.evaluators:
            result = ev.evaluate(data)
            results.append({
                "name": ev.__class__.__name__,
                "score": result["score"],
                "reason": result["reason"],
            })
        return self.aggregator.aggregate(results)
