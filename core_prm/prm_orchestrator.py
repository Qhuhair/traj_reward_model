import os
import yaml
from data_adapters.llm_prompt_adapter import LLMPromptAdapter
from estimators.progress_calc import ProgressCalculator
from estimators.td_gae_engine import TDGAEEngine


class PRM_Orchestrator:
    """核心调度器 (Facade): 串联适配、算法、输出的全流程"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base, "config", "prm_params.yaml")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self._adapter = LLMPromptAdapter()
        self._estimator = self._resolve_estimator()

    def _resolve_estimator(self):
        name = self.config.get("estimator", "td_gae")
        registry = {
            "progress": ProgressCalculator,
            "td_gae": TDGAEEngine,
        }
        cls = registry.get(name, TDGAEEngine)
        return cls()

    def run(self, raw_data) -> dict:
        """执行完整流水线：适配 → 计算 → 输出"""
        adapted = self._adapter.parse(raw_data)

        if adapted.errors:
            for e in adapted.errors:
                self._log("WARN", e)

        result = self._estimator.evaluate(
            q_sequence=adapted.q_sequence,
            rewards=adapted.rewards,
            config=self.config,
        )

        return self._package(adapted, result)

    @staticmethod
    def _package(adapted, result) -> dict:
        return {
            "q_sequence": adapted.q_sequence,
            "rewards": adapted.rewards,
            "progress": result.progress,
            "td_errors": result.td_errors,
            "advantages": result.advantages,
            "estimator": result.method,
            "warnings": adapted.errors,
        }

    @staticmethod
    def _log(level: str, msg: str):
        print(f"[PRM] {level}: {msg}")
