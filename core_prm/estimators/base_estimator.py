from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EstimationResult:
    progress: list
    td_errors: list
    advantages: list
    method: str


class BaseEstimator(ABC):
    @abstractmethod
    def evaluate(self, q_sequence: list, rewards: list, config: dict) -> EstimationResult:
        """输入 Q 序列和奖励序列，返回估计结果"""
        pass
