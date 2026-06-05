from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, data: dict) -> dict:
        """返回 { 'score': float, 'reason': str }"""
        pass
