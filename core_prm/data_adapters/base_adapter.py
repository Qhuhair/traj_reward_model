from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AdaptedData:
    q_sequence: list
    rewards: list
    errors: list


class BaseAdapter(ABC):
    @abstractmethod
    def parse(self, raw_data) -> AdaptedData:
        """将原始标注转化为标准化的 Q 序列、奖励序列及错误报告"""
        pass
