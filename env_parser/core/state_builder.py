from abc import ABC, abstractmethod
from ..utils.perception_utils import build_state_text


class BaseStateBuilder(ABC):
    """策略基类 — 对扩展开放，对修改关闭"""

    @abstractmethod
    def build(self, perception: list, config: dict) -> str:
        pass


class SimpleStateBuilder(BaseStateBuilder):
    """直接拼接感知元素为自然语言描述"""

    def build(self, perception: list, config: dict) -> str:
        cfg = dict(config)
        cfg["cluster_by_area"] = False
        return build_state_text(perception, cfg)


class ClusteredStateBuilder(BaseStateBuilder):
    """按屏幕区域(top/mid/bottom)聚类后组织描述"""

    def build(self, perception: list, config: dict) -> str:
        cfg = dict(config)
        cfg["cluster_by_area"] = True
        return build_state_text(perception, cfg)


def create_state_builder(config: dict) -> BaseStateBuilder:
    """工厂函数 — 根据配置选择策略"""
    use_cluster = config.get("cluster_by_area", True)
    if use_cluster:
        return ClusteredStateBuilder()
    return SimpleStateBuilder()
