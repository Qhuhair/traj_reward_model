"""API 抽象层的接口协议。"""

from typing import Protocol

from .schema import APIRequest, APIResponse, InputMode, TaskType


class Handler(Protocol):
    """单一能力处理器；具体实现后续由各模块适配。"""

    @property
    def task_type(self) -> TaskType:
        """处理器支持的任务类型。"""

    @property
    def input_modes(self) -> set[InputMode]:
        """处理器支持的输入形态。"""

    def handle(self, request: APIRequest) -> APIResponse:
        """处理请求并返回统一响应。"""


class FrameworkService(Protocol):
    """整体框架对外服务接口。"""

    def execute(self, request: APIRequest) -> APIResponse:
        """执行一次框架级请求。"""
