"""处理器注册表，负责按任务类型和输入形态分发。"""

from .errors import CapabilityNotFoundError
from .interfaces import Handler
from .schema import InputMode, TaskType


class HandlerRegistry:
    """维护能力到处理器的映射，避免 API 层硬编码具体模块。"""

    def __init__(self) -> None:
        self._handlers: dict[tuple[TaskType, InputMode], Handler] = {}

    def register(self, handler: Handler) -> None:
        """注册一个处理器到其声明的所有输入形态。"""
        for mode in handler.input_modes:
            self._handlers[(handler.task_type, mode)] = handler

    def get(self, task_type: TaskType, input_mode: InputMode) -> Handler:
        """查找处理器；未注册时给出明确异常。"""
        key = (task_type, input_mode)
        if key not in self._handlers:
            raise CapabilityNotFoundError(
                f"No handler for task_type={task_type.value}, input_mode={input_mode.value}"
            )
        return self._handlers[key]

    def capabilities(self) -> list[dict[str, str]]:
        """列出当前已注册能力，供未来 HTTP/CLI 层暴露。"""
        return [
            {"task_type": task.value, "input_mode": mode.value}
            for task, mode in sorted(self._handlers, key=lambda item: (item[0].value, item[1].value))
        ]
