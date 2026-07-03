"""整体框架对外 API 的抽象入口。"""

from .errors import APIError, CapabilityNotFoundError
from .interfaces import FrameworkService, Handler
from .registry import HandlerRegistry
from .schema import APIRequest, APIResponse, InputMode, TaskType
from .service import ModularFrameworkService

__all__ = [
    "APIError",
    "APIRequest",
    "APIResponse",
    "CapabilityNotFoundError",
    "FrameworkService",
    "Handler",
    "HandlerRegistry",
    "InputMode",
    "ModularFrameworkService",
    "TaskType",
]
