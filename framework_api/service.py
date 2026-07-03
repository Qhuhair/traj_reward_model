"""框架 API 服务编排层。"""

from .errors import APIError
from .registry import HandlerRegistry
from .schema import APIRequest, APIResponse


class ModularFrameworkService:
    """最小服务实现：只负责分发请求，不绑定具体业务实现。"""

    def __init__(self, registry: HandlerRegistry | None = None) -> None:
        self.registry = registry or HandlerRegistry()

    def execute(self, request: APIRequest) -> APIResponse:
        """执行请求；业务异常会被转换成统一失败响应。"""
        try:
            handler = self.registry.get(request.task_type, request.input_mode)
            return handler.handle(request)
        except APIError as exc:
            return APIResponse(ok=False, error=str(exc), request_id=request.request_id)

    def capabilities(self) -> list[dict[str, str]]:
        """返回已注册能力列表。"""
        return self.registry.capabilities()
