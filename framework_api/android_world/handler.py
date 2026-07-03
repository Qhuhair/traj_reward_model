"""AndroidWorld 评测 handler。"""

from framework_api.interfaces import Handler
from framework_api.schema import APIRequest, APIResponse, InputMode, TaskType

from .command import command_preview, model_env
from .config import AndroidWorldEvalConfig


class AndroidWorldEvalHandler(Handler):
    """返回 AndroidWorld 评测计划；暂不直接启动模拟器和长任务。"""

    task_type = TaskType.RUN_PIPELINE
    input_modes = {InputMode.JSON}

    def handle(self, request: APIRequest) -> APIResponse:
        """把 API 请求转换为 AndroidWorld 评测计划。"""
        config = AndroidWorldEvalConfig.from_payload(request.payload, request.options)
        return APIResponse(
            ok=True,
            data={
                "android_world_root": str(config.android_world_root),
                "command": command_preview(config),
                "env": model_env(config),
                "note": "当前 handler 只生成评测计划，不直接执行 AndroidWorld。",
            },
            request_id=request.request_id,
        )
