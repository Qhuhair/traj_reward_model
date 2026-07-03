"""对外 API 层的异常类型。"""


class APIError(Exception):
    """API 抽象层基础异常。"""


class CapabilityNotFoundError(APIError):
    """请求的能力未注册时抛出。"""
