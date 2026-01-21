from __future__ import annotations

import logging
from typing import Any

try:
    # pythonjsonlogger>=3 moved JsonFormatter here
    from pythonjsonlogger.json import JsonFormatter  # type: ignore
except Exception:  # pragma: no cover
    from pythonjsonlogger import jsonlogger  # type: ignore

    JsonFormatter = jsonlogger.JsonFormatter  # type: ignore

from src.shared.request_id import get_request_id


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler()
    formatter = JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    # 重置 handlers，避免重复输出
    root.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

# 兼容旧代码直接导入 logger 的情况
logger = get_logger("app")



def log_extra(**kwargs: Any) -> dict[str, Any]:
    # 统一将附加信息写入 structured logging 的 extra 字段
    return {"extra": kwargs}
