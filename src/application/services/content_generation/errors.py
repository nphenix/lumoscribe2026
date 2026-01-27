"""内容生成模块错误定义（T042）。"""

from __future__ import annotations

from typing import Any

from src.shared.errors import AppError


class ContentGenerationError(AppError):
    """内容生成错误。"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "content_generation_error",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
        )

