"""内容生成服务（兼容导出层）。

说明：
- 为满足项目宪章（单文件 <= 2000 行）并降低耦合，真实实现已拆分至：
  `src/application/services/content_generation/` 子包
- 本文件仅保留对外导入路径的稳定性（re-export），不包含任何业务逻辑。
"""

from __future__ import annotations

from src.application.services.content_generation.errors import ContentGenerationError
from src.application.services.content_generation.service import ContentGenerationService
from src.application.services.content_generation.types import (
    ContentGenerationResult,
    OutlineItem,
    SectionGenerationResult,
    TemplateSection,
)

__all__ = [
    "ContentGenerationService",
    "ContentGenerationError",
    "ContentGenerationResult",
    "SectionGenerationResult",
    "TemplateSection",
    "OutlineItem",
]
