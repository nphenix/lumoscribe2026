"""提示词相关常量。

本文件仅作为提示词注册表，聚合各业务模块的提示词定义。
"""

from src.application.services.document_cleaning.prompts import (
    PROMPTS as DOC_CLEANING_PROMPTS,
)
from src.application.services.document_cleaning.prompts import (
    SCOPE_CHART_EXTRACTION,
    SCOPE_DOC_CLEANING,
)
from src.application.services.content_generation.prompts import (
    PROMPTS as CONTENT_GENERATION_PROMPTS,
)
from src.application.services.content_generation.prompts import (
    SCOPE_CONTENT_GENERATION_SECTION,
    SCOPE_CONTENT_GENERATION_SECTION_POLISH,
    SCOPE_CONTENT_GENERATION_SECTION_STRUCTURED,
)
from src.application.services.outline_polish.prompts import (
    PROMPTS as OUTLINE_POLISH_PROMPTS,
)
from src.application.services.outline_polish.prompts import (
    SCOPE_OUTLINE_POLISH,
)

# 导出 Scope 常量供其他模块使用
__all__ = [
    "SCOPE_DOC_CLEANING",
    "SCOPE_CHART_EXTRACTION",
    "SCOPE_CONTENT_GENERATION_SECTION",
    "SCOPE_CONTENT_GENERATION_SECTION_STRUCTURED",
    "SCOPE_CONTENT_GENERATION_SECTION_POLISH",
    "SCOPE_OUTLINE_POLISH",
    "DEFAULT_PROMPTS",
]

# 聚合所有提示词种子
DEFAULT_PROMPTS = {
    **DOC_CLEANING_PROMPTS,
    **CONTENT_GENERATION_PROMPTS,
    **OUTLINE_POLISH_PROMPTS,
    # 未来在此处聚合其他模块的提示词
    # **ANOTHER_MODULE_PROMPTS,
}
