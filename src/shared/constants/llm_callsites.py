"""LLM 调用点（CallSite）相关常量。

本文件仅作为 callsite 注册表，聚合各业务模块的 callsite 定义，用于 seed 数据库。
"""

from src.application.services.content_generation.callsites import (
    CALLSITES as CONTENT_GENERATION_CALLSITES,
)
from src.application.services.document_cleaning.callsites import (
    CALLSITES as DOC_CLEANING_CALLSITES,
)
from src.application.services.hybrid_search.callsites import (
    CALLSITES as HYBRID_SEARCH_CALLSITES,
)
from src.application.services.mineru.callsites import (
    CALLSITES as MINERU_CALLSITES,
)
from src.application.services.vector_storage.callsites import (
    CALLSITES as VECTOR_STORAGE_CALLSITES,
)

DEFAULT_CALLSITES = {
    **DOC_CLEANING_CALLSITES,
    **CONTENT_GENERATION_CALLSITES,
    **VECTOR_STORAGE_CALLSITES,
    **HYBRID_SEARCH_CALLSITES,
    **MINERU_CALLSITES,
    # 未来在此处聚合其他模块的 callsite
}

