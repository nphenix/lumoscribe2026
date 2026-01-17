"""Document Cleaning 模块 LLM 调用点定义。"""

from __future__ import annotations

from src.application.services.document_cleaning.prompts import (
    SCOPE_CHART_EXTRACTION,
    SCOPE_DOC_CLEANING,
)

# 说明：
# - key 建议与 prompt scope 对齐，便于统一管理
# - prompt_scope 为空时运行时默认使用 key
CALLSITES = {
    SCOPE_DOC_CLEANING: {
        "expected_model_kind": "chat",
        "description": "文档清洗（规则过滤后进行 LLM 清洗）",
        "prompt_scope": SCOPE_DOC_CLEANING,
    },
    SCOPE_CHART_EXTRACTION: {
        "expected_model_kind": "multimodal",
        "description": "图表图片 -> JSON（多模态识别）",
        "prompt_scope": SCOPE_CHART_EXTRACTION,
    },
}

