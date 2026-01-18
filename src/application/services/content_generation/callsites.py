"""Content Generation 模块 LLM 调用点定义。"""

from __future__ import annotations

from src.application.services.content_generation.prompts import (
    SCOPE_CONTENT_GENERATION_SECTION,
    SCOPE_OUTLINE_POLISH,
)

CALLSITES = {
    SCOPE_CONTENT_GENERATION_SECTION: {
        "expected_model_kind": "chat",
        "description": "按模板章节生成内容（RAG 上下文 + 模板片段）",
        "prompt_scope": SCOPE_CONTENT_GENERATION_SECTION,
    },
    SCOPE_OUTLINE_POLISH: {
        "expected_model_kind": "chat",
        "description": "用户自定义大纲的 LLM 润色",
        "prompt_scope": SCOPE_OUTLINE_POLISH,
    },
}
