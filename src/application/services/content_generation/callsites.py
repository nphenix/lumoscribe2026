"""Content Generation 模块 LLM 调用点定义。"""

from __future__ import annotations

from src.application.services.content_generation.prompts import (
    SCOPE_CONTENT_GENERATION_SECTION,
    SCOPE_CONTENT_GENERATION_SECTION_POLISH,
    SCOPE_CONTENT_GENERATION_SECTION_STRUCTURED,
)

CALLSITES = {
    SCOPE_CONTENT_GENERATION_SECTION: {
        "expected_model_kind": "chat",
        "description": "按模板章节生成内容（RAG 上下文 + 模板片段）",
        "prompt_scope": SCOPE_CONTENT_GENERATION_SECTION,
    },
    SCOPE_CONTENT_GENERATION_SECTION_STRUCTURED: {
        "expected_model_kind": "chat",
        "description": "按大纲条目生成结构化内容与图表绑定（用于确定性插入锚点）",
        "prompt_scope": SCOPE_CONTENT_GENERATION_SECTION_STRUCTURED,
    },
    SCOPE_CONTENT_GENERATION_SECTION_POLISH: {
        "expected_model_kind": "chat",
        "description": "对白皮书章节 Markdown 做语言润色（不新增事实）",
        "prompt_scope": SCOPE_CONTENT_GENERATION_SECTION_POLISH,
    },
}
