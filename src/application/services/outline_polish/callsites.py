"""大纲优化模块 LLM 调用点定义。"""

from __future__ import annotations

from src.application.services.outline_polish.prompts import SCOPE_OUTLINE_POLISH

# 大纲优化的调用点配置
CALLSITES = {
    SCOPE_OUTLINE_POLISH: {
        "expected_model_kind": "chat",
        "description": "大纲润色与优化（基于行业配置的结构化输出）",
        "prompt_scope": SCOPE_OUTLINE_POLISH,
        # 可选：参数覆盖配置
        # "temperature": 0.7,
        # "max_tokens": 4000,
    },
}
