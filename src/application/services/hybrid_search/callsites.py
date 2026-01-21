"""Hybrid Search 模块 LLM 调用点定义（Rerank）。"""

from __future__ import annotations

CALLSITES = {
    "hybrid_search:rerank": {
        "expected_model_kind": "rerank",
        "description": "混合检索：候选结果重排序（Rerank）",
        "prompt_scope": None,
    }
}

