"""Vector Storage 模块 LLM 调用点定义（Embedding）。"""

from __future__ import annotations

CALLSITES = {
    "vector_storage:embed_text": {
        "expected_model_kind": "embedding",
        "description": "向量化：文本 -> Embedding（用于入库与检索）",
        "prompt_scope": None,
    }
}

