"""MinerU 模块调用点定义（OCR/解析）。"""

from __future__ import annotations

CALLSITES = {
    "mineru:parse_document": {
        "expected_model_kind": "ocr",
        "description": "PDF 解析与 OCR（MinerU 在线服务）",
        "prompt_scope": None,
    }
}

