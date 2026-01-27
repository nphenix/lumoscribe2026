"""T095: 知识库构建（服务端）功能测试。

约束（对齐项目要求）：
- 使用真实数据（依赖 T094 产出 data/intermediates/**/pic_to_json），不使用 mock
- 核心逻辑在服务端（API + Service），测试只做调用与验收
- 检索策略固定：hybrid + rerank（通过 T023 的 callsite 注入 embedding/rerank）

运行：
    pytest tests/test_knowledge_base.py -v --tb=short
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.interfaces.api.app import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pick_any_chart_name(input_root: Path) -> str:
    # 从真实 chart_json 中取一个可检索的 chart_name（避免读取巨大 chart_index.json）
    chart_files = sorted(input_root.rglob("chart_json/*.json"))
    if not chart_files:
        raise RuntimeError(f"未找到任何 chart_json/*.json: {input_root}")
    for p in chart_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("is_chart") is True:
            name = obj.get("_chart_name") or obj.get("chart_name") or obj.get("description")
            if isinstance(name, str) and name.strip():
                return name.strip()
    raise RuntimeError(f"未找到可用的图表名称（_chart_name/chart_name/description）: {chart_files[0]}")


def test_t095_kb_build_and_query_hybrid_rerank():
    # 使用相对路径（以项目根目录为工作目录时生效）
    input_root = Path("data") / "intermediates"
    if not input_root.exists():
        raise RuntimeError(f"缺少真实数据目录，请先跑 T094：{input_root.as_posix()}")

    query = _pick_any_chart_name(input_root)
    collection_name = f"t095_test_{uuid4().hex[:8]}"

    # 1) 建库端口（kb_admin 模式）：不加载上传路由，避免 python-multipart 等无关依赖影响
    os.environ["LUMO_API_MODE"] = "kb_admin"
    app_admin = create_app()
    client_admin = TestClient(app_admin)

    # 1) 构建知识库（只取少量文档，避免全量耗时）
    resp = client_admin.post(
        "/v1/kb/build",
        json={
            "input_root": input_root.as_posix(),
            "collection_name": collection_name,
            "recreate": True,
            "max_docs": 1,
            "chunk_size": 900,
            "chunk_overlap": 80,
        },
    )
    assert resp.status_code == 200, resp.text
    build = resp.json()
    assert build["success"] is True
    assert build["collection_name"] == collection_name
    assert build["docs_indexed"] >= 1
    assert build["chunks_indexed"] > 0

    # 2) 查询端口（kb_query 模式）：独立注入 embedding + rerank
    os.environ["LUMO_API_MODE"] = "kb_query"
    app_query = create_app()
    client_query = TestClient(app_query)

    resp2 = client_query.post(
        "/v1/kb/query",
        json={
            "query": query,
            "collection_name": collection_name,
            "top_k": 5,
            "rerank_top_n": 5,
        },
    )
    assert resp2.status_code == 200, resp2.text
    data = resp2.json()
    assert data["query"] == query
    assert data["metrics"]["rerank_applied"] is True
    assert len(data["results"]) > 0
    # T095 优化：BM25 使用预建索引（可观测）
    assert data["metrics"].get("bm25_index_used") is True

    top = data["results"][0]
    assert "metadata" in top and isinstance(top["metadata"], dict)
    # 可追溯字段（至少应包含 doc_title/doc_rel_path）
    assert top["metadata"].get("doc_title")
    assert top["metadata"].get("doc_rel_path")
    # rerank 后应写入 rerank_score（由 HybridSearchService 注入）
    assert "rerank_score" in top["metadata"]

