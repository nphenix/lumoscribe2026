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
import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.interfaces.api.app import create_app
from src.application.schemas.ingest import ChunkType, KBChunk
from src.application.services.hybrid_search_service import HybridSearchService
from src.application.services.vector_storage_service import VectorStorageService
from src.application.schemas.ingest import HybridSearchOptions
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.schema import TextNode


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


def test_recursive_retrieval_expands_chart_children():
    collection_name = f"t095_recursive_{uuid4().hex[:8]}"
    class _TestEmbedding:
        def get_text_embedding(self, text: str):
            t = (text or "").lower()
            if "revenue" in t or "trend" in t:
                return [1.0, 0.0, 0.0]
            if "cost" in t or "structure" in t:
                return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]

        def embed_documents(self, texts: list[str]):
            return [self.get_text_embedding(t) for t in texts]

    vs = VectorStorageService(embedding_model=_TestEmbedding())

    parent1 = KBChunk(
        chunk_id="p1",
        content="revenue trend discussion for 2024",
        chunk_type=ChunkType.PARAGRAPH,
        source_file_id="sf1",
        metadata={"doc_title": "d", "doc_rel_path": "x", "chart_ids": ["chart_c1"]},
        chunk_index=0,
    )
    parent2 = KBChunk(
        chunk_id="p2",
        content="cost structure analysis",
        chunk_type=ChunkType.PARAGRAPH,
        source_file_id="sf1",
        metadata={"doc_title": "d", "doc_rel_path": "x", "chart_ids": ["chart_c2"]},
        chunk_index=1,
    )
    chart1 = KBChunk(
        chunk_id="chart_c1",
        content="[Chart: c1]\nchart_name: revenue trend\nchart_type: line",
        chunk_type=ChunkType.CHART,
        source_file_id="sf1",
        metadata={"doc_title": "d", "doc_rel_path": "x", "chart_id": "c1"},
        chunk_index=0,
    )
    chart2 = KBChunk(
        chunk_id="chart_c2",
        content="[Chart: c2]\nchart_name: cost structure\nchart_type: bar",
        chunk_type=ChunkType.CHART,
        source_file_id="sf1",
        metadata={"doc_title": "d", "doc_rel_path": "x", "chart_id": "c2"},
        chunk_index=0,
    )

    asyncio.run(vs.delete_collection(collection_name))
    asyncio.run(
        vs.upsert_vectors([parent1, parent2, chart1, chart2], collection_name=collection_name)
    )

    hs = HybridSearchService(vector_service=vs, collection_name=collection_name, reranker=None)
    resp = asyncio.run(
        hs.search(
            query="revenue trend",
            collection_name=collection_name,
            options=HybridSearchOptions(top_k=1, use_rerank=False),
        )
    )
    assert resp.results
    ids = [r.chunk_id for r in resp.results]
    assert "p1" in ids
    assert "chart_c1" in ids
    assert "p2" not in ids
    assert "chart_c2" not in ids

    parent_count = sum(1 for r in resp.results if r.metadata.get("chunk_type") != "chart")
    assert parent_count == 1

    asyncio.run(vs.delete_collection(collection_name))


def test_parent_relationship_promotes_to_parent_node():
    collection_name = f"t095_parent_{uuid4().hex[:8]}"

    class _TestEmbedding:
        def get_text_embedding(self, text: str):
            t = (text or "").lower()
            if "revenue" in t:
                return [1.0, 0.0, 0.0]
            return [0.0, 0.0, 1.0]

        def embed_documents(self, texts: list[str]):
            return [self.get_text_embedding(t) for t in texts]

    vs = VectorStorageService(embedding_model=_TestEmbedding())
    asyncio.run(vs.delete_collection(collection_name))

    parent_id = "parent_sf1_0"
    parent_node = TextNode(id_=parent_id, text="context container", metadata={"doc_title": "d", "doc_rel_path": "x"})

    leaf = KBChunk(
        chunk_id="leaf1",
        content="revenue trend discussion",
        chunk_type=ChunkType.PARAGRAPH,
        source_file_id="sf1",
        metadata={"doc_title": "d", "doc_rel_path": "x", "parent_node_id": parent_id},
        chunk_index=0,
    )

    asyncio.run(
        vs.upsert_vectors(
            [leaf],
            collection_name=collection_name,
            docstore_only_nodes=[parent_node],
        )
    )

    hs = HybridSearchService(vector_service=vs, collection_name=collection_name, reranker=None)
    resp = asyncio.run(
        hs.search(
            query="revenue trend",
            collection_name=collection_name,
            options=HybridSearchOptions(top_k=1, use_rerank=False),
        )
    )
    assert resp.results
    assert resp.results[0].chunk_id == parent_id

    asyncio.run(vs.delete_collection(collection_name))

