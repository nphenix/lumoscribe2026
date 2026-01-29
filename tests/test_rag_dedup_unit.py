from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from src.application.services.hybrid_search_service import HybridSearchService
from src.application.services.rag_dedup import dedup_group_id, stable_fingerprint


def test_stable_fingerprint_is_deterministic():
    a = stable_fingerprint("A  B，C。")
    b = stable_fingerprint("a b c")
    assert a
    assert a == b


def test_dedup_group_id_chart_prefers_chart_id():
    gid = dedup_group_id(content="whatever", chunk_type="chart", metadata={"chart_id": "X"})
    assert gid == "chart:X"


def test_hybrid_search_dedup_by_group_keeps_one():
    svc = HybridSearchService()
    n1 = TextNode(
        text="t1",
        id_="n1",
        metadata={"dedup_group_id": "p:aaa", "doc_rel_path": "a.md", "doc_title": "A"},
    )
    n2 = TextNode(
        text="t2",
        id_="n2",
        metadata={"dedup_group_id": "p:aaa", "doc_rel_path": "b.md", "doc_title": "B"},
    )
    out = svc._dedup_by_group([NodeWithScore(node=n1, score=1.0), NodeWithScore(node=n2, score=2.0)])
    assert len(out) == 1
    meta = dict(getattr(out[0].node, "metadata", {}) or {})
    cs = meta.get("citation_sources")
    assert isinstance(cs, list)
    assert {x.get("doc_rel_path") for x in cs if isinstance(x, dict)} == {"a.md", "b.md"}
