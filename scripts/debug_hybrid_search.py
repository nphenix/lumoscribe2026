import asyncio
import json
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.application.schemas.ingest import HybridSearchOptions
from src.application.services.hybrid_search_service import HybridSearchService
from src.application.services.vector_storage_service import VectorStorageService


def _latest_kb_meta() -> dict:
    con = sqlite3.connect("F:/lumoscribe2026/.runtime/sqlite/lumoscribe.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    row = cur.execute(
        "SELECT extra_metadata FROM intermediate_artifacts WHERE type='KB_CHUNKS' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return {}
    raw = row["extra_metadata"]
    if not raw:
        return {}
    try:
        meta = json.loads(raw)
    except Exception:
        return {}
    return meta if isinstance(meta, dict) else {}


async def main() -> None:
    meta = _latest_kb_meta()
    bm25_path = meta.get("bm25_index_storage_path")
    collection = meta.get("collection_name") or "default"
    print("collection", collection)
    print("bm25_path", bm25_path)

    svc = HybridSearchService(
        vector_service=VectorStorageService(),
        bm25_index_storage_paths=[bm25_path] if bm25_path else [],
    )
    resp = await svc.search(
        query="图表",
        collection_name=collection,
        options=HybridSearchOptions(top_k=5, use_rerank=False),
    )
    print("results", len(resp.results), "bm25_used", resp.metrics.bm25_index_used)
    for r in resp.results[:15]:
        meta = r.metadata or {}
        head = (r.content or "").replace("\n", " ")[:80]
        print(r.rank, r.chunk_id, meta.get("chunk_type"), "chart_ids" in meta, head)


if __name__ == "__main__":
    asyncio.run(main())
