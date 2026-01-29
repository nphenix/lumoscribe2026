from __future__ import annotations

import pytest

from src.application.schemas.ingest import ChunkType, ChunkingConfig, ChunkingOptions, ChunkingStrategy
from src.application.services.chunking_service import DocumentChunkingService


@pytest.mark.asyncio
async def test_structure_aware_table_is_atomic_and_not_split():
    svc = DocumentChunkingService(config=ChunkingConfig(chunk_size=120, chunk_overlap=10, min_chunk_size=10))
    text = (
        "前言段落。\n\n"
        "| 指标 | 2023 | 2024 |\n"
        "|---|---:|---:|\n"
        "| 收入 | 10 | 12 |\n"
        "| 利润 | 1 | 2 |\n"
        "| 现金流 | 3 | 4 |\n"
        "\n\n后续段落。"
    )
    chunks = await svc.chunk_document(
        text=text,
        metadata={"source_file_id": "t"},
        options=ChunkingOptions(strategy=ChunkingStrategy.STRUCTURE_AWARE, chunk_size=120, chunk_overlap=10),
    )
    table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
    assert len(table_chunks) == 1
    assert "| 指标 |" in table_chunks[0].content
