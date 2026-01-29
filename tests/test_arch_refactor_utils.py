from __future__ import annotations

import asyncio

import pytest

from src.application.services.content_generation.recall_utils import gather_with_concurrency
from src.application.services.content_generation.section_context import (
    build_outline_item_heading,
    choose_source_display,
    is_bad_title,
    truncate_chars,
)
from src.application.services.embedding_adapter import to_llamaindex_embedding
from src.application.services.outline_polish.outline_polish_service import OutlinePolishService


def test_is_bad_title():
    assert is_bad_title("") is True
    assert is_bad_title("  ") is True
    assert is_bad_title("2026") is True
    assert is_bad_title("123") is True
    assert is_bad_title("报告") is True
    assert is_bad_title("证券研究报告") is True
    assert is_bad_title("市场分析") is False


def test_choose_source_display_priority():
    assert (
        choose_source_display(
            doc_title="市场研究报告",
            original_filename="a.pdf",
            doc_rel_path="x/y.md",
        )
        == "市场研究报告"
    )
    assert (
        choose_source_display(
            doc_title="报告",
            original_filename="a.pdf",
            doc_rel_path="x/y.md",
        )
        == "a"
    )
    assert (
        choose_source_display(
            doc_title="报告",
            original_filename="",
            doc_rel_path="x/y.md",
        )
        == "y.md"
    )


def test_truncate_chars():
    assert truncate_chars("", max_chars=10) == ""
    assert truncate_chars("abc", max_chars=10) == "abc"
    assert truncate_chars("abcdef", max_chars=3) == "abc"


def test_build_outline_item_heading():
    assert build_outline_item_heading(item_display="1.1 市场", depth=1).startswith("### ")
    assert build_outline_item_heading(item_display="1.1 市场", depth=2).startswith("#### ")


@pytest.mark.asyncio
async def test_gather_with_concurrency_progress_callback_awaitable():
    items = list(range(5))
    calls: list[tuple[int, int]] = []

    async def worker(i: int) -> int:
        await asyncio.sleep(0.01)
        return i * 2

    async def on_progress(done: int, total: int):
        calls.append((done, total))

    out = await gather_with_concurrency(
        items,
        worker=worker,
        concurrency=2,
        on_progress=on_progress,
        progress_every=2,
    )
    assert sorted(out) == [0, 2, 4, 6, 8]
    assert calls[-1] == (5, 5)


def test_to_llamaindex_embedding_adapter():
    class Dummy:
        def embed_query(self, text: str):
            return [1.0, 0.0, 0.0]

        def embed_documents(self, texts: list[str]):
            return [[1.0, 0.0, 0.0] for _ in texts]

    li = to_llamaindex_embedding(Dummy())
    q = li.get_query_embedding("x")
    d = li.get_text_embedding("y")
    ds = li.get_text_embedding_batch(["a", "b"])
    assert q == [1.0, 0.0, 0.0]
    assert d == [1.0, 0.0, 0.0]
    assert ds == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_outline_polish_prompt_render_is_brace_safe():
    svc = OutlinePolishService(prompt_service=object(), llm_call_site_repository=object(), llm_runtime_service=object())
    template = '{"a": 1, "industry": "{industry}"}'
    out = svc._render_system_prompt(  # noqa: SLF001
        template,
        input_data=type(
            "X",
            (),
            {
                "industry": "电力",
                "report_type": None,
                "language": None,
                "style": None,
            },
        )(),
    )
    assert '{"a": 1' in out
    assert "电力" in out
