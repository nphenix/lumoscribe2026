"""内容生成：coverage/audit 结构的构建工具。"""

from __future__ import annotations

from typing import Any


def coverage_outline_item_error(
    *,
    item_display: str,
    depth: int,
    query: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "type": "outline_item",
        "item": item_display,
        "depth": depth,
        "query": query,
        "hits": 0,
        "max_score": None,
        "uncovered": True,
        "error": {"type": error.__class__.__name__, "message": str(error)},
    }


def coverage_outline_item_ok(
    *,
    item_display: str,
    depth: int,
    query: str,
    hits: int,
    max_score: float | None,
    uncovered: bool,
    bm25_index_used: Any,
    rerank_applied: Any,
) -> dict[str, Any]:
    return {
        "type": "outline_item",
        "item": item_display,
        "depth": depth,
        "query": query,
        "hits": hits,
        "max_score": max_score,
        "uncovered": uncovered,
        "bm25_index_used": bm25_index_used,
        "rerank_applied": rerank_applied,
    }


def coverage_section_title_error(*, query: str, error: Exception) -> dict[str, Any]:
    return {
        "type": "section_title",
        "query": query,
        "hits": 0,
        "max_score": None,
        "uncovered": True,
        "error": {"type": error.__class__.__name__, "message": str(error)},
    }


def coverage_section_title_ok(
    *,
    query: str,
    hits: int,
    max_score: float | None,
    uncovered: bool,
    bm25_index_used: Any,
    rerank_applied: Any,
) -> dict[str, Any]:
    return {
        "type": "section_title",
        "query": query,
        "hits": hits,
        "max_score": max_score,
        "uncovered": uncovered,
        "bm25_index_used": bm25_index_used,
        "rerank_applied": rerank_applied,
    }
