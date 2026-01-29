from __future__ import annotations

from src.application.services.content_generation.table_formatter import (
    format_table_for_context,
    is_markdown_table_block,
    markdown_table_to_csv,
)


def test_markdown_table_to_csv_basic():
    md = """| 指标 | 2023 | 2024 |\n|---|---:|---:|\n| 收入 | 10 | 12 |\n| 利润 | 1 | 2 |"""
    assert is_markdown_table_block(md) is True
    csv_txt = markdown_table_to_csv(md)
    assert "指标" in csv_txt
    assert "收入" in csv_txt
    assert "2024" in csv_txt


def test_format_table_for_context_includes_csv_and_raw():
    md = """| A | B |\n|---|---|\n| 1 | 2 |"""
    out = format_table_for_context(md, max_rows=10, include_raw_fallback=True, max_raw_chars=1000)
    assert "【表格CSV】" in out
    assert "【表格原始】" in out
