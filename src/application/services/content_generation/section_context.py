"""内容生成：检索上下文组装的通用工具。"""

from __future__ import annotations

import re
from pathlib import Path

from src.application.schemas.ingest import SearchResult
from src.application.services.content_generation.text_utils import UUID_RE
from src.application.services.content_generation.table_formatter import format_table_for_context


def is_bad_title(title: str) -> bool:
    s = (title or "").strip()
    if not s:
        return True
    if UUID_RE.match(s):
        return True
    if re.fullmatch(r"\d{4}", s) or re.fullmatch(r"\d+", s):
        return True
    if s in {"证券研究报告", "研究报告", "年度报告", "报告"}:
        return True
    if len(s) < 4:
        return True
    return False


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def choose_source_display(
    *,
    doc_title: str | None,
    original_filename: str | None,
    doc_rel_path: str | None,
) -> str:
    dt_key = normalize_space(str(doc_title or ""))
    ofn_key = normalize_space(str(original_filename or ""))
    dr_key = normalize_space(str(doc_rel_path or ""))

    if dt_key and not is_bad_title(dt_key):
        return dt_key
    if ofn_key:
        return Path(ofn_key).stem
    if dr_key:
        return Path(dr_key).name
    return ""


def truncate_chars(text: str, *, max_chars: int) -> str:
    if not text:
        return ""
    n = int(max_chars or 0)
    if n <= 0:
        return ""
    s = str(text)
    return s if len(s) <= n else s[:n]


def pick_unique_hits(hits: list[SearchResult], *, per_top: int) -> list[SearchResult]:
    if per_top <= 0 or not hits:
        return []
    picked: list[SearchResult] = []
    seen_docs: set[str] = set()
    for r in hits:
        doc_rel = r.metadata.get("doc_rel_path") if isinstance(r.metadata, dict) else None
        src_id = r.metadata.get("source_file_id") if isinstance(r.metadata, dict) else None
        key = str(doc_rel or src_id or r.chunk_id or "").strip()
        if key and key in seen_docs:
            continue
        if key:
            seen_docs.add(key)
        picked.append(r)
        if len(picked) >= per_top:
            break
    return picked


def build_outline_item_heading(*, item_display: str, depth: int) -> str:
    lvl = max(0, int(depth) - 1)
    return f"###{'#' * lvl} {str(item_display or '').strip()}".strip()


def build_source_block(
    *,
    source_display: str,
    content: str,
    max_chars: int,
) -> str:
    text = format_table_for_context((content or "").strip())
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "…"
    header = f"【来源】{source_display}" if source_display else "【来源】"
    return "\n".join([header, text]).strip()
