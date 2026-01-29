from __future__ import annotations

import csv
import io
import re


_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def is_markdown_table_block(text: str) -> bool:
    s = (text or "").replace("\r\n", "\n").strip()
    if not s:
        return False
    lines = [ln.rstrip() for ln in s.split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    if "|" not in lines[0]:
        return False
    return _TABLE_SEP_RE.match(lines[1].strip()) is not None


def _split_row(line: str) -> list[str]:
    s = (line or "").strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    parts = [p.strip() for p in s.split("|")]
    return parts


def markdown_table_to_csv(text: str, *, max_rows: int = 80) -> str:
    s = (text or "").replace("\r\n", "\n").strip()
    if not is_markdown_table_block(s):
        return ""
    raw_lines = [ln.rstrip() for ln in s.split("\n") if ln.strip()]
    header = _split_row(raw_lines[0])
    rows: list[list[str]] = []
    for ln in raw_lines[2:]:
        if max_rows > 0 and len(rows) >= max_rows:
            break
        rows.append(_split_row(ln))

    width = max(len(header), max((len(r) for r in rows), default=0))
    header = header + [""] * max(0, width - len(header))
    norm_rows: list[list[str]] = []
    for r in rows:
        norm_rows.append(r + [""] * max(0, width - len(r)))

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for r in norm_rows:
        w.writerow(r)
    return buf.getvalue().strip()


def format_table_for_context(
    text: str,
    *,
    max_rows: int = 80,
    include_raw_fallback: bool = True,
    max_raw_chars: int = 1200,
) -> str:
    s = (text or "").strip()
    if not is_markdown_table_block(s):
        return s
    csv_txt = markdown_table_to_csv(s, max_rows=max_rows)
    parts: list[str] = []
    if csv_txt:
        parts.append("【表格CSV】")
        parts.append(csv_txt)
    if include_raw_fallback:
        raw = s if max_raw_chars <= 0 else (s[:max_raw_chars] + "…" if len(s) > max_raw_chars else s)
        parts.append("【表格原始】")
        parts.append(raw)
    return "\n".join([p for p in parts if p]).strip()

