from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.application.schemas.chart_spec import ChartConfig
from src.application.services.content_generation.chart_json_converter import ChartJsonConverter


def _parse_chart_id_from_text(text: str) -> str | None:
    s = (text or "").replace("\r\n", "\n").strip()
    if not s:
        return None
    first = s.split("\n", 1)[0].strip()
    if first.startswith("[Chart:") and first.endswith("]"):
        inner = first[len("[Chart:") : -1].strip()
        return inner or None
    return None


def extract_chart_candidates_from_vector_results(
    items: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    ids: list[str] = []
    for it in items or []:
        meta = it.get("metadata") if isinstance(it, dict) else None
        meta = meta if isinstance(meta, dict) else {}
        cid = str(meta.get("chart_id") or "").strip() or None
        if not cid:
            cid = _parse_chart_id_from_text(str(it.get("document") or ""))
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(it)
        ids.append(cid)
        if len(out) >= max_candidates:
            break
    return out, ids


def build_chart_candidates_block(
    candidates: list[dict[str, Any]],
    *,
    max_chars_per_chart: int = 600,
) -> str:
    lines: list[str] = []
    for it in candidates or []:
        meta = it.get("metadata") if isinstance(it, dict) else None
        meta = meta if isinstance(meta, dict) else {}
        cid = str(meta.get("chart_id") or "").strip() or _parse_chart_id_from_text(str(it.get("document") or "")) or ""
        name = str(meta.get("chart_name") or meta.get("_chart_name") or "").strip()
        text = str(it.get("document") or "").replace("\r\n", "\n").strip()
        body = text
        if body.startswith("[Chart:"):
            body = body.split("\n", 1)[-1].strip()
        body = body[: max(0, int(max_chars_per_chart or 0))] if max_chars_per_chart > 0 else body
        header = f"[Chart: {cid}] {name}".strip()
        lines.append(header)
        if body:
            lines.append(body)
        lines.append("")
    return "\n".join(lines).strip()


def chart_configs_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    kb_input_root: Path | None,
    converter: ChartJsonConverter,
) -> list[ChartConfig]:
    if kb_input_root is None:
        return []
    out: list[ChartConfig] = []
    for it in candidates or []:
        meta = it.get("metadata") if isinstance(it, dict) else None
        meta = meta if isinstance(meta, dict) else {}
        json_path = str(meta.get("chart_json_path") or "").strip()
        if not json_path:
            continue
        p = Path(json_path)
        if not p.is_absolute():
            p = (kb_input_root / p).resolve()
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict) or obj.get("is_chart") is not True:
            continue
        cid = str(obj.get("_chart_id") or meta.get("chart_id") or "").strip()
        cfgs = converter.chart_json_to_configs(obj) or []
        for cfg in cfgs:
            extra = cfg.extra if isinstance(cfg.extra, dict) else {}
            if cid:
                extra = {**extra, "chart_anchor_id": cid}
            cfg.extra = extra or None
            out.append(cfg)
    return out
