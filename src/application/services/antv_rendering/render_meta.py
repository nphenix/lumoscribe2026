from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RENDER_META_FILENAME = "render_meta.json"


@dataclass(frozen=True)
class RenderMetaEntry:
    chart_id: str
    json_hash: str
    theme: str
    render_version: str
    files: dict[str, str]
    engine: str | None = None


def sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def load_render_meta(chart_dir: Path) -> dict[str, Any]:
    p = chart_dir / RENDER_META_FILENAME
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def save_render_meta(chart_dir: Path, meta: dict[str, Any]) -> None:
    p = chart_dir / RENDER_META_FILENAME
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def get_entry(meta: dict[str, Any], chart_id: str) -> dict[str, Any] | None:
    charts = meta.get("charts")
    if not isinstance(charts, dict):
        return None
    it = charts.get(chart_id)
    return it if isinstance(it, dict) else None


def upsert_entry(
    meta: dict[str, Any],
    *,
    chart_id: str,
    json_hash: str,
    theme: str,
    render_version: str,
    files: dict[str, str],
    engine: str | None = None,
) -> dict[str, Any]:
    if "charts" not in meta or not isinstance(meta.get("charts"), dict):
        meta["charts"] = {}
    charts = meta["charts"]
    assert isinstance(charts, dict)
    payload: dict[str, Any] = {
        "chart_id": chart_id,
        "json_hash": json_hash,
        "theme": theme,
        "render_version": render_version,
        "files": files,
    }
    if engine:
        payload["engine"] = engine
    charts[chart_id] = payload
    meta["version"] = 1
    return payload

