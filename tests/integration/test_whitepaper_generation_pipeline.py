from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interfaces.api.app import create_app


def _pick_outline_draft() -> Path:
    drafts_dir = Path("data") / "Templates" / "drafts"
    items = sorted(drafts_dir.glob("*.md")) if drafts_dir.exists() else []
    if not items:
        raise RuntimeError("drafts 目录为空或不存在")
    return items[0]


def _norm_para(s: str) -> str:
    s2 = re.sub(r"<[^>]+>", "", s or "")
    s2 = s2.replace("\xa0", " ")
    s2 = re.sub(r"\s+", " ", s2).strip().lower()
    return s2


def _extract_paragraphs(html: str) -> list[str]:
    paras = re.findall(r"(?s)<p>(.*?)</p>", html or "")
    out: list[str] = []
    for p in paras:
        n = _norm_para(p)
        if len(n) >= 120 and "[chart:" not in n:
            out.append(n)
    return out


@pytest.mark.integration
def test_whitepaper_pipeline_structure_dedup_and_chart_placeholders():
    if os.getenv("LUMO_RUN_INTEGRATION") != "1":
        pytest.skip("integration test disabled (set LUMO_RUN_INTEGRATION=1)")

    input_root = Path("data") / "intermediates"
    if not input_root.exists():
        pytest.skip("缺少 data/intermediates，无法跑端到端集成测试")

    outline_path = _pick_outline_draft()
    collection_name = f"t204_{uuid4().hex[:8]}"

    os.environ["LUMO_API_MODE"] = "kb_admin"
    app_admin = create_app()
    client_admin = TestClient(app_admin)
    resp = client_admin.post(
        "/v1/kb/build",
        json={
            "input_root": input_root.as_posix(),
            "collection_name": collection_name,
            "recreate": True,
            "max_docs": 4,
            "chunk_size": 900,
            "chunk_overlap": 80,
        },
    )
    assert resp.status_code == 200, resp.text

    os.environ["LUMO_API_MODE"] = "full"
    app_full = create_app()
    client_full = TestClient(app_full)
    resp2 = client_full.post(
        "/v1/targets/whitepaper/generate",
        json={
            "workspace_id": "default",
            "collection_name": collection_name,
            "outline_filename": outline_path.name,
            "polish_outline": False,
            "polish_sections": False,
            "top_k": 6,
            "rerank_top_n": 5,
            "score_threshold": 0.15,
        },
    )
    assert resp2.status_code == 200, resp2.text
    payload = resp2.json()
    storage_path = payload["storage_path"]
    html_path = Path("data") / storage_path
    assert html_path.exists()

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    assert "<!DOCTYPE html>" in html

    paras = _extract_paragraphs(html)
    counts: dict[str, int] = {}
    for p in paras:
        counts[p] = counts.get(p, 0) + 1
    dupes = [p for p, c in counts.items() if c >= 2]
    assert not dupes, f"检测到疑似重复段落（exact match after normalization）：{len(dupes)}"

    assert "[Chart:" not in html, "HTML 中不应残留 Chart 占位符文本"

    rendered_total = 0
    for s in (payload.get("coverage", {}).get("sections") or []):
        rendered_total += len(s.get("rendered_charts") or [])
    chart_blocks = len(re.findall(r"class='chart-container'", html))
    assert chart_blocks == rendered_total

    tid = payload["target_id"]
    resp3 = client_full.delete(f"/v1/targets/{tid}")
    assert resp3.status_code == 200, resp3.text
    resp4 = client_admin.delete(f"/v1/kb/collections/{collection_name}")
    assert resp4.status_code == 200, resp4.text

