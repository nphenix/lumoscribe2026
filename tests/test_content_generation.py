"""T096: 白皮书生成（服务端）功能测试。

约束（对齐项目要求）：
- 使用真实数据（依赖 T094 产出 data/intermediates/**/pic_to_json），不使用 mock
- 核心逻辑在服务端（API + Service），测试只做调用与验收
- 生成流程：按章节/子章节先做 RAG 召回，覆盖不足由 LLM 补全，输出单 HTML 并写入 target_files

运行：
    pytest tests/test_content_generation.py -v --tb=short
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# 允许直接用 `python tests/test_content_generation.py` 运行（不依赖 pytest 自动注入 PYTHONPATH）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.interfaces.api.app import create_app


def _pick_outline_draft() -> Path:
    drafts_dir = Path("data") / "Templates" / "drafts"
    if not drafts_dir.exists():
        raise RuntimeError(f"缺少 drafts 目录: {drafts_dir.as_posix()}")
    items = sorted(drafts_dir.glob("*.md"))
    if not items:
        raise RuntimeError(f"drafts 目录下未找到任何 .md: {drafts_dir.as_posix()}")
    return items[0]


def _extract_chapter_titles(md: str) -> list[str]:
    titles: list[str] = []
    for line in (md or "").replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if s.startswith("## "):
            titles.append(s[3:].strip())
    return [t for t in titles if t]


def _extract_some_outline_items(md: str, limit: int = 6) -> list[str]:
    out: list[str] = []
    for line in (md or "").replace("\r\n", "\n").split("\n"):
        s = line.strip()
        # 兼容多种大纲格式：
        # - Markdown 列表：- xxx / * xxx
        # - 有序列表：1. xxx
        # - 子标题：### xxx（作为 probe）
        if s.startswith("-") or s.startswith("*"):
            txt = s.lstrip("-*").strip()
            if txt:
                out.append(txt)
        elif len(s) >= 3 and s[:2].isdigit() and s[2:3] == ".":
            txt = s.split(".", 1)[1].strip()
            if txt:
                out.append(txt)
        elif s.startswith("### "):
            txt = s[4:].strip()
            if txt:
                out.append(txt)
        if len(out) >= limit:
            break
    return out


def test_t096_whitepaper_generate_from_drafts_real_data_real_llm():
    # 新路径：T094 产出落在 data/intermediates/{source_file_id}/pic_to_json/
    input_root = Path("data") / "intermediates"
    if not input_root.exists():
        raise RuntimeError(f"缺少真实数据目录，请先跑 T094：{input_root.as_posix()}")

    outline_path = _pick_outline_draft()
    outline_text = outline_path.read_text(encoding="utf-8", errors="replace")
    chapter_titles = _extract_chapter_titles(outline_text)
    assert chapter_titles, "outline drafts 中未解析到任何章节（## ...）"
    probe_items = _extract_some_outline_items(outline_text, limit=6)

    collection_name = f"t096_test_{uuid4().hex[:8]}"

    # 1) 先建库（kb_admin）
    os.environ["LUMO_API_MODE"] = "kb_admin"
    app_admin = create_app()
    client_admin = TestClient(app_admin)
    resp = client_admin.post(
        "/v1/kb/build",
        json={
            "input_root": input_root.as_posix(),
            "collection_name": collection_name,
            "recreate": True,
            # 覆盖多篇文档（从 data/intermediates/**/pic_to_json/*.md 选择，每目录唯一）
            "max_docs": 6,
            "chunk_size": 900,
            "chunk_overlap": 80,
        },
    )
    assert resp.status_code == 200, resp.text

    # 2) 生成白皮书（full）
    os.environ["LUMO_API_MODE"] = "full"
    app_full = create_app()
    client_full = TestClient(app_full)
    resp2 = client_full.post(
        "/v1/targets/whitepaper/generate",
        json={
            "workspace_id": "default",
            "collection_name": collection_name,
            "outline_filename": outline_path.name,
            # 为了保证“严格按模板标题”可断言，测试中关闭润色（润色可能改变标题措辞）
            "polish_outline": False,
            "top_k": 6,
            "rerank_top_n": 5,
            "score_threshold": 0.15,
        },
    )
    assert resp2.status_code == 200, resp2.text
    payload = resp2.json()
    assert payload.get("target_id")
    assert payload.get("storage_path")
    assert payload.get("coverage") and isinstance(payload["coverage"], dict)

    # 3) 校验 HTML 文件已落盘
    storage_path = payload["storage_path"]
    html_path = Path("data") / storage_path
    assert html_path.exists(), f"HTML 未落盘: {html_path.as_posix()}"
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    assert "<!DOCTYPE html>" in html

    # 4) 章节标题必须全部出现（由模板解析产生，不能缺失）
    for t in chapter_titles:
        assert t in html, f"章节标题缺失: {t}"

    # 5) 抽查若干子章节条目必须出现（严格对齐大纲骨架）
    if probe_items:
        for it in probe_items[:3]:
            # 允许 HTML 标签包裹，但文本应出现
            assert it in html, f"子章节条目缺失或被改写: {it}"

    # 6) 覆盖信息可观测（用于评估 RAG 召回是否理想）
    sections = payload["coverage"].get("sections") or []
    assert sections and isinstance(sections, list)
    # 至少应包含 outline_item 覆盖记录
    any_item = False
    any_bm25_used = False
    for s in sections:
        cov = s.get("coverage") or []
        for c in cov:
            if c.get("type") == "outline_item":
                any_item = True
                if c.get("bm25_index_used") is True:
                    any_bm25_used = True
                break
    assert any_item is True
    assert any_bm25_used is True

    # 7) 清理：删除生成的目标文件
    tid = payload["target_id"]
    resp3 = client_full.delete(f"/v1/targets/{tid}")
    assert resp3.status_code == 200, resp3.text

    # 8) 清理：删除测试 collection（避免污染）
    resp4 = client_admin.delete(f"/v1/kb/collections/{collection_name}")
    assert resp4.status_code == 200, resp4.text


def test_t096_outline_polish_real_llm_smoke():
    # 该测试验证 callsite `content_generation:polish_outline` 可在真实环境执行
    os.environ["LUMO_API_MODE"] = "full"

    # 通过生成端点间接触发 polish（使用 drafts 文件作为输入）
    outline_path = _pick_outline_draft()
    input_root = Path("data") / "intermediates"
    if not input_root.exists():
        pytest.skip("缺少 data/intermediates，无法跑端到端真实环境测试")

    collection_name = f"t096_polish_{uuid4().hex[:8]}"

    os.environ["LUMO_API_MODE"] = "kb_admin"
    app_admin = create_app()
    client_admin = TestClient(app_admin)
    resp = client_admin.post(
        "/v1/kb/build",
        json={
            "input_root": input_root.as_posix(),
            "collection_name": collection_name,
            "recreate": True,
            "max_docs": 6,
        },
    )
    assert resp.status_code == 200, resp.text

    os.environ["LUMO_API_MODE"] = "full"
    app_full2 = create_app()
    client_full = TestClient(app_full2)
    resp2 = client_full.post(
        "/v1/targets/whitepaper/generate",
        json={
            "workspace_id": "default",
            "collection_name": collection_name,
            "outline_filename": outline_path.name,
            "polish_outline": True,
            "top_k": 3,
            "rerank_top_n": 3,
        },
    )
    assert resp2.status_code == 200, resp2.text
    payload = resp2.json()
    assert payload.get("document_title")

    # 清理：删除目标文件与 collection
    tid = payload["target_id"]
    client_full.delete(f"/v1/targets/{tid}")
    client_admin.delete(f"/v1/kb/collections/{collection_name}")


def run_t096_e2e(
    *,
    outline_filename: str,
    workspace_id: str = "default",
    collection_name: str | None = None,
    input_root: str = "data/intermediates",
    max_docs: int = 6,
    recreate: bool = True,
    polish_outline: bool = False,
    top_k: int = 10,
    rerank_top_n: int = 5,
    score_threshold: float | None = 0.15,
    cleanup: bool = False,
) -> dict:
    """一键 E2E：建库→生成 HTML→打印结果（可选清理）。"""
    root = Path(input_root)
    if not root.exists():
        raise RuntimeError(f"缺少真实数据目录，请先跑 T094：{root.as_posix()}")
    drafts_dir = Path("data") / "Templates" / "drafts"
    outline_path = drafts_dir / outline_filename
    if not outline_path.exists():
        raise RuntimeError(f"缺少大纲模板文件：{outline_path.as_posix()}")

    col = collection_name or f"t096_{uuid4().hex[:8]}"

    # 1) kb_admin build（进程内）
    os.environ["LUMO_API_MODE"] = "kb_admin"
    app_admin = create_app()
    client_admin = TestClient(app_admin)
    resp = client_admin.post(
        "/v1/kb/build",
        json={
            "input_root": root.as_posix(),
            "collection_name": col,
            "recreate": recreate,
            "max_docs": max_docs,
            "chunk_size": 900,
            "chunk_overlap": 80,
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"kb build failed: {resp.status_code} {resp.text}")

    # 2) full generate（进程内）
    os.environ["LUMO_API_MODE"] = "full"
    app_full = create_app()
    client_full = TestClient(app_full)
    resp2 = client_full.post(
        "/v1/targets/whitepaper/generate",
        json={
            "workspace_id": workspace_id,
            "collection_name": col,
            "outline_filename": outline_filename,
            # 严格按模板列表格式生成：默认不润色
            "polish_outline": polish_outline,
            "top_k": top_k,
            "rerank_top_n": rerank_top_n,
            "score_threshold": score_threshold,
        },
    )
    if resp2.status_code != 200:
        raise RuntimeError(f"whitepaper generate failed: {resp2.status_code} {resp2.text}")

    payload = resp2.json()
    target_id = payload.get("target_id")
    storage_path = payload.get("storage_path")
    if not target_id or not storage_path:
        raise RuntimeError(f"invalid generate response: {payload}")

    html_path = Path("data") / str(storage_path)
    if not html_path.exists():
        raise RuntimeError(f"HTML 未落盘: {html_path.as_posix()}")

    # 3) 可选清理
    if cleanup:
        client_full.delete(f"/v1/targets/{target_id}")
        client_admin.delete(f"/v1/kb/collections/{col}")

    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(description="T096 one-shot E2E (kb build -> generate html)")
    parser.add_argument(
        "--outline",
        default="outline_template_89e9bb6f-ba0d-4366-b41d-9f679bfb158d.md",
        help="drafts 目录下的大纲文件名",
    )
    parser.add_argument("--collection", default="", help="collection 名称（为空则自动生成）")
    parser.add_argument("--max-docs", type=int, default=6, help="建库最多处理多少个 pic_to_json 主文档（默认 6）")
    parser.add_argument("--no-recreate", action="store_true", help="建库时不重建 collection")
    parser.add_argument("--polish-outline", action="store_true", help="是否润色大纲（可能改变标题，不建议严格模式下开启）")
    parser.add_argument("--cleanup", action="store_true", help="完成后清理 target 与 collection（默认不清理，会保留生成的 html）")
    args = parser.parse_args()

    payload = run_t096_e2e(
        outline_filename=args.outline,
        collection_name=(args.collection.strip() or None),
        max_docs=max(1, int(args.max_docs)),
        recreate=(not args.no_recreate),
        polish_outline=bool(args.polish_outline),
        cleanup=bool(args.cleanup),
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    sp = payload.get("storage_path")
    if sp:
        print("LOCAL_HTML:", str((Path.cwd() / "data" / str(sp)).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
