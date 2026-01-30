from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_tags(s: str) -> str:
    p = _TextExtractor()
    p.feed(s or "")
    p.close()
    return unescape(p.get_text())


def _normalize_text(s: str) -> str:
    s2 = _strip_tags(s)
    s2 = s2.replace("\u00a0", " ")
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _has_visible_text(s: str) -> bool:
    return re.search(r"[A-Za-z0-9\u4e00-\u9fff]", s or "") is not None


def _extract_paragraphs(html: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"<p>(.*?)</p>", html or "", flags=re.DOTALL | re.IGNORECASE):
        txt = _normalize_text(m.group(1) or "")
        if txt:
            out.append(txt)
    return out


def _extract_sections(html: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    section_re = re.compile(r"<section\b[^>]*>(.*?)</section>", flags=re.DOTALL | re.IGNORECASE)
    title_re = re.compile(
        r"<h2\b[^>]*class=['\"]section-title['\"][^>]*>(.*?)</h2>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    content_re = re.compile(
        r"<div\b[^>]*class=['\"]section-content['\"][^>]*>(.*?)</div>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    chart_re = re.compile(
        r"<div\b[^>]*class=['\"]chart-container['\"][^>]*\bid=['\"]([^'\"]+)['\"][^>]*>",
        flags=re.IGNORECASE,
    )
    for idx, m in enumerate(section_re.finditer(html or "")):
        section_html = m.group(1) or ""
        title_m = title_re.search(section_html)
        title = _normalize_text(title_m.group(1) if title_m else "") or f"SECTION_{idx + 1}"
        content_m = content_re.search(section_html)
        content_html = content_m.group(1) if content_m else ""
        text = _normalize_text(content_html)
        charts = [c.strip() for c in chart_re.findall(content_html or "") if str(c or "").strip()]
        unresolved_anchors = [
            x.strip()
            for x in re.findall(r"\[Chart:\s*([^\]]+?)\s*\]", content_html or "", flags=re.IGNORECASE)
            if str(x or "").strip()
        ]

        first_chart_pos = (content_html or "").find("class='chart-container'")
        if first_chart_pos < 0:
            first_chart_pos = (content_html or "").find('class="chart-container"')

        first_text_pos = None
        for pm in re.finditer(r"<p>(.*?)</p>", content_html or "", flags=re.DOTALL | re.IGNORECASE):
            pt = _normalize_text(pm.group(1) or "")
            if _has_visible_text(pt):
                first_text_pos = pm.start()
                break

        chart_before_text = False
        if first_chart_pos >= 0 and first_text_pos is not None:
            chart_before_text = first_chart_pos < first_text_pos

        out.append(
            {
                "title": title,
                "text_len": len(text),
                "has_text": _has_visible_text(text),
                "charts": charts,
                "unresolved_chart_anchors": unresolved_anchors,
                "chart_before_text": chart_before_text,
            }
        )
    return out


def _compute_dup_metrics(paragraphs: list[str]) -> dict[str, Any]:
    normalized = [_normalize_text(p).lower() for p in (paragraphs or [])]
    counts: dict[str, int] = {}
    for p in normalized:
        if not p:
            continue
        counts[p] = counts.get(p, 0) + 1

    dup_total = sum(max(0, c - 1) for c in counts.values() if c > 1)
    total = len([p for p in normalized if p])
    dup_ratio = (dup_total / total) if total else 0.0

    top = sorted(((k, v) for k, v in counts.items() if v > 1), key=lambda x: (-x[1], -len(x[0])))[:20]
    top_items = []
    for txt, c in top:
        top_items.append({"count": c, "sample": (txt[:220] + ("..." if len(txt) > 220 else ""))})

    return {
        "paragraph_total": total,
        "duplicate_paragraph_total": dup_total,
        "duplicate_ratio": dup_ratio,
        "top_duplicate_paragraphs": top_items,
    }


def analyze_whitepaper_html(*, html: str, html_path: str | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    paragraphs = _extract_paragraphs(html)
    sections = _extract_sections(html)
    unresolved_anchors = [
        x.strip()
        for x in re.findall(r"\[Chart:\s*([^\]]+?)\s*\]", html or "", flags=re.IGNORECASE)
        if str(x or "").strip()
    ]
    charts = [
        x.strip()
        for x in re.findall(
            r"<div\b[^>]*class=['\"]chart-container['\"][^>]*\bid=['\"]([^'\"]+)['\"][^>]*>",
            html or "",
            flags=re.IGNORECASE,
        )
        if str(x or "").strip()
    ]

    empty_sections = [s["title"] for s in sections if s.get("has_text") is False]
    sections_chart_before_text = [s["title"] for s in sections if s.get("chart_before_text") is True]

    dup = _compute_dup_metrics(paragraphs)

    return {
        "generated_at": now,
        "input_html_path": html_path,
        "metrics": {
            **dup,
            "section_total": len(sections),
            "empty_section_total": len(empty_sections),
            "empty_sections": empty_sections,
            "chart_container_total": len(charts),
            "unresolved_chart_anchor_total": len(unresolved_anchors),
            "unresolved_chart_anchors": sorted(set(unresolved_anchors)),
            "sections_chart_before_text_total": len(sections_chart_before_text),
            "sections_chart_before_text": sections_chart_before_text,
        },
        "sections": sections,
    }


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_output_dir() -> Path:
    return Path("docs") / "process" / "ai-doc-platform-phase2"


def _render_md_report(report: dict[str, Any]) -> str:
    m = report.get("metrics") or {}
    html_path = report.get("input_html_path") or ""
    title = "阶段1：白皮书生成质量基线（自动统计报告）"
    lines = [
        f"# {title}",
        "",
        f"- 生成时间: {report.get('generated_at')}",
        f"- 输入 HTML: {html_path}",
        "",
        "## 指标汇总",
        "",
        f"- 段落数: {m.get('paragraph_total')}",
        f"- 重复段落数（精确重复）: {m.get('duplicate_paragraph_total')}",
        f"- 重复率（精确重复）: {m.get('duplicate_ratio')}",
        f"- section 数: {m.get('section_total')}",
        f"- 空 section 数（无可见文本）: {m.get('empty_section_total')}",
        f"- 图表容器数: {m.get('chart_container_total')}",
        f"- 未解析图表锚点数（[Chart: id] 残留）: {m.get('unresolved_chart_anchor_total')}",
        f"- 图表先于正文出现的 section 数: {m.get('sections_chart_before_text_total')}",
        "",
        "## 明细",
        "",
        "### 空 section",
        "",
    ]
    for t in (m.get("empty_sections") or []):
        lines.append(f"- {t}")
    if not (m.get("empty_sections") or []):
        lines.append("- 无")

    lines.extend(["", "### 未解析图表锚点", ""])
    for x in (m.get("unresolved_chart_anchors") or []):
        lines.append(f"- {x}")
    if not (m.get("unresolved_chart_anchors") or []):
        lines.append("- 无")

    lines.extend(["", "### 重复段落 Top（精确重复）", ""])
    top = m.get("top_duplicate_paragraphs") or []
    if not top:
        lines.append("- 无")
    else:
        for it in top:
            lines.append(f"- x{it.get('count')}: {it.get('sample')}")

    return "\n".join(lines) + "\n"


@dataclass
class _DemoTemplate:
    original_filename: str = "demo-template"


def _build_demo_html() -> str:
    from src.application.services.content_generation.html_renderer import WhitepaperHtmlRenderer

    renderer = WhitepaperHtmlRenderer()
    section = SimpleNamespace(
        section_id="s1",
        title="示例章节",
        content="这是一段正文。\n\n[Chart: demo_chart]\n\n这是一段重复正文。\n\n这是一段重复正文。",
        rendered_charts={
            "demo_chart": SimpleNamespace(
                container_html="<div>DEMO_CHART</div>",
                script_html="<script>console.log('demo')</script>",
            )
        },
        sources=["demo-source-1"],
    )
    return renderer.render(template=_DemoTemplate(), section_results=[section], document_title="DEMO")


def _try_run_t096_e2e(*, outline_filename: str, max_docs: int, cleanup: bool) -> dict[str, Any]:
    try:
        from tests.test_content_generation import run_t096_e2e
    except Exception as e:
        raise RuntimeError(f"无法导入 tests/test_content_generation.py: {e}") from e
    return run_t096_e2e(outline_filename=outline_filename, max_docs=max_docs, cleanup=cleanup)


def main() -> int:
    parser = argparse.ArgumentParser(description="T201: 白皮书生成质量基线统计（阶段1）")
    parser.add_argument("--html", default="", help="已生成的白皮书 HTML 文件路径（优先）")
    parser.add_argument("--run-e2e", action="store_true", help="调用 T096 E2E 生成白皮书并统计（需要 data/intermediates 与 LLM 配置）")
    parser.add_argument("--outline", default="", help="drafts 目录下的大纲文件名（配合 --run-e2e）")
    parser.add_argument("--max-docs", type=int, default=6, help="建库最多处理多少个 pic_to_json 主文档（默认 6）")
    parser.add_argument("--cleanup", action="store_true", help="生成完成后清理 target 与 collection（默认不清理）")
    parser.add_argument("--out-dir", default="", help="输出目录（默认 docs/process/ai-doc-platform-phase2）")
    parser.add_argument("--demo", action="store_true", help="生成内置 demo HTML 并跑统计（用于验证脚本可运行）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path: Path | None = None
    html: str = ""

    if args.demo:
        html = _build_demo_html()
    elif args.html.strip():
        html_path = Path(args.html.strip())
        if not html_path.exists():
            raise RuntimeError(f"HTML 文件不存在：{html_path.as_posix()}")
        html = html_path.read_text(encoding="utf-8", errors="ignore")
    elif args.run_e2e:
        outline = (args.outline or "").strip()
        if not outline:
            raise RuntimeError("使用 --run-e2e 时必须提供 --outline（data/Templates/drafts 下的文件名）")
        payload = _try_run_t096_e2e(outline_filename=outline, max_docs=max(1, int(args.max_docs)), cleanup=bool(args.cleanup))
        storage_path = payload.get("storage_path")
        if not storage_path:
            raise RuntimeError(f"T096 返回缺少 storage_path: {payload}")
        html_path = Path("data") / str(storage_path)
        if not html_path.exists():
            raise RuntimeError(f"HTML 未落盘: {html_path.as_posix()}")
        html = html_path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise RuntimeError("必须指定 --html 或 --run-e2e 或 --demo")

    report = analyze_whitepaper_html(html=html, html_path=(html_path.as_posix() if html_path else None))

    ts = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"t201-baseline-{ts}.json"
    md_path = out_dir / f"t201-baseline-{ts}.md"
    _write_json(json_path, report)
    _write_text(md_path, _render_md_report(report))

    print(json_path.as_posix())
    print(md_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
