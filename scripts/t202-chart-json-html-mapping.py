from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _normalize_text(s: str) -> str:
    s2 = re.sub(r"<[^>]+>", "", s or "")
    s2 = s2.replace("&nbsp;", " ")
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _extract_section_chart_map(html: str) -> list[dict[str, Any]]:
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
    chart_id_re = re.compile(
        r"<div\b[^>]*class=['\"]chart-container['\"][^>]*\bid=['\"]([^'\"]+)['\"][^>]*>",
        flags=re.IGNORECASE,
    )

    for idx, m in enumerate(section_re.finditer(html or "")):
        section_html = m.group(1) or ""
        title_m = title_re.search(section_html)
        title = _normalize_text(title_m.group(1) if title_m else "") or f"SECTION_{idx + 1}"
        content_m = content_re.search(section_html)
        content_html = content_m.group(1) if content_m else ""
        chart_ids = [x.strip() for x in chart_id_re.findall(content_html or "") if str(x or "").strip()]
        out.append({"section_title": title, "chart_ids": chart_ids})
    return out


def _resolve_chart_json_paths(*, chart_id: str, intermediates_root: Path) -> list[str]:
    if not intermediates_root.exists():
        return []
    hits = list(intermediates_root.glob(f"**/pic_to_json/chart_json/{chart_id}.json"))
    return [p.as_posix() for p in sorted(hits)]


def main() -> int:
    parser = argparse.ArgumentParser(description="T202: 抽样分析 chart_json 与 HTML 图表位置对应关系")
    parser.add_argument("--html", required=True, help="已生成的白皮书 HTML 文件路径")
    parser.add_argument("--intermediates-root", default="data/intermediates", help="intermediates 根目录（用于反查 chart_json 路径）")
    parser.add_argument("--out", default="", help="输出 JSON 文件路径（默认写到同级 .json）")
    args = parser.parse_args()

    html_path = Path(args.html)
    if not html_path.exists():
        raise RuntimeError(f"HTML 文件不存在：{html_path.as_posix()}")

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    mapping = _extract_section_chart_map(html)

    intermediates_root = Path(args.intermediates_root)
    enriched: list[dict[str, Any]] = []
    for it in mapping:
        rows = []
        for cid in it.get("chart_ids") or []:
            rows.append(
                {
                    "chart_id": cid,
                    "chart_json_paths": _resolve_chart_json_paths(chart_id=cid, intermediates_root=intermediates_root),
                }
            )
        enriched.append({"section_title": it.get("section_title"), "charts": rows})

    out_path = Path(args.out) if args.out else html_path.with_suffix(".chart-map.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

