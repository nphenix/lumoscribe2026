"""白皮书 HTML 渲染（T096）。

目标：
- 将每个 section 的 Markdown 内容渲染为 HTML
- 挂载图表 snippet 与参考来源

说明：
- 这是纯渲染层：不做检索、不做 LLM、不做建库相关逻辑
"""

from __future__ import annotations

import html as _html
import re
from typing import Any

from src.domain.entities.template import Template


class WhitepaperHtmlRenderer:
    """白皮书 HTML 渲染器（轻量、无依赖）。"""

    def render(
        self,
        *,
        template: Template,
        section_results: list[Any],
        document_title: str | None = None,
    ) -> str:
        doc_title = (document_title or "").strip() or template.original_filename
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='zh-CN'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{_html.escape(doc_title)}</title>",
            "<style>",
            self._get_default_styles(),
            "</style>",
            "</head>",
            "<body>",
            "<div class='container'>",
            f"<h1 class='document-title'>{_html.escape(doc_title)}</h1>",
        ]

        for section in section_results:
            html_parts.extend(self._render_section_html(section))

        html_parts.extend(["</div>", "</body>", "</html>"])
        return "\n".join(html_parts)

    def _render_section_html(self, section: Any) -> list[str]:
        section_id = getattr(section, "section_id", "")
        title = getattr(section, "title", "")
        content = getattr(section, "content", "") or ""
        rendered_charts = getattr(section, "rendered_charts", {}) or {}
        sources = getattr(section, "sources", []) or []

        parts = [
            f"<section id='{_html.escape(str(section_id))}' class='section'>",
            f"<h2 class='section-title'>{_html.escape(str(title))}</h2>",
        ]

        content_html = self._markdown_to_html(str(content))

        # 图表插入位置：尽量“靠近正文”，避免全部堆在 section 末尾（不依赖 LLM 指定位置）
        chart_blocks_by_key: dict[str, str] = {}
        for chart_key in sorted(rendered_charts.keys()):
            chart = rendered_charts.get(chart_key)
            if chart is None:
                continue
            blk = [
                f"<div class='chart-container' id='{_html.escape(str(chart_key))}'>",
                (getattr(chart, "container_html", "") or ""),
                (getattr(chart, "script_html", "") or ""),
                "</div>",
            ]
            chart_blocks_by_key[str(chart_key)] = "\n".join(blk)

        content_html = self._inject_charts_into_content(content_html, chart_blocks_by_key)
        parts.append(f"<div class='section-content'>{content_html}</div>")

        # 参考来源
        if sources:
            parts.append("<div class='sources'>")
            parts.append("<h4>参考来源</h4>")
            parts.append("<ol>")
            for s in sources:
                parts.append(f"<li>{_html.escape(str(s))}</li>")
            parts.append("</ol>")
            parts.append("</div>")

        parts.append("</section>")
        return parts

    def _markdown_to_html(self, markdown: str) -> str:
        """非常轻量的 Markdown -> HTML。

        约束：仅支持本项目白皮书输出的常见子集（标题/列表/段落/粗斜体）。
        """
        raw = (markdown or "").replace("\r\n", "\n")
        # 先把 Markdown 表格块转为占位符（避免被整体 escape 成一行文本）
        placeholders: list[str] = []
        tables: list[str] = []
        lines = raw.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            nxt = lines[i + 1] if i + 1 < len(lines) else ""

            def _is_table_sep(s: str) -> bool:
                ss = (s or "").strip()
                if "|" not in ss:
                    return False
                # 允许形如 | --- | ---: | :--- |
                ss2 = ss.replace("|", "").replace(":", "").replace("-", "").strip()
                return ss2 == ""

            def _parse_row(s: str) -> list[str]:
                ss = (s or "").strip()
                # 去掉首尾 |
                if ss.startswith("|"):
                    ss = ss[1:]
                if ss.endswith("|"):
                    ss = ss[:-1]
                cells = [c.strip() for c in ss.split("|")]
                return [c for c in cells if c != ""]

            if "|" in line and _is_table_sep(nxt):
                header = _parse_row(line)
                i += 2
                rows: list[list[str]] = []
                while i < len(lines):
                    r = lines[i]
                    if not r.strip():
                        break
                    if "|" not in r:
                        break
                    rows.append(_parse_row(r))
                    i += 1
                # 生成 HTML table（对单元格做 escape）
                ths = "".join([f"<th>{_html.escape(c)}</th>" for c in header])
                body_rows = []
                for rr in rows:
                    tds = "".join([f"<td>{_html.escape(c)}</td>" for c in rr])
                    body_rows.append(f"<tr>{tds}</tr>")
                table_html = (
                    "<table class='md-table'>"
                    f"<thead><tr>{ths}</tr></thead>"
                    f"<tbody>{''.join(body_rows)}</tbody>"
                    "</table>"
                )
                idx = len(tables)
                token = f"__TABLE_BLOCK_{idx}__"
                tables.append(table_html)
                placeholders.append(token)
                lines.insert(i, "")  # 保证块级分隔
                lines.insert(i, token)
                lines.insert(i, "")
                i += 3
                continue

            i += 1

        preprocessed = "\n".join(lines)
        html_content = _html.escape(preprocessed)

        # 标题（从深到浅，避免 #### 被 ### 吃掉）
        html_content = re.sub(r"^###### (.+)$", r"<h6>\1</h6>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^##### (.+)$", r"<h5>\1</h5>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html_content, flags=re.MULTILINE)

        # 粗体
        html_content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_content)
        html_content = re.sub(r"__(.+?)__", r"<strong>\1</strong>", html_content)

        # 斜体（避免与粗体冲突：保持简化实现）
        html_content = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html_content)
        html_content = re.sub(r"_(.+?)_", r"<em>\1</em>", html_content)

        # 列表（简化版：生成 li；包装 ul 在下方做一次块级修正）
        html_content = re.sub(r"^- (.+)$", r"<li>\1</li>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^(\d+)\. (.+)$", r"<li>\2</li>", html_content, flags=re.MULTILINE)

        # 段落：先按空行切段
        html_content = re.sub(r"\n\n", r"</p><p>", html_content)
        html_content = f"<p>{html_content}</p>"

        # 清理空段落/标题被 p 包裹
        html_content = re.sub(r"<p>\s*</p>", "", html_content)
        html_content = re.sub(r"<p>(<h[1-6]>.*</h[1-6]>)</p>", r"\1", html_content)

        # 把连续 li 包成 ul（避免裸 li）
        html_content = re.sub(
            r"(?s)(?:<li>.*?</li>\s*){2,}",
            lambda m: "<ul>\n" + m.group(0).strip() + "\n</ul>",
            html_content,
        )

        # 表格占位符回填（去掉可能的 <p> 包裹）
        for idx, table_html in enumerate(tables):
            token = f"__TABLE_BLOCK_{idx}__"
            html_content = html_content.replace(f"<p>{token}</p>", table_html)
            html_content = html_content.replace(token, table_html)

        return html_content

    def _inject_charts_into_content(self, content_html: str, chart_blocks_by_key: dict[str, str]) -> str:
        """把图表块插入到正文 HTML 内（仅按锚点插入）。

        约束：只替换正文中显式出现的 `[Chart: <id>]` 锚点，禁止“启发式”插入，
        以避免图表重复/错位。
        """
        if not chart_blocks_by_key:
            return content_html
        html_s = content_html or ""
        used: set[str] = set()

        # 1) 锚点独占段落：<p>[Chart: id]</p> 直接替换为图表块
        anchor_only_p = re.compile(r"<p>\s*\[Chart:\s*([^\]]+)\s*\]\s*</p>")

        def _replace_anchor_only_p(m: re.Match[str]) -> str:
            cid = (m.group(1) or "").strip()
            blk = chart_blocks_by_key.get(cid)
            if not blk:
                return m.group(0)
            used.add(cid)
            return blk

        html_s = anchor_only_p.sub(_replace_anchor_only_p, html_s)

        # 2) 锚点位于段落末尾（推荐形态）：在该段落结束后插入图表块
        anchor_re = re.compile(r"\[Chart:\s*([^\]]+?)\s*\]")
        p_re = re.compile(r"<p>(.*?)</p>", flags=re.DOTALL)

        def _process_p(m: re.Match[str]) -> str:
            inner = m.group(1) or ""
            ids = [x.strip() for x in anchor_re.findall(inner) if str(x or "").strip()]
            if not ids:
                return m.group(0)
            blocks: list[str] = []
            # 移除锚点文本
            new_inner = anchor_re.sub("", inner)
            new_inner = new_inner.strip()
            for cid in ids:
                blk = chart_blocks_by_key.get(cid)
                if not blk:
                    continue
                if cid in used:
                    continue
                used.add(cid)
                blocks.append(blk)
            if not blocks:
                return m.group(0)
            out = []
            if new_inner:
                out.append(f"<p>{new_inner}</p>")
            out.extend(blocks)
            return "\n".join(out)

        html_s = p_re.sub(_process_p, html_s)

        # 不再启发式插入 remaining charts（避免错位/重复）
        return html_s

    def _get_default_styles(self) -> str:
        return """
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .document-title {
                font-size: 28px;
                margin-bottom: 24px;
                padding-bottom: 12px;
                border-bottom: 2px solid #eee;
                color: #111;
            }
            .section {
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }
            .section:last-child {
                border-bottom: none;
            }
            .section-title {
                font-size: 24px;
                color: #1a1a1a;
                margin-bottom: 16px;
                padding-bottom: 8px;
                border-bottom: 2px solid #5470c6;
            }
            .section-content {
                font-size: 16px;
                color: #333;
            }
            .section-content h3 {
                font-size: 20px;
                margin-top: 24px;
                margin-bottom: 12px;
                color: #2c3e50;
            }
            .section-content h4 {
                font-size: 18px;
                margin-top: 20px;
                margin-bottom: 10px;
                color: #2c3e50;
            }
            .section-content h5 {
                font-size: 16px;
                margin-top: 16px;
                margin-bottom: 8px;
                color: #2c3e50;
            }
            .section-content h6 {
                font-size: 16px;
                margin-top: 16px;
                margin-bottom: 8px;
                color: #2c3e50;
            }
            .section-content p {
                margin-bottom: 16px;
                text-indent: 2em;
                text-align: justify;
                text-justify: inter-ideograph;
            }
            .section-content li {
                margin-left: 24px;
                margin-bottom: 8px;
            }
            .section-content li p {
                /* 列表项中的段落不做首行缩进，避免双重缩进 */
                text-indent: 0;
            }
            .section-content .md-table {
                width: 100%;
                border-collapse: collapse;
                margin: 16px 0;
                font-size: 14px;
                background: #fff;
            }
            .section-content .md-table th,
            .section-content .md-table td {
                border: 1px solid #e5e7eb;
                padding: 8px 10px;
                vertical-align: top;
            }
            .section-content .md-table th {
                background: #eef2ff;
                color: #1f2937;
                font-weight: 600;
            }
            .section-content .md-table td {
                color: #111827;
            }
            .chart-container {
                margin: 24px 0;
                padding: 16px;
                background-color: #fafafa;
                border-radius: 8px;
            }
            .sources {
                margin-top: 24px;
                padding-top: 16px;
                border-top: 1px dashed #ddd;
                font-size: 14px;
                color: #666;
            }
            .sources h4 {
                margin-bottom: 12px;
                color: #888;
            }
            .sources ol {
                margin-left: 20px;
            }
        """

