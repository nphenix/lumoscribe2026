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
        parts.append(f"<div class='section-content'>{content_html}</div>")

        # 图表（chart_renderer_service 产出 container_html/script_html）
        for chart_id, chart in rendered_charts.items():
            parts.append(f"<div class='chart-container' id='{_html.escape(str(chart_id))}'>")
            parts.append(getattr(chart, "container_html", "") or "")
            parts.append(getattr(chart, "script_html", "") or "")
            parts.append("</div>")

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
        html_content = _html.escape(markdown or "")

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

        return html_content

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

