"""内容生成服务（T042）。

按模板 section 生成内容，输出单 HTML。
整合 RAG 检索（混合检索 + Rerank）和图表渲染能力。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from src.application.repositories.template_repository import TemplateRepository
from src.application.schemas.chart_spec import (
    ChartConfig,
    ChartRenderRequest,
    ChartTemplateSnippet,
)
from src.application.schemas.ingest import HybridSearchOptions, SearchResult
from src.application.schemas.template import PreprocessResponse
from src.application.services.chart_renderer_service import ChartRendererService
from src.application.services.hybrid_search_service import (
    HybridSearchService,
    HybridSearchServiceError,
)
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.application.services.template_service import TemplateService
from src.domain.entities.template import Template, TemplateType
from src.shared.errors import AppError
from src.shared.constants.prompts import (
    SCOPE_CONTENT_GENERATION_SECTION,
    SCOPE_OUTLINE_POLISH,
)


class ContentGenerationError(AppError):
    """内容生成错误。"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "content_generation_error",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
        )


@dataclass
class TemplateSection:
    """模板 section（章节）。"""

    section_id: str
    title: str
    content: str
    placeholders: list[str]
    chart_placeholders: list[str]
    order: int


@dataclass
class SectionGenerationResult:
    """章节生成结果。"""

    section_id: str
    title: str
    content: str
    rendered_charts: dict[str, ChartTemplateSnippet]
    sources: list[str]
    tokens_used: int
    generation_time_ms: float


@dataclass
class ContentGenerationResult:
    """内容生成最终结果。"""

    template_id: str
    template_name: str
    html_content: str
    sections: list[SectionGenerationResult]
    total_tokens: int
    total_time_ms: float
    generated_at: datetime


class ContentGenerationService:
    """内容生成服务。"""

    # Markdown 占位符正则
    PLACEHOLDER_PATTERN = re.compile(r"\{\{(\w+)\}\}")

    # 图表占位符正则
    CHART_PLACEHOLDER_PATTERN = re.compile(r"\{\{chart:(\w+)\}\}")

    # 默认检索选项
    DEFAULT_TOP_K = 10
    DEFAULT_RERANK_TOP = 5

    def __init__(
        self,
        template_service: TemplateService,
        hybrid_search_service: HybridSearchService,
        llm_runtime_service: LLMRuntimeService,
        chart_renderer_service: ChartRendererService,
        template_repository: TemplateRepository,
    ):
        self.template_service = template_service
        self.hybrid_search_service = hybrid_search_service
        self.llm_runtime_service = llm_runtime_service
        self.chart_renderer_service = chart_renderer_service
        self.template_repository = template_repository

    def parse_template_sections(self, template: Template) -> list[TemplateSection]:
        """解析模板为多个 section。"""
        storage_path = Path("data") / template.storage_path

        if not storage_path.exists():
            raise ContentGenerationError(
                message="模板文件不存在",
                code="template_not_found",
                details={"storage_path": str(storage_path)},
            )

        content = storage_path.read_text(encoding="utf-8")

        if template.type == TemplateType.CUSTOM:
            return self._parse_markdown_sections(content, template.id)
        else:
            # Office 文档暂时不支持 section 解析
            return [
                TemplateSection(
                    section_id="main",
                    title=template.original_filename,
                    content=content,
                    placeholders=self._extract_placeholders(content),
                    chart_placeholders=[],
                    order=0,
                )
            ]

    def _parse_markdown_sections(self, content: str, template_id: str) -> list[TemplateSection]:
        """解析 Markdown 模板为章节。"""
        sections = []
        current_order = 0

        # 按 # 标题分割
        lines = content.split("\n")
        current_section_content = []
        current_title = None

        for line in lines:
            if line.startswith("## "):
                # 保存上一个 section
                if current_title is not None:
                    full_content = "\n".join(current_section_content)
                    sections.append(
                        TemplateSection(
                            section_id=f"{template_id}-section-{current_order}",
                            title=current_title,
                            content=full_content,
                            placeholders=self._extract_placeholders(full_content),
                            chart_placeholders=self._extract_chart_placeholders(full_content),
                            order=current_order,
                        )
                    )
                    current_order += 1

                current_title = line[3:].strip()
                current_section_content = []
            elif line.startswith("# ") and current_title is None:
                # 文档标题，不作为 section
                pass
            else:
                if current_title is not None:
                    current_section_content.append(line)

        # 保存最后一个 section
        if current_title is not None:
            full_content = "\n".join(current_section_content)
            sections.append(
                TemplateSection(
                    section_id=f"{template_id}-section-{current_order}",
                    title=current_title,
                    content=full_content,
                    placeholders=self._extract_placeholders(full_content),
                    chart_placeholders=self._extract_chart_placeholders(full_content),
                    order=current_order,
                )
            )

        return sections

    def _extract_placeholders(self, content: str) -> list[str]:
        """提取占位符。"""
        return list(set(self.PLACEHOLDER_PATTERN.findall(content)))

    def _extract_chart_placeholders(self, content: str) -> list[str]:
        """提取图表占位符。"""
        return list(set(self.CHART_PLACEHOLDER_PATTERN.findall(content)))

    def generate_section_content(
        self,
        section: TemplateSection,
        context: str,
        chart_configs: dict[str, ChartConfig],
        collection_name: str = "default",
    ) -> SectionGenerationResult:
        """生成单个 section 的内容。"""
        import time

        start_time = time.time()

        # 替换文本占位符
        generated_content = section.content
        tokens_used = 0

        # 准备上下文
        truncated_context = context[:15000] if len(context) > 15000 else context

        # 调用 LLM 生成内容
        try:
            runnable = self.llm_runtime_service.build_runnable_for_callsite(
                SCOPE_CONTENT_GENERATION_SECTION
            )
            result = runnable.invoke(
                {
                    "title": section.title,
                    "template_content": section.content,
                    "context": truncated_context,
                }
            )
            generated_content = result if result else section.content
            # 估算 token 使用量
            tokens_used = len(generated_content) // 4
        except Exception as e:
            raise ContentGenerationError(
                message=f"LLM 调用失败: {str(e)}",
                code="llm_invocation_failed",
                details={"section_id": section.section_id},
            )

        # 渲染图表
        rendered_charts: dict[str, ChartTemplateSnippet] = {}
        for chart_name in section.chart_placeholders:
            if chart_name in chart_configs:
                try:
                    snippet = self.chart_renderer_service.render_template_snippet(
                        chart_configs[chart_name],
                        library="echarts",
                    )
                    rendered_charts[chart_name] = snippet
                except Exception as e:
                    # 图表渲染失败不影响整体生成
                    pass

        # 提取来源
        sources = self._extract_sources_from_context(context)

        generation_time_ms = (time.time() - start_time) * 1000

        return SectionGenerationResult(
            section_id=section.section_id,
            title=section.title,
            content=generated_content,
            rendered_charts=rendered_charts,
            sources=sources,
            tokens_used=tokens_used,
            generation_time_ms=generation_time_ms,
        )

    def _extract_sources_from_context(self, context: str) -> list[str]:
        """从上下文中提取来源信息。"""
        sources = []
        # 简单的来源提取，实际可能需要更复杂的处理
        if "来源:" in context or "source:" in context.lower():
            # 提取来源行
            pass
        return sources

    def assemble_context_for_section(
        self,
        section: TemplateSection,
        collection_name: str = "default",
        options: HybridSearchOptions | None = None,
    ) -> str:
        """为指定 section 组装 RAG 检索上下文。"""
        options = options or HybridSearchOptions(
            top_k=self.DEFAULT_TOP_K,
            use_rerank=True,
            rerank_top_n=self.DEFAULT_RERANK_TOP,
        )

        # 对每个占位符进行检索
        all_results: list[SearchResult] = []

        for placeholder in section.placeholders:
            try:
                search_result = self.hybrid_search_service.search(
                    query=f"{section.title} {placeholder}",
                    collection_name=collection_name,
                    options=options,
                )
                all_results.extend(search_result.results)
            except HybridSearchServiceError:
                # 检索失败继续处理其他占位符
                continue

        # 按分数排序并去重
        seen_contents = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.score, reverse=True):
            if result.content not in seen_contents:
                seen_contents.add(result.content)
                unique_results.append(result)

        # 组装上下文
        context_parts = []
        for idx, result in enumerate(unique_results[:20], 1):
            source_info = f"[{idx}]"
            if result.metadata.get("source_file_id"):
                source_info = f"[来源: {result.metadata['source_file_id']}]"
            context_parts.append(f"{source_info}\n{result.content}\n")

        return "\n".join(context_parts)

    def render_final_html(
        self,
        template: Template,
        section_results: list[SectionGenerationResult],
    ) -> str:
        """渲染最终 HTML。"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='zh-CN'>",
            "<head>",
            f"<meta charset='UTF-8'>",
            f"<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{template.original_filename}</title>",
            "<style>",
            self._get_default_styles(),
            "</style>",
            "</head>",
            "<body>",
            "<div class='container'>",
        ]

        # 渲染每个 section
        for section in section_results:
            html_parts.extend(self._render_section_html(section))

        html_parts.extend(["</div>", "</body>", "</html>"])

        return "\n".join(html_parts)

    def _render_section_html(self, section: SectionGenerationResult) -> list[str]:
        """渲染单个 section 为 HTML。"""
        parts = [
            f"<section id='{section.section_id}' class='section'>",
            f"<h2 class='section-title'>{section.title}</h2>",
        ]

        # 渲染内容（Markdown 转 HTML）
        content_html = self._markdown_to_html(section.content)
        parts.append(f"<div class='section-content'>{content_html}</div>")

        # 渲染图表
        for chart_id, chart in section.rendered_charts.items():
            parts.append(f"<div class='chart-container' id='{chart_id}'>")
            parts.append(chart.container_html)
            parts.append(chart.script_html)
            parts.append("</div>")

        # 渲染来源
        if section.sources:
            parts.append("<div class='sources'>")
            parts.append("<h4>参考来源</h4>")
            parts.append("<ol>")
            for source in section.sources:
                parts.append(f"<li>{source}</li>")
            parts.append("</ol>")
            parts.append("</div>")

        parts.append("</section>")

        return parts

    def _markdown_to_html(self, markdown: str) -> str:
        """简单的 Markdown 转 HTML。"""
        import html

        html_content = html.escape(markdown)

        # 标题
        html_content = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html_content, flags=re.MULTILINE)

        # 粗体
        html_content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_content)
        html_content = re.sub(r"__(.+?)__", r"<strong>\1</strong>", html_content)

        # 斜体
        html_content = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html_content)
        html_content = re.sub(r"_(.+?)_", r"<em>\1</em>", html_content)

        # 列表
        html_content = re.sub(r"^- (.+)$", r"<li>\1</li>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^(\d+)\. (.+)$", r"<li>\2</li>", html_content, flags=re.MULTILINE)

        # 段落
        html_content = re.sub(r"\n\n", r"</p><p>", html_content)
        html_content = f"<p>{html_content}</p>"

        # 清理空段落
        html_content = re.sub(r"<p>\s*</p>", "", html_content)
        html_content = re.sub(r"<p>(<h[1-6]>.*</h[1-6]>)</p>", r"\1", html_content)

        return html_content

    def _get_default_styles(self) -> str:
        """获取默认样式。"""
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
            .section-content p {
                margin-bottom: 16px;
            }
            .section-content li {
                margin-left: 24px;
                margin-bottom: 8px;
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

    def generate_content(
        self,
        template_id: str,
        collection_name: str = "default",
        chart_configs: dict[str, ChartConfig] | None = None,
        search_options: HybridSearchOptions | None = None,
    ) -> ContentGenerationResult:
        """生成完整内容。"""
        import time

        start_time = time.time()

        # 获取模板
        template = self.template_service.get_template(template_id)
        if template is None:
            raise ContentGenerationError(
                message="模板不存在",
                code="template_not_found",
                details={"template_id": template_id},
            )

        # 预处理校验
        preprocess_result = self.template_service.preprocess_template(template_id)
        if preprocess_result["overall_status"] == "error":
            raise ContentGenerationError(
                message=f"模板预处理失败: {preprocess_result['message']}",
                code="template_preprocess_failed",
                details=preprocess_result,
            )

        # 解析 sections
        sections = self.parse_template_sections(template)

        # 生成每个 section
        section_results: list[SectionGenerationResult] = []
        total_tokens = 0

        chart_configs = chart_configs or {}

        for section in sections:
            # 检索上下文
            context = self.assemble_context_for_section(
                section, collection_name, search_options
            )

            # 生成内容
            result = self.generate_section_content(
                section, context, chart_configs, collection_name
            )
            section_results.append(result)
            total_tokens += result.tokens_used

        # 渲染最终 HTML
        html_content = self.render_final_html(template, section_results)

        total_time_ms = (time.time() - start_time) * 1000

        return ContentGenerationResult(
            template_id=template_id,
            template_name=template.original_filename,
            html_content=html_content,
            sections=section_results,
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
            generated_at=datetime.now(),
        )

    def polish_outline(self, outline: str) -> str:
        """对用户自定义大纲进行 LLM 润色。

        Args:
            outline: 原始大纲（Markdown 格式）

        Returns:
            润色后的大纲（Markdown 格式）

        Raises:
            ContentGenerationError: 当 LLM 调用失败时
        """
        if not outline or not outline.strip():
            raise ContentGenerationError(
                message="大纲内容不能为空",
                code="outline_empty",
                status_code=400,
            )

        try:
            runnable = self.llm_runtime_service.build_runnable_for_callsite(
                SCOPE_OUTLINE_POLISH
            )
            result = runnable.invoke({"outline": outline})
            polished_outline = result if result else outline
            return polished_outline.strip()
        except Exception as e:
            raise ContentGenerationError(
                message=f"LLM 润色失败: {str(e)}",
                code="outline_polish_failed",
                details={"original_outline": outline[:200]},  # 只记录前200字符
            )
