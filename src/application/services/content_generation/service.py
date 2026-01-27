"""内容生成服务实现（T042/T096）。

说明：
- 该文件承载 ContentGenerationService 的主编排逻辑
- 具体实现拆分到同目录下的模块（text/chart/outline/llm），降低耦合与单文件体积
"""

from __future__ import annotations

import asyncio
import inspect
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.application.repositories.template_repository import TemplateRepository
from src.application.schemas.chart_spec import ChartConfig, ChartTemplateSnippet
from src.application.schemas.ingest import HybridSearchOptions, SearchResult
from src.application.schemas.template import PreprocessResponse
from src.application.services.chart_renderer_service import ChartRendererService
from src.application.services.content_generation.auto_chart_resolver import AutoChartResolver
from src.application.services.content_generation.chart_json_converter import ChartJsonConverter
from src.application.services.content_generation.html_renderer import WhitepaperHtmlRenderer
from src.application.services.content_generation.llm_section import (
    GLOSSARY_PLACEHOLDER_RE,
    SectionLLMGenerator,
)
from src.application.services.content_generation.outline_utils import (
    build_outline_skeleton_markdown,
    parse_outline_items_from_section_content,
)
from src.application.services.content_generation.text_utils import UUID_RE
from src.application.services.content_generation.types import (
    ContentGenerationResult,
    OutlineItem,
    SectionGenerationResult,
    TemplateSection,
)
from src.application.services.hybrid_search_service import HybridSearchService
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.application.services.template_service import TemplateService
from src.domain.entities.template import Template
from src.shared.constants.prompts import SCOPE_CONTENT_GENERATION_SECTION, SCOPE_OUTLINE_POLISH
from src.shared.logging import get_logger, log_extra

from .errors import ContentGenerationError

log = get_logger(__name__)


def _extract_md_title(md_text: str) -> str | None:
    if not md_text:
        return None
    for line in md_text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip() or None
        if s:
            break
    return None


class ContentGenerationService:
    """内容生成服务。"""

    # 默认检索选项
    DEFAULT_TOP_K = 10
    DEFAULT_RERANK_TOP = 5
    DEFAULT_PER_QUERY_TOP_N = 3
    DEFAULT_MAX_CONTEXT_CHARS = 15000
    DEFAULT_MAX_AUTO_CHARTS_PER_SECTION = 8

    _MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")

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

        self._chart_converter = ChartJsonConverter()
        self._auto_chart_resolver = AutoChartResolver(converter=self._chart_converter)
        self._section_llm = SectionLLMGenerator(
            llm_runtime_service=self.llm_runtime_service,
            callsite_scope=SCOPE_CONTENT_GENERATION_SECTION,
        )

    async def _emit(
        self,
        on_event: Callable[[dict[str, Any]], Any] | None,
        payload: dict[str, Any],
    ) -> None:
        if on_event is None:
            return
        try:
            r = on_event(payload)
            if inspect.isawaitable(r):
                await r  # type: ignore[misc]
        except Exception:
            return

    # -------------------------
    # Template parsing
    # -------------------------

    def parse_template_sections(self, template: Template, *, content_override: str | None = None) -> list[TemplateSection]:
        """解析模板为多个 section。"""
        if content_override is None:
            content = template.content or ""
        else:
            content = content_override

        return self._parse_markdown_sections(content, template.id)

    def _parse_markdown_sections(self, content: str, template_id: str) -> list[TemplateSection]:
        """解析 Markdown 内容为多个 section。"""
        lines = (content or "").splitlines()
        sections: list[TemplateSection] = []
        current_title: str | None = None
        current_section_content: list[str] = []
        current_order = 0

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
                            order=current_order,
                        )
                    )
                    current_order += 1

                current_title = line[3:].strip()
                current_section_content = []
            elif line.startswith("# ") and current_title is None:
                # 文档标题，不作为 section
                continue
            else:
                if current_title is not None:
                    current_section_content.append(line)

        if current_title is not None:
            full_content = "\n".join(current_section_content)
            sections.append(
                TemplateSection(
                    section_id=f"{template_id}-section-{current_order}",
                    title=current_title,
                    content=full_content,
                    order=current_order,
                )
            )

        return sections

    # -------------------------
    # RAG context assembly
    # -------------------------

    async def assemble_context_for_section(
        self,
        section: TemplateSection,
        collection_name: str = "default",
        options: HybridSearchOptions | None = None,
        *,
        coverage_score_threshold: float | None = None,
        kb_input_root: Path | None = None,
        max_auto_charts_per_section: int | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[
        str,
        list[dict[str, Any]],
        str | None,
        list[OutlineItem],
        list[str],
        list[ChartConfig],
        dict[str, str],
    ]:
        """为指定 section 组装 RAG 检索上下文，并返回覆盖评估。"""
        options = options or HybridSearchOptions(
            top_k=self.DEFAULT_TOP_K,
            use_rerank=True,
            rerank_top_n=self.DEFAULT_RERANK_TOP,
        )

        coverage: list[dict[str, Any]] = []
        sources_set: set[str] = set()
        item_context_map: dict[str, str] = {}
        max_auto = max_auto_charts_per_section or self.DEFAULT_MAX_AUTO_CHARTS_PER_SECTION
        seen_chart_json: set[str] = set()

        outline_items = parse_outline_items_from_section_content(section.content)
        template_skeleton: str | None = None
        if outline_items:
            template_skeleton = build_outline_skeleton_markdown(outline_items)
            log.info(
                "content_generation.section_outline_parsed",
                extra=log_extra(
                    section_id=section.section_id,
                    section_title=section.title,
                    outline_items=len(outline_items),
                    collection_name=collection_name,
                ),
            )

            def _is_bad_title(t: str) -> bool:
                s = (t or "").strip()
                if not s:
                    return True
                if UUID_RE.match(s):
                    return True
                if re.fullmatch(r"\d{4}", s) or re.fullmatch(r"\d+", s):
                    return True
                if s in {"证券研究报告", "研究报告", "年度报告", "报告"}:
                    return True
                if len(s) < 4:
                    return True
                return False

            async def _search_one(item: OutlineItem):
                base = f"{section.title} {item.title}".strip()
                if section.title.strip() == "附录":
                    base = item.title.strip() or item.display.strip()
                if "名词解释" in base:
                    base = f"{base} 术语 定义".strip()
                q = base
                try:
                    resp = await self.hybrid_search_service.search(
                        query=q,
                        collection_name=collection_name,
                        options=options,
                    )
                    return item, q, resp
                except Exception as exc:  # noqa: BLE001
                    return item, q, exc

            sem = asyncio.Semaphore(6)

            async def _guarded(it: OutlineItem):
                async with sem:
                    return await _search_one(it)

            tasks = [asyncio.create_task(_guarded(it)) for it in outline_items]
            results: list[tuple[OutlineItem, str, Any]] = []
            done = 0
            total = len(tasks)
            for fut in asyncio.as_completed(tasks):
                results.append(await fut)
                done += 1
                if done == total or (done % 10) == 0:
                    log.info(
                        "content_generation.section_recall_progress",
                        extra=log_extra(
                            section_id=section.section_id,
                            section_title=section.title,
                            done=done,
                            total=total,
                            collection_name=collection_name,
                        ),
                    )
                    await self._emit(
                        on_event,
                        {
                            "type": "section_recall_progress",
                            "section_id": section.section_id,
                            "section_title": section.title,
                            "done": done,
                            "total": total,
                        },
                    )

            seen_chunk_ids: set[str] = set()
            context_parts: list[str] = []
            chart_configs: list[ChartConfig] = []

            for item, q, resp in results:
                if isinstance(resp, Exception):
                    coverage.append(
                        {
                            "type": "outline_item",
                            "item": item.display,
                            "depth": item.depth,
                            "query": q,
                            "hits": 0,
                            "max_score": None,
                            "uncovered": True,
                            "error": {"type": resp.__class__.__name__, "message": str(resp)},
                        }
                    )
                    item_context_map[item.display] = ""
                    continue

                hits = resp.results or []
                try:
                    max_score = max((r.score for r in hits), default=None)
                except Exception:
                    max_score = None
                uncovered = not bool(hits)
                if coverage_score_threshold is not None and max_score is not None:
                    if float(max_score) < float(coverage_score_threshold):
                        uncovered = True

                coverage.append(
                    {
                        "type": "outline_item",
                        "item": item.display,
                        "depth": item.depth,
                        "query": q,
                        "hits": len(hits),
                        "max_score": max_score,
                        "uncovered": uncovered,
                        "bm25_index_used": getattr(resp.metrics, "bm25_index_used", None),
                        "rerank_applied": getattr(resp.metrics, "rerank_applied", None),
                    }
                )

                per_top = min(self.DEFAULT_PER_QUERY_TOP_N, len(hits))
                if per_top <= 0:
                    item_context_map[item.display] = ""
                    continue

                block_lines = [f"###{'#' * max(0, item.depth - 1)} {item.display}".strip()]

                picked: list[SearchResult] = []
                seen_docs: set[str] = set()
                for r in hits:
                    doc_rel = r.metadata.get("doc_rel_path") if isinstance(r.metadata, dict) else None
                    src_id = r.metadata.get("source_file_id") if isinstance(r.metadata, dict) else None
                    key = str(doc_rel or src_id or r.chunk_id or "").strip()
                    if key and key in seen_docs:
                        continue
                    if key:
                        seen_docs.add(key)
                    picked.append(r)
                    if len(picked) >= per_top:
                        break

                # auto chart extract（对 picked hits 做一次抽取）
                if kb_input_root is not None and len(chart_configs) < max_auto:
                    auto_cfgs = self._auto_chart_resolver.extract_auto_chart_configs(
                        hits=picked,
                        kb_input_root=kb_input_root,
                        md_image_re=self._MD_IMAGE_RE,
                        max_auto=max_auto - len(chart_configs),
                        seen_chart_json=seen_chart_json,
                        sources_set=sources_set,
                        hint_text=q,
                        audit=coverage,
                    )
                    chart_configs.extend(auto_cfgs)

                for r in picked:
                    if r.chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(r.chunk_id)

                    doc_title = r.metadata.get("doc_title") if isinstance(r.metadata, dict) else None
                    doc_rel = r.metadata.get("doc_rel_path") if isinstance(r.metadata, dict) else None
                    ofn = r.metadata.get("original_filename") if isinstance(r.metadata, dict) else None

                    dt_key = str(doc_title or "").strip()
                    ofn_key = str(ofn or "").strip()
                    dr_key = str(doc_rel or "").strip()

                    display = ""
                    if not _is_bad_title(dt_key):
                        display = dt_key
                    elif ofn_key:
                        display = Path(ofn_key).stem
                    elif dr_key:
                        display = Path(dr_key).name
                    display = re.sub(r"\s+", " ", (display or "").strip())

                    header = f"【来源】{display}" if display else "【来源】"
                    text = (r.content or "").strip()
                    if len(text) > 1200:
                        text = text[:1200] + "…"
                    block_lines.append(header)
                    block_lines.append(text)
                    block_lines.append("")

                    if display:
                        sources_set.add(display)

                block = "\n".join(block_lines).strip()
                item_context_map[item.display] = block
                context_parts.append(block)

            context = "\n\n".join([p for p in context_parts if p]).strip()
            if len(context) > self.DEFAULT_MAX_CONTEXT_CHARS:
                context = context[: self.DEFAULT_MAX_CONTEXT_CHARS]

            uncovered_count = sum(1 for c in coverage if c.get("type") == "outline_item" and c.get("uncovered") is True)
            log.info(
                "content_generation.section_context_ready",
                extra=log_extra(
                    section_id=section.section_id,
                    section_title=section.title,
                    context_chars=len(context),
                    coverage_items=len(coverage),
                    uncovered_items=uncovered_count,
                ),
            )
            await self._emit(
                on_event,
                {
                    "type": "section_context_ready",
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "context_chars": len(context),
                    "coverage_items": len(coverage),
                },
            )
            if chart_configs:
                coverage.append({"type": "auto_charts", "count": len(chart_configs)})

            return (
                context,
                coverage,
                template_skeleton,
                outline_items,
                sorted(sources_set),
                chart_configs,
                item_context_map,
            )

        # 无大纲条目：仍尝试用章节标题做一次检索；无知识库时会降级为空上下文
        resp = None
        try:
            resp = await self.hybrid_search_service.search(
                query=section.title,
                collection_name=collection_name,
                options=options,
            )
            hits = resp.results or []
        except Exception as exc:  # noqa: BLE001
            hits = []
            coverage.append(
                {
                    "type": "section_title",
                    "query": section.title,
                    "hits": 0,
                    "max_score": None,
                    "uncovered": True,
                    "error": {"type": exc.__class__.__name__, "message": str(exc)},
                }
            )

        def _is_bad_title(t: str) -> bool:
            s = (t or "").strip()
            if not s:
                return True
            if UUID_RE.match(s):
                return True
            if re.fullmatch(r"\d{4}", s) or re.fullmatch(r"\d+", s):
                return True
            if s in {"证券研究报告", "研究报告", "年度报告", "报告"}:
                return True
            if len(s) < 4:
                return True
            return False

        context_parts: list[str] = []
        chart_configs: list[ChartConfig] = []
        if kb_input_root is not None and hits:
            chart_configs = self._auto_chart_resolver.extract_auto_chart_configs(
                hits=hits[: min(3, len(hits))],
                kb_input_root=kb_input_root,
                md_image_re=self._MD_IMAGE_RE,
                max_auto=max_auto,
                seen_chart_json=seen_chart_json,
                sources_set=sources_set,
                hint_text=section.title,
                audit=coverage,
            )

        for r in hits[: min(3, len(hits))]:
            doc_title = r.metadata.get("doc_title") if isinstance(r.metadata, dict) else None
            doc_rel = r.metadata.get("doc_rel_path") if isinstance(r.metadata, dict) else None
            ofn = r.metadata.get("original_filename") if isinstance(r.metadata, dict) else None

            dt_key = str(doc_title or "").strip()
            ofn_key = str(ofn or "").strip()
            dr_key = str(doc_rel or "").strip()

            display = ""
            if not _is_bad_title(dt_key):
                display = dt_key
            elif ofn_key:
                display = Path(ofn_key).stem
            elif dr_key:
                display = Path(dr_key).name
            display = re.sub(r"\s+", " ", (display or "").strip())
            if display:
                sources_set.add(display)

            txt = (r.content or "").strip()
            if len(txt) > 1600:
                txt = txt[:1600] + "…"
            if display:
                context_parts.append(f"【来源】{display}\n{txt}".strip())
            else:
                context_parts.append(txt)

        context = "\n\n".join([p for p in context_parts if p]).strip()
        if len(context) > self.DEFAULT_MAX_CONTEXT_CHARS:
            context = context[: self.DEFAULT_MAX_CONTEXT_CHARS]

        if hits:
            try:
                max_score = max((r.score for r in hits), default=None)
            except Exception:
                max_score = None
            coverage.append(
                {
                    "type": "section_title",
                    "query": section.title,
                    "hits": len(hits),
                    "max_score": max_score,
                    "uncovered": False if max_score is not None else not bool(hits),
                    "bm25_index_used": getattr(getattr(resp, "metrics", None), "bm25_index_used", None),
                    "rerank_applied": getattr(getattr(resp, "metrics", None), "rerank_applied", None),
                }
            )
        if chart_configs:
            coverage.append({"type": "auto_charts", "count": len(chart_configs)})

        log.info(
            "content_generation.section_context_ready",
            extra=log_extra(
                section_id=section.section_id,
                section_title=section.title,
                context_chars=len(context),
                coverage_items=len(coverage),
                uncovered_items=sum(1 for c in coverage if c.get("uncovered") is True),
                sources=len(sources_set),
                auto_charts=len(chart_configs),
            ),
        )
        await self._emit(
            on_event,
            {
                "type": "section_context_ready",
                "section_id": section.section_id,
                "section_title": section.title,
                "context_chars": len(context),
                "coverage_items": len(coverage),
            },
        )

        return context, coverage, None, [], sorted(sources_set), chart_configs, {}

    # -------------------------
    # Section generation
    # -------------------------

    async def _generate_section(
        self,
        *,
        section: TemplateSection,
        context: str,
        template_content_override: str | None,
        outline_items: list[OutlineItem],
        document_title: str | None,
        sources: list[str],
        auto_chart_configs: list[ChartConfig],
        coverage: list[dict[str, Any]] | None,
        on_event: Callable[[dict[str, Any]], Any] | None,
        stream_tokens: bool,
    ) -> SectionGenerationResult:
        import time

        start_time = time.time()
        truncated_context = context[: self.DEFAULT_MAX_CONTEXT_CHARS] if len(context) > self.DEFAULT_MAX_CONTEXT_CHARS else context

        payload = {
            "title": section.title,
            "template_content": template_content_override or section.content,
            "context": truncated_context,
            "document_title": document_title or "",
            "outline_items": "\n".join([it.display for it in (outline_items or [])]).strip(),
        }

        try:
            generated_content, tokens_used = await self._section_llm.generate(
                section_id=section.section_id,
                section_title=section.title,
                payload=payload,
                stream_tokens=bool(stream_tokens),
                on_event=on_event,
            )
        except Exception as e:  # noqa: BLE001
            raise ContentGenerationError(
                message=f"LLM 调用失败: {str(e)}",
                code="llm_invocation_failed",
                details={"section_id": section.section_id},
            )

        # 附录名词解释：避免占位符模板内容污染最终产物（无召回时宁可留空）
        if section.title.strip() == "附录" and "名词解释" in generated_content:
            if GLOSSARY_PLACEHOLDER_RE.search(generated_content):
                lines = generated_content.replace("\r\n", "\n").split("\n")
                out: list[str] = []
                in_glossary = False
                for line in lines:
                    if re.match(r"^#{1,6}\s*C\.\s*名词解释\s*$", line.strip()):
                        in_glossary = True
                        out.append(line.rstrip())
                        continue
                    if in_glossary:
                        continue
                    out.append(line.rstrip())
                generated_content = "\n".join([x for x in out]).strip()

        # 兜底：严格按骨架时补齐缺失标题（仅在确实有召回上下文时才追加）
        if template_content_override:
            required_items: list[str] = []
            for line in template_content_override.replace("\r\n", "\n").split("\n"):
                s = line.strip()
                if s.startswith("- "):
                    required_items.append(line.rstrip("\n"))

            def _has_item(item_line: str) -> bool:
                plain = (item_line or "").strip()
                if plain.startswith("- "):
                    plain = plain[2:].strip()
                if not plain:
                    return False
                pat = re.compile(rf"(?m)^(?:[-*]\s+|#{1,6}\s+)?{re.escape(plain)}\s*$")
                return pat.search(generated_content) is not None

            missing = [h for h in required_items if h and not _has_item(h)]
            if missing:
                log.warning(
                    "content_generation.section_missing_headings",
                    extra=log_extra(
                        section_id=section.section_id,
                        section_title=section.title,
                        missing_count=len(missing),
                    ),
                )
                if truncated_context.strip():
                    parts = [generated_content.rstrip(), ""]
                    for h in missing:
                        parts.append(h)
                        parts.append("")
                    generated_content = "\n".join(parts).strip()

        rendered_charts: dict[str, ChartTemplateSnippet] = {}
        for idx, cfg in enumerate(auto_chart_configs or []):
            try:
                snippet = self.chart_renderer_service.render_template_snippet(cfg, library="echarts")
                rendered_charts[f"auto_chart_{idx+1}"] = snippet
            except Exception:
                continue

        generation_time_ms = (time.time() - start_time) * 1000
        return SectionGenerationResult(
            section_id=section.section_id,
            title=section.title,
            content=generated_content,
            rendered_charts=rendered_charts,
            sources=sources or [],
            coverage=coverage or [],
            tokens_used=tokens_used,
            generation_time_ms=generation_time_ms,
        )

    # -------------------------
    # Public APIs
    # -------------------------

    async def generate_content(
        self,
        *,
        template_id: str,
        collection_name: str = "default",
        chart_configs: dict[str, ChartConfig] | None = None,
        search_options: HybridSearchOptions | None = None,
        template_content_override: str | None = None,
        document_title: str | None = None,
        coverage_score_threshold: float | None = None,
        kb_input_root: Path | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
        stream_tokens: bool = False,
    ) -> ContentGenerationResult:
        """生成内容并输出单 HTML。"""
        import time

        start_time = time.time()
        chart_configs = chart_configs or {}
        search_options = search_options or HybridSearchOptions(
            top_k=self.DEFAULT_TOP_K,
            use_rerank=True,
            rerank_top_n=self.DEFAULT_RERANK_TOP,
        )

        template = self.template_service.get_template(template_id)  # type: ignore[attr-defined]
        if template is None:
            raise ContentGenerationError(
                message=f"模板不存在: {template_id}",
                status_code=404,
                code="template_not_found",
            )

        sections = self.parse_template_sections(template, content_override=template_content_override)
        if not sections:
            raise ContentGenerationError(
                message="模板未解析到任何章节（## ...）",
                status_code=400,
                code="template_sections_empty",
                details={"template_id": template_id},
            )

        # 文档标题：优先传入；否则从大纲 override 推断；再回退模板名
        doc_title = (document_title or "").strip()
        if not doc_title and template_content_override:
            doc_title = _extract_md_title(template_content_override) or ""
        if not doc_title:
            doc_title = template.original_filename

        section_results: list[SectionGenerationResult] = []
        total_tokens = 0

        for idx, sec in enumerate(sections, start=1):
            await self._emit(
                on_event,
                {
                    "type": "section_start",
                    "section_id": sec.section_id,
                    "section_title": sec.title,
                    "index": idx,
                    "total": len(sections),
                },
            )

            context, coverage, skeleton, outline_items, sources, auto_charts, _ = await self.assemble_context_for_section(
                sec,
                collection_name=collection_name,
                options=search_options,
                coverage_score_threshold=coverage_score_threshold,
                kb_input_root=kb_input_root,
                on_event=on_event,
            )

            # 若识别到大纲骨架，则用 skeleton 约束 LLM 输出结构
            template_override_for_section = skeleton if skeleton else None

            r = await self._generate_section(
                section=sec,
                context=context,
                template_content_override=template_override_for_section,
                outline_items=outline_items,
                document_title=doc_title,
                sources=sources,
                auto_chart_configs=auto_charts,
                coverage=coverage,
                on_event=on_event,
                stream_tokens=bool(stream_tokens),
            )
            section_results.append(r)
            total_tokens += r.tokens_used

            await self._emit(
                on_event,
                {
                    "type": "section_done",
                    "section_id": sec.section_id,
                    "section_title": sec.title,
                    "tokens_used": r.tokens_used,
                    "rendered_charts": list(r.rendered_charts.keys()),
                },
            )

        html = WhitepaperHtmlRenderer().render(
            template=template,
            section_results=section_results,
            document_title=doc_title,
        )

        total_time_ms = (time.time() - start_time) * 1000
        return ContentGenerationResult(
            template_id=template.id,
            template_name=template.original_filename,
            html_content=html,
            sections=section_results,
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
            generated_at=datetime.now(),
            document_title=doc_title,
        )

    def preprocess(self, *, template_id: str) -> PreprocessResponse:
        """预处理：保留旧接口（兼容 generate.py 使用）。"""
        template = self.template_service.get_template(template_id)  # type: ignore[attr-defined]
        if template is None:
            raise ContentGenerationError(
                message=f"模板不存在: {template_id}",
                status_code=404,
                code="template_not_found",
            )
        sections = self.parse_template_sections(template)
        return PreprocessResponse(
            template_id=template_id,
            sections=[{"id": s.section_id, "title": s.title, "order": s.order} for s in sections],
        )

    def polish_outline(self, outline: str) -> str:
        """大纲润色（调用点：content_generation:polish_outline）。"""
        try:
            runnable = self.llm_runtime_service.build_runnable_for_callsite(SCOPE_OUTLINE_POLISH)
            out = runnable.invoke({"outline": outline or ""})
            return str(out or "").strip()
        except Exception as e:  # noqa: BLE001
            raise ContentGenerationError(
                message=f"大纲润色失败: {str(e)}",
                code="outline_polish_failed",
            )

