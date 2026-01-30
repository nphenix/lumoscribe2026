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
from src.application.services.content_generation.section_structured import SectionStructuredGenerator
from src.application.services.content_generation.outline_utils import (
    build_outline_skeleton_markdown,
    parse_outline_items_from_section_content,
)
from src.application.services.content_generation.coverage_utils import (
    coverage_outline_item_error,
    coverage_outline_item_ok,
    coverage_section_title_error,
    coverage_section_title_ok,
)
from src.application.services.content_generation.chart_candidates import (
    build_chart_candidates_block,
    chart_configs_from_candidates,
    extract_chart_candidates_from_vector_results,
)
from src.application.services.content_generation.chart_anchor_postprocessor import place_chart_anchors
from src.application.services.content_generation.recall_utils import gather_with_concurrency
from src.application.services.content_generation.section_context import (
    build_outline_item_heading,
    build_source_block,
    choose_source_display,
    pick_unique_hits,
    truncate_chars,
)
from src.application.services.content_generation.text_utils import (
    UUID_RE,
    strip_leading_title_heading,
    strip_orphan_outline_list_items,
)
from src.application.services.content_generation.pipeline import (
    PipelineOptions,
    SectionGenerationPipeline,
)
from src.application.services.content_generation.types import (
    ContentGenerationResult,
    OutlineItem,
    SectionGenerationResult,
    TemplateSection,
)
from src.application.services.hybrid_search_service import HybridSearchService
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.application.services.outline_polish.outline_polish_service import OutlinePolishService
from src.application.services.outline_polish.schema import OutlinePolishInput
from src.application.services.template_service import TemplateService
from src.domain.entities.template import Template
from src.shared.config import get_settings
from src.shared.constants.prompts import (
    SCOPE_CONTENT_GENERATION_SECTION,
    SCOPE_CONTENT_GENERATION_SECTION_POLISH,
)
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
    DEFAULT_PER_QUERY_TOP_N = 5
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
        outline_polish_service: OutlinePolishService,
    ):
        self.template_service = template_service
        self.hybrid_search_service = hybrid_search_service
        self.llm_runtime_service = llm_runtime_service
        self.chart_renderer_service = chart_renderer_service
        self.template_repository = template_repository
        self._outline_polish_service = outline_polish_service

        self._chart_converter = ChartJsonConverter()
        self._auto_chart_resolver = AutoChartResolver(converter=self._chart_converter)
        self._section_llm = SectionLLMGenerator(
            llm_runtime_service=self.llm_runtime_service,
            callsite_scope=SCOPE_CONTENT_GENERATION_SECTION,
        )
        self._section_polisher = SectionLLMGenerator(
            llm_runtime_service=self.llm_runtime_service,
            callsite_scope=SCOPE_CONTENT_GENERATION_SECTION_POLISH,
        )
        self._section_structured = SectionStructuredGenerator(
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
        global_seen_chart_json: set[str] | None = None,
        global_seen_chart_ids: set[str] | None = None,
        exclude_chunk_ids: set[str] | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[
        str,
        list[dict[str, Any]],
        str | None,
        list[OutlineItem],
        list[str],
        list[ChartConfig],
        dict[str, str],
        list[str],
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
        seen_chart_json: set[str] = global_seen_chart_json if global_seen_chart_json is not None else set()
        seen_chart_ids: set[str] = global_seen_chart_ids if global_seen_chart_ids is not None else set()
        exclude_chunk_ids = exclude_chunk_ids or set()

        outline_items = parse_outline_items_from_section_content(section.content)
        if outline_items:
            return await self._assemble_context_with_outline(
                section=section,
                outline_items=outline_items,
                collection_name=collection_name,
                options=options,
                coverage_score_threshold=coverage_score_threshold,
                kb_input_root=kb_input_root,
                max_auto=max_auto,
                seen_chart_json=seen_chart_json,
                seen_chart_ids=seen_chart_ids,
                coverage=coverage,
                sources_set=sources_set,
                item_context_map=item_context_map,
                exclude_chunk_ids=exclude_chunk_ids,
                on_event=on_event,
            )

        return await self._assemble_context_without_outline(
            section=section,
            collection_name=collection_name,
            options=options,
            kb_input_root=kb_input_root,
            max_auto=max_auto,
            seen_chart_json=seen_chart_json,
            seen_chart_ids=seen_chart_ids,
            coverage=coverage,
            sources_set=sources_set,
            item_context_map=item_context_map,
            exclude_chunk_ids=exclude_chunk_ids,
            on_event=on_event,
        )

    async def _assemble_context_with_outline(
        self,
        *,
        section: TemplateSection,
        outline_items: list[OutlineItem],
        collection_name: str,
        options: HybridSearchOptions,
        coverage_score_threshold: float | None,
        kb_input_root: Path | None,
        max_auto: int,
        seen_chart_json: set[str],
        seen_chart_ids: set[str],
        coverage: list[dict[str, Any]],
        sources_set: set[str],
        item_context_map: dict[str, str],
        exclude_chunk_ids: set[str],
        on_event: Callable[[dict[str, Any]], Any] | None,
    ):
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

        async def _on_progress(done: int, total: int):
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

        results = await gather_with_concurrency(
            outline_items,
            worker=_search_one,
            concurrency=6,
            on_progress=_on_progress,
            progress_every=10,
        )

        seen_chunk_ids: set[str] = set()
        used_chunk_ids: list[str] = []
        context_parts: list[str] = []
        chart_configs: list[ChartConfig] = []

        for item, q, resp in results:
            if isinstance(resp, Exception):
                coverage.append(
                    coverage_outline_item_error(
                        item_display=item.display,
                        depth=item.depth,
                        query=q,
                        error=resp,
                    )
                )
                item_context_map[item.display] = ""
                continue

            hits = resp.results or []
            if exclude_chunk_ids:
                hits = [r for r in hits if r.chunk_id not in exclude_chunk_ids]
            try:
                max_score = max((r.score for r in hits), default=None)
            except Exception:
                max_score = None

            uncovered = not bool(hits)
            if coverage_score_threshold is not None and max_score is not None:
                if float(max_score) < float(coverage_score_threshold):
                    uncovered = True

            coverage.append(
                coverage_outline_item_ok(
                    item_display=item.display,
                    depth=item.depth,
                    query=q,
                    hits=len(hits),
                    max_score=max_score,
                    uncovered=uncovered,
                    bm25_index_used=getattr(resp.metrics, "bm25_index_used", None),
                    rerank_applied=getattr(resp.metrics, "rerank_applied", None),
                )
            )

            per_top = min(self.DEFAULT_PER_QUERY_TOP_N, len(hits))
            if per_top <= 0:
                item_context_map[item.display] = ""
                continue

            block_lines = [build_outline_item_heading(item_display=item.display, depth=item.depth)]
            picked = pick_unique_hits(hits, per_top=per_top)

            if kb_input_root is not None and len(chart_configs) < max_auto:
                picked_non_chart = [r for r in picked if (r.metadata or {}).get("chunk_type") != "chart"]
                auto_cfgs = self._auto_chart_resolver.extract_auto_chart_configs(
                    hits=picked_non_chart,
                    kb_input_root=kb_input_root,
                    md_image_re=self._MD_IMAGE_RE,
                    max_auto=max_auto - len(chart_configs),
                    seen_chart_json=seen_chart_json,
                    sources_set=sources_set,
                    hint_text=q,
                    audit=coverage,
                )
                if auto_cfgs and seen_chart_ids:
                    kept_cfgs: list[ChartConfig] = []
                    for cfg in auto_cfgs:
                        extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                        cid = str(extra.get("chart_anchor_id") or "").strip()
                        if cid and cid in seen_chart_ids:
                            continue
                        kept_cfgs.append(cfg)
                    auto_cfgs = kept_cfgs
                chart_configs.extend(auto_cfgs)

                # 若识别出了图表（cfg.extra.chart_anchor_id），优先把对应 chart 语义 chunk 放入上下文，
                # 让 LLM 能看到 `[Chart: <id>]` 并在正文引用处落锚点（段落末尾）。
                anchor_ids: list[str] = []
                seen_a: set[str] = set()
                for cfg in auto_cfgs or []:
                    extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                    cid = str(extra.get("chart_anchor_id") or "").strip()
                    if cid and cid not in seen_a:
                        seen_a.add(cid)
                        anchor_ids.append(cid)
                cid = ""
                if anchor_ids:
                    cid = anchor_ids[0]
                    if seen_chart_ids and cid in seen_chart_ids:
                        cid = ""
                if cid:
                    chart_node = next(
                        (
                            rr
                            for rr in hits
                            if (rr.metadata or {}).get("chunk_type") == "chart"
                            and str((rr.metadata or {}).get("chart_id") or "").strip() == cid
                        ),
                        None,
                    )
                    if chart_node is not None and chart_node not in picked:
                        picked.append(chart_node)

            for r in picked:
                if r.chunk_id in seen_chunk_ids:
                    continue
                if exclude_chunk_ids and r.chunk_id in exclude_chunk_ids:
                    continue
                seen_chunk_ids.add(r.chunk_id)
                used_chunk_ids.append(r.chunk_id)

                doc_title = r.metadata.get("doc_title") if isinstance(r.metadata, dict) else None
                doc_rel = r.metadata.get("doc_rel_path") if isinstance(r.metadata, dict) else None
                ofn = r.metadata.get("original_filename") if isinstance(r.metadata, dict) else None

                display = choose_source_display(
                    doc_title=doc_title,
                    original_filename=ofn,
                    doc_rel_path=doc_rel,
                )

                block_lines.append(build_source_block(source_display=display, content=(r.content or ""), max_chars=1200))
                block_lines.append("")

                if display:
                    sources_set.add(display)
                cs = (r.metadata or {}).get("citation_sources") if isinstance(r.metadata, dict) else None
                if isinstance(cs, list):
                    for it in cs[:5]:
                        if not isinstance(it, dict):
                            continue
                        d2 = choose_source_display(
                            doc_title=it.get("doc_title"),
                            original_filename=it.get("original_filename"),
                            doc_rel_path=it.get("doc_rel_path"),
                        )
                        if d2:
                            sources_set.add(d2)

            block = "\n".join(block_lines).strip()
            item_context_map[item.display] = block
            context_parts.append(block)

        context = "\n\n".join([p for p in context_parts if p]).strip()
        context = truncate_chars(context, max_chars=self.DEFAULT_MAX_CONTEXT_CHARS)

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
            used_chunk_ids,
        )

    def _outline_heading_prefix(self, depth: int) -> str:
        d = max(1, int(depth or 1))
        lvl = min(6, 2 + d)
        return "#" * lvl

    def _build_outline_query(self, *, section_title: str, item: OutlineItem, ancestors: list[OutlineItem]) -> str:
        parts = [section_title]
        for a in ancestors:
            if a and a.title:
                parts.append(a.title)
        if item and item.title:
            parts.append(item.title)
        base = " ".join([p.strip() for p in parts if str(p or "").strip()]).strip()
        if section_title.strip() == "附录":
            base = (item.title.strip() or item.display.strip()) if item else base
        if "名词解释" in base:
            base = f"{base} 术语 定义".strip()
        return base

    async def _generate_section_hierarchical(
        self,
        *,
        section: TemplateSection,
        collection_name: str,
        options: HybridSearchOptions,
        coverage_score_threshold: float | None,
        kb_input_root: Path | None,
        document_title: str,
        on_event: Callable[[dict[str, Any]], Any] | None,
        stream_tokens: bool,
        max_auto_charts_per_section: int | None,
    ) -> SectionGenerationResult:
        import time

        start_time = time.time()
        outline_items = parse_outline_items_from_section_content(section.content)
        if not outline_items:
            context, coverage, _skeleton, _items, sources, auto_charts, _item_ctx, _used_chunk_ids = await self.assemble_context_for_section(
                section,
                collection_name=collection_name,
                options=options,
                coverage_score_threshold=coverage_score_threshold,
                kb_input_root=kb_input_root,
                max_auto_charts_per_section=max_auto_charts_per_section,
                on_event=on_event,
            )
            r = await self._generate_section(
                section=section,
                context=context,
                template_content_override=None,
                outline_items=[],
                document_title=document_title,
                sources=sources,
                auto_chart_configs=auto_charts,
                coverage=coverage,
                on_event=on_event,
                stream_tokens=bool(stream_tokens),
            )
            return r

        log.info(
            "content_generation.section_hierarchical_start",
            extra=log_extra(
                section_id=section.section_id,
                section_title=section.title,
                outline_items=len(outline_items),
                collection_name=collection_name,
            ),
        )
        await self._emit(
            on_event,
            {
                "type": "section_hierarchical_start",
                "section_id": section.section_id,
                "section_title": section.title,
                "outline_items": len(outline_items),
            },
        )

        max_auto = max_auto_charts_per_section or self.DEFAULT_MAX_AUTO_CHARTS_PER_SECTION
        seen_chart_json: set[str] = set()
        sources_set: set[str] = set()
        coverage: list[dict[str, Any]] = []
        rendered_charts: dict[str, ChartTemplateSnippet] = {}
        global_auto_chart_idx = 0
        total_tokens = 0

        def _render_charts(cfgs: list[ChartConfig], *, section_key_prefix: str) -> None:
            for cfg in cfgs or []:
                chart_key = None
                extra = cfg.extra if isinstance(cfg.extra, dict) else None
                if isinstance(extra, dict):
                    chart_key = str(extra.get("chart_anchor_id") or "").strip() or None
                if not chart_key:
                    continue
                if chart_key in rendered_charts:
                    n = 2
                    while f"{chart_key}__{n}" in rendered_charts:
                        n += 1
                    chart_key = f"{chart_key}__{n}"
                rendered_charts[chart_key] = ChartTemplateSnippet(
                    chart_id=str(chart_key),
                    container_html="",
                    script_html="",
                )

        intro_text = ""
        try:
            resp = await self.hybrid_search_service.search(
                query=section.title,
                collection_name=collection_name,
                options=options,
            )
            hits = resp.results or []
        except Exception as exc:  # noqa: BLE001
            hits = []
            coverage.append(coverage_section_title_error(query=section.title, error=exc))
            resp = None

        if hits:
            try:
                max_score = max((r.score for r in hits), default=None)
            except Exception:
                max_score = None
            coverage.append(
                coverage_section_title_ok(
                    query=section.title,
                    hits=len(hits),
                    max_score=max_score,
                    uncovered=False if max_score is not None else not bool(hits),
                    bm25_index_used=getattr(getattr(resp, "metrics", None), "bm25_index_used", None),
                    rerank_applied=getattr(getattr(resp, "metrics", None), "rerank_applied", None),
                )
            )

        picked_intro = pick_unique_hits(hits, per_top=min(self.DEFAULT_PER_QUERY_TOP_N, len(hits)))
        if kb_input_root is not None and picked_intro and max_auto > 0:
            picked_intro_non_chart = [r for r in picked_intro if (r.metadata or {}).get("chunk_type") != "chart"]
            intro_cfgs = self._auto_chart_resolver.extract_auto_chart_configs(
                hits=picked_intro_non_chart,
                kb_input_root=kb_input_root,
                md_image_re=self._MD_IMAGE_RE,
                max_auto=max_auto,
                seen_chart_json=seen_chart_json,
                sources_set=sources_set,
                hint_text=section.title,
                audit=coverage,
            )
            if intro_cfgs:
                coverage.append({"type": "auto_charts", "count": len(intro_cfgs)})
            _render_charts(intro_cfgs, section_key_prefix=section.section_id)

        intro_context_parts: list[str] = []
        for r in picked_intro:
            doc_title = r.metadata.get("doc_title") if isinstance(r.metadata, dict) else None
            doc_rel = r.metadata.get("doc_rel_path") if isinstance(r.metadata, dict) else None
            ofn = r.metadata.get("original_filename") if isinstance(r.metadata, dict) else None
            display = choose_source_display(
                doc_title=doc_title,
                original_filename=ofn,
                doc_rel_path=doc_rel,
            )
            if display:
                sources_set.add(display)
            txt = truncate_chars((r.content or "").strip(), max_chars=1600)
            intro_context_parts.append((f"【来源】{display}\n{txt}".strip() if display else txt).strip())
        intro_context = truncate_chars("\n\n".join([p for p in intro_context_parts if p]).strip(), max_chars=self.DEFAULT_MAX_CONTEXT_CHARS)

        intro_chart_candidates_block = ""
        intro_chart_candidate_ids: list[str] = []
        if kb_input_root is not None and max_auto > 0:
            try:
                raw_chart_hits = await self.hybrid_search_service.vector_service.search(
                    query=section.title,
                    collection_name=collection_name,
                    top_k=12,
                    filter_metadata={"chunk_type": "chart"},
                )
                filtered, ids = extract_chart_candidates_from_vector_results(raw_chart_hits, max_candidates=min(8, max_auto))
                intro_chart_candidate_ids = ids
                intro_chart_candidates_block = build_chart_candidates_block(filtered)
                cfgs_chart = chart_configs_from_candidates(
                    filtered,
                    kb_input_root=kb_input_root,
                    converter=self._chart_converter,
                )
                if cfgs_chart:
                    _render_charts(cfgs_chart, section_key_prefix=section.section_id)
            except Exception:
                intro_chart_candidates_block = ""
                intro_chart_candidate_ids = []

        intro_skeleton = f"- {section.title}\n    "
        intro_payload = {
            "title": section.title,
            "template_content": intro_skeleton,
            "context": intro_context,
            "document_title": document_title or "",
            "outline_items": section.title,
        }
        try:
            await self._emit(
                on_event,
                {
                    "type": "outline_node_start",
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "node_type": "intro",
                    "node_title": section.title,
                },
            )
            intro_chart_ids: list[str] = []
            try:
                for cfg in intro_cfgs or []:
                    extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                    cid = str(extra.get("chart_anchor_id") or "").strip()
                    if cid and cid not in intro_chart_ids:
                        intro_chart_ids.append(cid)
            except Exception:
                intro_chart_ids = []
            for cid in intro_chart_candidate_ids or []:
                c = str(cid or "").strip()
                if c and c not in intro_chart_ids:
                    intro_chart_ids.append(c)

            raw_intro = ""
            intro_tokens = 0
            try:
                out, mode = await self._section_structured.generate_outline_item(
                    outline_key=section.title,
                    title=section.title,
                    skeleton=intro_skeleton,
                    context=intro_context,
                    chart_candidates_block=intro_chart_candidates_block,
                    document_title=document_title or "",
                    required_chart_ids=intro_chart_ids,
                    available_chart_ids=intro_chart_ids,
                )
                allowed = set(intro_chart_ids)
                chosen: list[str] = []
                for cid in out.chart_ids or []:
                    c = str(cid or "").strip()
                    if not c or c not in allowed:
                        continue
                    if c in chosen:
                        continue
                    chosen.append(c)
                    if len(chosen) >= 3:
                        break
                body = (out.markdown_body or "").replace("\r\n", "\n").strip()
                body_lines = body.split("\n") if body else [""]
                out_lines = [f"- {section.title}"]
                out_lines.extend([("    " + ln if ln.strip() else "    ") for ln in body_lines])
                out_lines.extend([f"    [Chart: {cid}]" for cid in chosen])
                raw_intro = "\n".join(out_lines).rstrip()
                intro_tokens = max(0, len(raw_intro) // 4)
                coverage.append(
                    {"type": "structured_output", "mode": mode, "node_id": f"{section.section_id}__intro", "ok": True}
                )
            except Exception as exc:
                coverage.append(
                    {"type": "structured_output", "mode": "failed", "node_id": f"{section.section_id}__intro", "ok": False, "error": str(exc)}
                )
                raw_intro, intro_tokens = await self._section_llm.generate(
                    section_id=f"{section.section_id}__intro",
                    section_title=section.title,
                    payload=intro_payload,
                    stream_tokens=bool(stream_tokens),
                    on_event=on_event,
                )
            total_tokens += intro_tokens
            tmp = f"## {section.title}\n\n{(raw_intro or '').strip()}".strip()
            tmp = strip_orphan_outline_list_items(tmp, skeleton=intro_skeleton)
            intro_text = strip_leading_title_heading(tmp, title=section.title).strip()
            await self._emit(
                on_event,
                {
                    "type": "outline_node_done",
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "node_type": "intro",
                    "node_title": section.title,
                    "tokens_used": intro_tokens,
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise ContentGenerationError(
                message=f"LLM 调用失败: {str(exc)}",
                code="llm_invocation_failed",
                details={"section_id": section.section_id, "node": "intro"},
            )

        generated_by_item: dict[str, str] = {}
        fragments: list[str] = []
        stack: list[OutlineItem] = []

        for idx, item in enumerate(outline_items, start=1):
            while stack and stack[-1].depth >= item.depth:
                stack.pop()
            ancestors = list(stack)

            q = self._build_outline_query(section_title=section.title, item=item, ancestors=ancestors)
            await self._emit(
                on_event,
                {
                    "type": "outline_item_start",
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "item": item.display,
                    "depth": item.depth,
                    "index": idx,
                    "total": len(outline_items),
                    "query": q,
                },
            )
            try:
                resp2 = await self.hybrid_search_service.search(
                    query=q,
                    collection_name=collection_name,
                    options=options,
                )
                hits2 = resp2.results or []
            except Exception as exc:  # noqa: BLE001
                hits2 = []
                coverage.append(
                    coverage_outline_item_error(
                        item_display=item.display,
                        depth=item.depth,
                        query=q,
                        error=exc,
                    )
                )
                resp2 = None

            try:
                max_score2 = max((r.score for r in hits2), default=None)
            except Exception:
                max_score2 = None

            uncovered = not bool(hits2)
            if coverage_score_threshold is not None and max_score2 is not None:
                if float(max_score2) < float(coverage_score_threshold):
                    uncovered = True

            coverage.append(
                coverage_outline_item_ok(
                    item_display=item.display,
                    depth=item.depth,
                    query=q,
                    hits=len(hits2),
                    max_score=max_score2,
                    uncovered=uncovered,
                    bm25_index_used=getattr(getattr(resp2, "metrics", None), "bm25_index_used", None),
                    rerank_applied=getattr(getattr(resp2, "metrics", None), "rerank_applied", None),
                )
            )

            per_top = min(self.DEFAULT_PER_QUERY_TOP_N, len(hits2))
            picked = pick_unique_hits(hits2, per_top=per_top)
            cfgs: list[ChartConfig] = []

            if kb_input_root is not None and picked and len(rendered_charts) < max_auto:
                remain = max(0, max_auto - len(rendered_charts))
                if remain > 0:
                    picked_non_chart = [r for r in picked if (r.metadata or {}).get("chunk_type") != "chart"]
                    cfgs = self._auto_chart_resolver.extract_auto_chart_configs(
                        hits=picked_non_chart,
                        kb_input_root=kb_input_root,
                        md_image_re=self._MD_IMAGE_RE,
                        max_auto=remain,
                        seen_chart_json=seen_chart_json,
                        sources_set=sources_set,
                        hint_text=q,
                        audit=coverage,
                    )
                    if cfgs:
                        coverage.append({"type": "auto_charts", "count": len(cfgs)})
                    _render_charts(cfgs, section_key_prefix=section.section_id)

            item_chart_candidates_block = ""
            item_chart_candidate_ids: list[str] = []
            if kb_input_root is not None and len(rendered_charts) < max_auto:
                try:
                    raw_chart_hits = await self.hybrid_search_service.vector_service.search(
                        query=q,
                        collection_name=collection_name,
                        top_k=12,
                        filter_metadata={"chunk_type": "chart"},
                    )
                    remain = max(0, max_auto - len(rendered_charts))
                    filtered, ids = extract_chart_candidates_from_vector_results(
                        raw_chart_hits,
                        max_candidates=min(8, max(1, remain)),
                    )
                    item_chart_candidate_ids = ids
                    item_chart_candidates_block = build_chart_candidates_block(filtered)
                    cfgs_chart = chart_configs_from_candidates(
                        filtered,
                        kb_input_root=kb_input_root,
                        converter=self._chart_converter,
                    )
                    if cfgs_chart:
                        _render_charts(cfgs_chart, section_key_prefix=section.section_id)
                except Exception:
                    item_chart_candidates_block = ""
                    item_chart_candidate_ids = []

            context_lines = [build_outline_item_heading(item_display=item.display, depth=item.depth)]
            if ancestors:
                parent_key = ancestors[-1].display
                parent_txt = (generated_by_item.get(parent_key) or "").strip()
                if parent_txt:
                    context_lines.append("【上级内容摘要】")
                    context_lines.append(truncate_chars(parent_txt, max_chars=900))
                    context_lines.append("")

            for r in picked:
                doc_title = r.metadata.get("doc_title") if isinstance(r.metadata, dict) else None
                doc_rel = r.metadata.get("doc_rel_path") if isinstance(r.metadata, dict) else None
                ofn = r.metadata.get("original_filename") if isinstance(r.metadata, dict) else None
                display = choose_source_display(
                    doc_title=doc_title,
                    original_filename=ofn,
                    doc_rel_path=doc_rel,
                )
                context_lines.append(build_source_block(source_display=display, content=(r.content or ""), max_chars=1200))
                context_lines.append("")
                if display:
                    sources_set.add(display)
                cs = (r.metadata or {}).get("citation_sources") if isinstance(r.metadata, dict) else None
                if isinstance(cs, list):
                    for it in cs[:5]:
                        if not isinstance(it, dict):
                            continue
                        d2 = choose_source_display(
                            doc_title=it.get("doc_title"),
                            original_filename=it.get("original_filename"),
                            doc_rel_path=it.get("doc_rel_path"),
                        )
                        if d2:
                            sources_set.add(d2)

            if item_chart_candidates_block:
                context_lines.append("【图表候选】")
                context_lines.append(item_chart_candidates_block)
                context_lines.append("")

            context_txt = truncate_chars("\n".join(context_lines).strip(), max_chars=self.DEFAULT_MAX_CONTEXT_CHARS)

            skeleton = f"- {item.display}\n    "
            payload = {
                "title": item.display,
                "template_content": skeleton,
                "context": context_txt,
                "document_title": document_title or "",
                "outline_items": item.display,
            }

            node_id = f"{section.section_id}__item_{idx}"
            try:
                item_required_chart_ids: list[str] = []
                item_available_chart_ids: list[str] = []
                try:
                    for cfg in cfgs or []:
                        extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                        cid = str(extra.get("chart_anchor_id") or "").strip()
                        if cid and cid not in item_available_chart_ids:
                            item_available_chart_ids.append(cid)
                    for cid in item_chart_candidate_ids or []:
                        c = str(cid or "").strip()
                        if c and c not in item_available_chart_ids:
                            item_available_chart_ids.append(c)
                    item_required_chart_ids = list(item_available_chart_ids)
                except Exception:
                    item_required_chart_ids = []
                    item_available_chart_ids = []

                structured_used = False
                raw = ""
                tokens_used = 0
                try:
                    out, mode = await self._section_structured.generate_outline_item(
                        outline_key=item.display,
                        title=item.display,
                        skeleton=skeleton,
                        context=context_txt,
                        chart_candidates_block=item_chart_candidates_block,
                        document_title=document_title or "",
                        required_chart_ids=item_required_chart_ids,
                        available_chart_ids=item_available_chart_ids,
                    )
                    allowed = set(item_available_chart_ids)
                    chosen: list[str] = []
                    for cid in out.chart_ids or []:
                        c = str(cid or "").strip()
                        if not c or c not in allowed:
                            continue
                        if c in chosen:
                            continue
                        chosen.append(c)
                        if len(chosen) >= 3:
                            break
                    body = (out.markdown_body or "").replace("\r\n", "\n").strip()
                    body_lines = body.split("\n") if body else [""]
                    out_lines = [f"- {item.display}"]
                    out_lines.extend([("    " + ln if ln.strip() else "    ") for ln in body_lines])
                    out_lines.extend([f"    [Chart: {cid}]" for cid in chosen])
                    raw = "\n".join(out_lines).rstrip()
                    tokens_used = max(0, len(raw) // 4)
                    structured_used = True
                    coverage.append({"type": "structured_output", "mode": mode, "node_id": node_id, "ok": True})
                except Exception as exc:
                    coverage.append(
                        {"type": "structured_output", "mode": "failed", "node_id": node_id, "ok": False, "error": str(exc)}
                    )
                    raw, tokens_used = await self._section_llm.generate(
                        section_id=node_id,
                        section_title=item.display,
                        payload=payload,
                        stream_tokens=bool(stream_tokens),
                        on_event=on_event,
                    )
            except Exception as exc:  # noqa: BLE001
                raise ContentGenerationError(
                    message=f"LLM 调用失败: {str(exc)}",
                    code="llm_invocation_failed",
                    details={"section_id": section.section_id, "item": item.display},
                )
            total_tokens += tokens_used

            prefix = self._outline_heading_prefix(item.depth)
            tmp = f"{prefix} {item.display}\n\n{(raw or '').strip()}".strip()
            tmp = strip_orphan_outline_list_items(tmp, skeleton=skeleton)
            fragments.append(tmp)
            generated_by_item[item.display] = tmp
            await self._emit(
                on_event,
                {
                    "type": "outline_item_done",
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "item": item.display,
                    "depth": item.depth,
                    "index": idx,
                    "total": len(outline_items),
                    "tokens_used": tokens_used,
                    "rendered_charts": list(rendered_charts.keys()),
                },
            )

            stack.append(item)

        content_parts = [p for p in [intro_text, *fragments] if (p or "").strip()]
        final_content = "\n\n".join(content_parts).strip()

        generation_time_ms = (time.time() - start_time) * 1000
        return SectionGenerationResult(
            section_id=section.section_id,
            title=section.title,
            content=final_content,
            rendered_charts=rendered_charts,
            sources=sorted(sources_set),
            coverage=coverage,
            tokens_used=total_tokens,
            generation_time_ms=generation_time_ms,
        )

    async def _assemble_context_without_outline(
        self,
        *,
        section: TemplateSection,
        collection_name: str,
        options: HybridSearchOptions,
        kb_input_root: Path | None,
        max_auto: int,
        seen_chart_json: set[str],
        seen_chart_ids: set[str],
        coverage: list[dict[str, Any]],
        sources_set: set[str],
        item_context_map: dict[str, str],
        exclude_chunk_ids: set[str],
        on_event: Callable[[dict[str, Any]], Any] | None,
    ):
        resp = None
        try:
            resp = await self.hybrid_search_service.search(
                query=section.title,
                collection_name=collection_name,
                options=options,
            )
            hits = resp.results or []
            if exclude_chunk_ids:
                hits = [r for r in hits if r.chunk_id not in exclude_chunk_ids]
        except Exception as exc:  # noqa: BLE001
            hits = []
            coverage.append(coverage_section_title_error(query=section.title, error=exc))

        context_parts: list[str] = []
        chart_configs: list[ChartConfig] = []
        used_chunk_ids: list[str] = []
        non_chart_hits = [r for r in hits if (r.metadata or {}).get("chunk_type") != "chart"]
        chart_hits = [r for r in hits if (r.metadata or {}).get("chunk_type") == "chart"]

        # 选择用于上下文的“主文本片段”（优先非图表 chunk）
        window_non_chart = non_chart_hits[: min(self.DEFAULT_PER_QUERY_TOP_N, len(non_chart_hits))]

        # 若召回中存在图表线索（chart_ids），尽量保证 window_non_chart 中至少包含一个带 chart_ids 的父 chunk，
        # 以便后续渲染至少 1 张图表（若确实没有，则不强求）。
        if window_non_chart and not any((r.metadata or {}).get("chart_ids") for r in window_non_chart):
            cand = next((r for r in non_chart_hits if (r.metadata or {}).get("chart_ids")), None)
            if cand is not None and cand not in window_non_chart:
                window_non_chart[-1] = cand

        if kb_input_root is not None and window_non_chart:
            window_non_chart_for_chart = [r for r in window_non_chart if (r.metadata or {}).get("chunk_type") != "chart"]
            chart_configs = self._auto_chart_resolver.extract_auto_chart_configs(
                hits=window_non_chart_for_chart,
                kb_input_root=kb_input_root,
                md_image_re=self._MD_IMAGE_RE,
                max_auto=max_auto,
                seen_chart_json=seen_chart_json,
                sources_set=sources_set,
                hint_text=section.title,
                audit=coverage,
            )
            if chart_configs and seen_chart_ids:
                kept_cfgs: list[ChartConfig] = []
                for cfg in chart_configs:
                    extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                    cid = str(extra.get("chart_anchor_id") or "").strip()
                    if cid and cid in seen_chart_ids:
                        continue
                    kept_cfgs.append(cfg)
                chart_configs = kept_cfgs

        # 选择用于上下文的图表 chunk：若渲染出了图表，则优先把对应 chart_id 的图表语义 chunk 放入上下文，
        # 让 LLM 明确看到 `[Chart: <id>]` 以便在正文中落锚点；否则最多带 1 个图表语义 chunk（如果存在）。
        selected_chart_hits: list[SearchResult] = []
        if chart_hits:
            anchor_ids: list[str] = []
            seen_a: set[str] = set()
            for cfg in chart_configs or []:
                extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                cid = str(extra.get("chart_anchor_id") or "").strip()
                if cid and cid not in seen_a:
                    seen_a.add(cid)
                    anchor_ids.append(cid)
            if anchor_ids:
                for cid in anchor_ids:
                    if seen_chart_ids and cid in seen_chart_ids:
                        continue
                    m = next((r for r in chart_hits if str((r.metadata or {}).get("chart_id") or "").strip() == cid), None)
                    if m is not None and m not in selected_chart_hits:
                        selected_chart_hits.append(m)
            if not selected_chart_hits:
                for rr in chart_hits:
                    cid = str((rr.metadata or {}).get("chart_id") or "").strip()
                    if seen_chart_ids and cid and cid in seen_chart_ids:
                        continue
                    selected_chart_hits.append(rr)
                    break

        # 若上下文里已经有 chart chunk，但 chart_configs 未覆盖（例如模型直接引用了 chart_id），
        # 这里尝试直接从 chart chunk 的 metadata 反查 chart_json，确保能渲染对应图表。
        if kb_input_root is not None and selected_chart_hits and len(chart_configs) < max_auto:
            for ch in selected_chart_hits:
                if len(chart_configs) >= max_auto:
                    break
                meta = ch.metadata if isinstance(ch.metadata, dict) else {}
                chart_id = str(meta.get("chart_id") or "").strip()
                if seen_chart_ids and chart_id and chart_id in seen_chart_ids:
                    continue
                chart_json_name = str(meta.get("chart_json_path") or "").strip()
                doc_rel = str(meta.get("doc_rel_path") or "").strip()
                if not chart_id or not chart_json_name or not doc_rel:
                    continue
                try:
                    doc_dir_rel = self._auto_chart_resolver._resolve_doc_dir_rel(doc_rel)  # type: ignore[attr-defined]
                    cj_path = (kb_input_root / doc_dir_rel / "chart_json" / chart_json_name).resolve()
                except Exception:
                    continue
                key = str(cj_path).replace("\\", "/")
                if key in seen_chart_json:
                    continue
                if not cj_path.exists():
                    continue
                try:
                    import json as _json
                    from src.application.services.content_generation.text_utils import read_text_best_effort as _read

                    obj = _json.loads(_read(cj_path))
                except Exception:
                    continue
                if not isinstance(obj, dict) or obj.get("is_chart") is not True:
                    continue
                cfgs = self._chart_converter.chart_json_to_configs(obj)
                for j, cfg in enumerate(cfgs):
                    if len(chart_configs) >= max_auto:
                        break
                    extra = cfg.extra if isinstance(cfg.extra, dict) else {}
                    cfg2 = cfg.model_copy(
                        update={
                            "extra": {
                                **(extra or {}),
                                "chart_anchor_id": chart_id,
                                "chart_anchor_index": j,
                            }
                        }
                    )
                    chart_configs.append(cfg2)
                seen_chart_json.add(key)

        selected_for_context = [*window_non_chart, *selected_chart_hits]
        for r in selected_for_context:
            used_chunk_ids.append(r.chunk_id)
            doc_title = r.metadata.get("doc_title") if isinstance(r.metadata, dict) else None
            doc_rel = r.metadata.get("doc_rel_path") if isinstance(r.metadata, dict) else None
            ofn = r.metadata.get("original_filename") if isinstance(r.metadata, dict) else None

            display = choose_source_display(
                doc_title=doc_title,
                original_filename=ofn,
                doc_rel_path=doc_rel,
            )
            if display:
                sources_set.add(display)
            cs = (r.metadata or {}).get("citation_sources") if isinstance(r.metadata, dict) else None
            if isinstance(cs, list):
                for it in cs[:5]:
                    if not isinstance(it, dict):
                        continue
                    d2 = choose_source_display(
                        doc_title=it.get("doc_title"),
                        original_filename=it.get("original_filename"),
                        doc_rel_path=it.get("doc_rel_path"),
                    )
                    if d2:
                        sources_set.add(d2)

            txt = truncate_chars((r.content or "").strip(), max_chars=1600)
            if display:
                context_parts.append(f"【来源】{display}\n{txt}".strip())
            else:
                context_parts.append(txt)

        context = "\n\n".join([p for p in context_parts if p]).strip()
        context = truncate_chars(context, max_chars=self.DEFAULT_MAX_CONTEXT_CHARS)

        if hits:
            try:
                max_score = max((r.score for r in hits), default=None)
            except Exception:
                max_score = None
            coverage.append(
                coverage_section_title_ok(
                    query=section.title,
                    hits=len(hits),
                    max_score=max_score,
                    uncovered=False if max_score is not None else not bool(hits),
                    bm25_index_used=getattr(getattr(resp, "metrics", None), "bm25_index_used", None),
                    rerank_applied=getattr(getattr(resp, "metrics", None), "rerank_applied", None),
                )
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

        return context, coverage, None, [], sorted(sources_set), chart_configs, item_context_map, used_chunk_ids

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

        chart_anchor_ids: list[str] = []
        seen_anchor: set[str] = set()
        for cfg in auto_chart_configs or []:
            extra = cfg.extra if isinstance(cfg.extra, dict) else {}
            cid = str(extra.get("chart_anchor_id") or "").strip()
            if not cid or cid in seen_anchor:
                continue
            seen_anchor.add(cid)
            chart_anchor_ids.append(cid)
        # 召回到几个可用图表锚点，就要求使用几个（不做硬上限）
        required_ids = chart_anchor_ids
        required_chart_ids = ", ".join([f"[Chart: {cid}]" for cid in required_ids]) if required_ids else ""
        available_chart_ids = ", ".join([f"[Chart: {cid}]" for cid in chart_anchor_ids]) if chart_anchor_ids else ""

        title_by_id: dict[str, str] = {}
        for cfg in auto_chart_configs or []:
            extra = cfg.extra if isinstance(cfg.extra, dict) else {}
            cid = str(extra.get("chart_anchor_id") or "").strip()
            if not cid:
                continue
            t = str(getattr(cfg, "title", "") or "").strip()
            title_by_id.setdefault(cid, t)
        chart_candidates_block = "\n".join(
            [f"[Chart: {cid}] {title_by_id.get(cid, '')}".strip() for cid in chart_anchor_ids]
        ).strip()

        payload = {
            "title": section.title,
            "template_content": template_content_override or section.content,
            "context": truncated_context,
            "document_title": document_title or "",
            "outline_items": "\n".join([it.display for it in (outline_items or [])]).strip(),
            "required_chart_ids": required_chart_ids,
            "available_chart_ids": available_chart_ids,
        }

        generated_content = ""
        tokens_used = 0
        try:
            out, mode = await self._section_structured.generate_outline_item(
                outline_key=section.section_id,
                title=section.title,
                skeleton=str(payload["template_content"] or ""),
                context=str(payload["context"] or ""),
                chart_candidates_block=chart_candidates_block,
                document_title=str(payload["document_title"] or ""),
                required_chart_ids=required_ids,
                available_chart_ids=chart_anchor_ids,
            )
            body = (out.markdown_body or "").replace("\r\n", "\n").strip()
            generated_content = place_chart_anchors(
                body,
                chart_ids=required_ids,
                hint_by_chart_id=title_by_id,
                min_paragraph_index=1,
            ).strip()
            tokens_used = max(0, len(generated_content) // 4)
            if isinstance(coverage, list):
                coverage.append({"type": "structured_output", "mode": mode, "node_id": section.section_id, "ok": True})
        except Exception as exc:
            if isinstance(coverage, list):
                coverage.append({"type": "structured_output", "mode": "failed", "node_id": section.section_id, "ok": False, "error": str(exc)})
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
            generated_content = place_chart_anchors(
                str(generated_content or ""),
                chart_ids=required_ids,
                hint_by_chart_id=title_by_id,
                min_paragraph_index=1,
            ).strip()
            tokens_used = max(0, int(tokens_used or 0))

        # 清理模型可能重复输出的文档/章节标题，避免外层 HTML 标题与正文重复导致结构混乱
        if document_title:
            generated_content = strip_leading_title_heading(generated_content, title=str(document_title))

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

        if template_content_override:
            generated_content = strip_orphan_outline_list_items(
                generated_content,
                skeleton=template_content_override,
            )

        rendered_charts: dict[str, ChartTemplateSnippet] = {}
        for idx, cfg in enumerate(auto_chart_configs or []):
            chart_key = None
            extra = cfg.extra if isinstance(cfg.extra, dict) else None
            if isinstance(extra, dict):
                chart_key = str(extra.get("chart_anchor_id") or "").strip() or None
            if not chart_key:
                continue
            if chart_key in rendered_charts:
                chart_key = f"{chart_key}__{idx+1}"
            rendered_charts[chart_key] = ChartTemplateSnippet(
                chart_id=str(chart_key),
                container_html="",
                script_html="",
            )

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
        max_auto_charts_per_section: int | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
        stream_tokens: bool = False,
        polish_sections: bool = False,
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

        embedder = None
        if bool(getattr(get_settings(), "whitepaper_enable_semantic_dedup", True)):
            try:
                embedder = self.llm_runtime_service.get_model_for_callsite("vector_storage:embed_text")
            except Exception:
                embedder = None

        opts = PipelineOptions(
            enable_semantic_dedup=bool(getattr(get_settings(), "whitepaper_enable_semantic_dedup", True))
            and embedder is not None,
            dedup_similarity_threshold=float(
                getattr(get_settings(), "whitepaper_dedup_similarity_threshold", 0.92)
            ),
            enable_strip_empty_outline_items=bool(
                getattr(get_settings(), "whitepaper_strip_empty_outline_items", True)
            ),
            enable_section_polish=bool(polish_sections)
            or bool(getattr(get_settings(), "whitepaper_polish_sections", False)),
        )
        pipeline = SectionGenerationPipeline(
            content_service=self,
            options=opts,
            embedding_model=embedder,
        )

        section_results, pstats, total_tokens = await pipeline.run(
            sections=sections,
            collection_name=collection_name,
            search_options=search_options,
            document_title=doc_title,
            coverage_score_threshold=coverage_score_threshold,
            kb_input_root=kb_input_root,
            max_auto_charts_per_section=max_auto_charts_per_section,
            on_event=on_event,
            stream_tokens=bool(stream_tokens),
        )

        if kb_input_root is not None:
            try:
                chart_ids: list[str] = []
                seen: set[str] = set()
                for s in section_results:
                    for cid in sorted((getattr(s, "rendered_charts", {}) or {}).keys()):
                        c = str(cid or "").strip()
                        if c and c not in seen:
                            seen.add(c)
                            chart_ids.append(c)
                    import re as _re

                    for m in _re.finditer(r"\[Chart:\s*([^\]]+?)\s*\]", str(getattr(s, "content", "") or "")):
                        c = str(m.group(1) or "").strip()
                        if c and c not in seen:
                            seen.add(c)
                            chart_ids.append(c)
                if chart_ids:
                    from src.application.services.antv_rendering import AntvRenderingService

                    AntvRenderingService().render_from_chart_json_dir(
                        kb_input_root=kb_input_root,
                        chart_ids=chart_ids,
                        theme="whitepaper-default",
                        force=False,
                    )
            except Exception as exc:
                log.warning(
                    "antv_render.pre_render_failed",
                    extra=log_extra(kb_input_root=str(kb_input_root), error=str(exc)),
                )

        html = WhitepaperHtmlRenderer().render(
            template=template,
            section_results=section_results,
            document_title=doc_title,
            kb_input_root=kb_input_root,
            chart_theme="whitepaper-default",
        )

        log.info(
            "content_generation.pipeline_stats",
            extra=log_extra(
                template_id=template_id,
                template_name=template.original_filename,
                sections=len(section_results),
                enable_semantic_dedup=opts.enable_semantic_dedup,
                enable_strip_empty_outline_items=opts.enable_strip_empty_outline_items,
                enable_section_polish=opts.enable_section_polish,
                dedup_removed_within_sections=pstats.dedup_removed_within_sections,
                dedup_removed_across_sections=pstats.dedup_removed_across_sections,
                empty_outline_items_removed=pstats.empty_outline_items_removed,
                polished_sections=pstats.polished_sections,
                polish_skipped_by_validator=pstats.polish_skipped_by_validator,
                postprocess_time_ms=round(pstats.postprocess_time_ms, 2),
                polish_time_ms=round(pstats.polish_time_ms, 2),
            ),
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

    async def _polish_section(
        self,
        *,
        section_id: str,
        section_title: str,
        skeleton: str,
        draft_markdown: str,
        document_title: str | None = None,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
    ) -> tuple[str, int]:
        payload = {
            "document_title": (document_title or "").strip(),
            "title": section_title,
            "template_content": skeleton,
            "draft_markdown": (draft_markdown or "").strip(),
        }
        return await self._section_polisher.generate(
            section_id=section_id,
            section_title=section_title,
            payload=payload,
            stream_tokens=False,
            on_event=on_event,
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

    async def polish_outline(self, outline: str) -> str:
        """大纲润色（委托 OutlinePolishService 执行）。"""
        try:
            input_data = OutlinePolishInput(outline=outline or "")
            result = await self._outline_polish_service.polish_outline(input_data)
            if result.success and result.output:
                return result.output.polished_outline
            else:
                raise ContentGenerationError(
                    message=f"大纲润色失败: {result.error or '未知错误'}",
                    code="outline_polish_failed",
                )
        except ContentGenerationError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ContentGenerationError(
                message=f"大纲润色失败: {str(e)}",
                code="outline_polish_failed",
            )
