from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.application.services.content_generation.postprocessors import (
    DedupStats,
    SemanticDeduplicator,
    strip_empty_outline_items,
)
from src.application.services.content_generation.types import (
    OutlineItem,
    SectionGenerationResult,
    TemplateSection,
)


@dataclass
class PipelineOptions:
    enable_semantic_dedup: bool = True
    dedup_similarity_threshold: float = 0.92
    enable_strip_empty_outline_items: bool = True
    enable_section_polish: bool = False


@dataclass
class PipelineStats:
    dedup_removed_within_sections: int = 0
    dedup_removed_across_sections: int = 0
    empty_outline_items_removed: int = 0
    polished_sections: int = 0
    polish_skipped_by_validator: int = 0
    postprocess_time_ms: float = 0.0
    polish_time_ms: float = 0.0


async def _emit(on_event: Callable[[dict[str, Any]], Any] | None, payload: dict[str, Any]) -> None:
    if on_event is None:
        return
    try:
        r = on_event(payload)
        if asyncio.iscoroutine(r):
            await r
    except Exception:
        return


def _extract_skeleton_list_lines(skeleton: str) -> list[str]:
    out: list[str] = []
    for ln in (skeleton or "").replace("\r\n", "\n").split("\n"):
        s = ln.rstrip()
        if s.lstrip(" ").startswith("- "):
            out.append(s.lstrip(" "))
    return out


def _polish_validator_ok(*, original: str, polished: str, skeleton: str) -> bool:
    sk_lines = _extract_skeleton_list_lines(skeleton)
    if not sk_lines:
        return True

    def _pick(md: str) -> list[str]:
        out: list[str] = []
        for ln in (md or "").replace("\r\n", "\n").split("\n"):
            s = ln.lstrip(" ").rstrip()
            if s.startswith("- "):
                out.append(s)
        return out

    orig_lines = _pick(original)
    pol_lines = _pick(polished)
    if orig_lines != pol_lines:
        return False

    for it in sk_lines:
        if it not in pol_lines:
            return False
    return True


class SectionGenerationPipeline:
    def __init__(
        self,
        *,
        content_service: Any,
        options: PipelineOptions,
        embedding_model: Any | None = None,
    ):
        self._svc = content_service
        self._opt = options
        self._embedding_model = embedding_model

    async def run(
        self,
        *,
        sections: list[TemplateSection],
        collection_name: str,
        search_options: Any,
        document_title: str,
        coverage_score_threshold: float | None,
        kb_input_root: Path | None,
        max_auto_charts_per_section: int | None,
        on_event: Callable[[dict[str, Any]], Any] | None,
        stream_tokens: bool,
    ) -> tuple[list[SectionGenerationResult], PipelineStats, int]:
        section_results: list[SectionGenerationResult] = []
        total_tokens = 0
        skeleton_by_section_id: dict[str, str] = {}
        global_used_chunk_ids: set[str] = set()
        global_seen_chart_json: set[str] = set()
        global_seen_chart_ids: set[str] = set()

        await _emit(
            on_event,
            {
                "type": "pipeline_start",
                "pipeline": "section_generation",
                "sections": len(sections),
            },
        )

        for idx, sec in enumerate(sections, start=1):
            await _emit(
                on_event,
                {
                    "type": "section_start",
                    "section_id": sec.section_id,
                    "section_title": sec.title,
                    "index": idx,
                    "total": len(sections),
                },
            )

            context, coverage, skeleton, outline_items2, sources, auto_charts, _item_ctx, used_chunk_ids = (
                await self._svc.assemble_context_for_section(
                    sec,
                    collection_name=collection_name,
                    options=search_options,
                    coverage_score_threshold=coverage_score_threshold,
                    kb_input_root=kb_input_root,
                    max_auto_charts_per_section=max_auto_charts_per_section,
                    global_seen_chart_json=global_seen_chart_json,
                    global_seen_chart_ids=global_seen_chart_ids,
                    exclude_chunk_ids=global_used_chunk_ids,
                    on_event=on_event,
                )
            )
            if used_chunk_ids:
                global_used_chunk_ids.update([x for x in used_chunk_ids if str(x or "").strip()])
            if skeleton:
                skeleton_by_section_id[str(sec.section_id)] = str(skeleton)

            template_override_for_section = skeleton if skeleton else None

            r: SectionGenerationResult = await self._svc._generate_section(
                section=sec,
                context=context,
                template_content_override=template_override_for_section,
                outline_items=outline_items2,
                document_title=document_title,
                sources=sources,
                auto_chart_configs=auto_charts,
                coverage=coverage,
                on_event=on_event,
                stream_tokens=bool(stream_tokens),
            )
            section_results.append(r)
            total_tokens += r.tokens_used
            for k in (r.rendered_charts or {}).keys():
                kk = str(k or "").strip()
                if kk:
                    global_seen_chart_ids.add(kk)

            await _emit(
                on_event,
                {
                    "type": "section_done",
                    "section_id": sec.section_id,
                    "section_title": sec.title,
                    "tokens_used": r.tokens_used,
                    "rendered_charts": list(r.rendered_charts.keys()),
                },
            )

        stats = PipelineStats()
        post_start = time.time()
        section_results = await self._postprocess_sections(
            section_results,
            skeleton_by_section_id=skeleton_by_section_id,
            stats=stats,
            on_event=on_event,
        )
        stats.postprocess_time_ms = (time.time() - post_start) * 1000

        polish_start = time.time()
        section_results, polish_tokens = await self._polish_sections(
            section_results,
            skeleton_by_section_id=skeleton_by_section_id,
            document_title=document_title,
            stats=stats,
            on_event=on_event,
        )
        stats.polish_time_ms = (time.time() - polish_start) * 1000
        total_tokens += polish_tokens

        await _emit(
            on_event,
            {
                "type": "pipeline_done",
                "pipeline": "section_generation",
                "sections": len(section_results),
                "stats": {
                    "dedup_removed_within_sections": stats.dedup_removed_within_sections,
                    "dedup_removed_across_sections": stats.dedup_removed_across_sections,
                    "empty_outline_items_removed": stats.empty_outline_items_removed,
                    "polished_sections": stats.polished_sections,
                    "polish_skipped_by_validator": stats.polish_skipped_by_validator,
                    "postprocess_time_ms": round(stats.postprocess_time_ms, 2),
                    "polish_time_ms": round(stats.polish_time_ms, 2),
                },
            },
        )

        return section_results, stats, total_tokens

    async def _postprocess_sections(
        self,
        section_results: list[SectionGenerationResult],
        *,
        skeleton_by_section_id: dict[str, str],
        stats: PipelineStats,
        on_event: Callable[[dict[str, Any]], Any] | None,
    ) -> list[SectionGenerationResult]:
        global_kept_embeddings: list[list[float]] = []
        deduper: SemanticDeduplicator | None = None
        if self._opt.enable_semantic_dedup and self._embedding_model is not None:
            deduper = SemanticDeduplicator(
                embedding_model=self._embedding_model,
                similarity_threshold=self._opt.dedup_similarity_threshold,
            )

        for s in section_results:
            skeleton = str(skeleton_by_section_id.get(str(s.section_id)) or "").strip()

            if self._opt.enable_strip_empty_outline_items and skeleton:
                cleaned, removed = strip_empty_outline_items(s.content, skeleton=skeleton)
                if removed:
                    stats.empty_outline_items_removed += removed
                    s.content = cleaned
                    await _emit(
                        on_event,
                        {
                            "type": "postprocess",
                            "stage": "strip_empty_outline_items",
                            "section_id": s.section_id,
                            "section_title": s.title,
                            "removed": removed,
                        },
                    )

            if deduper is not None:
                cleaned2, dstat, global_kept_embeddings = await deduper.dedup_markdown(
                    s.content,
                    global_kept_embeddings=global_kept_embeddings,
                )
                if dstat.removed_within_section or dstat.removed_across_sections:
                    stats.dedup_removed_within_sections += dstat.removed_within_section
                    stats.dedup_removed_across_sections += dstat.removed_across_sections
                    s.content = cleaned2
                    await _emit(
                        on_event,
                        {
                            "type": "postprocess",
                            "stage": "semantic_dedup",
                            "section_id": s.section_id,
                            "section_title": s.title,
                            "removed_within_section": dstat.removed_within_section,
                            "removed_across_sections": dstat.removed_across_sections,
                        },
                    )

        return section_results

    async def _polish_sections(
        self,
        section_results: list[SectionGenerationResult],
        *,
        skeleton_by_section_id: dict[str, str],
        document_title: str,
        stats: PipelineStats,
        on_event: Callable[[dict[str, Any]], Any] | None,
    ) -> tuple[list[SectionGenerationResult], int]:
        if not self._opt.enable_section_polish:
            return section_results, 0

        polish_tokens = 0
        for s in section_results:
            skeleton = str(skeleton_by_section_id.get(str(s.section_id)) or "").strip()
            if not skeleton:
                continue

            await _emit(
                on_event,
                {
                    "type": "polish_start",
                    "section_id": s.section_id,
                    "section_title": s.title,
                },
            )

            polished, used = await self._svc._polish_section(
                section_id=s.section_id,
                section_title=s.title,
                skeleton=skeleton,
                draft_markdown=s.content,
                document_title=document_title,
                on_event=on_event,
            )
            polish_tokens += used

            if polished and _polish_validator_ok(original=s.content, polished=polished, skeleton=skeleton):
                s.content = polished
                stats.polished_sections += 1
                await _emit(
                    on_event,
                    {
                        "type": "polish_done",
                        "section_id": s.section_id,
                        "section_title": s.title,
                        "tokens_used": used,
                    },
                )
            else:
                stats.polish_skipped_by_validator += 1
                await _emit(
                    on_event,
                    {
                        "type": "polish_skipped",
                        "section_id": s.section_id,
                        "section_title": s.title,
                        "reason": "validator_rejected",
                        "tokens_used": used,
                    },
                )

        return section_results, polish_tokens
