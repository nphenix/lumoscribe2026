"""文档清洗服务（T031）。

提供两层清洗策略：
1. 规则过滤：移除广告、噪声、重复内容
2. LLM 智能清洗：使用推理 LLM 保留结构，移除噪声
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any
from collections.abc import AsyncIterator

from src.application.repositories.intermediate_artifact_repository import (
    IntermediateArtifactRepository,
)
from src.application.schemas.document_cleaning import (
    ChartType,
    ChartData,
    ChartJSONArtifact,
    ChartExtractionResult,
    CleaningOptions,
    CleaningResult,
    CleaningStats,
    CleanedDocumentArtifact,
)
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.application.services.prompt_service import PromptService
from src.domain.entities.intermediate_artifact import IntermediateType
from src.shared.constants.prompts import (
    DEFAULT_PROMPTS,
    SCOPE_CHART_EXTRACTION,
    SCOPE_DOC_CLEANING,
)
from src.shared.logging import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class DocumentCleaningService:
    """文档清洗服务。

    提供两层清洗策略，产出 cleaned_doc 中间产物。
    """

    # 广告特征模式
    AD_PATTERNS = [
        r"广告",  # 通用广告标记
        r"Sponsored",  # 赞助内容
        r"广告位",  # 广告位占位
        r"推广",  # 推广内容
        r"VIP",  # VIP 会员广告
        r"开通会员",  # 会员推广
        r"扫码关注",  # 二维码广告
        r"微信公众号",  # 公众号推广
        r"客服电话",  # 客服广告
        r"咨询热线",  # 咨询热线广告
    ]

    # 噪声模式（页眉页脚、页码、目录等）
    NOISE_PATTERNS = [
        r"第\s*\d+\s*页",  # 页码
        r"Page\s*\d+",  # 英文页码
        r"^\s*[\d\-]+\s*$",  # 纯数字行（可能是页码）
        r"^[-—=]{3,}$",  # 分隔线
        r"版权声明",  # 版权信息
        r"版权所有",  # 版权信息
        r"联系电话：\d+",  # 联系电话
        r"地址：",  # 地址信息
        r"^目\s*录\s*$",  # 目录标题
        r"^表\s*次\s*$",  # 目录标题变体
        r"^目\s*次\s*$",  # 目录标题变体
        r"^第[一二三四五六七八九十]+章",  # 章节标题（目录条目）
        r"^第[一二三四五六七八九十]+节",  # 小节标题（目录条目）
        r"^\d+[.、]\s+\S+.{0,30}$",  # 目录条目格式（数字+点+内容）
        r"^\s*\.{3,}\s*\d+\s*$",  # 目录页码引导线
    ]

    # 仅在文档开头区域删除的噪声（避免误伤正文）
    # 说明：这类信息常见于封面/首页（如“汇报人：xxx”）
    HEAD_NOISE_PATTERNS = [
        r"^\s*(汇报人|报告人|主讲人|演讲人)\s*[:：].*$",
    ]

    # Markdown 图片链接（用于结构保护）
    # 例：![](images/xxx.jpg) 或 ![alt](images/xxx.jpg)
    MD_IMAGE_TOKEN_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")

    def __init__(
        self,
        llm_runtime: LLMRuntimeService,
        artifact_repository: IntermediateArtifactRepository,
        prompt_service: PromptService,
    ):
        self.llm_runtime = llm_runtime
        self.artifact_repository = artifact_repository
        self.prompt_service = prompt_service

    async def clean_document(
        self,
        text: str,
        source_file_id: int,
        options: CleaningOptions | None = None,
    ) -> CleaningResult:
        """清洗文档，返回清洗后的内容和元数据。

        Args:
            text: 原始文本
            source_file_id: 源文件 ID
            options: 清洗选项

        Returns:
            清洗结果
        """
        try:
            options = options or CleaningOptions()
            logger.info(f"开始清洗文档 source_file_id={source_file_id}")

            # 第一层：规则过滤
            rule_cleaned = self.rule_based_clean(text, options)
            rule_stats = self._calculate_stats(text, rule_cleaned)

            # 第二层：LLM 智能清洗
            llm_cleaned = await self.llm_clean(rule_cleaned, options)
            final_stats = self._calculate_stats(rule_cleaned, llm_cleaned)
            final_stats = self._merge_stats(rule_stats, final_stats)

            # 保存产物
            output_path = self._save_artifact(
                source_file_id=source_file_id,
                stats=final_stats,
            )

            # 构建产物数据
            artifact = CleanedDocumentArtifact(
                source_file_id=source_file_id,
                original_text=text,
                cleaned_text=llm_cleaned,
                cleaning_stats=final_stats,
                output_path=output_path,
            )

            # 写入文件
            self._write_artifact_file(output_path, artifact)

            logger.info(
                f"文档清洗完成 source_file_id={source_file_id}, "
                f"移除字符数={final_stats.removed_chars}"
            )

            return CleaningResult(
                success=True,
                cleaned_text=llm_cleaned,
                stats=final_stats,
                error=None,
            )

        except Exception as e:
            logger.error(f"文档清洗失败 source_file_id={source_file_id}: {e}")
            return CleaningResult(
                success=False,
                cleaned_text=None,
                stats=None,
                error=str(e),
            )

    async def clean_document_stream(
        self,
        text: str,
        source_file_id: int,
        options: CleaningOptions | None = None,
        *,
        include_final_result: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """以流式方式清洗文档。

        说明：
        - 该方法会持续产出 `delta` 事件（用于 UI/SSE/WebSocket 实时显示）
        - 由于流式过程中无法“回撤”已输出内容，最终会再产出 `final` 事件，携带与
          `clean_document()` **完全一致**的最终清洗结果（建议以 final 为准落盘/展示）
        """
        options = options or CleaningOptions()
        logger.info(f"开始流式清洗文档 source_file_id={source_file_id}")

        # 第一层：规则过滤（同步，快速完成）
        yield {"type": "stage", "stage": "rule_clean_start"}
        rule_cleaned = self.rule_based_clean(text, options)
        rule_stats = self._calculate_stats(text, rule_cleaned)
        yield {"type": "stage", "stage": "rule_clean_done"}

        # 第二层：LLM 清洗（流式输出）
        yield {"type": "stage", "stage": "llm_clean_start"}
        llm_cleaned: str | None = None
        async for evt in self.llm_clean_stream(rule_cleaned, options):
            if evt.get("type") == "delta":
                yield evt
            elif evt.get("type") == "final":
                llm_cleaned = str(evt.get("cleaned_text") or "")
            else:
                # 透传 stage/error 等事件
                yield evt

        if llm_cleaned is None:
            # 理论上 llm_clean_stream 必然产出 final；这里做兜底避免 None
            llm_cleaned = ""
        final_stats = self._calculate_stats(rule_cleaned, llm_cleaned)
        final_stats = self._merge_stats(rule_stats, final_stats)

        # 保存产物（与 clean_document 一致）
        output_path = self._save_artifact(
            source_file_id=source_file_id,
            stats=final_stats,
        )
        artifact = CleanedDocumentArtifact(
            source_file_id=source_file_id,
            original_text=text,
            cleaned_text=llm_cleaned,
            cleaning_stats=final_stats,
            output_path=output_path,
        )
        self._write_artifact_file(output_path, artifact)

        yield {"type": "stage", "stage": "llm_clean_done"}

        if include_final_result:
            yield {
                "type": "final",
                "success": True,
                "cleaned_text": llm_cleaned,
                "stats": final_stats.model_dump(),
                "output_path": output_path,
            }

    def rule_based_clean(self, text: str, options: CleaningOptions) -> str:
        """规则过滤。

        Args:
            text: 原始文本
            options: 清洗选项

        Returns:
            规则过滤后的文本
        """
        result = text

        # 移除广告
        if options.remove_ads:
            for pattern in self.AD_PATTERNS:
                result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        # 移除噪声
        if options.remove_noise:
            # 头部噪声：仅对前 N 行应用（避免误伤正文）
            head_limit = 80
            head_lines = result.split("\n")[:head_limit]
            tail_lines = result.split("\n")[head_limit:]
            for pattern in self.HEAD_NOISE_PATTERNS:
                head_lines = [
                    re.sub(pattern, "", line, flags=re.MULTILINE) for line in head_lines
                ]
            result = "\n".join(head_lines + tail_lines)

            for pattern in self.NOISE_PATTERNS:
                result = re.sub(pattern, "", result, flags=re.MULTILINE)

        # 标准化空白字符
        if options.normalize_whitespace:
            # 将多个连续空白字符替换为单个空格
            result = re.sub(r"[ \t]+", " ", result)
            # 移除每行首尾空白
            result = "\n".join(line.strip() for line in result.split("\n"))
            # 将多个连续换行替换为单个换行
            result = re.sub(r"\n{3,}", "\n\n", result)

        # 移除重复内容
        if options.remove_duplicates:
            result = self._remove_duplicates(result)

        return result.strip()

    async def llm_clean(
        self, text: str, options: CleaningOptions, *, strict: bool = False
    ) -> str:
        """LLM 智能清洗。

        使用 callsite `doc_cleaning:clean_text` 调用推理 LLM，保留文档结构，移除噪声。

        Args:
            text: 规则过滤后的文本
            options: 清洗选项

        Returns:
            LLM 清洗后的文本
        """
        segments = self._split_markdown_by_image_tokens(text)

        # 如果存在图片链接：LLM 只处理文本段，图片段原样拼回（避免被误删）
        if len(segments) > 1:
            out: list[str] = []
            for seg_type, seg_text in segments:
                if seg_type == "image":
                    out.append(seg_text)
                else:
                    out.append(
                        await self._llm_clean_text_segment(
                            seg_text, options, strict=strict
                        )
                    )
            return "".join(out).strip()

        # 无图片链接：单段清洗
        return (await self._llm_clean_text_segment(text, options, strict=strict)).strip()

    async def llm_clean_stream(
        self, text: str, options: CleaningOptions, *, strict: bool = False
    ) -> AsyncIterator[dict[str, Any]]:
        """仅 LLM 层的流式清洗（不含规则过滤）。

        产出事件：
        - {"type": "delta", "text": "..."}: 逐段增量文本（最佳努力）
        - {"type": "stage", "stage": "..."}: 阶段信号
        - {"type": "error", "error": "..."}: 发生错误（strict=False 时会降级输出原文）
        """
        yield {"type": "stage", "stage": "segment_split"}
        segments = self._split_markdown_by_image_tokens(text)
        yield {
            "type": "stage",
            "stage": "segment_stream_start",
            "segments": len(segments),
        }

        final_parts: list[str] = []

        for seg_type, seg_text in segments:
            if seg_type == "image":
                # 图片 token 必须原样透传（结构保护）
                if seg_text:
                    yield {"type": "delta", "text": seg_text}
                final_parts.append(seg_text)
                continue

            seg_final: str | None = None
            async for evt in self._llm_clean_text_segment_stream(
                seg_text, options, strict=strict
            ):
                if evt.get("type") == "segment_final":
                    seg_final = str(evt.get("text") or "")
                    continue
                yield evt
            final_parts.append(seg_final if seg_final is not None else "")

        final_text = "".join(final_parts).strip()
        yield {"type": "stage", "stage": "segment_stream_done"}
        yield {"type": "final", "cleaned_text": final_text}

    def _split_markdown_by_image_tokens(self, text: str) -> list[tuple[str, str]]:
        """按 Markdown 图片 token 将文本分段。

        Returns:
            list of (type, content), where type in {"text", "image"}
        """
        if not text:
            return [("text", "")]

        parts: list[tuple[str, str]] = []
        pos = 0
        for m in self.MD_IMAGE_TOKEN_RE.finditer(text):
            if m.start() > pos:
                parts.append(("text", text[pos : m.start()]))
            parts.append(("image", m.group(0)))
            pos = m.end()
        if pos < len(text):
            parts.append(("text", text[pos:]))
        return parts if parts else [("text", text)]

    async def _llm_clean_text_segment(
        self, text: str, options: CleaningOptions, *, strict: bool
    ) -> str:
        """清洗单个文本段（该段不包含 Markdown 图片链接 token）。"""
        if not text.strip():
            return text

        # 保留段落边界：把首尾空白摘出来，避免 LLM strip 破坏拼接
        prefix_match = re.match(r"^\s*", text)
        suffix_match = re.search(r"\s*$", text)
        prefix = prefix_match.group(0) if prefix_match else ""
        suffix = suffix_match.group(0) if suffix_match else ""
        core = text[len(prefix) : len(text) - len(suffix)]
        if not core.strip():
            return text

        try:
            prompt_text = self._build_llm_clean_prompt(core, options, strict=strict)

            llm = self._bind_llm_for_streaming(
                self.llm_runtime.get_model_for_callsite(SCOPE_DOC_CLEANING)
            )

            prompt_template = ChatPromptTemplate.from_messages([("human", "{input}")])
            chain = prompt_template | llm | StrOutputParser()

            # 使用流式传输聚合结果（内部聚合），确保对外 clean_document() 语义稳定
            cleaned_chunks: list[str] = []
            async for chunk in chain.astream({"input": prompt_text}):
                if chunk:
                    cleaned_chunks.append(str(chunk))
            cleaned = "".join(cleaned_chunks)

            cleaned_text = cleaned.strip()
            return f"{prefix}{cleaned_text}{suffix}"

        except Exception as e:
            if strict:
                raise
            logger.warning(f"LLM 清洗失败，返回原文本段: {e}")
            return text

    def _bind_llm_for_streaming(self, llm: Any) -> Any:
        """尽可能为底层 LLM 启用流式输出（LangChain 推荐：stream/astream + streaming=True）。"""
        try:
            # OpenAI Compatible: ChatOpenAI 需要 streaming=True 才会真正 token streaming
            if hasattr(llm, "streaming"):
                return llm.bind(streaming=True)
            # Ollama: 默认支持 streaming；若用户配置了 disable_streaming，则显式关闭
            if hasattr(llm, "disable_streaming"):
                return llm.bind(disable_streaming=False)
        except Exception:
            return llm
        return llm

    async def _llm_clean_text_segment_stream(
        self, text: str, options: CleaningOptions, *, strict: bool
    ) -> AsyncIterator[dict[str, Any]]:
        """流式清洗单个文本段（该段不包含 Markdown 图片链接 token）。"""
        if not text.strip():
            if text:
                yield {"type": "delta", "text": text}
            return

        # 保留段落边界：把首尾空白摘出来，避免 LLM strip 破坏拼接
        prefix_match = re.match(r"^\s*", text)
        suffix_match = re.search(r"\s*$", text)
        prefix = prefix_match.group(0) if prefix_match else ""
        suffix = suffix_match.group(0) if suffix_match else ""
        core = text[len(prefix) : len(text) - len(suffix)]
        if not core.strip():
            yield {"type": "delta", "text": text}
            yield {"type": "segment_final", "text": text}
            return

        try:
            prompt_text = self._build_llm_clean_prompt(core, options, strict=strict)
            llm = self._bind_llm_for_streaming(
                self.llm_runtime.get_model_for_callsite(SCOPE_DOC_CLEANING)
            )
            prompt_template = ChatPromptTemplate.from_messages([("human", "{input}")])
            chain = prompt_template | llm | StrOutputParser()

            if prefix:
                yield {"type": "delta", "text": prefix}

            core_chunks: list[str] = []
            async for chunk in chain.astream({"input": prompt_text}):
                if chunk:
                    s = str(chunk)
                    core_chunks.append(s)
                    yield {"type": "delta", "text": s}

            if suffix:
                yield {"type": "delta", "text": suffix}

            segment_final = f"{prefix}{''.join(core_chunks).strip()}{suffix}"
            yield {"type": "segment_final", "text": segment_final}

        except Exception as e:
            if strict:
                raise
            logger.warning(f"LLM 流式清洗失败，降级输出原文本段: {e}")
            yield {"type": "error", "error": str(e)}
            yield {"type": "delta", "text": text}
            yield {"type": "segment_final", "text": text}

    def _build_llm_clean_prompt(
        self, text: str, options: CleaningOptions, *, strict: bool = False
    ) -> str:
        """构建 LLM 清洗提示词。"""
        instructions = []

        if options.remove_ads:
            instructions.append("- 移除所有广告内容、推广信息")

        if options.remove_noise:
            instructions.append("- 移除页眉、页脚、页码等噪声")
            instructions.append("- 移除版权声明、联系方式等非正文内容")

        if options.remove_duplicates:
            instructions.append("- 移除重复的内容段落")

        if options.preserve_structure:
            instructions.append("- 保留文档的标题层级结构")
            instructions.append("- 保留段落、列表等格式")
            instructions.append("- 保持内容的逻辑连贯性")

        instructions_str = "\n".join(instructions)

        # 获取提示词模板
        prompt_entity = self.prompt_service.get_active_prompt(SCOPE_DOC_CLEANING)
        if prompt_entity and prompt_entity.content and prompt_entity.content.strip():
            template = prompt_entity.content
        else:
            if strict:
                raise RuntimeError(
                    f"未找到激活提示词: {SCOPE_DOC_CLEANING}（请检查 SQLite prompts 配置）"
                )
            template = DEFAULT_PROMPTS[SCOPE_DOC_CLEANING]["content"]

        # 渲染提示词
        # 简单替换 {instructions} 和 {text}，避免 format 报错
        prompt = template.replace("{instructions}", instructions_str).replace(
            "{text}", text
        )

        return prompt

    def _remove_duplicates(self, text: str) -> str:
        """移除重复内容。"""
        lines = text.split("\n")
        seen = set()
        unique_lines = []

        for line in lines:
            # 标准化行内容（去除首尾空白，转小写）
            normalized = line.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)
            elif normalized and normalized in seen:
                # 跳过重复行
                pass

        return "\n".join(unique_lines)

    def _calculate_stats(
        self, original: str, cleaned: str
    ) -> CleaningStats:
        """计算清洗统计信息。"""
        original_chars = len(original)
        cleaned_chars = len(cleaned)

        original_paragraphs = len([p for p in original.split("\n") if p.strip()])
        cleaned_paragraphs = len([p for p in cleaned.split("\n") if p.strip()])

        return CleaningStats(
            original_chars=original_chars,
            cleaned_chars=cleaned_chars,
            removed_chars=max(0, original_chars - cleaned_chars),
            original_paragraphs=original_paragraphs,
            cleaned_paragraphs=cleaned_paragraphs,
            removed_paragraphs=max(0, original_paragraphs - cleaned_paragraphs),
            ads_removed=0,  # 简化估算
            noise_removed=0,
            duplicates_removed=0,
        )

    def _merge_stats(
        self, rule_stats: CleaningStats, llm_stats: CleaningStats
    ) -> CleaningStats:
        """合并两次清洗的统计信息。"""
        return CleaningStats(
            original_chars=rule_stats.original_chars,
            cleaned_chars=llm_stats.cleaned_chars,
            removed_chars=rule_stats.removed_chars + llm_stats.removed_chars,
            original_paragraphs=rule_stats.original_paragraphs,
            cleaned_paragraphs=llm_stats.cleaned_paragraphs,
            removed_paragraphs=rule_stats.removed_paragraphs
            + llm_stats.removed_paragraphs,
            ads_removed=rule_stats.ads_removed + llm_stats.ads_removed,
            noise_removed=rule_stats.noise_removed + llm_stats.noise_removed,
            duplicates_removed=rule_stats.duplicates_removed
            + llm_stats.duplicates_removed,
        )

    def _save_artifact(
        self,
        source_file_id: int,
        stats: CleaningStats,
    ) -> str:
        """保存中间产物到数据库。

        Returns:
            相对存储路径
        """
        import uuid
        from datetime import datetime
        from src.domain.entities.intermediate_artifact import IntermediateArtifact

        batch_id = datetime.now().strftime("%Y%m%d")
        artifact_id = str(uuid.uuid4())
        relative_path = f"intermediates/{source_file_id}/cleaned_doc/{artifact_id}.json"

        artifact = IntermediateArtifact(
            id=artifact_id,
            workspace_id="default",  # TODO: 从上下文获取
            source_id=str(source_file_id),
            type=IntermediateType.CLEANED_DOC,
            storage_path=relative_path,
            deletable=True,
            extra_metadata=json.dumps(
                {
                    "batch_id": batch_id,
                    "original_chars": stats.original_chars,
                    "cleaned_chars": stats.cleaned_chars,
                },
                ensure_ascii=False,
            ),
        )
        self.artifact_repository.create(artifact)

        return relative_path

    def _write_artifact_file(
        self, output_path: str, artifact: CleanedDocumentArtifact
    ) -> None:
        """写入产物文件。"""
        base_path = Path("data") / output_path
        base_path.parent.mkdir(parents=True, exist_ok=True)

        with open(base_path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2, ensure_ascii=False))


class ChartExtractionService:
    """图表提取与转换服务（T032）。

    从 MinerU 输出中提取图表，调用多模态模型将图表转为 JSON。
    """

    def __init__(
        self,
        llm_runtime: LLMRuntimeService,
        artifact_repository: IntermediateArtifactRepository,
        prompt_service: PromptService,
    ):
        self.llm_runtime = llm_runtime
        self.artifact_repository = artifact_repository
        self.prompt_service = prompt_service

    async def extract_charts(
        self,
        mineru_output: dict[str, Any],
        source_file_id: int,
    ) -> ChartExtractionResult:
        """提取图表并转换为 JSON。

        Args:
            mineru_output: MinerU 输出结果
            source_file_id: 源文件 ID

        Returns:
            图表提取结果
        """
        try:
            logger.info(f"开始提取图表 source_file_id={source_file_id}")

            # 从 MinerU 输出中提取图表信息
            charts = await self._extract_from_mineru(mineru_output, source_file_id)

            if not charts:
                logger.info(f"未发现图表 source_file_id={source_file_id}")
                return ChartExtractionResult(
                    success=True,
                    charts=[],
                    error=None,
                )

            # 调用多模态模型转换为 JSON
            json_charts = []
            for chart in charts:
                chart_json = await self.chart_to_json(
                    chart_image_path=chart["image_path"],
                    chart_type=chart["type"],
                )
                if chart_json:
                    json_charts.append(chart_json)

            # 保存产物
            output_path = self._save_artifact(
                source_file_id=source_file_id,
                charts=json_charts,
            )

            # 写入文件
            artifact = ChartJSONArtifact(
                source_file_id=source_file_id,
                charts=json_charts,
                output_path=output_path,
            )
            self._write_artifact_file(output_path, artifact)

            logger.info(
                f"图表提取完成 source_file_id={source_file_id}, "
                f"提取图表数={len(json_charts)}"
            )

            return ChartExtractionResult(
                success=True,
                charts=json_charts,
                error=None,
            )

        except Exception as e:
            logger.error(f"图表提取失败 source_file_id={source_file_id}: {e}")
            return ChartExtractionResult(
                success=False,
                charts=None,
                error=str(e),
            )

    async def _extract_from_mineru(
        self, mineru_output: dict[str, Any], source_file_id: int
    ) -> list[dict[str, Any]]:
        """从 MinerU 输出中提取图表信息。"""
        charts = []

        # MinerU 输出结构：
        # - images: 页面图片列表
        # - markdown: 文本内容
        # - pdf_structure: PDF 结构信息

        images = mineru_output.get("images", [])
        pdf_structure = mineru_output.get("pdf_structure", {})

        # 从 images 中提取图表
        for idx, image_info in enumerate(images):
            image_path = image_info.get("path", "")
            image_type = image_info.get("type", "")

            # 判断是否为图表类型
            if self._is_chart_image(image_info):
                charts.append(
                    {
                        "chart_id": f"chart_{source_file_id}_{idx}",
                        "image_path": image_path,
                        "type": self._detect_chart_type(image_info),
                        "page_number": image_info.get("page", idx + 1),
                    }
                )

        # 从 pdf_structure 中提取表格和图表
        for element in pdf_structure.get("elements", []):
            if element.get("type") in ["table", "figure", "chart"]:
                charts.append(
                    {
                        "chart_id": f"chart_{source_file_id}_{element.get('index', 0)}",
                        "image_path": element.get("image_path", ""),
                        "type": element.get("type", "other"),
                        "page_number": element.get("page", 1),
                    }
                )

        return charts

    def _is_chart_image(self, image_info: dict[str, Any]) -> bool:
        """判断图片是否为图表。"""
        image_type = image_info.get("type", "").lower()
        chart_indicators = ["chart", "figure", "graph", "table", "plot"]

        # 检查类型字段
        if image_type in chart_indicators:
            return True

        # 检查文件名
        image_path = image_info.get("path", "").lower()
        if any(indicator in image_path for indicator in chart_indicators):
            return True

        # 检查宽度高度比例（图表通常有特定的宽高比）
        width = image_info.get("width", 0)
        height = image_info.get("height", 0)
        if width > 0 and height > 0:
            ratio = width / height
            # 图表通常不是正方形或极窄长
            if 0.3 < ratio < 5:
                return True

        return False

    def _detect_chart_type(self, image_info: dict[str, Any]) -> ChartType:
        """检测图表类型。"""
        image_type = image_info.get("type", "").lower()
        image_path = image_info.get("path", "").lower()

        if "table" in image_type or "table" in image_path:
            return ChartType.TABLE

        if "bar" in image_type or "bar" in image_path:
            return ChartType.BAR

        if "line" in image_type or "line" in image_path:
            return ChartType.LINE

        if "pie" in image_type or "pie" in image_path:
            return ChartType.PIE

        if "scatter" in image_type or "scatter" in image_path:
            return ChartType.SCATTER

        if "area" in image_type or "area" in image_path:
            return ChartType.AREA

        if "radar" in image_type or "radar" in image_path:
            return ChartType.RADAR

        if "heatmap" in image_type or "heatmap" in image_path:
            return ChartType.HEATMAP

        return ChartType.OTHER

    async def chart_to_json(
        self,
        chart_image_path: str,
        chart_type: str,
        *,
        strict_json: bool = False,
    ) -> ChartData | None:
        """图表转 JSON。

        调用多模态模型将图表图片转换为结构化 JSON。

        Args:
            chart_image_path: 图表图片路径
            chart_type: 图表类型

        Returns:
            图表 JSON 数据
        """
        try:
            # 检查文件是否存在
            if not Path(chart_image_path).exists():
                logger.warning(f"图表文件不存在: {chart_image_path}")
                return None

            # 构建提示词
            prompt_text = self._build_chart_prompt(chart_type)

            # 获取模型实例
            llm = self.llm_runtime.get_model_for_callsite(SCOPE_CHART_EXTRACTION)

            # 构建链
            # 注意：多模态模型通常需要特定的 Prompt 结构来传递图片
            # 这里假设 LLMRuntime 返回的 ChatProtocol 支持 standard LangChain message format
            from langchain_core.messages import HumanMessage

            # 构造多模态消息
            # LangChain 标准格式：content=[{"type": "text", ...}, {"type": "image_url", ...}]
            # 但具体的 LLM 实现可能有所不同，这里假设 LLMRuntimeService 会处理好适配
            # 实际上，如果 LLM 是 ChatOpenAI，它支持 base64 或 url
            # 为了简单起见，我们这里使用 LLMRuntimeService 提供的 build_runnable_for_capability
            # 但是它绑定了 Prompt... 且多模态的 Prompt 处理比较复杂
            
            # 回退方案：如果 capability 为 chart_ocr，我们期望 LLMRuntimeService 能正确处理
            # 但是我们的 prompt 是动态的 (chart_type)。
            
            # 我们手动构建多模态消息（base64 data URL）
            with open(chart_image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

            mime_type = mimetypes.guess_type(chart_image_path)[0] or "image/jpeg"
            if not mime_type.startswith("image/"):
                mime_type = "image/jpeg"
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_string}"
                        },
                    },
                ]
            )
            
            # 兼容性检查：确保模型支持 async invoke
            if not hasattr(llm, "ainvoke"):
                raise RuntimeError(
                    f"多模态调用点返回的模型不支持 ainvoke: {type(llm).__name__}"
                )

            try:
                result = await llm.ainvoke([message])
            except Exception as exc:
                # 严格模式：直接失败并暴露真实原因（例如 provider 不支持图像输入）
                raise RuntimeError(
                    f"多模态 LLM 调用失败: model={type(llm).__name__}, image={chart_image_path}, error={exc}"
                ) from exc
            
            # 解析结果
            json_content = self._parse_chart_result(
                result.content if hasattr(result, "content") else result,
                strict=strict_json,
            )

            return ChartData(
                chart_id=f"chart_{Path(chart_image_path).stem}",
                chart_type=ChartType(chart_type) if chart_type in [e.value for e in ChartType] else ChartType.OTHER,
                json_content=json_content,
                source_image_path=chart_image_path,
                confidence=0.9,  # TODO: 从模型响应中获取置信度
                page_number=None,
            )

        except Exception as e:
            logger.error(f"图表转 JSON 失败: {chart_image_path}, error: {e}")
            if strict_json:
                raise
            return None

    def _build_chart_prompt(self, chart_type: str) -> str:
        """构建图表识别提示词。"""
        type_descriptions = {
            "bar": "柱状图",
            "line": "折线图",
            "pie": "饼图",
            "table": "表格",
            "scatter": "散点图",
            "area": "面积图",
            "radar": "雷达图",
            "heatmap": "热力图",
            "other": "图表",
        }

        description = type_descriptions.get(chart_type, "图表")

        # 获取提示词模板
        prompt_entity = self.prompt_service.get_active_prompt(SCOPE_CHART_EXTRACTION)
        if prompt_entity and prompt_entity.content:
            template = prompt_entity.content
        else:
            template = DEFAULT_PROMPTS[SCOPE_CHART_EXTRACTION]["content"]

        # 渲染提示词
        prompt = template.replace("{description}", description).replace(
            "{chart_type}", chart_type
        )

        return prompt

    def _parse_chart_result(
        self, result: Any, *, strict: bool = False
    ) -> dict[str, Any]:
        """解析图表识别结果。"""
        if isinstance(result, str):
            try:
                # 清理可能的前缀/后缀
                raw_original = result
                raw = result.strip()
                if raw.startswith("```json"):
                    raw = raw[7:]
                if raw.startswith("```"):
                    raw = raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

                if not raw:
                    raise json.JSONDecodeError(
                        "Empty response after stripping fences", raw, 0
                    )

                return json.loads(raw)
            except json.JSONDecodeError as exc:
                if strict:
                    # 尝试从混杂文本中提取首个 JSON 块（常见：前后有解释文字）
                    extracted = self._try_extract_json_block(raw_original)
                    if extracted is not None:
                        return json.loads(extracted)
                    raise
                # 如果解析失败，返回原始文本
                return {"raw_text": result, "error": f"JSON 解析失败: {exc}"}

        if isinstance(result, dict):
            return result

        return {"raw_result": str(result)}

    def _try_extract_json_block(self, text: str | None) -> str | None:
        """从任意文本中提取第一个可能的 JSON 对象/数组字符串（用于严格解析前的最后尝试）。"""
        if not text:
            return None
        s = text.strip()
        if not s:
            return None

        # 去掉 markdown code fences（保留内部内容）
        if s.startswith("```json"):
            s = s[7:]
        if s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

        # 寻找 JSON 起始符
        start_candidates = [i for i in (s.find("{"), s.find("[")) if i != -1]
        if not start_candidates:
            return None
        start = min(start_candidates)

        opener = s[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1].strip()
                    return candidate if candidate else None
            # 兼容嵌套对象/数组
            elif ch in "{[":
                # 进入嵌套：以其自身 opener/closer 计算，这里用统一 depth 处理可能不严谨，
                # 但对于“首个完整块”的提取足够使用（后续 json.loads 会再次校验）
                pass
        return None

    def _save_artifact(
        self,
        source_file_id: int,
        charts: list[ChartData],
    ) -> str:
        """保存中间产物到数据库。

        Returns:
            相对存储路径
        """
        from datetime import datetime
        import uuid

        from src.domain.entities.intermediate_artifact import IntermediateArtifact

        batch_id = datetime.now().strftime("%Y%m%d")
        artifact_id = str(uuid.uuid4())
        relative_path = f"intermediates/{source_file_id}/chart_json/{artifact_id}.json"

        # 保存到数据库
        artifact = IntermediateArtifact(
            id=artifact_id,
            workspace_id="default",  # TODO: 从上下文获取
            source_id=str(source_file_id),
            type=IntermediateType.CHART_JSON,
            storage_path=relative_path,
            deletable=True,
            extra_metadata=json.dumps(
                {"batch_id": batch_id, "chart_count": len(charts)},
                ensure_ascii=False,
            ),
        )
        self.artifact_repository.create(artifact)

        return relative_path

    def _write_artifact_file(
        self, output_path: str, artifact: ChartJSONArtifact
    ) -> None:
        """写入产物文件。"""
        base_path = Path("data") / output_path
        base_path.parent.mkdir(parents=True, exist_ok=True)

        with open(base_path, "w", encoding="utf-8") as f:
            f.write(artifact.model_dump_json(indent=2, ensure_ascii=False))

    # =========================
    # T094: intermediates_cleaned -> pic_to_json workspace pipeline
    # =========================

    _MD_IMAGE_PATH_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")

    def _normalize_ref_path(self, raw: str) -> str:
        s = (raw or "").strip().strip('"').strip("'")
        # 去掉 query/hash，避免误判
        s = s.split("#", 1)[0].split("?", 1)[0].strip()
        s = s.replace("\\", "/")
        if s.startswith("./"):
            s = s[2:]
        return s

    def _resolve_ref_to_abs(self, base_dir: Path, ref: str) -> Path | None:
        """将 markdown/json 中的路径引用解析为绝对路径（用于与 removed 集合对齐）。"""
        ref_norm = self._normalize_ref_path(ref)
        if not ref_norm:
            return None
        lower = ref_norm.lower()
        if lower.startswith(("http://", "https://", "data:")):
            return None
        p = Path(ref_norm)
        try:
            if p.is_absolute():
                return p.resolve()
            return (base_dir / p).resolve()
        except OSError:
            return None

    def _extract_is_chart_strict(self, json_obj: dict[str, Any], *, image: Path) -> bool:
        if not isinstance(json_obj, dict):
            raise RuntimeError(f"图转 JSON 输出不是对象: image={image}")
        if "is_chart" not in json_obj:
            raise RuntimeError(f"图转 JSON 输出缺少 is_chart: image={image}")
        v = json_obj.get("is_chart")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in {"true", "yes", "y", "1"}:
                return True
            if vv in {"false", "no", "n", "0"}:
                return False
        raise RuntimeError(f"is_chart 字段类型非法: image={image}, value={v!r}")

    def _prune_json_obj(
        self,
        obj: Any,
        *,
        base_dir: Path,
        removed_abs: set[Path],
    ) -> tuple[Any, bool]:
        """递归移除 JSON 中对 removed 图片的引用。

        设计目标：不依赖固定 schema，尽可能小范围删除字段/列表项，同时确保不残留引用。
        """
        if isinstance(obj, str):
            abs_path = self._resolve_ref_to_abs(base_dir, obj)
            if abs_path is not None and abs_path in removed_abs:
                return None, True
            return obj, False

        if isinstance(obj, list):
            changed = False
            new_list: list[Any] = []
            for item in obj:
                # 列表元素若“包含”引用，倾向删除整个元素（常见：images 列表/元素对象）
                if self._json_contains_removed(item, base_dir=base_dir, removed_abs=removed_abs):
                    changed = True
                    continue
                new_item, item_changed = self._prune_json_obj(
                    item, base_dir=base_dir, removed_abs=removed_abs
                )
                if item_changed:
                    changed = True
                new_list.append(new_item)
            return new_list, changed

        if isinstance(obj, dict):
            changed = False
            new_dict: dict[str, Any] = {}
            for k, v in obj.items():
                if isinstance(v, str):
                    abs_path = self._resolve_ref_to_abs(base_dir, v)
                    if abs_path is not None and abs_path in removed_abs:
                        changed = True
                        continue
                    new_dict[k] = v
                    continue

                if isinstance(v, (dict, list)):
                    # 若该字段的值整体是“图片对象/列表项”，这里不直接删除整个字段，
                    # 而是递归剔除内部引用（列表内部会删除引用元素）。
                    new_v, v_changed = self._prune_json_obj(
                        v, base_dir=base_dir, removed_abs=removed_abs
                    )
                    if v_changed:
                        changed = True
                    new_dict[k] = new_v
                    continue

                new_dict[k] = v
            return new_dict, changed

        return obj, False

    def _json_contains_removed(
        self, obj: Any, *, base_dir: Path, removed_abs: set[Path]
    ) -> bool:
        """判断任意 JSON 结构是否包含对 removed 图片的引用。"""
        if isinstance(obj, str):
            abs_path = self._resolve_ref_to_abs(base_dir, obj)
            return abs_path is not None and abs_path in removed_abs
        if isinstance(obj, list):
            return any(
                self._json_contains_removed(x, base_dir=base_dir, removed_abs=removed_abs)
                for x in obj
            )
        if isinstance(obj, dict):
            return any(
                self._json_contains_removed(v, base_dir=base_dir, removed_abs=removed_abs)
                for v in obj.values()
            )
        return False

    def _remove_md_refs(self, md_text: str, *, md_dir: Path, removed_abs: set[Path]) -> tuple[str, int]:
        removed_count = 0

        def _repl(m: re.Match) -> str:
            nonlocal removed_count
            raw = m.group(1)
            abs_path = self._resolve_ref_to_abs(md_dir, raw)
            if abs_path is not None and abs_path in removed_abs:
                removed_count += 1
                return ""
            return m.group(0)

        new_text = self._MD_IMAGE_PATH_RE.sub(_repl, md_text or "")
        return new_text, removed_count

    def _extract_markdown_image_context(
        self, *, md_text: str, image_rel_posix: str
    ) -> tuple[str | None, str | None]:
        """从 Markdown 中提取指定图片的上下文（alt 文本 + 最近标题）。

        Args:
            md_text: Markdown 全文
            image_rel_posix: 形如 images/xxx.jpg 的相对路径（posix）
        """
        if not md_text:
            return None, None

        # 统一换行
        lines = md_text.replace("\r\n", "\n").split("\n")

        # 逐行寻找图片引用
        image_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        target = self._normalize_ref_path(image_rel_posix)

        for idx, line in enumerate(lines):
            m = image_re.search(line)
            if not m:
                continue
            alt = (m.group(1) or "").strip()
            ref = self._normalize_ref_path(m.group(2))
            if ref != target:
                continue

            # 向上找最近标题
            heading = None
            for j in range(idx - 1, -1, -1):
                s = (lines[j] or "").strip()
                if not s:
                    continue
                if s.startswith("#"):
                    heading = s.lstrip("#").strip()
                    break
            return alt or None, heading or None

        return None, None

    def _extract_layout_image_name(self, *, doc_dir: Path, image_filename: str) -> str | None:
        """从 layout.json 中严格提取指定图片的标题（按 MinerU layout 结构）。

        依据实例结构：
        - pdf_info[].preproc_blocks[] 中存在 type=="image" 的块；
        - 该块包含 blocks[]，其中：
          - image_body: spans[].image_path == <filename>
          - image_caption / image_title: spans[].content == 标题
        """
        layout_path = doc_dir / "layout.json"
        if not layout_path.exists():
            return None

        try:
            data = json.loads(layout_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        def _collect_span_contents(block: dict[str, Any]) -> str:
            # 某些 layout 可能直接在块级别给 content
            v = block.get("content")
            if isinstance(v, str) and v.strip():
                return v.strip()

            parts: list[str] = []
            lines = block.get("lines")
            if isinstance(lines, list):
                for line in lines:
                    if not isinstance(line, dict):
                        continue
                    spans = line.get("spans")
                    if not isinstance(spans, list):
                        continue
                    for sp in spans:
                        if not isinstance(sp, dict):
                            continue
                        # 实例里使用 content；兼容部分输出使用 text
                        c = sp.get("content")
                        if isinstance(c, str) and c.strip():
                            parts.append(c.strip())
                            continue
                        t = sp.get("text")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
            return " ".join(parts).strip()

        def _image_body_has_path(block: dict[str, Any]) -> tuple[bool, int | None]:
            if block.get("type") != "image_body":
                return False, None
            group_id = block.get("group_id")
            group_id_int = int(group_id) if isinstance(group_id, (int, float)) else None
            lines = block.get("lines")
            if not isinstance(lines, list):
                return False, group_id_int
            for line in lines:
                if not isinstance(line, dict):
                    continue
                spans = line.get("spans")
                if not isinstance(spans, list):
                    continue
                for sp in spans:
                    if not isinstance(sp, dict):
                        continue
                    ip = sp.get("image_path")
                    if isinstance(ip, str) and ip.strip() == image_filename:
                        return True, group_id_int
            return False, group_id_int

        pdf_info = data.get("pdf_info")
        if not isinstance(pdf_info, list):
            return None

        for page in pdf_info:
            if not isinstance(page, dict):
                continue
            blocks = page.get("preproc_blocks")
            if not isinstance(blocks, list):
                continue

            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") != "image":
                    continue
                sub_blocks = b.get("blocks")
                if not isinstance(sub_blocks, list):
                    continue

                # 先确认该 image 块是否对应目标图片，并拿到 group_id
                matched = False
                matched_group: int | None = None
                for sb in sub_blocks:
                    if not isinstance(sb, dict):
                        continue
                    ok, gid = _image_body_has_path(sb)
                    if ok:
                        matched = True
                        matched_group = gid
                        break
                if not matched:
                    continue

                # 严格按结构取 caption（同 group_id 优先）
                captions: list[str] = []
                for sb in sub_blocks:
                    if not isinstance(sb, dict):
                        continue
                    sb_type = sb.get("type")
                    if sb_type not in {"image_caption", "image_title"}:
                        continue
                    gid = sb.get("group_id")
                    gid_int = int(gid) if isinstance(gid, (int, float)) else None
                    if matched_group is not None and gid_int is not None and gid_int != matched_group:
                        continue
                    txt = _collect_span_contents(sb)
                    if txt:
                        captions.append(txt)

                if captions:
                    # 多段 caption 合并（去重）
                    seen: set[str] = set()
                    deduped: list[str] = []
                    for t in captions:
                        if t in seen:
                            continue
                        seen.add(t)
                        deduped.append(t)
                    return " ".join(deduped).strip() or None

                return None

        return None

    def _derive_chart_name(
        self,
        *,
        model_json: dict[str, Any],
        doc_dir: Path,
        image_rel_posix: str,
        per_doc_counter: dict[str, int],
    ) -> str:
        """生成用于 RAG 展示的图表名称（避免 uuid）。"""
        image_filename = Path(image_rel_posix).name

        # 1) 优先：从 layout.json 提取该图片的原始名称/标题
        layout_name = self._extract_layout_image_name(
            doc_dir=doc_dir, image_filename=image_filename
        )
        if layout_name:
            return layout_name

        # 2) 若 layout 为空：由 LLM 给出有意义名称（优先 description/title 等字段）
        for k in ("description", "title", "chart_title", "name"):
            v = model_json.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # 3) 次选：从 Markdown 提取 alt/标题（更贴近文档上下文）
        md_files = sorted(doc_dir.glob("*.md"))
        for md_path in md_files:
            try:
                md_text = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            alt, heading = self._extract_markdown_image_context(
                md_text=md_text, image_rel_posix=image_rel_posix
            )
            if alt:
                return alt
            if heading:
                return heading

        # 4) 兜底：不要带“图1/图2...”编号（避免 KB/RAG 召回污染）
        chart_type = model_json.get("chart_type")
        if isinstance(chart_type, str) and chart_type.strip():
            return f"图表（{chart_type.strip()}）"
        return "图表"

    def _validate_no_deleted_refs_in_md(self, md_text: str, *, md_dir: Path, removed_abs: set[Path]) -> None:
        for m in self._MD_IMAGE_PATH_RE.finditer(md_text or ""):
            abs_path = self._resolve_ref_to_abs(md_dir, m.group(1))
            if abs_path is not None and abs_path in removed_abs:
                raise RuntimeError(f"Markdown 仍引用已删除图片: md_dir={md_dir}, ref={m.group(1)}")

    def _validate_no_deleted_refs_in_json(self, obj: Any, *, base_dir: Path, removed_abs: set[Path]) -> None:
        if isinstance(obj, str):
            abs_path = self._resolve_ref_to_abs(base_dir, obj)
            if abs_path is not None and abs_path in removed_abs:
                raise RuntimeError(f"JSON 仍引用已删除图片: base_dir={base_dir}, ref={obj}")
            return
        if isinstance(obj, list):
            for x in obj:
                self._validate_no_deleted_refs_in_json(x, base_dir=base_dir, removed_abs=removed_abs)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                self._validate_no_deleted_refs_in_json(v, base_dir=base_dir, removed_abs=removed_abs)
            return

    async def run_t094_pic_to_json(
        self,
        *,
        input_root: str | Path = Path("data") / "intermediates_cleaned",
        output_root: str | Path = Path("data") / "pic_to_json",
        chart_json_dirname: str = "chart_json",
        max_images: int | None = None,
        concurrency: int = 1,
        strict: bool = True,
        progress_callback: Any | None = None,
        progress_every: int = 10,
        resume: bool = False,
    ) -> dict[str, Any]:
        """T094: 在 `data/pic_to_json/` 下完成复制/图转JSON/删非图表/清理引用。

        关键约束：
        - 先完整复制 `data/intermediates_cleaned/` 到 `data/pic_to_json/workspace/` 再处理
        - 仅使用 SQLite 中配置的多模态 LLM（callsite: chart_extraction:extract_json）
        - 不使用 mock；strict=True 时遇到任何解析问题直接失败（不降级）
        """
        from datetime import datetime

        input_root_p = Path(input_root).resolve()
        output_root_p = Path(output_root).resolve()
        # 新方案：不再创建 workspace/results 两层目录。
        # 直接将 input_root 完整复制到 output_root，后续所有处理都在 output_root 内完成。
        workspace_root = output_root_p

        if not input_root_p.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_root_p}")

        # 严格模式下，必须从 SQLite 获取激活 Prompt（不允许静默回退到代码 Seed）
        if strict:
            active_prompt = self.prompt_service.get_active_prompt(SCOPE_CHART_EXTRACTION)
            if (
                active_prompt is None
                or not getattr(active_prompt, "content", None)
                or not str(active_prompt.content).strip()
            ):
                raise RuntimeError(
                    f"未找到激活提示词: {SCOPE_CHART_EXTRACTION}（请先在 SQLite prompts 中发布激活版本，或运行 scripts/update-chart-extraction-prompt.py）"
                )

        # 1) 复制（先完整复制再处理）
        # 说明：output_root 作为“工作目录快照”，因此每次运行会覆盖重建（避免旧结果污染）。
        t0 = time.monotonic()
        
        should_copy = True
        if resume and output_root_p.exists():
            should_copy = False
            if callable(progress_callback):
                progress_callback(
                    {
                        "stage": "copy_start",
                        "ts": datetime.now().isoformat(),
                        "input_root": str(input_root_p),
                        "output_root": str(output_root_p),
                        "msg": "Resuming mode: skipping copy",
                    }
                )
        
        if should_copy:
            if callable(progress_callback):
                progress_callback(
                    {
                        "stage": "copy_start",
                        "ts": datetime.now().isoformat(),
                        "input_root": str(input_root_p),
                        "output_root": str(output_root_p),
                    }
                )
            if output_root_p.exists():
                shutil.rmtree(output_root_p)
            shutil.copytree(input_root_p, output_root_p, dirs_exist_ok=False)
            if callable(progress_callback):
                progress_callback(
                    {
                        "stage": "copy_done",
                        "ts": datetime.now().isoformat(),
                        "duration_s": round(time.monotonic() - t0, 3),
                    }
                )

        # 2) 扫描 workspace 中的 images
        t1 = time.monotonic()
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        images: list[Path] = []
        for p in workspace_root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            if not any(part.lower() == "images" for part in p.parts):
                continue
            images.append(p)

        images.sort()
        if max_images is not None and max_images > 0:
            images = images[: int(max_images)]
        if not images:
            raise RuntimeError(f"workspace 中未找到任何 images 图片: {workspace_root}")
        if callable(progress_callback):
            progress_callback(
                {
                    "stage": "scan_done",
                    "ts": datetime.now().isoformat(),
                    "duration_s": round(time.monotonic() - t1, 3),
                    "total_images": len(images),
                }
            )

        # ===== 2.5) 断点续跑状态文件（JSONL）=====
        # 目的：让 --resume 不仅能跳过“已生成 chart_json 的图”，也能跳过“已判定为非图表/错误的图”。
        state_path = (workspace_root / "t094_state.jsonl").resolve()

        def _image_sig(p: Path) -> dict[str, int] | None:
            try:
                st = p.stat()
                return {"size": int(st.st_size), "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))}
            except OSError:
                return None

        def _load_state_latest(path: Path) -> dict[str, dict[str, Any]]:
            if not path.exists():
                return {}
            latest: dict[str, dict[str, Any]] = {}
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    src = obj.get("source_image")
                    if isinstance(src, str) and src.strip():
                        latest[src] = obj
            except OSError:
                return {}
            return latest

        async def _append_state(path: Path, record: dict[str, Any], *, lock: asyncio.Lock) -> None:
            # append-only，抗中断；多并发下用 lock 防止写入交错
            line = json.dumps(record, ensure_ascii=False)
            async with lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

        state_latest: dict[str, dict[str, Any]] = _load_state_latest(state_path) if resume else {}
        state_lock = asyncio.Lock()

        # 如果 --resume 且没有 state 文件，则先用现有 chart_json 进行最小回填（至少能跳过已落盘的图表）
        if resume and not state_path.exists():
            for jf in sorted(workspace_root.rglob(f"{chart_json_dirname}/*.json")):
                try:
                    obj = json.loads(jf.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                src = obj.get("_source_image")
                if not isinstance(src, str) or not src.strip():
                    continue
                img_abs = (workspace_root / Path(src)).resolve()
                sig = _image_sig(img_abs)
                rec = {
                    "ts": datetime.now().isoformat(),
                    "source_image": src,
                    "status": "chart",
                    "image_sig": sig,
                    "chart_json": jf.relative_to(workspace_root).as_posix(),
                    "chart_id": obj.get("_chart_id"),
                    "chart_name": obj.get("_chart_name"),
                    "chart_type": obj.get("chart_type"),
                }
                await _append_state(state_path, rec, lock=state_lock)
                state_latest[src] = rec

        # 3) 批量图转 JSON（严格模式下要求输出可解析 JSON 且包含 is_chart）
        started_at = datetime.now().isoformat()
        items: list[dict[str, Any]] = []
        non_chart_images: list[Path] = []
        per_doc_counter: dict[str, int] = {}

        semaphore = asyncio.Semaphore(max(1, int(concurrency)))

        async def _one(image_path: Path) -> dict[str, Any]:
            async with semaphore:
                rel_image = image_path.relative_to(workspace_root).as_posix()
                # 结果不写入 images/，而是写入同文档目录下的 chart_json/
                # 例如：<doc_dir>/images/a.jpg -> <doc_dir>/chart_json/a.json
                if image_path.parent.name.lower() != "images":
                    raise RuntimeError(f"图片不在 images 目录下: {image_path}")
                doc_dir = image_path.parent.parent
                chart_json_dir = doc_dir / chart_json_dirname
                chart_json_dir.mkdir(parents=True, exist_ok=True)
                out_json_path = (chart_json_dir / f"{image_path.stem}.json").resolve()

                # Resume 逻辑（JSONL）：chart / non_chart / error 都可跳过
                if resume:
                    # 安全加固：若 chart_json 文件已存在，优先视为 chart，避免 state 误标 non_chart 导致误删
                    if out_json_path.exists():
                        try:
                            model_json = json.loads(out_json_path.read_text(encoding="utf-8"))
                            return {
                                "source_image": rel_image,
                                "result_json": out_json_path.relative_to(workspace_root).as_posix(),
                                "is_chart": True,
                                "status": "chart",
                                "should_delete": False,
                                "chart_id": model_json.get("_chart_id"),
                                "chart_name": model_json.get("_chart_name"),
                            }
                        except Exception:
                            # JSON 已损坏：删除后走正常 LLM 流程重建
                            try:
                                out_json_path.unlink()
                            except OSError:
                                pass
                    sig_now = _image_sig(image_path)
                    last = state_latest.get(rel_image)
                    if isinstance(last, dict) and last.get("image_sig") == sig_now:
                        status = last.get("status")
                        if status == "chart":
                            # 以 state 中的 chart_json 为准；若缺失则尝试默认 out_json_path
                            rel_json = last.get("chart_json") or out_json_path.relative_to(workspace_root).as_posix()
                            if rel_json and (workspace_root / Path(rel_json)).exists():
                                return {
                                    "source_image": rel_image,
                                    "result_json": rel_json,
                                    "is_chart": True,
                                    "status": "chart",
                                    "should_delete": False,
                                    "chart_id": last.get("chart_id"),
                                    "chart_name": last.get("chart_name"),
                                }
                        elif status == "non_chart":
                            return {
                                "source_image": rel_image,
                                "result_json": None,
                                "is_chart": False,
                                "status": "non_chart",
                                "should_delete": True,
                                "chart_id": None,
                                "chart_name": None,
                            }
                        elif status == "error":
                            # 用户选择：error 也跳过（不删除）
                            return {
                                "source_image": rel_image,
                                "result_json": None,
                                "is_chart": False,
                                "status": "error",
                                "should_delete": False,
                                "error": last.get("error"),
                                "chart_id": None,
                                "chart_name": None,
                            }

                chart_data = await self.chart_to_json(
                    chart_image_path=str(image_path),
                    chart_type="other",
                    strict_json=strict,
                )
                if chart_data is None:
                    raise RuntimeError(f"图转 JSON 返回空: {image_path}")
                model_json = chart_data.json_content
                is_chart = (
                    self._extract_is_chart_strict(model_json, image=image_path)
                    if strict
                    else bool(model_json.get("is_chart"))
                )

                # 生成稳定 chart_id（避免使用图片文件名/uuid）
                import hashlib

                chart_id = "chart_" + hashlib.sha1(rel_image.encode("utf-8")).hexdigest()[:12]

                # 生成 RAG 展示用的图表名称
                chart_name = self._derive_chart_name(
                    model_json=model_json,
                    doc_dir=doc_dir,
                    image_rel_posix=f"images/{image_path.name}",
                    per_doc_counter=per_doc_counter,
                )

                # 将元数据写入 JSON（不破坏原字段，供 KB/RAG 直接使用）
                model_json = {
                    **model_json,
                    "_chart_id": chart_id,
                    "_chart_name": chart_name,
                    "_source_image": rel_image,
                }

                # 仅对图表落盘 JSON；非图表不落盘，避免 KB 误收录
                if is_chart:
                    out_json_path.write_text(
                        json.dumps(model_json, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    result_json_rel = out_json_path.relative_to(workspace_root).as_posix()
                    status = "chart"
                    should_delete = False
                else:
                    # 确保没有残留旧文件
                    if out_json_path.exists():
                        out_json_path.unlink()
                    result_json_rel = None
                    status = "non_chart"
                    should_delete = True

                # 写入 JSONL 状态（append-only）
                sig_now = _image_sig(image_path)
                rec = {
                    "ts": datetime.now().isoformat(),
                    "source_image": rel_image,
                    "status": status,
                    "image_sig": sig_now,
                    "chart_json": result_json_rel,
                    "chart_id": chart_id,
                    "chart_name": chart_name if is_chart else None,
                    "chart_type": model_json.get("chart_type") if isinstance(model_json, dict) else None,
                }
                await _append_state(state_path, rec, lock=state_lock)
                state_latest[rel_image] = rec

                return {
                    "source_image": rel_image,
                    "result_json": result_json_rel,
                    "is_chart": is_chart,
                    "status": status,
                    "should_delete": should_delete,
                    "chart_id": chart_id,
                    "chart_name": chart_name,
                }

        # 并发收集（并发上限由 semaphore 控制）；使用 as_completed 以便实时输出进度
        t2 = time.monotonic()
        async def _one_with_state(img: Path) -> dict[str, Any]:
            try:
                return await _one(img)
            except Exception as exc:
                # 记录错误到 JSONL，便于 --resume 直接跳过该图
                rel_image = img.relative_to(workspace_root).as_posix()
                sig_now = _image_sig(img)
                rec = {
                    "ts": datetime.now().isoformat(),
                    "source_image": rel_image,
                    "status": "error",
                    "image_sig": sig_now,
                    "error": str(exc),
                }
                try:
                    await _append_state(state_path, rec, lock=state_lock)
                    state_latest[rel_image] = rec
                except Exception:
                    # 记录失败不应覆盖原始异常语义
                    pass
                raise

        tasks = [asyncio.create_task(_one_with_state(img)) for img in images]
        done_count = 0
        try:
            for fut in asyncio.as_completed(tasks):
                row = await fut
                items.append(row)
                done_count += 1
                if row.get("should_delete"):
                    non_chart_images.append(workspace_root / Path(row["source_image"]))
                if callable(progress_callback) and (
                    done_count == 1
                    or done_count % max(1, int(progress_every)) == 0
                    or done_count == len(images)
                ):
                    progress_callback(
                        {
                            "stage": "chart_to_json_progress",
                            "ts": datetime.now().isoformat(),
                            "done": done_count,
                            "total": len(images),
                            "last_image": row.get("source_image"),
                            "last_is_chart": row.get("is_chart"),
                            "last_status": row.get("status"),
                            "elapsed_s": round(time.monotonic() - t2, 3),
                        }
                    )
                
                # 周期性卸载 Ollama 模型以释放显存 (每 60 张)
                # 注意：这可能会稍微影响性能（重新加载需要几秒），但能防止显存耗尽卡死
                if done_count % 60 == 0:
                    unloaded = await self.llm_runtime.unload_model_if_ollama(SCOPE_CHART_EXTRACTION)
                    if unloaded:
                        if callable(progress_callback):
                            progress_callback(
                                {
                                    "stage": "ollama_unload",
                                    "ts": datetime.now().isoformat(),
                                    "msg": "Unloaded Ollama model to free GPU memory",
                                }
                            )
                        # 给一点时间让系统回收资源
                        await asyncio.sleep(2.0)
        except Exception:
            # 若任一任务失败，取消剩余任务，保证严格失败语义
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise

        removed_abs: set[Path] = {p.resolve() for p in non_chart_images}

        # 4) 删除非图表图片（仅在 workspace）
        t3 = time.monotonic()
        deleted_count = 0
        for p in non_chart_images:
            try:
                if p.exists():
                    p.unlink()
                    deleted_count += 1
            except OSError as exc:
                raise RuntimeError(f"删除图片失败: {p}: {exc}") from exc
        if callable(progress_callback):
            progress_callback(
                {
                    "stage": "delete_non_charts_done",
                    "ts": datetime.now().isoformat(),
                    "duration_s": round(time.monotonic() - t3, 3),
                    "deleted_images": deleted_count,
                }
            )

        # 5) 清理 md / json 引用（仅在 workspace）
        t4 = time.monotonic()
        md_changed = 0
        md_refs_removed = 0
        for md_path in sorted(workspace_root.rglob("*.md")):
            original = md_path.read_text(encoding="utf-8")
            updated, removed_n = self._remove_md_refs(
                original, md_dir=md_path.parent, removed_abs=removed_abs
            )
            if removed_n > 0 and updated != original:
                md_path.write_text(updated, encoding="utf-8")
                md_changed += 1
                md_refs_removed += removed_n

        json_changed = 0
        json_refs_removed = 0
        for jf in sorted(workspace_root.rglob("*.json")):
            raw = jf.read_text(encoding="utf-8")
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSON 文件解析失败: {jf}: {exc}") from exc

            new_obj, changed = self._prune_json_obj(
                obj, base_dir=jf.parent, removed_abs=removed_abs
            )
            if changed:
                jf.write_text(
                    json.dumps(new_obj, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                json_changed += 1
                json_refs_removed += 1  # 粗略计数：至少发生过一次删除
        if callable(progress_callback):
            progress_callback(
                {
                    "stage": "prune_refs_done",
                    "ts": datetime.now().isoformat(),
                    "duration_s": round(time.monotonic() - t4, 3),
                    "md_files_changed": md_changed,
                    "md_refs_removed": md_refs_removed,
                    "json_files_changed": json_changed,
                }
            )

        # 6) 一致性校验：确保 md/json 不再引用已删除图片
        t5 = time.monotonic()
        for md_path in sorted(workspace_root.rglob("*.md")):
            self._validate_no_deleted_refs_in_md(
                md_path.read_text(encoding="utf-8"),
                md_dir=md_path.parent,
                removed_abs=removed_abs,
            )
        for jf in sorted(workspace_root.rglob("*.json")):
            obj = json.loads(jf.read_text(encoding="utf-8"))
            self._validate_no_deleted_refs_in_json(
                obj, base_dir=jf.parent, removed_abs=removed_abs
            )
        if callable(progress_callback):
            progress_callback(
                {
                    "stage": "validate_done",
                    "ts": datetime.now().isoformat(),
                    "duration_s": round(time.monotonic() - t5, 3),
                }
            )

        finished_at = datetime.now().isoformat()
        report = {
            "task": "T094",
            "input_root": str(input_root_p),
            "output_root": str(output_root_p),
            "workspace_root": str(workspace_root),
            "started_at": started_at,
            "finished_at": finished_at,
            "total_images_scanned": len(images),
            "total_results": len(items),
            "non_chart_images": [p.relative_to(workspace_root).as_posix() for p in non_chart_images],
            "deleted_images": deleted_count,
            "md_files_changed": md_changed,
            "md_refs_removed": md_refs_removed,
            "json_files_changed": json_changed,
            "json_refs_removed_estimate": json_refs_removed,
            "items": items,
        }
        (workspace_root / "run_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (workspace_root / "chart_index.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report
