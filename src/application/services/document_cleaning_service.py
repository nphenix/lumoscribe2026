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
import shutil
import re
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
    MD_IMAGE_TOKEN_EXTRACT_RE = re.compile(r"!\[([^\]]*)]\(([^)]+)\)")
    MD_TABLE_HTML_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
    MD_TABLE_BLOCK_RE = re.compile(
        r"(?:^\s*\|.+\|\s*\n\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\s*\|?\s*(?:\n\s*\|.*\|\s*)+)",
        re.MULTILINE,
    )
    MD_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\s*\|?\s*$")
    QR_CONTEXT_PATTERNS = [
        r"二维码",
        r"扫码",
        r"扫一扫",
        r"公众号",
        r"微信",
        r"qr\\s*code",
        r"qrcode",
    ]

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
        source_file_id: str,
        options: CleaningOptions | None = None,
        mineru_output_dir: Path | None = None,
    ) -> CleaningResult:
        """清洗文档，返回清洗后的内容和元数据。

        Args:
            text: 原始文本
            source_file_id: 源文件 ID
            options: 清洗选项
            mineru_output_dir: MinerU 输出目录（用于复制图片）

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
            llm_cleaned = await self.llm_clean(rule_cleaned, options, original_text=text)
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
            self._write_artifact_file(output_path, artifact, mineru_output_dir)

            logger.info(
                f"文档清洗完成 source_file_id={source_file_id}, "
                f"移除字符数={final_stats.removed_chars}"
            )
            try:
                logger.info(
                    "doc_cleaning_final",
                    extra={
                        "source_file_id": source_file_id,
                        "output_path": output_path,
                        "removed_chars": final_stats.removed_chars,
                    },
                )
            except Exception:
                pass

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
        source_file_id: str,
        options: CleaningOptions | None = None,
        mineru_output_dir: Path | None = None,
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
        async for evt in self.llm_clean_stream(rule_cleaned, options, original_text=text):
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
        self._write_artifact_file(output_path, artifact, mineru_output_dir)

        yield {"type": "stage", "stage": "llm_clean_done"}

        try:
            logger.info(
                "doc_cleaning_final",
                extra={
                    "source_file_id": source_file_id,
                    "output_path": output_path,
                    "removed_chars": final_stats.removed_chars,
                },
            )
        except Exception:
            pass

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

        def _extract_table_spans(src: str) -> list[tuple[int, int]]:
            spans: list[tuple[int, int]] = []
            try:
                from markdown_it import MarkdownIt
                md = MarkdownIt("gfm-like")
                tokens = md.parse(src)
                lines = src.splitlines(True)
                starts: list[int] = []
                pos = 0
                for ln in lines:
                    starts.append(pos)
                    pos += len(ln)
                open_stack: list[tuple[int, int]] = []
                for tk in tokens:
                    if tk.type == "table_open" and tk.map:
                        open_stack.append((tk.map[0], tk.map[1]))
                    elif tk.type == "table_close" and open_stack:
                        lb, le = open_stack.pop(0)
                        s = starts[lb] if lb < len(starts) else 0
                        e = (starts[le - 1] + len(lines[le - 1])) if (le - 1) < len(lines) else len(src)
                        spans.append((s, e))
            except Exception:
                pass
            for m in self.MD_TABLE_HTML_RE.finditer(src):
                spans.append((m.start(), m.end()))
            # 轻量级 Markdown 管道表回退识别
            lines = src.splitlines(True)
            starts: list[int] = []
            pos = 0
            for ln in lines:
                starts.append(pos)
                pos += len(ln)
            i = 0
            while i + 1 < len(lines):
                header = lines[i]
                sep = lines[i + 1]
                if ("|" in header) and self.MD_TABLE_SEP_RE.fullmatch(sep.strip()):
                    j = i + 2
                    while j < len(lines):
                        if not lines[j].strip():
                            break
                        if "|" not in lines[j]:
                            break
                        j += 1
                    s = starts[i]
                    e = starts[j - 1] + len(lines[j - 1])
                    spans.append((s, e))
                    i = j
                    continue
                i += 1
            spans.sort(key=lambda x: (x[0], x[1]))
            merged: list[tuple[int, int]] = []
            for s, e in spans:
                if not merged or s >= merged[-1][1]:
                    merged.append((s, e))
                else:
                    ls, le = merged[-1]
                    if e > le:
                        merged[-1] = (ls, e)
            return merged

        table_spans = _extract_table_spans(result)
        placeholders: list[tuple[str, str]] = []
        if table_spans:
            buf = []
            last = 0
            for idx, (s, e) in enumerate(table_spans):
                ph = f"\u0000TABLE_BLOCK_{idx}\u0000"
                buf.append(result[last:s])
                buf.append(ph)
                placeholders.append((ph, result[s:e]))
                last = e
            buf.append(result[last:])
            result = "".join(buf)

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

        if placeholders:
            for ph, original_block in placeholders:
                # 使用原始表格文本替换占位符，确保结构不被破坏
                result = result.replace(ph, original_block)

        return result.strip()

    async def llm_clean(
        self,
        text: str,
        options: CleaningOptions,
        *,
        strict: bool = False,
        original_text: str | None = None,
    ) -> str:
        """LLM 智能清洗。

        使用 callsite `doc_cleaning:clean_text` 调用推理 LLM，保留文档结构，移除噪声。

        Args:
            text: 规则过滤后的文本
            options: 清洗选项

        Returns:
            LLM 清洗后的文本
        """
        segments = self._split_markdown_by_special_tokens(text, original_text=original_text)
        max_len = 8192

        # 如果存在图片链接：LLM 只处理文本段，图片段原样拼回（避免被误删）
        if len(segments) > 1:
            out: list[str] = []
            for seg_type, seg_text in segments:
                if seg_type in {"image", "table", "chart"}:
                    out.append(seg_text)
                elif seg_type == "qr_image":
                    continue
                else:
                    if len(seg_text) <= max_len:
                        out.append(
                            await self._llm_clean_text_segment(
                                seg_text, options, strict=strict
                            )
                        )
                    else:
                        for sub in self._split_text_by_length(seg_text, max_len):
                            out.append(
                                await self._llm_clean_text_segment(
                                    sub, options, strict=strict
                                )
                            )
            return "".join(out).strip()

        # 无图片链接：单段清洗
        if len(text) <= max_len:
            return (await self._llm_clean_text_segment(text, options, strict=strict)).strip()
        out2: list[str] = []
        for sub in self._split_text_by_length(text, max_len):
            out2.append(await self._llm_clean_text_segment(sub, options, strict=strict))
        return "".join(out2).strip()

    async def llm_clean_stream(
        self,
        text: str,
        options: CleaningOptions,
        *,
        strict: bool = False,
        original_text: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """仅 LLM 层的流式清洗（不含规则过滤）。

        产出事件：
        - {"type": "delta", "text": "..."}: 逐段增量文本（最佳努力）
        - {"type": "stage", "stage": "..."}: 阶段信号
        - {"type": "error", "error": "..."}: 发生错误（strict=False 时会降级输出原文）
        """
        yield {"type": "stage", "stage": "segment_split"}
        segments = self._split_markdown_by_special_tokens(text, original_text=original_text)
        yield {
            "type": "stage",
            "stage": "segment_stream_start",
            "segments": len(segments),
        }

        max_len = 8192
        concurrency_limit = 1
        final_parts: list[str] = [""] * len(segments)

        queue: asyncio.Queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def process_segment(idx: int, seg_type: str, seg_text: str) -> None:
            if seg_type in {"image", "table", "chart"}:
                if seg_text:
                    await queue.put({"type": "delta", "text": seg_text})
                final_parts[idx] = seg_text
                await queue.put({"type": "progress", "segment_index": idx, "status": "done"})
                return
            if seg_type == "qr_image":
                final_parts[idx] = ""
                await queue.put({"type": "progress", "segment_index": idx, "status": "done"})
                return

            def _subs_for(text_in: str) -> list[str]:
                if len(text_in) <= max_len:
                    return [text_in]
                return self._split_text_by_length(text_in, max_len)

            combined: list[str] = []
            for sub in _subs_for(seg_text):
                async for evt in self._llm_clean_text_segment_stream(sub, options, strict=strict):
                    if evt.get("type") == "segment_final":
                        combined.append(str(evt.get("text") or ""))
                        continue
                    await queue.put(evt)
            final_parts[idx] = "".join(combined).strip()
            await queue.put({"type": "progress", "segment_index": idx, "status": "done"})

        async def run_with_semaphore(coro):
            async with semaphore:
                await coro

        tasks = [
            asyncio.create_task(run_with_semaphore(process_segment(i, t, s)))
            for i, (t, s) in enumerate(segments)
        ]

        done_count = 0
        while done_count < len(segments):
            evt = await queue.get()
            if isinstance(evt, dict) and evt.get("type") == "progress":
                done_count += 1
            else:
                yield evt
        # 排空队列中可能遗留的事件
        while not queue.empty():
            evt = await queue.get()
            if isinstance(evt, dict) and evt.get("type") == "progress":
                continue
            yield evt
        await asyncio.gather(*tasks)

        final_text = "".join(final_parts).strip()
        yield {"type": "stage", "stage": "segment_stream_done"}
        yield {"type": "final", "cleaned_text": final_text}

    def _split_markdown_by_special_tokens(self, text: str, *, original_text: str | None = None) -> list[tuple[str, str]]:
        if not text:
            return [("text", "")]
        spans: list[tuple[int, int, str]] = []
        # 优先尝试 AST 解析（如安装了 markdown-it-py）
        try:
            from markdown_it import MarkdownIt
            md = MarkdownIt("gfm-like")
            tokens = md.parse(text)
            lines = text.splitlines(True)
            starts: list[int] = []
            pos = 0
            for ln in lines:
                starts.append(pos)
                pos += len(ln)
            open_stack: list[tuple[int, int]] = []
            for tk in tokens:
                if tk.type == "table_open" and tk.map:
                    open_stack.append((tk.map[0], tk.map[1]))
                elif tk.type == "table_close" and open_stack:
                    lb, le = open_stack.pop(0)
                    s = starts[lb] if lb < len(starts) else 0
                    e = (starts[le - 1] + len(lines[le - 1])) if (le - 1) < len(lines) else len(text)
                    spans.append((s, e, "table"))
                elif tk.type in {"fence", "code_block"} and tk.map:
                    info = getattr(tk, "info", "") or ""
                    lang = str(info).strip().lower()
                    if lang in {
                        "mermaid",
                        "vega",
                        "vegalite",
                        "graphviz",
                        "dot",
                        "plantuml",
                        "echarts",
                        "chart",
                        "sequence-diagram",
                    }:
                        lb, le = tk.map
                        s = starts[lb] if lb < len(starts) else 0
                        e = (starts[le - 1] + len(lines[le - 1])) if (le - 1) < len(lines) else len(text)
                        spans.append((s, e, "chart"))
        except Exception:
            pass
        for m in self.MD_TABLE_HTML_RE.finditer(text):
            spans.append((m.start(), m.end(), "table"))
        for m in self.MD_TABLE_BLOCK_RE.finditer(text):
            spans.append((m.start(), m.end(), "table"))
        lines = text.splitlines(True)
        starts: list[int] = []
        pos = 0
        for ln in lines:
            starts.append(pos)
            pos += len(ln)
        i = 0
        while i + 1 < len(lines):
            header = lines[i]
            sep = lines[i + 1]
            if ("|" in header) and self.MD_TABLE_SEP_RE.fullmatch(sep.strip()):
                j = i + 2
                while j < len(lines):
                    if not lines[j].strip():
                        break
                    if "|" not in lines[j]:
                        break
                    j += 1
                s = starts[i]
                e = starts[j - 1] + len(lines[j - 1])
                spans.append((s, e, "table"))
                i = j
                continue
            i += 1
        spans.sort(key=lambda x: (x[0], x[1]))
        merged: list[tuple[int, int, str]] = []
        for s, e, t in spans:
            if not merged or s >= merged[-1][1]:
                merged.append((s, e, t))
            else:
                ls, le, lt = merged[-1]
                if e > le:
                    merged[-1] = (ls, e, lt)
        for m in self.MD_IMAGE_TOKEN_RE.finditer(text):
            ms, me = m.start(), m.end()
            inside = False
            for ts, te, _ in merged:
                if ms >= ts and me <= te:
                    inside = True
                    break
            if not inside:
                token = text[ms:me]
                seg_type = "image"
                em = self.MD_IMAGE_TOKEN_EXTRACT_RE.match(token)
                alt = em.group(1) if em else ""
                link = em.group(2) if em else ""
                ctx_left = text[max(0, ms - 200) : ms]
                ctx_right = text[me : min(len(text), me + 200)]
                if original_text:
                    idx = original_text.find(token)
                    if idx != -1:
                        ctx_left = original_text[max(0, idx - 200) : idx]
                        ctx_right = original_text[idx + len(token) : min(len(original_text), idx + len(token) + 200)]
                def _has_qr_ctx(s: str) -> bool:
                    for p in self.QR_CONTEXT_PATTERNS:
                        if re.search(p, s, flags=re.IGNORECASE):
                            return True
                    return False
                fname = Path(link).name.lower() if link else ""
                if _has_qr_ctx(alt) or _has_qr_ctx(ctx_left) or _has_qr_ctx(ctx_right) or ("qr" in fname or "qrcode" in fname):
                    seg_type = "qr_image"
                merged.append((ms, me, seg_type))
        merged.sort(key=lambda x: (x[0], x[1]))
        parts: list[tuple[str, str]] = []
        pos = 0
        for s, e, t in merged:
            if s > pos:
                parts.append(("text", text[pos:s]))
            parts.append((t, text[s:e]))
            pos = e
        if pos < len(text):
            parts.append(("text", text[pos:]))
        return parts if parts else [("text", text)]

    def _split_text_by_length(self, text: str, max_chars: int) -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        length = 0
        tokens = re.split(r"(\n{2,}|[。！？!?；;]+)", text)
        for tok in tokens:
            if not tok:
                continue
            if length + len(tok) > max_chars:
                if buf:
                    parts.append("".join(buf))
                buf = [tok]
                length = len(tok)
            else:
                buf.append(tok)
                length += len(tok)
        if buf:
            parts.append("".join(buf))
        return parts

    async def _llm_clean_text_segment(
        self, text: str, options: CleaningOptions, *, strict: bool
    ) -> str:
        """清洗单个文本段（该段不包含 Markdown 图片链接 token）。"""
        if not text.strip():
            return text

        # 尝试从上下文中获取 source_file_id (通过检查调用栈或简单地依赖外部传入)
        # 由于当前架构限制，暂时只记录文本摘要
        # 优化日志可读性：移除换行转义，避免 json logger 再次转义导致无法阅读
        text_preview = text[:80].replace("\n", " ").strip()
        logger.info(f"开始清洗文本段（长度：{len(text)}，预览：{text_preview}…）")

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
            chunk_count = 0
            async for chunk in chain.astream({"input": prompt_text}):
                if chunk:
                    chunk_str = str(chunk)
                    cleaned_chunks.append(chunk_str)
                    chunk_count += 1
                    # 每收到 10 个 chunk 打印一次日志，避免刷屏
                    if chunk_count % 10 == 0:
                        logger.info(f"清洗进度：已接收 {chunk_count} 个片段")
            
            logger.info(f"文本段清洗完成（片段数：{chunk_count}）")
            cleaned = "".join(cleaned_chunks)

            cleaned_text = cleaned.strip()
            return f"{prefix}{cleaned_text}{suffix}"

        except Exception as e:
            if strict:
                raise
            logger.warning(f"文本段清洗失败，已降级为原文（错误：{e}）")
            return text

    def _bind_llm_for_streaming(self, llm: Any) -> Any:
        """尽可能为底层 LLM 启用流式输出。"""
        # 移除所有自动 bind 操作，避免向底层客户端传递不支持的参数（如 disable_streaming 传给 OpenAI）
        # LangChain 的 .astream() 方法会自动处理大部分流式需求
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
            logger.warning(f"流式清洗失败，已降级输出原文本段（错误：{e}）")
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
        source_file_id: str,
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
        # T097 要求使用 data/intermediates/{id}/cleaned_doc/ 目录
        # 这里存储路径是相对路径，相对于 DATA_ROOT (f:\lumoscribe2026\data)
        # 所以是 intermediates/{source_file_id}/cleaned_doc/{artifact_id}.json
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

    def _filter_json_content(
        self,
        json_content: list[dict[str, Any]],
        cleaned_text: str,
        options: CleaningOptions,
    ) -> list[dict[str, Any]]:
        """过滤 JSON 内容。
        
        1. 过滤未在 cleaned_text 中引用的图片
        2. 过滤符合规则的噪声文本
        """
        filtered = []
        
        # 提取 cleaned_text 中的所有图片链接
        used_images = set()
        for m in self.MD_IMAGE_TOKEN_RE.finditer(cleaned_text):
            # 提取路径: ![](images/xxx.jpg) -> images/xxx.jpg
            # 简单提取括号内的内容
            link = m.group(0).split("(", 1)[1].rstrip(")")
            used_images.add(Path(link).name)

        for item in json_content:
            item_type = item.get("type")
            
            # 处理图片/表格/公式 (image, table, formula 等可能包含 img_path)
            img_path = item.get("img_path")
            if img_path:
                img_name = Path(img_path).name
                # 如果图片未在 cleaned_text 中引用，则移除
                if img_name not in used_images:
                    continue
                caps = item.get("image_caption") or []
                caps_text = " ".join([str(c) for c in caps])
                def _has_qr_hint(s: str) -> bool:
                    for p in self.QR_CONTEXT_PATTERNS:
                        if re.search(p, s, flags=re.IGNORECASE):
                            return True
                    return False
                if _has_qr_hint(img_name) or _has_qr_hint(caps_text):
                    continue
            
            # 处理文本
            if item_type == "text":
                text = item.get("text", "")
                # 应用规则过滤
                if self._is_noise(text, options):
                    continue
            
            # 处理已废弃内容 (discarded)
            if item_type == "discarded":
                continue

            filtered.append(item)
            
        return filtered

    def _is_noise(self, text: str, options: CleaningOptions) -> bool:
        """判断文本是否为噪声 (基于规则)。"""
        if not text.strip():
            return True
            
        if options.remove_ads:
            for pattern in self.AD_PATTERNS:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return True
                    
        if options.remove_noise:
            # 简单判断：如果整段匹配噪声模式
            for pattern in self.NOISE_PATTERNS:
                # 这里的模式有些是行级的，有些是段落级的
                # 简单起见，如果匹配到任何噪声模式，且文本较短，则认为是噪声
                if re.search(pattern, text, flags=re.MULTILINE):
                    # 如果包含"第x页"且很短
                    if len(text) < 20: 
                        return True
                    # 如果完全匹配
                    if re.fullmatch(pattern, text.strip(), flags=re.MULTILINE):
                        return True
                        
            # 头部噪声 (汇报人等)
            for pattern in self.HEAD_NOISE_PATTERNS:
                if re.search(pattern, text):
                    return True

        return False

    def _write_artifact_file(
        self,
        output_path: str,
        artifact: CleanedDocumentArtifact,
        mineru_output_dir: Path | None = None,
    ) -> None:
        """写入产物文件。
        
        按照 T097 和 T031 要求，必须保留完整的 MinerU 产物结构以便后续处理（如图表提取 T094）。
        
        产出文件清单：
        1. `cleaned.md`: 清洗后的 Markdown 文本（去噪、去广告，保留图片链接）
        2. `cleaned.json`: 清洗任务元数据（统计信息、原始文本引用等）
        3. `content_list.json`: (复制) MinerU 原始内容结构列表，用于图表定位
        4. `layout.json`: (复制) MinerU 布局信息
        5. `images/`: (复制) 图片文件夹，确保 Markdown 图片链接有效
        """
        base_path = Path("data") / output_path
        output_dir = base_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 写入清洗后的 Markdown (核心产物)
        # 使用 artifact_id 命名或固定命名，这里保持与 output_path 一致的命名风格
        md_path = base_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(artifact.cleaned_text)

        # 2. 复制 MinerU 关键结构文件 (content_list 和 layout)
        if mineru_output_dir:
            # 复制 images 目录
            src_images_dir = mineru_output_dir / "images"
            if src_images_dir.exists() and src_images_dir.is_dir():
                dst_images_dir = output_dir / "images"
                if dst_images_dir.exists():
                    shutil.rmtree(dst_images_dir)
                shutil.copytree(src_images_dir, dst_images_dir)
                logger.info(f"已复制图片目录: {src_images_dir} -> {dst_images_dir}")

            # 复制 content_list.json
            content_list_files = list(mineru_output_dir.glob("*_content_list.json"))
            if content_list_files:
                src_content_list = content_list_files[0]
                dst_content_list = output_dir / "content_list.json"
                shutil.copy2(src_content_list, dst_content_list)
                logger.info(f"已复制结构文件: {src_content_list.name} -> content_list.json")
            
            # 复制 layout.json
            layout_files = list(mineru_output_dir.glob("layout.json"))
            if layout_files:
                src_layout = layout_files[0]
                dst_layout = output_dir / "layout.json"
                shutil.copy2(src_layout, dst_layout)
                logger.info(f"已复制布局文件: {src_layout.name} -> layout.json")

        # 3. 写入元数据 JSON (CleanedDocumentArtifact)
        # 包含清洗统计、状态等，并关联上述文件
        final_json_data = artifact.model_dump(mode="json")
        
        # 注入关联文件信息，方便后续步骤查找
        final_json_data["related_files"] = {
            "markdown": md_path.name,
            "content_list": "content_list.json" if mineru_output_dir and list(mineru_output_dir.glob("*_content_list.json")) else None,
            "layout": "layout.json" if mineru_output_dir and list(mineru_output_dir.glob("layout.json")) else None,
            "images_dir": "images"
        }

        with open(base_path, "w", encoding="utf-8") as f:
            json.dump(final_json_data, f, indent=2, ensure_ascii=False)


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

    # -------------------------
    # Chart JSON 正式归一化（供 T032/T094 共用）
    # -------------------------

    _CHART_TYPE_ALIASES: dict[str, str] = {
        # bar
        "bar": "bar",
        "bar_chart": "bar",
        "column": "bar",
        "column_chart": "bar",
        "histogram": "bar",
        # line
        "line": "line",
        "line_chart": "line",
        # pie
        "pie": "pie",
        "pie_chart": "pie",
        "donut": "pie",
        "donut_chart": "pie",
        # stacked area
        "stacked_area": "stacked_area",
        "stacked_area_chart_with_legend": "stacked_area",
        "area": "stacked_area",
        "area_chart": "stacked_area",
        # sankey / flow
        "sankey": "sankey",
        "flow": "sankey",
        "flow_chart_with_categories": "sankey",
        "process_flow": "sankey",
        "process_flow_chart": "sankey",
        "process_flow_diagram": "sankey",
        "flow_diagram": "sankey",
        # table / comparison
        "table": "table",
        "table_chart": "table",
        "comparison_table": "table",
        "comparison_chart": "table",
        "energy_storage_comparison_chart": "table",
        # other chart families
        "scatter": "scatter",
        "scatter_plot": "scatter",
        "heatmap": "heatmap",
        "radar": "radar",
        # unknown / none
        "unknown": "unknown",
        "none": "none",
    }

    _CHART_TYPE_ALLOWED: set[str] = {
        "bar",
        "line",
        "pie",
        "stacked_area",
        "sankey",
        "table",
        "scatter",
        "heatmap",
        "radar",
        "unknown",
        "none",
    }

    def _parse_bool_maybe(self, v: Any) -> bool | None:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1", "yes", "y"}:
                return True
            if s in {"false", "0", "no", "n"}:
                return False
        return None

    def _normalize_chart_type_str(self, raw: Any, *, hint: str | None = None) -> str:
        s = str(raw).strip().lower() if raw is not None else ""
        if s:
            s = self._CHART_TYPE_ALIASES.get(s, s)
        if not s and hint:
            hs = str(hint).strip().lower()
            s = self._CHART_TYPE_ALIASES.get(hs, hs)
        if not s:
            return "unknown"
        return s if s in self._CHART_TYPE_ALLOWED else "unknown"

    def _normalize_chart_json(
        self,
        obj: dict[str, Any],
        *,
        hint_type: str | None,
        strict: bool,
        image_path: Path,
    ) -> dict[str, Any]:
        """将模型输出归一化为“受控 schema”（不改动额外字段，但确保关键字段稳定）。

        约束：
        - strict=True 时：必须包含可解析的 is_chart；若 is_chart=true 则 description 不可空。
        - chart_type 永远落到受控集合（否则置为 unknown）。
        """
        out = dict(obj or {})

        # schema_version
        sv = out.get("schema_version")
        if not isinstance(sv, int):
            out["schema_version"] = 1

        # is_chart
        is_chart = self._parse_bool_maybe(out.get("is_chart"))
        if is_chart is None:
            if strict:
                raise RuntimeError(f"图转 JSON 输出缺少/非法 is_chart: image={image_path}")
            # 非严格：尽量不篡改原输出
            return out
        out["is_chart"] = bool(is_chart)

        if not is_chart:
            out["chart_type"] = "none"
            out["description"] = ""
            out["chart_count"] = 0
            out["chart_data"] = []
            return out

        # chart_type
        out["chart_type"] = self._normalize_chart_type_str(out.get("chart_type"), hint=hint_type)

        # description：优先使用 description，其次兼容 title/name
        desc = out.get("description")
        if not (isinstance(desc, str) and desc.strip()):
            for k in ("title", "name", "chart_title", "_chart_name", "chart_name"):
                v = out.get(k)
                if isinstance(v, str) and v.strip():
                    desc = v.strip()
                    break
        desc_s = desc.strip() if isinstance(desc, str) else ""
        if strict and not desc_s:
            raise RuntimeError(f"图转 JSON 输出缺少 description（is_chart=true）: image={image_path}")
        out["description"] = desc_s

        # chart_count
        cc = out.get("chart_count")
        if not isinstance(cc, int):
            out["chart_count"] = 1

        # chart_data
        if "chart_data" not in out:
            if strict:
                raise RuntimeError(f"图转 JSON 输出缺少 chart_data: image={image_path}")
            out["chart_data"] = {}

        return out

    async def chart_to_json(
        self,
        chart_image_path: str,
        chart_type: str,
        *,
        strict_json: bool = False,
        timeout_seconds: int | None = None,
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
                import asyncio as _asyncio
                result = await _asyncio.wait_for(llm.ainvoke([message]), timeout=float(timeout_seconds or 60))
            except _asyncio.TimeoutError as exc:
                raise RuntimeError(
                    f"多模态 LLM 调用超时: model={type(llm).__name__}, image={chart_image_path}, timeout={timeout_seconds or 60}s"
                ) from exc
            except Exception as exc:
                # 严格模式：直接失败并暴露真实原因（例如 provider 不支持图像输入）
                # 优先使用 str(exc)，如果为空则使用 repr(exc)
                err_msg = str(exc) or repr(exc)
                raise RuntimeError(
                    f"多模态 LLM 调用失败: model={type(llm).__name__}, image={chart_image_path}, error={err_msg}"
                ) from exc
            
            # 解析结果
            json_content = self._parse_chart_result(
                result.content if hasattr(result, "content") else result,
                strict=strict_json,
            )

            # 归一化关键字段，保证下游（KB/渲染）稳定
            try:
                if isinstance(json_content, dict):
                    json_content = self._normalize_chart_json(
                        json_content,
                        hint_type=chart_type,
                        strict=bool(strict_json),
                        image_path=Path(chart_image_path),
                    )
            except Exception:
                if strict_json:
                    raise

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
        # T094 输出路径：intermediates/{id}/pic_to_json/chart_json/
        relative_path = f"intermediates/{source_file_id}/pic_to_json/chart_json/{artifact_id}.json"

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

    # 说明：T094 属于批处理工作流，已迁移至独立模块：
    # `src/application/services/document_cleaning/t094_pic_to_json_pipeline.py`
    # 由脚本 `scripts/t094-pic-to-json.py` 调用，避免把测试/批处理逻辑固化进核心 Service。
