"""文档切分服务（T033）。

基于 LlamaIndex 实现分层文档切分：
- 结构感知切分：识别标题、段落、表格
- 语义切分：使用嵌入模型检测语义边界
- 句子级别优化：保持句子完整性
- 长度约束：确保 chunk size 符合配置
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from typing import Any

from llama_index.core import Document as LIDocument
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.core.schema import TextNode

from src.application.schemas.ingest import (
    ChunkType,
    ChunkingConfig,
    ChunkingOptions,
    ChunkingStrategy,
    KBChunk,
)
from src.application.services.embedding_adapter import to_llamaindex_embedding
from src.shared.errors import AppError


class ChunkingServiceError(AppError):
    """切分服务错误。"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "chunking_error",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
        )


class DocumentChunkingService:
    """文档切分服务。"""
 
    _MD_TABLE_BLOCK_RE = re.compile(
        r"(?:^\s*\|.+\|\s*\n\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\s*\|?\s*(?:\n\s*\|.*\|\s*)+)",
        re.MULTILINE,
    )

    def __init__(self, config: ChunkingConfig | None = None, *, embedding_model: Any | None = None):
        """初始化切分服务。

        Args:
            config: 切分配置
        """
        self.config = config or ChunkingConfig()
        self._embed_model: BaseEmbedding | None = None
        if embedding_model is not None:
            self._embed_model = to_llamaindex_embedding(embedding_model)
        self._semantic_splitter = None
 
    def _truncate_table_block(self, table_md: str, *, max_chars: int) -> tuple[str, bool]:
        s = (table_md or "").replace("\r\n", "\n").strip()
        if not s:
            return "", False
        if max_chars <= 0 or len(s) <= max_chars:
            return s, False
        lines = [ln.rstrip() for ln in s.split("\n")]
        if len(lines) <= 3:
            cut = s[:max_chars].rstrip()
            return (cut + " …（截断）").strip(), True
        header = lines[:2]
        out: list[str] = []
        out.extend(header)
        remain = max_chars - len("\n".join(out))
        truncated = False
        for ln in lines[2:]:
            if remain <= 0:
                truncated = True
                break
            if len(ln) + 1 > remain:
                truncated = True
                break
            out.append(ln)
            remain -= len(ln) + 1
        if truncated:
            out.append("…（表格过长已截断，保留表头与部分行）")
        return "\n".join(out).strip(), truncated

    async def _get_embed_model(self):
        """获取嵌入模型。"""
        if self._embed_model is None:
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                self._embed_model = HuggingFaceEmbedding(
                    model_name=self.config.embed_model_name,
                )
            except ImportError as e:
                raise ChunkingServiceError(
                    message=f"无法导入嵌入模型: {e}",
                    code="embed_model_import_error",
                )
        return self._embed_model

    async def _get_semantic_splitter(
        self,
        embed_model=None,
    ) -> SemanticSplitterNodeParser:
        """获取语义切分器。"""
        if self._semantic_splitter is None:
            model = embed_model or await self._get_embed_model()
            self._semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=self.config.semantic_threshold * 100,
                embed_model=model,
            )
        return self._semantic_splitter

    def _generate_chunk_id(self, source_file_id: str, chunk_index: int) -> str:
        """生成 chunk ID。

        Args:
            source_file_id: 源文件 ID
            chunk_index: chunk 索引

        Returns:
            chunk ID
        """
        unique_str = f"{source_file_id}_{chunk_index}_{uuid.uuid4().hex[:8]}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]

    def _split_into_paragraphs(self, text: str) -> list[dict[str, Any]]:
        """将文本分割成段落块。

        Args:
            text: 输入文本

        Returns:
            段落块列表，每个块包含内容和类型信息
        """
        # 识别代码块
        code_pattern = r"```[\s\S]*?```|`[^`]+`"

        blocks = []
        remaining = text

        # 提取代码块
        code_matches = list(re.finditer(code_pattern, remaining))
        if code_matches:
            last_end = 0
            for match in code_matches:
                # 添加代码块前的内容
                before = remaining[last_end : match.start()].strip()
                if before:
                    blocks.extend(self._split_paragraph_text(before))

                # 添加代码块
                blocks.append(
                    {
                        "content": match.group(),
                        "type": ChunkType.CODE,
                    }
                )
                last_end = match.end()

            # 添加剩余内容
            after = remaining[last_end:].strip()
            if after:
                blocks.extend(self._split_paragraph_text(after))
        else:
            blocks.extend(self._split_paragraph_text(remaining))

        return blocks

    def _split_plain_paragraphs(self, text: str) -> list[dict[str, Any]]:
        paragraphs = re.split(r"\n\n+", text)
        blocks: list[dict[str, Any]] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if re.match(r"^(#{1,6}\s+|[\w\u4e00-\u9fff].{0,50}$)", para):
                blocks.append({"content": para, "type": ChunkType.HEADING})
            else:
                blocks.append({"content": para, "type": ChunkType.PARAGRAPH})
        return blocks
 
    def _split_paragraph_text(self, text: str) -> list[dict[str, Any]]:
        """分割普通段落文本。

        Args:
            text: 段落文本

        Returns:
            文本块列表
        """
        s = (text or "").replace("\r\n", "\n")
        matches = list(self._MD_TABLE_BLOCK_RE.finditer(s))
        if not matches:
            return self._split_plain_paragraphs(s)
        blocks: list[dict[str, Any]] = []
        last_end = 0
        for m in matches:
            before = s[last_end : m.start()].strip()
            if before:
                blocks.extend(self._split_plain_paragraphs(before))
            tbl = (m.group() or "").strip()
            if tbl:
                blocks.append({"content": tbl, "type": ChunkType.TABLE})
            last_end = m.end()
        after = s[last_end:].strip()
        if after:
            blocks.extend(self._split_plain_paragraphs(after))
        return blocks

    def _split_by_sentence(self, text: str) -> list[str]:
        """按句子分割文本。

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 句末标点
        sentence_endings = re.compile(r"[.!?！？\n]+")
        sentences = sentence_endings.split(text)

        # 过滤空句子并重组
        result = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 如果加上这个句子不超过限制，继续累积
            if len(current) + len(sentence) < self.config.chunk_size:
                current += " " + sentence if current else sentence
            else:
                if current:
                    result.append(current)
                current = sentence

        if current:
            result.append(current)

        return result

    def _merge_small_chunks(
        self,
        chunks: list[dict[str, Any]],
        min_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """合并过小的 chunks。

        Args:
            chunks: chunks 列表
            min_size: 最小大小

        Returns:
            合并后的 chunks
        """
        min_size = min_size or self.config.min_chunk_size
        if min_size <= 0:
            return chunks

        merged = []
        buffer = ""

        for chunk in chunks:
            buffer += ("\n\n" if buffer else "") + chunk["content"]

            if len(buffer) >= min_size:
                merged.append(
                    {
                        "content": buffer,
                        "type": chunk.get("type", ChunkType.MIXED),
                    }
                )
                buffer = ""

        # 处理剩余内容
        if buffer:
            if merged and len(buffer) < min_size:
                # 合并到最后一个 chunk
                merged[-1]["content"] += "\n\n" + buffer
            else:
                merged.append(
                    {
                        "content": buffer,
                        "type": ChunkType.MIXED,
                    }
                )

        return merged

    def _create_chunks_from_blocks(
        self,
        blocks: list[dict[str, Any]],
        source_file_id: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[KBChunk]:
        """从文本块创建 chunks。

        Args:
            blocks: 文本块列表
            source_file_id: 源文件 ID
            chunk_size: chunk 大小
            chunk_overlap: 重叠大小

        Returns:
            KBChunk 列表
        """
        chunk_size = chunk_size or self.config.chunk_size
        chunk_overlap = chunk_overlap or self.config.chunk_overlap

        chunks = []
        current_content = ""
        current_type = ChunkType.PARAGRAPH
        start_char = 0
        chunk_index = 0

        def flush_current():
            nonlocal current_content, start_char, chunk_index

            if not current_content:
                return
 
            if current_type == ChunkType.TABLE:
                txt, truncated = self._truncate_table_block(current_content, max_chars=chunk_size)
                chunks.append(
                    KBChunk(
                        chunk_id=self._generate_chunk_id(source_file_id, chunk_index),
                        content=txt,
                        metadata={
                            "chunk_type": current_type.value,
                            "source": "document",
                            "table_truncated": bool(truncated),
                        },
                        source_file_id=source_file_id,
                        chunk_type=current_type,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(txt),
                    )
                )
                chunk_index += 1
                start_char += len(txt)
                current_content = ""
                start_char = max(0, start_char - chunk_overlap)
                return

            # 按大小限制分割
            while len(current_content) > chunk_size:
                chunk_text = current_content[:chunk_size]
                remaining = current_content[chunk_size:]

                # 尝试在句子边界分割
                last_period = max(
                    chunk_text.rfind("."),
                    chunk_text.rfind("!"),
                    chunk_text.rfind("?"),
                    chunk_text.rfind(".\n"),
                    chunk_text.rfind("!\n"),
                    chunk_text.rfind("?\n"),
                )

                if last_period > chunk_size * 0.5:
                    chunk_text = chunk_text[: last_period + 1]
                    remaining = current_content[last_period + 1 :]

                chunks.append(
                    KBChunk(
                        chunk_id=self._generate_chunk_id(source_file_id, chunk_index),
                        content=chunk_text.strip(),
                        metadata={
                            "chunk_type": current_type.value,
                            "source": "document",
                        },
                        source_file_id=source_file_id,
                        chunk_type=current_type,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                    )
                )

                chunk_index += 1
                start_char += len(chunk_text) - chunk_overlap
                current_content = remaining

            # 处理剩余内容
            if current_content:
                chunks.append(
                    KBChunk(
                        chunk_id=self._generate_chunk_id(source_file_id, chunk_index),
                        content=current_content.strip(),
                        metadata={
                            "chunk_type": current_type.value,
                            "source": "document",
                        },
                        source_file_id=source_file_id,
                        chunk_type=current_type,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=start_char + len(current_content),
                    )
                )

                chunk_index += 1
                start_char += len(current_content)

            current_content = ""
            start_char = max(0, start_char - chunk_overlap)

        for block in blocks:
            # 如果新块类型不同，先 flush 当前内容
            if current_content and block["type"] != current_type:
                flush_current()

            current_content += ("\n\n" if current_content else "") + block["content"]
            current_type = block.get("type", ChunkType.MIXED)

            # 如果内容超过限制，flush
            if len(current_content) >= chunk_size:
                flush_current()

        # Flush 剩余内容
        flush_current()

        return chunks

    async def chunk_document(
        self,
        text: str,
        metadata: dict[str, Any],
        options: ChunkingOptions | None = None,
    ) -> list[KBChunk]:
        """切分文档，返回 chunk 列表。

        Args:
            text: 文档文本
            metadata: 元数据
            options: 切分选项

        Returns:
            KBChunk 列表

        Raises:
            ChunkingServiceError: 切分失败
        """
        options = options or ChunkingOptions()
        source_file_id = (
            str(metadata.get("source_file_id")).strip()
            if metadata.get("source_file_id") is not None
            else ""
        )
        if not source_file_id:
            source_file_id = "unknown"

        try:
            # 根据策略选择切分方法
            if options.strategy == ChunkingStrategy.STRUCTURE_AWARE:
                return await self._chunk_structure_aware(text, metadata, options)
            elif options.strategy == ChunkingStrategy.SEMANTIC:
                return await self._chunk_semantic(text, metadata, options)
            elif options.strategy == ChunkingStrategy.SENTENCE:
                return await self._chunk_by_sentence(text, metadata, options)
            else:
                return await self._chunk_by_length(text, metadata, options)

        except Exception as e:
            raise ChunkingServiceError(
                message=f"文档切分失败: {e}",
                code="chunking_failed",
                details={
                    "source_file_id": source_file_id,
                    "strategy": options.strategy.value,
                },
            ) from e

    async def _chunk_structure_aware(
        self,
        text: str,
        metadata: dict[str, Any],
        options: ChunkingOptions,
    ) -> list[KBChunk]:
        """结构感知切分。

        Args:
            text: 文档文本
            metadata: 元数据
            options: 切分选项

        Returns:
            KBChunk 列表
        """
        # 步骤 1: 识别结构块
        blocks = self._split_into_paragraphs(text)

        # 步骤 2: 合并小 chunks
        merged_blocks = self._merge_small_chunks(blocks)

        # 步骤 3: 创建 chunks
        chunk_size = options.chunk_size or self.config.chunk_size
        chunk_overlap = options.chunk_overlap or self.config.chunk_overlap
        source_file_id = (
            str(metadata.get("source_file_id")).strip()
            if metadata.get("source_file_id") is not None
            else ""
        )
        if not source_file_id:
            source_file_id = "unknown"

        chunks = self._create_chunks_from_blocks(
            merged_blocks,
            source_file_id,
            chunk_size,
            chunk_overlap,
        )

        # 添加元数据
        for chunk in chunks:
            chunk.metadata.update(metadata)

        return chunks

    async def _chunk_semantic(
        self,
        text: str,
        metadata: dict[str, Any],
        options: ChunkingOptions,
    ) -> list[KBChunk]:
        """语义切分。

        Args:
            text: 文档文本
            metadata: 元数据
            options: 切分选项

        Returns:
            KBChunk 列表
        """
        embed_model = await self._get_embed_model()
        splitter = await self._get_semantic_splitter(embed_model)

        # 创建 LlamaIndex 文档
        doc = LIDocument(text=text, metadata=metadata)

        # 使用语义切分器
        nodes = splitter.get_nodes_from_documents([doc])

        source_file_id = (
            str(metadata.get("source_file_id")).strip()
            if metadata.get("source_file_id") is not None
            else ""
        )
        if not source_file_id:
            source_file_id = "unknown"
        chunks = []

        for idx, node in enumerate(nodes):
            # 确定 chunk 类型
            node_text = node.get_content()
            chunk_type = ChunkType.PARAGRAPH
            if node_text.startswith("```"):
                chunk_type = ChunkType.CODE
            elif node_text.startswith("|") and node_text.count("|") > 1:
                chunk_type = ChunkType.TABLE
            elif node_text.startswith("#"):
                chunk_type = ChunkType.HEADING

            chunks.append(
                KBChunk(
                    chunk_id=self._generate_chunk_id(source_file_id, idx),
                    content=node_text,
                    metadata={
                        **metadata,
                        "chunk_type": chunk_type.value,
                        "node_id": node.node_id,
                    },
                    source_file_id=source_file_id,
                    chunk_type=chunk_type,
                    chunk_index=idx,
                    start_char=node.start_char_idx or 0,
                    end_char=node.end_char_idx or len(node_text),
                )
            )

        return chunks

    async def _chunk_by_sentence(
        self,
        text: str,
        metadata: dict[str, Any],
        options: ChunkingOptions,
    ) -> list[KBChunk]:
        """按句子切分。

        Args:
            text: 文档文本
            metadata: 元数据
            options: 切分选项

        Returns:
            KBChunk 列表
        """
        chunk_size = options.chunk_size or self.config.chunk_size
        chunk_overlap = options.chunk_overlap or self.config.chunk_overlap

        # 使用 SentenceSplitter
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
            sentence_separator=".!?！？",
        )

        doc = LIDocument(text=text, metadata=metadata)
        nodes = splitter.get_nodes_from_documents([doc])

        source_file_id = (
            str(metadata.get("source_file_id")).strip()
            if metadata.get("source_file_id") is not None
            else ""
        )
        if not source_file_id:
            source_file_id = "unknown"
        chunks = []

        for idx, node in enumerate(nodes):
            chunks.append(
                KBChunk(
                    chunk_id=self._generate_chunk_id(source_file_id, idx),
                    content=node.get_content(),
                    metadata={
                        **metadata,
                        "chunk_type": ChunkType.PARAGRAPH.value,
                        "node_id": node.node_id,
                    },
                    source_file_id=source_file_id,
                    chunk_type=ChunkType.PARAGRAPH,
                    chunk_index=idx,
                    start_char=node.start_char_idx or 0,
                    end_char=node.end_char_idx or len(node.get_content()),
                )
            )

        return chunks

    async def _chunk_by_length(
        self,
        text: str,
        metadata: dict[str, Any],
        options: ChunkingOptions,
    ) -> list[KBChunk]:
        """按长度切分。

        Args:
            text: 文档文本
            metadata: 元数据
            options: 切分选项

        Returns:
            KBChunk 列表
        """
        chunk_size = options.chunk_size or self.config.chunk_size
        source_file_id = (
            str(metadata.get("source_file_id")).strip()
            if metadata.get("source_file_id") is not None
            else ""
        )
        if not source_file_id:
            source_file_id = "unknown"

        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # 尝试在句子边界分割
            if end < len(text):
                for boundary in [".", "!", "?", ".\n", "!\n", "?\n"]:
                    last_boundary = text[start:end].rfind(boundary)
                    if last_boundary > chunk_size * 0.3:
                        end = start + last_boundary + 1
                        break

            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(
                    KBChunk(
                        chunk_id=self._generate_chunk_id(source_file_id, idx),
                        content=chunk_text.strip(),
                        metadata={
                            **metadata,
                            "chunk_type": ChunkType.PARAGRAPH.value,
                        },
                        source_file_id=source_file_id,
                        chunk_type=ChunkType.PARAGRAPH,
                        chunk_index=idx,
                        start_char=start,
                        end_char=end,
                    )
                )
                idx += 1

            start = end - (options.chunk_overlap or self.config.chunk_overlap)
            start = max(0, start)

        return chunks

    async def chunk_batch(
        self,
        documents: list[tuple[str, dict[str, Any]]],
        options: ChunkingOptions | None = None,
    ) -> list[KBChunk]:
        """批量切分文档。

        Args:
            documents: 文档列表，格式为 [(text, metadata), ...]
            options: 切分选项

        Returns:
            所有文档的 KBChunk 列表
        """
        all_chunks: list[KBChunk] = []
        semaphore = asyncio.Semaphore(10)

        async def chunk_with_semaphore(
            text: str,
            metadata: dict[str, Any],
        ) -> list[KBChunk]:
            async with semaphore:
                return await self.chunk_document(text, metadata, options)

        tasks = [chunk_with_semaphore(text, meta) for text, meta in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                raise ChunkingServiceError(
                    message=f"批量切分失败: {result}",
                    code="batch_chunking_failed",
                ) from result
            all_chunks.extend(result)

        return all_chunks

    def get_statistics(self, chunks: list[KBChunk]) -> dict[str, Any]:
        """获取 chunks 统计信息。

        Args:
            chunks: chunks 列表

        Returns:
            统计信息
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "avg_chunk_size": 0,
                "chunk_type_distribution": {},
            }

        total_chars = sum(len(c.content) for c in chunks)
        type_dist: dict[str, int] = {}

        for chunk in chunks:
            type_key = chunk.chunk_type.value
            type_dist[type_key] = type_dist.get(type_key, 0) + 1

        return {
            "total_chunks": len(chunks),
            "total_chars": total_chars,
            "avg_chunk_size": total_chars / len(chunks),
            "chunk_type_distribution": type_dist,
        }
