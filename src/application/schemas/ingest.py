"""文档摄入与检索 Schema 定义。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    """文档切分策略。"""

    STRUCTURE_AWARE = "structure_aware"  # 结构感知切分
    SEMANTIC = "semantic"  # 语义切分
    SENTENCE = "sentence"  # 句子级别切分
    LENGTH = "length"  # 长度约束切分


class ChunkType(str, Enum):
    """Chunk 类型。"""

    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADING = "heading"
    CODE = "code"
    MIXED = "mixed"


class SearchStrategy(str, Enum):
    """检索策略。"""

    VECTOR = "vector"  # 向量检索
    BM25 = "bm25"  # BM25 关键词检索
    HYBRID = "hybrid"  # 混合检索


class RerankStrategy(str, Enum):
    """重排序策略。"""

    NONE = "none"  # 不使用重排序
    CROSS_ENCODER = "cross_encoder"  # Cross-Encoder 重排序
    LL_RERANK = "ll_rerank"  # LLM 重排序


class ChunkingConfig(BaseModel):
    """文档切分配置类。"""

    chunk_size: int = Field(
        default=1024,
        ge=100,
        le=8192,
        description="每个 chunk 的最大字符数",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=512,
        description="连续 chunk 之间的重叠字符数",
    )
    embed_model_name: str = Field(
        default="BAAI/bge-large-zh-v1.5",
        description="嵌入模型名称",
    )
    semantic_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="语义切分阈值（余弦相似度）",
    )
    min_chunk_size: int = Field(
        default=100,
        ge=10,
        description="最小 chunk 字符数",
    )
    keep_structure: bool = Field(
        default=True,
        description="是否保持文档结构",
    )
    split_by_sentence: bool = Field(
        default=True,
        description="是否按句子边界切分",
    )


class ChunkingOptions(BaseModel):
    """文档切分选项。"""

    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.STRUCTURE_AWARE,
        description="切分策略",
    )
    chunk_size: int | None = Field(
        default=None,
        ge=100,
        le=8192,
        description="覆盖默认 chunk_size",
    )
    chunk_overlap: int | None = Field(
        default=None,
        ge=0,
        le=512,
        description="覆盖默认 chunk_overlap",
    )
    extract_images: bool = Field(
        default=False,
        description="是否提取图片中的文字",
    )
    table_to_text: bool = Field(
        default=True,
        description="是否将表格转换为文本",
    )
    preserve_formatting: bool = Field(
        default=True,
        description="是否保留格式信息",
    )


class KBChunk(BaseModel):
    """知识库 Chunk 模型。"""

    chunk_id: str = Field(
        ...,
        description="Chunk 唯一标识",
    )
    content: str = Field(
        ...,
        description="Chunk 内容",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="元数据",
    )
    embedding: list[float] | None = Field(
        default=None,
        description="向量嵌入",
    )
    source_file_id: int = Field(
        ...,
        description="源文件 ID",
    )
    chunk_type: ChunkType = Field(
        default=ChunkType.PARAGRAPH,
        description="Chunk 类型",
    )
    chunk_index: int = Field(
        default=0,
        description="Chunk 在文档中的索引",
    )
    start_char: int = Field(
        default=0,
        description="起始字符位置",
    )
    end_char: int = Field(
        default=0,
        description="结束字符位置",
    )


class KBChunkArtifact(BaseModel):
    """知识库 Chunk 产物。"""

    source_file_id: int = Field(
        ...,
        description="源文件 ID",
    )
    chunks: list[KBChunk] = Field(
        ...,
        description="Chunk 列表",
    )
    output_path: str = Field(
        ...,
        description="输出路径",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间",
    )
    total_chunks: int = Field(
        default=0,
        description="总 Chunk 数",
    )
    total_chars: int = Field(
        default=0,
        description="总字符数",
    )


class VectorCollectionInfo(BaseModel):
    """向量集合信息。"""

    name: str = Field(
        ...,
        description="集合名称",
    )
    chunk_count: int = Field(
        default=0,
        description="Chunk 数量",
    )
    embedding_model: str = Field(
        default="",
        description="嵌入模型",
    )
    dimension: int = Field(
        default=0,
        description="向量维度",
    )
    created_at: datetime | None = Field(
        None,
        description="创建时间",
    )
    updated_at: datetime | None = Field(
        None,
        description="更新时间",
    )


class SearchResult(BaseModel):
    """检索结果。"""

    chunk_id: str = Field(
        ...,
        description="Chunk ID",
    )
    content: str = Field(
        ...,
        description="Chunk 内容",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="相关性分数",
    )
    search_type: SearchStrategy = Field(
        ...,
        description="检索类型",
    )
    source_file_id: int = Field(
        ...,
        description="源文件 ID",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="元数据",
    )
    rank: int = Field(
        default=0,
        description="排名",
    )


class HybridSearchOptions(BaseModel):
    """混合检索选项。"""

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="返回结果数量",
    )
    vector_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="向量检索权重",
    )
    bm25_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="BM25 检索权重",
    )
    use_rerank: bool = Field(
        default=False,
        description="是否使用重排序",
    )
    rerank_model: str | None = Field(
        default=None,
        description="重排序模型名称",
    )
    rerank_top_n: int = Field(
        default=5,
        ge=1,
        le=50,
        description="重排序后返回数量",
    )
    filter_metadata: dict[str, Any] | None = Field(
        default=None,
        description="元数据过滤条件",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="分数阈值",
    )


class SearchMetrics(BaseModel):
    """检索指标。"""

    total_results: int = Field(
        default=0,
        description="总结果数",
    )
    vector_results: int = Field(
        default=0,
        description="向量检索结果数",
    )
    bm25_results: int = Field(
        default=0,
        description="BM25 检索结果数",
    )
    rerank_applied: bool = Field(
        default=False,
        description="是否使用重排序",
    )
    query_time_ms: float = Field(
        default=0.0,
        description="查询耗时（毫秒）",
    )
    fusion_method: str = Field(
        default="rrf",
        description="融合方法",
    )


class HybridSearchResponse(BaseModel):
    """混合检索响应。"""

    query: str = Field(
        ...,
        description="搜索查询",
    )
    results: list[SearchResult] = Field(
        ...,
        description="检索结果",
    )
    metrics: SearchMetrics = Field(
        default_factory=SearchMetrics,
        description="检索指标",
    )
    total_time_ms: float = Field(
        default=0.0,
        description="总耗时（毫秒）",
    )


class IndexBuildProgress(BaseModel):
    """索引构建进度。"""

    collection_name: str = Field(
        ...,
        description="集合名称",
    )
    stage: str = Field(
        ...,
        description="当前阶段",
    )
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="进度百分比",
    )
    processed_chunks: int = Field(
        default=0,
        description="已处理 Chunk 数",
    )
    total_chunks: int = Field(
        default=0,
        description="总 Chunk 数",
    )
    status: str = Field(
        default="running",
        description="状态",
    )
    error: str | None = Field(
        default=None,
        description="错误信息",
    )


class IndexBuildResult(BaseModel):
    """索引构建结果。"""

    success: bool = Field(
        ...,
        description="是否成功",
    )
    collection_name: str = Field(
        ...,
        description="集合名称",
    )
    chunks_indexed: int = Field(
        default=0,
        description="已索引 Chunk 数",
    )
    vectors_stored: int = Field(
        default=0,
        description="已存储向量数",
    )
    duration_seconds: float = Field(
        default=0.0,
        description="耗时（秒）",
    )
    error: str | None = Field(
        default=None,
        description="错误信息",
    )
    collection_info: VectorCollectionInfo | None = Field(
        default=None,
        description="集合信息",
    )
