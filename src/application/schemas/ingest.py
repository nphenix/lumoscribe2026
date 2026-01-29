"""摄入 API Pydantic 模型定义。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.application.schemas.mineru import MinerUIngestOptions


class ChunkType(str, Enum):
    """Chunk 类型枚举。"""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    CODE = "code"
    TABLE = "table"
    CHART = "chart"
    MIXED = "mixed"


class ChunkingStrategy(str, Enum):
    """切分策略枚举。"""

    STRUCTURE_AWARE = "structure_aware"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    LENGTH = "length"


class SearchStrategy(str, Enum):
    """搜索策略枚举。"""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class ChunkingConfig(BaseModel):
    """切分配置。"""

    chunk_size: int = Field(default=1024, ge=100, le=8192, description="Chunk 大小")
    chunk_overlap: int = Field(default=200, ge=0, le=512, description="Chunk 重叠大小")
    min_chunk_size: int = Field(default=100, ge=10, le=500, description="最小 chunk 大小")
    semantic_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="语义切分阈值")
    embed_model_name: str = Field(default="BAAI/bge-small-zh", description="嵌入模型名称")


class ChunkingOptions(BaseModel):
    """切分选项。"""

    strategy: ChunkingStrategy = Field(default=ChunkingStrategy.STRUCTURE_AWARE, description="切分策略")
    chunk_size: int | None = Field(default=None, ge=100, le=8192, description="覆盖默认 chunk 大小")
    chunk_overlap: int | None = Field(default=None, ge=0, le=512, description="覆盖默认 chunk 重叠")


class KBChunk(BaseModel):
    """知识库 Chunk。"""

    chunk_id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="Chunk 内容")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    source_file_id: str = Field(default="", description="源文件 ID")
    chunk_type: ChunkType = Field(default=ChunkType.PARAGRAPH, description="Chunk 类型")
    chunk_index: int = Field(default=0, description="Chunk 索引")
    start_char: int = Field(default=0, description="起始字符位置")
    end_char: int = Field(default=0, description="结束字符位置")
    embedding: list[float] | None = Field(default=None, description="嵌入向量")


class VectorCollectionInfo(BaseModel):
    """向量集合信息。"""

    name: str = Field(..., description="集合名称")
    chunk_count: int = Field(default=0, description="Chunk 数量")
    embedding_model: str = Field(default="", description="嵌入模型")
    dimension: int = Field(default=0, description="向量维度")
    created_at: datetime | None = Field(None, description="创建时间")
    updated_at: datetime | None = Field(None, description="更新时间")


class HybridSearchOptions(BaseModel):
    """混合搜索选项。"""

    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    use_rerank: bool = Field(default=True, description="是否使用重排序")
    rerank_top_n: int | None = Field(default=None, ge=1, le=100, description="重排序返回数量")
    filter_metadata: dict[str, Any] | None = Field(default=None, description="元数据过滤")
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0, description="分数阈值")
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="向量检索权重")
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="BM25 检索权重")
    max_total_charts: int | None = Field(default=None, ge=0, le=50, description="最多附带的图表结果数量")
    max_charts_per_parent: int | None = Field(default=None, ge=0, le=20, description="每个父节点最多附带的图表数量")


class SearchResult(BaseModel):
    """搜索结果。"""

    chunk_id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="内容")
    score: float = Field(default=0.0, description="分数")
    search_type: SearchStrategy = Field(default=SearchStrategy.HYBRID, description="搜索类型")
    source_file_id: str = Field(default="", description="源文件 ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    rank: int = Field(default=0, description="排名")


class SearchMetrics(BaseModel):
    """搜索指标。"""

    total_results: int = Field(default=0, description="总结果数")
    vector_results: int = Field(default=0, description="向量检索结果数")
    bm25_results: int = Field(default=0, description="BM25 检索结果数")
    bm25_index_used: bool = Field(default=False, description="是否使用了 BM25 索引")
    rerank_applied: bool = Field(default=False, description="是否应用了重排序")
    query_time_ms: float = Field(default=0.0, description="查询时间（毫秒）")


class HybridSearchResponse(BaseModel):
    """混合搜索响应。"""

    query: str = Field(..., description="查询文本")
    results: list[SearchResult] = Field(default_factory=list, description="搜索结果")
    metrics: SearchMetrics = Field(default_factory=SearchMetrics, description="搜索指标")
    total_time_ms: float = Field(default=0.0, description="总耗时（毫秒）")


class IngestTriggerRequest(BaseModel):
    """触发摄入任务请求。"""

    workspace_id: str | None = Field(
        default=None,
        max_length=36,
        description="工作空间 ID，不传则处理所有工作空间的 ACTIVE 文件",
    )
    source_file_ids: list[str] | None = Field(
        default=None,
        description="源文件 ID 列表，指定则仅处理这些文件",
    )
    options: MinerUIngestOptions | None = Field(
        default=None,
        description="MinerU 摄入选项配置",
    )
    wait_for_completion: bool = Field(
        default=False,
        description="是否同步等待任务完成（不建议用于大批量）",
    )


class IngestJobResponse(BaseModel):
    """摄入任务详情响应。"""

    id: int = Field(..., description="任务 ID")
    job_type: str = Field(..., description="任务类型")
    status: str = Field(..., description="任务状态")
    progress: int = Field(default=0, ge=0, le=100, description="进度百分比")
    celery_task_id: str | None = Field(None, description="Celery 任务 ID")
    input_summary: dict[str, Any] | None = Field(None, description="输入摘要")
    result_summary: dict[str, Any] | None = Field(None, description="结果摘要")
    error_code: str | None = Field(None, description="错误码")
    error_message: str | None = Field(None, description="错误信息")
    created_at: datetime = Field(..., description="创建时间")
    started_at: datetime | None = Field(None, description="开始时间")
    finished_at: datetime | None = Field(None, description="完成时间")


class IngestTriggerResponse(BaseModel):
    """触发摄入任务响应。"""

    job_id: int = Field(..., description="任务 ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="状态信息")
    input_summary: dict[str, Any] = Field(..., description="输入摘要")
    created_at: datetime = Field(..., description="创建时间")


class IngestJobListResponse(BaseModel):
    """摄入任务列表响应。"""

    items: list[IngestJobResponse] = Field(..., description="任务列表")
    total: int = Field(..., description="总数")


class IngestProgressResponse(BaseModel):
    """摄入进度响应。"""

    job_id: int = Field(..., description="任务 ID")
    status: str = Field(..., description="任务状态")
    progress: int = Field(default=0, ge=0, le=100, description="进度百分比")
    processed_count: int = Field(default=0, description="已处理文件数")
    total_count: int = Field(default=0, description="总文件数")
    success_count: int = Field(default=0, description="成功数")
    failed_count: int = Field(default=0, description="失败数")
    current_file: str | None = Field(None, description="当前处理的文件名")
    result_urls: list[str] | None = Field(None, description="结果下载链接")
    error_message: str | None = Field(None, description="错误信息")
