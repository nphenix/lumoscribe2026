"""文档清洗与图表提取 Schema 定义。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ChartType(str, Enum):
    """图表类型枚举。"""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    TABLE = "table"
    SCATTER = "scatter"
    AREA = "area"
    RADAR = "radar"
    HEATMAP = "heatmap"
    OTHER = "other"


class CleaningStats(BaseModel):
    """清洗统计信息。"""

    original_chars: int = Field(..., description="原始字符数")
    cleaned_chars: int = Field(..., description="清洗后字符数")
    removed_chars: int = Field(..., description="移除的字符数")
    original_paragraphs: int = Field(..., description="原始段落数")
    cleaned_paragraphs: int = Field(..., description="清洗后段落数")
    removed_paragraphs: int = Field(..., description="移除的段落数")
    ads_removed: int = Field(default=0, description="移除的广告数量")
    noise_removed: int = Field(default=0, description="移除的噪声数量")
    duplicates_removed: int = Field(default=0, description="移除的重复内容数量")


class CleaningOptions(BaseModel):
    """文档清洗选项。"""

    remove_ads: bool = Field(default=True, description="是否移除广告")
    remove_noise: bool = Field(default=True, description="是否移除噪声（页眉页脚、页码等）")
    remove_duplicates: bool = Field(default=True, description="是否移除重复内容")
    normalize_whitespace: bool = Field(default=True, description="是否标准化空白字符")
    preserve_structure: bool = Field(default=True, description="是否保留文档结构")
    language: str = Field(default="zh", description="文档语言")


class CleanedDocumentArtifact(BaseModel):
    """清洗后的文档产物。"""

    source_file_id: str = Field(..., description="源文件 ID")
    original_text: str = Field(..., description="原始文本")
    cleaned_text: str = Field(..., description="清洗后文本")
    cleaning_stats: CleaningStats = Field(..., description="清洗统计信息")
    output_path: str = Field(..., description="输出文件路径")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


class ChartData(BaseModel):
    """单个图表数据。"""

    chart_id: str = Field(..., description="图表唯一标识")
    chart_type: ChartType = Field(..., description="图表类型")
    json_content: dict[str, Any] = Field(..., description="图表 JSON 内容")
    source_image_path: str = Field(..., description="源图片路径")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    page_number: int | None = Field(None, description="页码")


class ChartJSONArtifact(BaseModel):
    """图表 JSON 产物。"""

    source_file_id: str = Field(..., description="源文件 ID")
    charts: list[ChartData] = Field(..., description="图表列表")
    output_path: str = Field(..., description="输出文件路径")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")


class CleaningResult(BaseModel):
    """文档清洗结果。"""

    success: bool = Field(..., description="是否成功")
    cleaned_text: str | None = Field(None, description="清洗后文本")
    stats: CleaningStats | None = Field(None, description="清洗统计")
    error: str | None = Field(None, description="错误信息")


class ChartExtractionResult(BaseModel):
    """图表提取结果。"""

    success: bool = Field(..., description="是否成功")
    charts: list[ChartData] | None = Field(None, description="图表列表")
    error: str | None = Field(None, description="错误信息")
