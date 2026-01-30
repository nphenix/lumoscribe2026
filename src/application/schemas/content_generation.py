"""内容生成 API Pydantic 模型定义。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.application.schemas.ingest import HybridSearchOptions


class GenerateRequest(BaseModel):
    """内容生成请求。"""

    template_id: str = Field(
        ...,
        min_length=36,
        max_length=36,
        description="模板 ID（UUID 格式）",
    )
    collection_name: str = Field(
        default="default",
        description="知识库 collection 名称",
    )
    document_title: str | None = Field(
        default=None,
        max_length=200,
        description="文档标题（用于 HTML 渲染）",
    )
    outline_polish_enabled: bool = Field(
        default=False,
        description="是否启用大纲润色（仅对自定义模板有效）",
    )
    polish_sections: bool = Field(
        default=False,
        description="是否启用章节级语言润色（不新增外部事实）",
    )
    search_options: HybridSearchOptions | None = Field(
        default=None,
        description="混合检索选项",
    )
    coverage_score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="覆盖评估分数阈值，低于此值的条目标记为未覆盖",
    )
    kb_input_root: str | None = Field(
        default=None,
        description="知识库输入根目录路径（用于自动图表渲染）",
    )
    max_auto_charts_per_section: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description="每个章节最多自动渲染的图表数量",
    )
    stream_tokens: bool = Field(
        default=False,
        description="是否启用 token 级别流式输出",
    )


class GenerateResponse(BaseModel):
    """内容生成响应。"""

    job_id: int = Field(..., description="任务 ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="状态信息")
    template_id: str = Field(..., description="模板 ID")
    template_name: str = Field(..., description="模板名称")
    document_title: str | None = Field(None, description="文档标题")
    created_at: datetime = Field(..., description="创建时间")


class GenerateResultResponse(BaseModel):
    """内容生成结果响应（同步返回或任务完成后查询）。"""

    job_id: int = Field(..., description="任务 ID")
    status: str = Field(..., description="任务状态")
    template_id: str = Field(..., description="模板 ID")
    template_name: str = Field(..., description="模板名称")
    document_title: str | None = Field(None, description="文档标题")
    html_content: str | None = Field(None, description="生成的 HTML 内容")
    sections: list[dict[str, Any]] | None = Field(None, description="章节生成结果")
    total_tokens: int = Field(default=0, description="总 token 使用量")
    total_time_ms: float = Field(default=0.0, description="总耗时（毫秒）")
    generated_at: datetime | None = Field(None, description="生成完成时间")
    error_code: str | None = Field(None, description="错误码")
    error_message: str | None = Field(None, description="错误信息")
    created_at: datetime = Field(..., description="创建时间")
    finished_at: datetime | None = Field(None, description="完成时间")


class GenerateProgressResponse(BaseModel):
    """内容生成进度响应。"""

    job_id: int = Field(..., description="任务 ID")
    status: str = Field(..., description="任务状态")
    progress: int = Field(default=0, ge=0, le=100, description="进度百分比")
    current_section: str | None = Field(None, description="当前处理的章节标题")
    section_index: int = Field(default=0, description="当前章节索引")
    section_total: int = Field(default=0, description="总章节数")
    tokens_used: int = Field(default=0, description="已使用的 token")
    error_message: str | None = Field(None, description="错误信息")


class OutlinePolishRequest(BaseModel):
    """大纲润色请求。"""

    outline: str = Field(
        ...,
        min_length=10,
        max_length=50000,
        description="原始大纲内容（Markdown 格式）",
    )


class OutlinePolishResponse(BaseModel):
    """大纲润色响应。"""

    polished_outline: str = Field(..., description="润色后的大纲内容")


class GenerateOptions(BaseModel):
    """内容生成选项（用于前端可选配置）。"""

    enable_outline_polish: bool = Field(
        default=False,
        description="是否启用大纲润色功能",
    )
    enable_auto_chart_rendering: bool = Field(
        default=True,
        description="是否启用自动图表渲染",
    )
    max_charts_per_section: int = Field(
        default=6,
        ge=0,
        le=20,
        description="每个章节最多渲染的图表数量",
    )
    coverage_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="覆盖评估阈值",
    )
