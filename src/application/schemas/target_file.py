"""目标文件 Pydantic 模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TargetFileResponse(BaseModel):
    """目标文件详情响应。"""

    id: str = Field(..., description="目标文件ID")
    workspace_id: str = Field(..., description="工作空间ID")
    template_id: str = Field(..., description="关联模板ID")
    kb_id: str | None = Field(None, description="关联知识库ID")
    job_id: str | None = Field(None, description="关联生成任务ID")
    output_filename: str = Field(..., description="输出文件名")
    storage_path: str = Field(..., description="存储路径")
    description: str | None = Field(None, description="文件描述")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")

    # 关联信息
    template_name: str | None = Field(None, description="模板名称")
    kb_name: str | None = Field(None, description="知识库名称")

    class Config:
        from_attributes = True


class TargetFileListItem(BaseModel):
    """目标文件列表项。"""

    id: str = Field(..., description="目标文件ID")
    workspace_id: str = Field(..., description="工作空间ID")
    template_id: str = Field(..., description="关联模板ID")
    kb_id: str | None = Field(None, description="关联知识库ID")
    job_id: str | None = Field(None, description="关联生成任务ID")
    output_filename: str = Field(..., description="输出文件名")
    storage_path: str = Field(..., description="存储路径")
    description: str | None = Field(None, description="文件描述")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")

    class Config:
        from_attributes = True


class TargetFileListResponse(BaseModel):
    """目标文件列表响应。"""

    items: list[TargetFileListItem] = Field(..., description="目标文件列表")
    total: int = Field(..., description="总数")


class DeleteTargetFileResponse(BaseModel):
    """删除目标文件响应。"""

    id: str = Field(..., description="目标文件ID")
    message: str = Field(..., description="操作结果消息")


class ErrorDetail(BaseModel):
    """错误详情。"""

    code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误消息")
    request_id: str | None = Field(None, description="请求ID")
    details: dict[str, Any] | None = Field(None, description="详细信息")


class ErrorResponse(BaseModel):
    """统一错误响应。"""

    error: ErrorDetail
