"""源文件 Pydantic 模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.domain.entities.source_file import SourceFileStatus


class SourceFileCreate(BaseModel):
    """创建源文件请求。"""

    workspace_id: str = Field(..., min_length=1, max_length=36, description="工作空间ID")
    original_filename: str = Field(..., min_length=1, max_length=512, description="原始文件名")
    file_hash: str = Field(..., min_length=64, max_length=64, description="SHA256 文件哈希")
    description: str | None = Field(None, max_length=2000, description="文件描述")


class SourceFileUpdate(BaseModel):
    """更新源文件请求。"""

    description: str | None = Field(None, max_length=2000, description="文件描述")


class SourceFileResponse(BaseModel):
    """源文件响应。"""

    id: str
    workspace_id: str
    original_filename: str
    file_hash: str
    file_size: int
    storage_path: str
    status: SourceFileStatus
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime
    description: str | None

    class Config:
        from_attributes = True


class SourceFileListItem(BaseModel):
    """源文件列表项。"""

    id: str
    workspace_id: str
    original_filename: str
    file_size: int
    storage_path: str
    status: SourceFileStatus
    created_at: datetime
    updated_at: datetime
    description: str | None

    class Config:
        from_attributes = True


class SourceFileListResponse(BaseModel):
    """源文件列表响应。"""

    items: list[SourceFileListItem]
    total: int


class SourceFileUploadResponse(BaseModel):
    """源文件上传响应。"""

    id: str
    workspace_id: str
    original_filename: str
    storage_path: str
    status: SourceFileStatus
    created_at: datetime


class ArchiveSourceFileResponse(BaseModel):
    """归档源文件响应。"""

    id: str
    workspace_id: str
    original_filename: str
    storage_path: str
    status: SourceFileStatus
    archived_at: datetime | None
    message: str


class UnarchiveSourceFileResponse(BaseModel):
    """取消归档源文件响应。"""

    id: str
    workspace_id: str
    original_filename: str
    storage_path: str
    status: SourceFileStatus
    message: str


class DeleteSourceFileResponse(BaseModel):
    """删除源文件响应。"""

    id: str
    message: str


class ErrorDetail(BaseModel):
    """错误详情。"""

    code: str
    message: str
    request_id: str | None
    details: dict[str, Any] | None


class ErrorResponse(BaseModel):
    """统一错误响应。"""

    error: ErrorDetail
