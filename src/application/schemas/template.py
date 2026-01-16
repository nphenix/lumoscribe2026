"""模板 Pydantic 模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TemplateCreate(BaseModel):
    """模板创建请求。"""

    workspace_id: str = Field(..., min_length=1, max_length=36, description="工作空间ID")
    description: str | None = Field(None, max_length=1024, description="模板描述")


class TemplateUpdate(BaseModel):
    """模板更新请求。"""

    description: str | None = Field(None, max_length=1024, description="模板描述")


class TemplateResponse(BaseModel):
    """模板响应。"""

    id: str
    workspace_id: str
    original_filename: str
    file_format: str
    type: str
    version: int
    locked: bool
    storage_path: str
    description: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TemplateListResponse(BaseModel):
    """模板列表响应。"""

    items: list[TemplateResponse]
    total: int
    limit: int
    offset: int


class TemplateUploadResponse(BaseModel):
    """模板上传响应。"""

    id: str
    workspace_id: str
    original_filename: str
    file_format: str
    type: str
    version: int
    locked: bool
    storage_path: str
    description: str | None
    created_at: datetime


class DeleteTemplateResponse(BaseModel):
    """删除模板响应。"""

    id: str
    message: str


class LockTemplateResponse(BaseModel):
    """锁定模板响应。"""

    id: str
    locked: bool
    message: str


class PreprocessCheck(BaseModel):
    """预处理检查项。"""

    type: str = Field(..., description="检查类型")
    status: str = Field(..., description="检查状态: passed, warning, error")
    message: str = Field(..., description="检查消息")
    details: dict | None = Field(None, description="检查详情")


class PreprocessResponse(BaseModel):
    """预处理校验响应。"""

    template_id: str
    template_type: str
    checks: list[PreprocessCheck]
    overall_status: str = Field(..., description="整体状态: passed, warning, error")
    message: str = Field(..., description="整体消息")

    class Config:
        from_attributes = True


class PreprocessDetail(BaseModel):
    """预处理详情。"""

    placeholder_name: str | None = None
    line_number: int | None = None
    context: str | None = None
