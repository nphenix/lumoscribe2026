"""中间态产物 Pydantic 模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.domain.entities.intermediate_artifact import IntermediateType


class IntermediateArtifactResponse(BaseModel):
    """中间态产物详情响应。"""

    id: str = Field(..., description="产物ID")
    workspace_id: str = Field(..., description="工作空间ID")
    source_id: str | None = Field(None, description="关联源文件ID")
    type: IntermediateType = Field(..., description="产物类型")
    storage_path: str = Field(..., description="相对存储路径")
    deletable: bool = Field(default=True, description="是否可删除")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    metadata: dict[str, Any] | None = Field(None, description="元数据")

    class Config:
        from_attributes = True


class IntermediateArtifactListItem(BaseModel):
    """中间态产物列表项。"""

    id: str = Field(..., description="产物ID")
    workspace_id: str = Field(..., description="工作空间ID")
    source_id: str | None = Field(None, description="关联源文件ID")
    type: IntermediateType = Field(..., description="产物类型")
    storage_path: str = Field(..., description="相对存储路径")
    deletable: bool = Field(default=True, description="是否可删除")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")

    class Config:
        from_attributes = True


class IntermediateArtifactListResponse(BaseModel):
    """中间态产物列表响应。"""

    items: list[IntermediateArtifactListItem]
    total: int


class DeleteIntermediateArtifactResponse(BaseModel):
    """删除中间态产物响应。"""

    id: str = Field(..., description="被删除的产物ID")
    message: str = Field(default="删除成功", description="操作结果信息")


class BatchDeleteIntermediateArtifactsResponse(BaseModel):
    """批量删除中间态产物响应。"""

    deleted_ids: list[str] = Field(..., description="已删除的产物ID列表")
    count: int = Field(..., description="删除数量")
    message: str = Field(default="批量删除成功", description="操作结果信息")
