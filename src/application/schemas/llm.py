"""LLM 配置相关 Schema。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LLMProviderCreate(BaseModel):
    """创建 Provider 请求。"""

    name: str = Field(..., min_length=1, max_length=128, description="Provider 名称")
    provider_type: str = Field(..., min_length=1, max_length=64, description="Provider 类型")
    base_url: str | None = Field(None, max_length=2048, description="基础 URL（可填写 endpoint）")
    api_key: str | None = Field(None, max_length=8192, description="API Key/Token 明文（可选）")
    api_key_env: str | None = Field(None, max_length=128, description="API Key 环境变量名（可选）")
    config: dict[str, Any] | None = Field(None, description="额外配置")
    enabled: bool = Field(True, description="是否启用")
    description: str | None = Field(None, max_length=1024, description="说明")


class LLMProviderUpdate(BaseModel):
    """更新 Provider 请求。"""

    name: str | None = Field(None, min_length=1, max_length=128, description="Provider 名称")
    provider_type: str | None = Field(None, min_length=1, max_length=64, description="Provider 类型")
    base_url: str | None = Field(None, max_length=2048, description="基础 URL（可填写 endpoint）")
    api_key: str | None = Field(None, max_length=8192, description="API Key/Token 明文（可选）")
    api_key_env: str | None = Field(None, max_length=128, description="API Key 环境变量名（可选）")
    config: dict[str, Any] | None = Field(None, description="额外配置")
    enabled: bool | None = Field(None, description="是否启用")
    description: str | None = Field(None, max_length=1024, description="说明")


class LLMProviderResponse(BaseModel):
    """Provider 响应。"""

    id: str
    name: str
    provider_type: str
    base_url: str | None
    api_key_env: str | None
    config: dict[str, Any] | None
    enabled: bool
    description: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LLMProviderListResponse(BaseModel):
    """Provider 列表响应。"""

    items: list[LLMProviderResponse]
    total: int
    limit: int
    offset: int


class LLMProviderDeleteResponse(BaseModel):
    """Provider 删除响应。"""

    id: str
    message: str


class LLMModelCreate(BaseModel):
    """创建模型请求。"""

    provider_id: str = Field(..., min_length=1, max_length=36, description="Provider ID")
    name: str = Field(..., min_length=1, max_length=128, description="模型名称")
    model_kind: str = Field(..., min_length=1, max_length=64, description="模型类型")
    config: dict[str, Any] | None = Field(None, description="模型参数")
    enabled: bool = Field(True, description="是否启用")
    description: str | None = Field(None, max_length=1024, description="说明")


class LLMModelUpdate(BaseModel):
    """更新模型请求。"""

    name: str | None = Field(None, min_length=1, max_length=128, description="模型名称")
    model_kind: str | None = Field(None, min_length=1, max_length=64, description="模型类型")
    config: dict[str, Any] | None = Field(None, description="模型参数")
    enabled: bool | None = Field(None, description="是否启用")
    description: str | None = Field(None, max_length=1024, description="说明")


class LLMModelResponse(BaseModel):
    """模型响应。"""

    id: str
    provider_id: str
    name: str
    model_kind: str
    config: dict[str, Any] | None
    enabled: bool
    description: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LLMModelListResponse(BaseModel):
    """模型列表响应。"""

    items: list[LLMModelResponse]
    total: int
    limit: int
    offset: int


class LLMModelDeleteResponse(BaseModel):
    """模型删除响应。"""

    id: str
    message: str


class LLMCapabilityUpsertItem(BaseModel):
    """能力映射 Upsert 请求项。"""

    id: str | None = Field(None, description="能力映射 ID")
    capability: str = Field(..., min_length=1, max_length=128, description="能力名称")
    model_id: str = Field(..., min_length=1, max_length=36, description="模型 ID")
    priority: int = Field(100, ge=0, description="优先级（越小越优先）")
    enabled: bool = Field(True, description="是否启用")
    description: str | None = Field(None, max_length=1024, description="说明")


class LLMCapabilityPatchRequest(BaseModel):
    """能力映射批量 Upsert 请求。"""

    items: list[LLMCapabilityUpsertItem]


class LLMCapabilityResponse(BaseModel):
    """能力映射响应。"""

    id: str
    capability: str
    model_id: str
    priority: int
    enabled: bool
    description: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LLMCapabilityListResponse(BaseModel):
    """能力映射列表响应。"""

    items: list[LLMCapabilityResponse]
    total: int
    limit: int
    offset: int


class LLMCapabilityPatchResponse(BaseModel):
    """能力映射批量 Upsert 响应。"""

    items: list[LLMCapabilityResponse]


# ================= CallSites =================


class LLMCallSiteCreate(BaseModel):
    """创建 CallSite 请求。"""

    key: str = Field(..., min_length=1, max_length=256, description="调用点 key（module:action）")
    expected_model_kind: str = Field(
        ..., min_length=1, max_length=64, description="期望模型类型（chat/embedding/rerank/multimodal）"
    )
    model_id: str | None = Field(None, max_length=36, description="绑定的模型 ID（可为空）")
    config: dict[str, Any] | None = Field(None, description="调用点参数覆盖（JSON）")
    prompt_scope: str | None = Field(None, max_length=256, description="提示词 scope（为空则默认使用 key）")
    enabled: bool = Field(True, description="是否启用")
    description: str | None = Field(None, max_length=1024, description="说明")


class LLMCallSiteUpdate(BaseModel):
    """更新 CallSite 请求（PATCH）。"""

    model_id: str | None = Field(None, max_length=36, description="绑定的模型 ID")
    config: dict[str, Any] | None = Field(None, description="调用点参数覆盖（JSON）")
    prompt_scope: str | None = Field(None, max_length=256, description="提示词 scope")
    enabled: bool | None = Field(None, description="是否启用")
    description: str | None = Field(None, max_length=1024, description="说明")


class LLMCallSiteResponse(BaseModel):
    """CallSite 响应。"""

    id: str
    key: str
    expected_model_kind: str
    model_id: str | None
    config: dict[str, Any] | None
    prompt_scope: str | None
    enabled: bool
    description: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LLMCallSiteListResponse(BaseModel):
    """CallSite 列表响应。"""

    items: list[LLMCallSiteResponse]
    total: int
    limit: int
    offset: int
