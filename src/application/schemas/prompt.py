"""提示词相关 Schema。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class PromptMessage(BaseModel):
    """聊天消息模板。"""

    role: str = Field(..., description="角色: system/user/assistant")
    content: str = Field(..., description="模板内容")


class PromptCreate(BaseModel):
    """创建提示词请求。"""

    scope: str = Field(..., min_length=1, max_length=128, description="提示词范围")
    format: str = Field("text", description="提示词格式: text/chat")
    content: str | None = Field(None, description="文本模板内容")
    messages: list[PromptMessage] | None = Field(None, description="聊天消息模板")
    active: bool = Field(True, description="是否激活")
    description: str | None = Field(None, max_length=1024, description="说明")

    @model_validator(mode="after")
    def validate_format(self):
        if self.format == "text":
            if not self.content:
                raise ValueError("text 格式必须提供 content")
            if self.messages:
                raise ValueError("text 格式不应包含 messages")
        if self.format == "chat":
            if not self.messages:
                raise ValueError("chat 格式必须提供 messages")
        return self


class PromptUpdate(BaseModel):
    """更新提示词请求。"""

    active: bool | None = Field(None, description="是否激活")
    description: str | None = Field(None, max_length=1024, description="说明")


class PromptResponse(BaseModel):
    """提示词响应。"""

    id: str
    scope: str
    format: str
    content: str | None
    messages: list[PromptMessage] | None
    version: int
    active: bool
    description: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PromptListResponse(BaseModel):
    """提示词列表响应。"""

    items: list[PromptResponse]
    total: int
    limit: int
    offset: int


class PromptScopeSummary(BaseModel):
    scope: str
    format: str | None
    latest_version: int
    active_version: int | None
    versions: int
    updated_at: datetime | None


class PromptScopeListResponse(BaseModel):
    items: list[PromptScopeSummary]
    total: int
    limit: int
    offset: int


class PromptDiffResponse(BaseModel):
    from_id: str
    to_id: str
    scope: str | None
    diff: str


class PromptDeleteResponse(BaseModel):
    """删除提示词响应。"""

    id: str
    message: str
