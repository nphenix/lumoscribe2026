"""LLM Provider 实体。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ...shared.db import Base

if TYPE_CHECKING:
    pass


class LLMProvider(Base):
    """LLM Provider 表。

    用于管理模型供应商连接信息与密钥环境变量名称。
    """

    __tablename__ = "llm_providers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    key: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )  # Provider 唯一标识（用于 seed / 引用）
    name: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True
    )  # Provider 名称
    provider_type: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # openai_compatible/ollama/gemini/flagembedding 等

    base_url: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )  # OpenAI-compatible base_url 等
    api_key: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # API Key/Token 明文（可选）
    api_key_env: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )  # API Key 环境变量名（不存明文）
    config_json: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # 额外配置 JSON（字符串）

    enabled: Mapped[bool] = mapped_column(default=True)  # 是否启用
    description: Mapped[str | None] = mapped_column(
        String(1024), nullable=True
    )  # 备注说明

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
