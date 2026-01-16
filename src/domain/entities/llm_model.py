"""LLM 模型实体。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ...shared.db import Base

if TYPE_CHECKING:
    pass


class LLMModel(Base):
    """LLM 模型表。

    用于记录具体模型参数与所属 provider。
    """

    __tablename__ = "llm_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    provider_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("llm_providers.id"), nullable=False, index=True
    )

    name: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True
    )  # 模型名称
    model_kind: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # chat/embedding/rerank 等
    config_json: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # 模型参数 JSON（字符串）

    enabled: Mapped[bool] = mapped_column(default=True)
    description: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
