"""LLM 能力映射实体。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ...shared.db import Base

if TYPE_CHECKING:
    pass


class LLMCapability(Base):
    """LLM 能力映射表。

    capability -> provider 的优先级映射，可配置启用状态。
    """

    __tablename__ = "llm_capabilities"
    __table_args__ = (
        UniqueConstraint("capability", "provider_id", name="uq_capability_provider"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    capability: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True
    )  # 能力名称
    provider_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("llm_providers.id"), nullable=False, index=True
    )

    priority: Mapped[int] = mapped_column(
        Integer, nullable=False, default=100
    )  # 优先级（数值越小优先级越高）
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
