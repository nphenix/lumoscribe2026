"""提示词实体。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ...shared.db import Base

if TYPE_CHECKING:
    pass


class Prompt(Base):
    """提示词表。

    支持按 scope 进行版本管理与 active 切换。
    """

    __tablename__ = "prompts"
    __table_args__ = (UniqueConstraint("scope", "version", name="uq_prompt_scope_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    scope: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True
    )  # 绑定能力或业务场景
    format: Mapped[str] = mapped_column(
        String(32), nullable=False, default="text"
    )  # text/chat
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    messages_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    active: Mapped[bool] = mapped_column(default=False)
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
