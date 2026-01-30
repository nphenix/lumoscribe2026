"""LLM 调用点（Callsite）实体。

用于把"某个业务调用点"绑定到具体 Provider 与参数覆盖配置：
- key: 稳定的调用点标识（建议 module:action）
- expected_model_kind: 期望的模型类型（chat/embedding/rerank/multimodal/ocr），运行时根据此字段构建对应类型的模型
- provider_id: 绑定的 Provider（可为空，表示已注册但未配置）
- config_json: 调用点级参数覆盖（优先级最高）
- prompt_scope: chat/multimodal 时使用的提示词 scope（为空则使用 key）
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ...shared.db import Base


class LLMCallSite(Base):
    """LLM 调用点表。"""

    __tablename__ = "llm_call_sites"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID
    key: Mapped[str] = mapped_column(
        String(256), nullable=False, unique=True, index=True
    )  # 调用点 key（module:action）

    expected_model_kind: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # chat/embedding/rerank/multimodal/ocr

    provider_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("llm_providers.id"), nullable=True, index=True
    )

    config_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_scope: Mapped[str | None] = mapped_column(String(256), nullable=True)
    max_concurrency: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )  # CallSite 级并发上限（为空表示继承 Provider）

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
