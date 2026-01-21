"""中间态产物实体。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, String
from sqlalchemy.orm import Mapped, mapped_column

from ...shared.db import Base

if TYPE_CHECKING:
    pass


class IntermediateType(str, PyEnum):
    """中间态产物类型枚举。"""

    MINERU_RAW = "mineru_raw"  # MinerU 原始输出
    CLEANED_DOC = "cleaned_doc"  # 清洗后文档
    CHART_JSON = "chart_json"  # 图表 JSON
    KB_CHUNKS = "kb_chunks"  # 知识库切块


class IntermediateArtifact(Base):
    """中间态产物表。

    用于存储和管理 Ingest 流水线的中间产物，支持观测与删除。
    """

    __tablename__ = "intermediate_artifacts"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True
    )  # UUID 格式
    workspace_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )  # 关联工作空间

    # 关联关系
    source_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )  # 关联源文件（可选）

    # 产物信息
    type: Mapped[str] = mapped_column(
        Enum(IntermediateType),
        nullable=False,
    )
    storage_path: Mapped[str] = mapped_column(
        String(1024), nullable=False
    )  # 相对存储路径

    # 删除标记
    deletable: Mapped[bool] = mapped_column(
        default=True
    )  # 是否可删除（系统产物可能不可删除）

    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # 可选元数据
    extra_metadata: Mapped[str | None] = mapped_column(
        String(4096), nullable=True
    )  # JSON 格式的额外元数据
