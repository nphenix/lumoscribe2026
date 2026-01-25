"""源文件实体。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ...shared.db import Base

if TYPE_CHECKING:
    pass


class SourceFileStatus(str, PyEnum):
    """源文件状态枚举。"""

    ACTIVE = "active"
    ARCHIVED = "archived"
    MINERU_PROCESSING = "mineru_processing"
    MINERU_COMPLETED = "mineru_completed"
    CLEANING_PROCESSING = "cleaning_processing"
    CLEANING_COMPLETED = "cleaning_completed"


class SourceFile(Base):
    """源文件元数据表。

    用于管理用户上传的 PDF 源文件，支持 CRUD 与归档操作。
    """

    __tablename__ = "source_files"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True
    )  # UUID 格式
    workspace_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )  # 关联工作空间

    # 文件信息
    original_filename: Mapped[str] = mapped_column(
        String(512), nullable=False
    )  # 原始文件名
    file_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )  # SHA256 哈希值
    file_size: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )  # 文件大小（字节）
    storage_path: Mapped[str] = mapped_column(
        String(1024), nullable=False
    )  # 相对存储路径（data/sources/{workspace_name}/{filename}）

    # 状态管理
    status: Mapped[str] = mapped_column(
        Enum(SourceFileStatus),
        nullable=False,
        default=SourceFileStatus.ACTIVE,
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

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

    # 可选描述
    description: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
