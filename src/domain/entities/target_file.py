"""目标文件实体。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from ...shared.db import Base

if TYPE_CHECKING:
    pass


class TargetFile(Base):
    """目标文件表。

    用于存储和管理生成的最终目标文件（单 HTML）。
    """

    __tablename__ = "target_files"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True
    )  # UUID 格式
    workspace_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )  # 关联工作空间

    # 关联关系
    template_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )  # 关联模板
    kb_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )  # 关联知识库（可选）
    job_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )  # 关联生成任务

    # 文件信息
    output_filename: Mapped[str] = mapped_column(
        String(512), nullable=False
    )  # 输出文件名
    storage_path: Mapped[str] = mapped_column(
        String(1024), nullable=False
    )  # 相对存储路径（data/targets/{workspace_name}/{filename}）

    # 可选描述
    description: Mapped[str | None] = mapped_column(
        String(1024), nullable=True
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
