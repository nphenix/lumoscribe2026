"""模板实体。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ...shared.db import Base

if TYPE_CHECKING:
    pass


class TemplateType(str, PyEnum):
    """模板类型枚举。"""

    CUSTOM = "custom"  # 用户自定义模板
    SYSTEM = "system"  # 系统预置模板


class Template(Base):
    """模板文件表。

    用于管理生成目标文件的模板，支持版本控制与锁定。
    """

    __tablename__ = "templates"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True
    )  # UUID 格式
    workspace_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )  # 关联工作空间

    # 文件信息
    original_filename: Mapped[str] = mapped_column(
        String(512), nullable=False
    )  # 原始模板文件名
    file_format: Mapped[str] = mapped_column(
        String(32), nullable=False, default="html"
    )  # 文件格式（html/jinja2/markdown等）

    # 模板属性
    type: Mapped[str] = mapped_column(
        Enum(TemplateType),
        nullable=False,
        default=TemplateType.CUSTOM,
    )
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )  # 版本号
    locked: Mapped[bool] = mapped_column(
        default=False
    )  # 锁定标记，锁定后不可修改
    storage_path: Mapped[str] = mapped_column(
        String(1024), nullable=False
    )  # 相对存储路径

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
