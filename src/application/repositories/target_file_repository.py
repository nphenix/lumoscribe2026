"""目标文件仓储层。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from sqlalchemy.orm import Session

from src.domain.entities.target_file import TargetFile


class TargetFileRepository:
    """目标文件仓储类。"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, target_file: TargetFile) -> TargetFile:
        """创建目标文件。"""
        self.db.add(target_file)
        self.db.commit()
        self.db.refresh(target_file)
        return target_file

    def get_by_id(self, target_id: str) -> TargetFile | None:
        """根据ID获取目标文件。"""
        return self.db.query(TargetFile).filter(TargetFile.id == target_id).first()

    def list(
        self,
        workspace_id: str | None = None,
        template_id: str | None = None,
        kb_id: str | None = None,
        job_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TargetFile]:
        """列出目标文件，支持多条件过滤。"""
        query = self.db.query(TargetFile)

        if workspace_id is not None:
            query = query.filter(TargetFile.workspace_id == workspace_id)

        if template_id is not None:
            query = query.filter(TargetFile.template_id == template_id)

        if kb_id is not None:
            query = query.filter(TargetFile.kb_id == kb_id)

        if job_id is not None:
            query = query.filter(TargetFile.job_id == job_id)

        if created_after is not None:
            query = query.filter(TargetFile.created_at >= created_after)

        if created_before is not None:
            query = query.filter(TargetFile.created_at <= created_before)

        return query.order_by(TargetFile.created_at.desc()).offset(offset).limit(limit).all()

    def count(
        self,
        workspace_id: str | None = None,
        template_id: str | None = None,
        kb_id: str | None = None,
        job_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> int:
        """统计目标文件数量。"""
        query = self.db.query(TargetFile)

        if workspace_id is not None:
            query = query.filter(TargetFile.workspace_id == workspace_id)

        if template_id is not None:
            query = query.filter(TargetFile.template_id == template_id)

        if kb_id is not None:
            query = query.filter(TargetFile.kb_id == kb_id)

        if job_id is not None:
            query = query.filter(TargetFile.job_id == job_id)

        if created_after is not None:
            query = query.filter(TargetFile.created_at >= created_after)

        if created_before is not None:
            query = query.filter(TargetFile.created_at <= created_before)

        return query.count()

    def update(self, target_file: TargetFile) -> TargetFile:
        """更新目标文件。"""
        target_file.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(target_file)
        return target_file

    def delete(self, target_id: str) -> bool:
        """删除目标文件。"""
        target_file = self.get_by_id(target_id)
        if target_file is None:
            return False

        self.db.delete(target_file)
        self.db.commit()
        return True
