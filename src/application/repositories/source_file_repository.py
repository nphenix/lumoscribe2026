"""源文件仓储层。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from sqlalchemy.orm import Session

from src.domain.entities.source_file import SourceFile, SourceFileStatus


class SourceFileRepository:
    """源文件仓储类。"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, source_file: SourceFile) -> SourceFile:
        """创建源文件。"""
        self.db.add(source_file)
        self.db.commit()
        self.db.refresh(source_file)
        return source_file

    def get_by_id(self, source_id: str) -> SourceFile | None:
        """根据ID获取源文件。"""
        return self.db.query(SourceFile).filter(SourceFile.id == source_id).first()

    def get_by_hash(self, file_hash: str) -> SourceFile | None:
        """根据文件哈希获取源文件。"""
        return self.db.query(SourceFile).filter(SourceFile.file_hash == file_hash).first()

    def list(
        self,
        workspace_id: str | None = None,
        status: SourceFileStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SourceFile]:
        """列出源文件。"""
        query = self.db.query(SourceFile)

        if workspace_id is not None:
            query = query.filter(SourceFile.workspace_id == workspace_id)

        if status is not None:
            query = query.filter(SourceFile.status == status)

        return query.order_by(SourceFile.created_at.desc()).offset(offset).limit(limit).all()

    def count(
        self,
        workspace_id: str | None = None,
        status: SourceFileStatus | None = None,
    ) -> int:
        """统计源文件数量。"""
        query = self.db.query(SourceFile)

        if workspace_id is not None:
            query = query.filter(SourceFile.workspace_id == workspace_id)

        if status is not None:
            query = query.filter(SourceFile.status == status)

        return query.count()

    def update(self, source_file: SourceFile) -> SourceFile:
        """更新源文件。"""
        source_file.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(source_file)
        return source_file

    def update_status(
        self,
        source_id: str,
        status: SourceFileStatus,
        archived_at: datetime | None = None,
    ) -> SourceFile | None:
        """更新源文件状态。"""
        source_file = self.get_by_id(source_id)
        if source_file is None:
            return None

        source_file.status = status
        source_file.archived_at = archived_at
        source_file.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(source_file)
        return source_file

    def update_storage_path(self, source_id: str, storage_path: str) -> SourceFile | None:
        """更新源文件存储路径。"""
        source_file = self.get_by_id(source_id)
        if source_file is None:
            return None

        source_file.storage_path = storage_path
        source_file.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(source_file)
        return source_file

    def delete(self, source_id: str) -> bool:
        """删除源文件。"""
        source_file = self.get_by_id(source_id)
        if source_file is None:
            return False

        self.db.delete(source_file)
        self.db.commit()
        return True
