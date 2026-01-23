"""中间态产物仓储层。"""

from __future__ import annotations

from typing import Literal

from sqlalchemy.orm import Session

from src.domain.entities.intermediate_artifact import IntermediateArtifact, IntermediateType


class IntermediateArtifactRepository:
    """中间态产物仓储类。"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, artifact: IntermediateArtifact) -> IntermediateArtifact:
        """创建中间态产物。"""
        self.db.add(artifact)
        self.db.commit()
        self.db.refresh(artifact)
        return artifact

    def update(self, artifact: IntermediateArtifact) -> IntermediateArtifact:
        """更新中间态产物。"""
        self.db.add(artifact)
        self.db.commit()
        self.db.refresh(artifact)
        return artifact

    def update_extra_metadata(
        self,
        artifact_id: str,
        *,
        extra_metadata: str | None,
    ) -> IntermediateArtifact | None:
        """仅更新 extra_metadata 字段（用于进度观测等高频更新）。"""
        artifact = self.get_by_id(artifact_id)
        if artifact is None:
            return None
        artifact.extra_metadata = extra_metadata
        self.db.add(artifact)
        self.db.commit()
        self.db.refresh(artifact)
        return artifact

    def get_by_id(self, artifact_id: str) -> IntermediateArtifact | None:
        """根据ID获取中间态产物。"""
        return self.db.query(IntermediateArtifact).filter(IntermediateArtifact.id == artifact_id).first()

    def list(
        self,
        workspace_id: str | None = None,
        artifact_type: IntermediateType | None = None,
        source_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[IntermediateArtifact]:
        """列出中间态产物。"""
        query = self.db.query(IntermediateArtifact)

        if workspace_id is not None:
            query = query.filter(IntermediateArtifact.workspace_id == workspace_id)

        if artifact_type is not None:
            query = query.filter(IntermediateArtifact.type == artifact_type)

        if source_id is not None:
            query = query.filter(IntermediateArtifact.source_id == source_id)

        return query.order_by(IntermediateArtifact.created_at.desc()).offset(offset).limit(limit).all()

    def count(
        self,
        workspace_id: str | None = None,
        artifact_type: IntermediateType | None = None,
        source_id: str | None = None,
    ) -> int:
        """统计中间态产物数量。"""
        query = self.db.query(IntermediateArtifact)

        if workspace_id is not None:
            query = query.filter(IntermediateArtifact.workspace_id == workspace_id)

        if artifact_type is not None:
            query = query.filter(IntermediateArtifact.type == artifact_type)

        if source_id is not None:
            query = query.filter(IntermediateArtifact.source_id == source_id)

        return query.count()

    def list_by_source(self, source_id: str) -> list[IntermediateArtifact]:
        """获取指定源文件产生的所有中间态产物。"""
        return (
            self.db.query(IntermediateArtifact)
            .filter(IntermediateArtifact.source_id == source_id)
            .order_by(IntermediateArtifact.created_at.desc())
            .all()
        )

    def delete(self, artifact_id: str) -> bool:
        """删除中间态产物（物理删除）。"""
        artifact = self.get_by_id(artifact_id)
        if artifact is None:
            return False

        self.db.delete(artifact)
        self.db.commit()
        return True
