"""中间态产物服务层。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from src.application.repositories.intermediate_artifact_repository import (
    IntermediateArtifactRepository,
)
from src.domain.entities.intermediate_artifact import IntermediateArtifact, IntermediateType
from src.shared.errors import AppError


class IntermediateArtifactService:
    """中间态产物服务类。"""

    def __init__(self, repository: IntermediateArtifactRepository):
        self.repository = repository

    def _parse_metadata(self, metadata: str | None) -> dict[str, Any] | None:
        """解析 JSON 元数据。"""
        if metadata is None:
            return None
        try:
            return json.loads(metadata)
        except json.JSONDecodeError:
            return None

    def get_artifact(self, artifact_id: str) -> IntermediateArtifact | None:
        """获取中间态产物详情。"""
        return self.repository.get_by_id(artifact_id)

    def list_artifacts(
        self,
        workspace_id: str | None = None,
        artifact_type: str | None = None,
        source_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[IntermediateArtifact], int]:
        """列出中间态产物。"""
        # 解析类型
        type_enum: IntermediateType | None = None
        if artifact_type is not None:
            try:
                type_enum = IntermediateType(artifact_type)
            except ValueError:
                raise AppError(
                    code="invalid_type",
                    message=f"无效的中间态类型: {artifact_type}",
                    status_code=400,
                )

        items = self.repository.list(
            workspace_id=workspace_id,
            artifact_type=type_enum,
            source_id=source_id,
            limit=limit,
            offset=offset,
        )
        total = self.repository.count(
            workspace_id=workspace_id,
            artifact_type=type_enum,
            source_id=source_id,
        )
        return items, total

    def delete_artifact(self, artifact_id: str) -> bool:
        """删除中间态产物（物理删除文件和数据库记录）。"""
        artifact = self.repository.get_by_id(artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="中间态产物不存在")

        if not artifact.deletable:
            raise HTTPException(status_code=403, detail="该中间态产物不可删除")

        # 删除存储文件
        storage_path = Path("data") / artifact.storage_path
        if storage_path.exists():
            if storage_path.is_dir():
                shutil.rmtree(storage_path)
            else:
                storage_path.unlink()

        # T095：kb_chunks 预建 BM25 索引文件（与报告同名同目录，后缀 .bm25.json）
        try:
            sp = (artifact.storage_path or "").replace("\\", "/")
            if "intermediates/kb_chunks/" in sp and sp.endswith(".json"):
                bm25_path = Path("data") / (sp[:-5] + ".bm25.json")
                if bm25_path.exists():
                    bm25_path.unlink()
        except Exception:
            pass

        # 删除数据库记录
        return self.repository.delete(artifact_id)

    def list_by_source(self, source_id: str) -> list[IntermediateArtifact]:
        """获取指定源文件产生的所有中间态产物。"""
        return self.repository.list_by_source(source_id)

    def batch_delete_by_source(self, source_id: str) -> list[str]:
        """批量删除指定源文件产生的所有中间态产物。"""
        artifacts = self.repository.list_by_source(source_id)
        deleted_ids = []

        for artifact in artifacts:
            if artifact.deletable:
                # 删除存储文件
                storage_path = Path("data") / artifact.storage_path
                if storage_path.exists():
                    if storage_path.is_dir():
                        shutil.rmtree(storage_path)
                    else:
                        storage_path.unlink()

                # 删除数据库记录
                if self.repository.delete(artifact.id):
                    deleted_ids.append(artifact.id)

        return deleted_ids
