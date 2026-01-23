"""中间态产物 API 路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.application.repositories.intermediate_artifact_repository import (
    IntermediateArtifactRepository,
)
from src.application.schemas.intermediate_artifact import (
    BatchDeleteIntermediateArtifactsResponse,
    DeleteIntermediateArtifactResponse,
    IntermediateArtifactListResponse,
    IntermediateArtifactResponse,
)
from src.application.services.intermediate_artifact_service import (
    IntermediateArtifactService,
)
from src.interfaces.api.deps import get_db


router = APIRouter()


def get_intermediate_artifact_service(db: Session = Depends(get_db)) -> IntermediateArtifactService:
    """获取中间态产物服务实例。"""
    repository = IntermediateArtifactRepository(db)
    return IntermediateArtifactService(repository)


@router.get(
    "/intermediates",
    response_model=IntermediateArtifactListResponse,
    summary="列出中间态产物",
    description="获取中间态产物列表，支持按类型、工作空间和来源过滤。",
)
def list_intermediates(
    workspace_id: Annotated[str | None, Query(max_length=36, description="工作空间ID")] = None,
    artifact_type: Annotated[
        str | None, Query(description="中间态类型: mineru_raw, cleaned_doc, chart_json, kb_chunks")
    ] = None,
    source_id: Annotated[str | None, Query(max_length=36, description="关联源文件ID")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: IntermediateArtifactService = Depends(get_intermediate_artifact_service),
):
    """列出中间态产物。"""
    items, total = service.list_artifacts(
        workspace_id=workspace_id,
        artifact_type=artifact_type,
        source_id=source_id,
        limit=limit,
        offset=offset,
    )

    return IntermediateArtifactListResponse(
        items=[
            {
                "id": item.id,
                "workspace_id": item.workspace_id,
                "source_id": item.source_id,
                "type": item.type,
                "storage_path": item.storage_path,
                "deletable": item.deletable,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in items
        ],
        total=total,
    )


@router.get(
    "/intermediates/{artifact_id}",
    response_model=IntermediateArtifactResponse,
    summary="获取中间态产物详情",
    description="获取指定中间态产物的详细信息，包括元数据。",
)
def get_intermediate(
    artifact_id: str,
    service: IntermediateArtifactService = Depends(get_intermediate_artifact_service),
):
    """获取中间态产物详情。"""
    artifact = service.get_artifact(artifact_id)
    if artifact is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="中间态产物不存在")

    # 解析元数据
    metadata = service._parse_metadata(artifact.extra_metadata)

    return IntermediateArtifactResponse(
        id=artifact.id,
        workspace_id=artifact.workspace_id,
        source_id=artifact.source_id,
        type=artifact.type,
        storage_path=artifact.storage_path,
        deletable=artifact.deletable,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
        metadata=metadata,
    )


@router.delete(
    "/intermediates/{artifact_id}",
    response_model=DeleteIntermediateArtifactResponse,
    summary="删除中间态产物",
    description="删除指定的中间态产物及其关联的存储文件。",
)
def delete_intermediate(
    artifact_id: str,
    service: IntermediateArtifactService = Depends(get_intermediate_artifact_service),
):
    """删除中间态产物。"""
    success = service.delete_artifact(artifact_id)
    if not success:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="中间态产物不存在")

    return DeleteIntermediateArtifactResponse(
        id=artifact_id,
        message="中间态产物已删除",
    )


@router.delete(
    "/intermediates/by-source/{source_id}",
    response_model=BatchDeleteIntermediateArtifactsResponse,
    summary="批量删除中间态产物",
    description="删除指定源文件产生的所有中间态产物。",
)
def batch_delete_intermediates_by_source(
    source_id: str,
    service: IntermediateArtifactService = Depends(get_intermediate_artifact_service),
):
    """批量删除中间态产物。"""
    deleted_ids = service.batch_delete_by_source(source_id)

    return BatchDeleteIntermediateArtifactsResponse(
        deleted_ids=deleted_ids,
        count=len(deleted_ids),
        message=f"已删除 {len(deleted_ids)} 个中间态产物",
    )
