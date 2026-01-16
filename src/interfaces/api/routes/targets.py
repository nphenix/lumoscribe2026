"""目标文件 API 路由。"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.application.repositories.target_file_repository import TargetFileRepository
from src.application.schemas.target_file import (
    DeleteTargetFileResponse,
    TargetFileListResponse,
    TargetFileResponse,
)
from src.application.services.target_file_service import TargetFileService
from src.interfaces.api.deps import get_db


router = APIRouter()


def get_target_file_service(db: Session = Depends(get_db)) -> TargetFileService:
    """获取目标文件服务实例。"""
    repository = TargetFileRepository(db)
    return TargetFileService(repository)


@router.get(
    "/targets",
    response_model=TargetFileListResponse,
    summary="列出目标文件",
    description="获取目标文件列表，支持按模板、任务、工作空间和时间过滤。",
)
def list_target_files(
    workspace_id: Annotated[str | None, Query(max_length=36, description="工作空间ID")] = None,
    template_id: Annotated[str | None, Query(max_length=36, description="模板ID")] = None,
    kb_id: Annotated[str | None, Query(max_length=36, description="知识库ID")] = None,
    job_id: Annotated[str | None, Query(max_length=36, description="生成任务ID")] = None,
    created_after: Annotated[datetime | None, Query(description="创建时间上限")] = None,
    created_before: Annotated[datetime | None, Query(description="创建时间下限")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: TargetFileService = Depends(get_target_file_service),
):
    """列出目标文件。"""
    items, total = service.list_target_files(
        workspace_id=workspace_id,
        template_id=template_id,
        kb_id=kb_id,
        job_id=job_id,
        created_after=created_after,
        created_before=created_before,
        limit=limit,
        offset=offset,
    )

    return TargetFileListResponse(
        items=[
            {
                "id": item.id,
                "workspace_id": item.workspace_id,
                "template_id": item.template_id,
                "kb_id": item.kb_id,
                "job_id": item.job_id,
                "output_filename": item.output_filename,
                "storage_path": item.storage_path,
                "description": item.description,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in items
        ],
        total=total,
    )


@router.get(
    "/targets/{target_id}",
    response_model=TargetFileResponse,
    summary="获取目标文件详情",
    description="获取指定目标文件的详细信息，包括关联的模板和知识库信息。",
)
def get_target_file(
    target_id: str,
    service: TargetFileService = Depends(get_target_file_service),
):
    """获取目标文件详情。"""
    target_file = service.get_target_file(target_id)
    if target_file is None:
        raise HTTPException(status_code=404, detail="目标文件不存在")

    return TargetFileResponse(
        id=target_file.id,
        workspace_id=target_file.workspace_id,
        template_id=target_file.template_id,
        kb_id=target_file.kb_id,
        job_id=target_file.job_id,
        output_filename=target_file.output_filename,
        storage_path=target_file.storage_path,
        description=target_file.description,
        created_at=target_file.created_at,
        updated_at=target_file.updated_at,
        template_name=None,  # TODO: 关联查询模板名称
        kb_name=None,  # TODO: 关联查询知识库名称
    )


@router.get(
    "/targets/{target_id}/download",
    summary="下载目标文件",
    description="下载目标 HTML 文件，支持自定义输出文件名。",
)
def download_target_file(
    target_id: str,
    filename: Annotated[str | None, Query(description="自定义输出文件名")] = None,
    service: TargetFileService = Depends(get_target_file_service),
):
    """下载目标文件。"""
    target_file = service.get_target_file(target_id)
    if target_file is None:
        raise HTTPException(status_code=404, detail="目标文件不存在")

    from pathlib import Path

    file_path = Path("data") / target_file.storage_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="目标文件不存在")

    # 使用自定义文件名或默认 output_filename
    output_filename = filename or target_file.output_filename

    def iter_file():
        yield file_path.read_bytes()

    return StreamingResponse(
        iter_file(),
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="{output_filename}"',
            "Content-Length": str(file_path.stat().st_size),
        },
    )


@router.get(
    "/targets/{target_id}/view",
    summary="查看目标文件",
    description="内联查看目标 HTML 文件（直接在浏览器中打开）。",
)
def view_target_file(
    target_id: str,
    service: TargetFileService = Depends(get_target_file_service),
):
    """查看目标文件。"""
    target_file = service.get_target_file(target_id)
    if target_file is None:
        raise HTTPException(status_code=404, detail="目标文件不存在")

    from pathlib import Path

    file_path = Path("data") / target_file.storage_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="目标文件不存在")

    def iter_file():
        yield file_path.read_bytes()

    return StreamingResponse(
        iter_file(),
        media_type="text/html",
        headers={
            "Content-Disposition": f'inline; filename="{target_file.output_filename}"',
            "Content-Length": str(file_path.stat().st_size),
        },
    )


@router.delete(
    "/targets/{target_id}",
    response_model=DeleteTargetFileResponse,
    summary="删除目标文件",
    description="删除目标文件及其关联的存储文件。",
)
def delete_target_file(
    target_id: str,
    service: TargetFileService = Depends(get_target_file_service),
):
    """删除目标文件。"""
    success = service.delete_target_file(target_id)
    if not success:
        raise HTTPException(status_code=404, detail="目标文件不存在")

    return DeleteTargetFileResponse(
        id=target_id,
        message="目标文件已删除",
    )
