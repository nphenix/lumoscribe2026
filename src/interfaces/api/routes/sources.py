"""源文件 API 路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from src.application.repositories.source_file_repository import SourceFileRepository
from src.application.schemas.source_file import (
    ArchiveSourceFileResponse,
    DeleteSourceFileResponse,
    SourceFileListResponse,
    SourceFileResponse,
    SourceFileUpdate,
    SourceFileUploadResponse,
    UnarchiveSourceFileResponse,
)
from src.application.services.source_file_service import SourceFileService
from src.domain.entities.source_file import SourceFileStatus
from src.interfaces.api.deps import get_db


router = APIRouter()


def get_source_file_service(db: Session = Depends(get_db)) -> SourceFileService:
    """获取源文件服务实例。"""
    repository = SourceFileRepository(db)
    return SourceFileService(repository)


@router.post(
    "/sources",
    response_model=SourceFileUploadResponse,
    status_code=201,
    summary="上传源文件",
    description="上传 PDF 源文件，支持工作空间隔离和文件去重。",
)
async def create_source_file(
    workspace_id: Annotated[str, Form(min_length=1, max_length=36)],
    file: Annotated[UploadFile, File(description="PDF 文件")],
    description: Annotated[str | None, Form(max_length=2000)] = None,
    service: SourceFileService = Depends(get_source_file_service),
):
    """创建源文件（上传）。"""
    # 验证文件类型
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    source_file = await service.create_source_file(
        workspace_id=workspace_id,
        file=file,
        description=description,
    )

    return SourceFileUploadResponse(
        id=source_file.id,
        workspace_id=source_file.workspace_id,
        original_filename=source_file.original_filename,
        storage_path=source_file.storage_path,
        status=source_file.status,
        created_at=source_file.created_at,
    )


@router.get(
    "/sources",
    response_model=SourceFileListResponse,
    summary="列出源文件",
    description="获取源文件列表，支持按工作空间和状态过滤。",
)
def list_source_files(
    workspace_id: Annotated[str | None, Query(max_length=36)] = None,
    status: Annotated[str | None, Query(description="状态: active, archived")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: SourceFileService = Depends(get_source_file_service),
):
    """列出源文件。"""
    items, total = service.list_source_files(
        workspace_id=workspace_id,
        status=status,
        limit=limit,
        offset=offset,
    )

    return SourceFileListResponse(
        items=[
            {
                "id": item.id,
                "workspace_id": item.workspace_id,
                "original_filename": item.original_filename,
                "file_size": item.file_size,
                "storage_path": item.storage_path,
                "status": item.status,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "description": item.description,
            }
            for item in items
        ],
        total=total,
    )


@router.get(
    "/sources/{source_id}",
    response_model=SourceFileResponse,
    summary="获取源文件详情",
    description="获取指定源文件的详细信息。",
)
def get_source_file(
    source_id: str,
    service: SourceFileService = Depends(get_source_file_service),
):
    """获取源文件详情。"""
    source_file = service.get_source_file(source_id)
    if source_file is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="源文件不存在")

    return SourceFileResponse(
        id=source_file.id,
        workspace_id=source_file.workspace_id,
        original_filename=source_file.original_filename,
        file_hash=source_file.file_hash,
        file_size=source_file.file_size,
        storage_path=source_file.storage_path,
        status=source_file.status,
        archived_at=source_file.archived_at,
        created_at=source_file.created_at,
        updated_at=source_file.updated_at,
        description=source_file.description,
    )


@router.patch(
    "/sources/{source_id}",
    response_model=SourceFileResponse,
    summary="更新源文件",
    description="更新源文件的元数据信息。",
)
def update_source_file(
    source_id: str,
    payload: SourceFileUpdate,
    service: SourceFileService = Depends(get_source_file_service),
):
    """更新源文件元数据。"""
    source_file = service.update_source_file(
        source_id=source_id,
        description=payload.description,
    )
    if source_file is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="源文件不存在")

    return SourceFileResponse(
        id=source_file.id,
        workspace_id=source_file.workspace_id,
        original_filename=source_file.original_filename,
        file_hash=source_file.file_hash,
        file_size=source_file.file_size,
        storage_path=source_file.storage_path,
        status=source_file.status,
        archived_at=source_file.archived_at,
        created_at=source_file.created_at,
        updated_at=source_file.updated_at,
        description=source_file.description,
    )


@router.delete(
    "/sources/{source_id}",
    response_model=DeleteSourceFileResponse,
    summary="删除源文件",
    description="删除源文件及其关联的存储文件。",
)
def delete_source_file(
    source_id: str,
    service: SourceFileService = Depends(get_source_file_service),
):
    """删除源文件。"""
    success = service.delete_source_file(source_id)
    if not success:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="源文件不存在")

    return DeleteSourceFileResponse(
        id=source_id,
        message="源文件已删除",
    )


@router.post(
    "/sources/{source_id}/archive",
    response_model=ArchiveSourceFileResponse,
    summary="归档源文件",
    description="将源文件移动到归档目录，添加 .archived 后缀。",
)
def archive_source_file(
    source_id: str,
    service: SourceFileService = Depends(get_source_file_service),
):
    """归档源文件。"""
    source_file = service.archive_source_file(source_id)
    if source_file is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="源文件不存在")

    return ArchiveSourceFileResponse(
        id=source_file.id,
        workspace_id=source_file.workspace_id,
        original_filename=source_file.original_filename,
        storage_path=source_file.storage_path,
        status=source_file.status,
        archived_at=source_file.archived_at,
        message="源文件已归档",
    )


@router.post(
    "/sources/{source_id}/unarchive",
    response_model=UnarchiveSourceFileResponse,
    summary="取消归档",
    description="将归档的源文件移回原工作空间目录。",
)
def unarchive_source_file(
    source_id: str,
    service: SourceFileService = Depends(get_source_file_service),
):
    """取消归档源文件。"""
    source_file = service.unarchive_source_file(source_id)
    if source_file is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="源文件不存在")

    return UnarchiveSourceFileResponse(
        id=source_file.id,
        workspace_id=source_file.workspace_id,
        original_filename=source_file.original_filename,
        storage_path=source_file.storage_path,
        status=source_file.status,
        message="源文件已取消归档",
    )
