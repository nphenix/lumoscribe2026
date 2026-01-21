"""模板 API 路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from src.application.repositories.template_repository import TemplateRepository
from src.application.schemas.template import (
    DeleteTemplateResponse,
    LockTemplateResponse,
    PreprocessResponse,
    TemplateCreate,
    TemplateListResponse,
    TemplateResponse,
    TemplateUpdate,
    TemplateUploadResponse,
)
from src.application.services.template_service import TemplateService
from src.domain.entities.template import TemplateType
from src.interfaces.api.deps import get_db


router = APIRouter()


def get_template_service(db: Session = Depends(get_db)) -> TemplateService:
    """获取模板服务实例。"""
    repository = TemplateRepository(db)
    return TemplateService(repository)


@router.post(
    "/templates",
    response_model=TemplateUploadResponse,
    status_code=201,
    summary="创建模板",
    description="上传模板文件，支持自定义模板(Markdown)和系统模板(Office文档)。",
)
async def create_template(
    workspace_id: Annotated[str, Form(min_length=1, max_length=36, description="工作空间ID")],
    file: Annotated[UploadFile, File(description="模板文件")],
    description: Annotated[str | None, Form(max_length=1024, description="模板描述")] = None,
    service: TemplateService = Depends(get_template_service),
):
    """创建模板（上传）。"""
    # 验证文件类型
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    template = await service.create_template(
        workspace_id=workspace_id,
        file=file,
        description=description,
    )

    return TemplateUploadResponse(
        id=template.id,
        workspace_id=template.workspace_id,
        original_filename=template.original_filename,
        file_format=template.file_format,
        type=template.type.value,
        version=template.version,
        locked=template.locked,
        storage_path=template.storage_path,
        description=template.description,
        created_at=template.created_at,
    )


@router.get(
    "/templates",
    response_model=TemplateListResponse,
    summary="列出模板",
    description="获取模板列表，支持按类型和工作空间过滤。",
)
def list_templates(
    workspace_id: Annotated[str | None, Query(max_length=36, description="工作空间ID")] = None,
    template_type: Annotated[str | None, Query(description="模板类型: custom, system")] = None,
    locked: Annotated[bool | None, Query(description="锁定状态")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: TemplateService = Depends(get_template_service),
):
    """列出模板。"""
    items, total = service.list_templates(
        workspace_id=workspace_id,
        template_type=template_type,
        locked=locked,
        limit=limit,
        offset=offset,
    )

    return TemplateListResponse(
        items=[
            TemplateResponse(
                id=item.id,
                workspace_id=item.workspace_id,
                original_filename=item.original_filename,
                file_format=item.file_format,
                type=item.type.value,
                version=item.version,
                locked=item.locked,
                storage_path=item.storage_path,
                description=item.description,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in items
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/templates/{template_id}",
    response_model=TemplateResponse,
    summary="获取模板详情",
    description="获取指定模板的详细信息。",
)
def get_template(
    template_id: str,
    service: TemplateService = Depends(get_template_service),
):
    """获取模板详情。"""
    template = service.get_template(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="模板不存在")

    return TemplateResponse(
        id=template.id,
        workspace_id=template.workspace_id,
        original_filename=template.original_filename,
        file_format=template.file_format,
        type=template.type.value,
        version=template.version,
        locked=template.locked,
        storage_path=template.storage_path,
        description=template.description,
        created_at=template.created_at,
        updated_at=template.updated_at,
    )


@router.patch(
    "/templates/{template_id}",
    response_model=TemplateResponse,
    summary="更新模板",
    description="更新模板的元数据信息。",
)
def update_template(
    template_id: str,
    payload: TemplateUpdate,
    service: TemplateService = Depends(get_template_service),
):
    """更新模板元数据。"""
    template = service.update_template(
        template_id=template_id,
        description=payload.description,
    )
    if template is None:
        raise HTTPException(status_code=404, detail="模板不存在")

    return TemplateResponse(
        id=template.id,
        workspace_id=template.workspace_id,
        original_filename=template.original_filename,
        file_format=template.file_format,
        type=template.type.value,
        version=template.version,
        locked=template.locked,
        storage_path=template.storage_path,
        description=template.description,
        created_at=template.created_at,
        updated_at=template.updated_at,
    )


@router.delete(
    "/templates/{template_id}",
    response_model=DeleteTemplateResponse,
    summary="删除模板",
    description="删除模板及其关联的存储文件。",
)
def delete_template(
    template_id: str,
    service: TemplateService = Depends(get_template_service),
):
    """删除模板。"""
    success = service.delete_template(template_id)
    if not success:
        raise HTTPException(status_code=404, detail="模板不存在")

    return DeleteTemplateResponse(
        id=template_id,
        message="模板已删除",
    )


@router.post(
    "/templates/{template_id}/preprocess",
    response_model=PreprocessResponse,
    summary="预处理校验",
    description="对模板进行预处理校验，检查占位符完整性和文档结构。",
)
def preprocess_template(
    template_id: str,
    service: TemplateService = Depends(get_template_service),
):
    """预处理校验模板。"""
    result = service.preprocess_template(template_id)

    return PreprocessResponse(
        template_id=result["template_id"],
        template_type=result["template_type"],
        checks=result["checks"],
        overall_status=result["overall_status"],
        message=result["message"],
    )


@router.post(
    "/templates/{template_id}/lock",
    response_model=LockTemplateResponse,
    summary="锁定模板",
    description="锁定模板，锁定后不可修改内容，但可用于生成任务。",
)
def lock_template(
    template_id: str,
    service: TemplateService = Depends(get_template_service),
):
    """锁定模板。"""
    template = service.lock_template(template_id, lock=True)
    if template is None:
        raise HTTPException(status_code=404, detail="模板不存在")

    return LockTemplateResponse(
        id=template.id,
        locked=template.locked,
        message="模板已锁定",
    )


@router.post(
    "/templates/{template_id}/unlock",
    response_model=LockTemplateResponse,
    summary="解锁模板",
    description="解锁模板，允许修改内容。",
)
def unlock_template(
    template_id: str,
    service: TemplateService = Depends(get_template_service),
):
    """解锁模板。"""
    template = service.lock_template(template_id, lock=False)
    if template is None:
        raise HTTPException(status_code=404, detail="模板不存在")

    return LockTemplateResponse(
        id=template.id,
        locked=template.locked,
        message="模板已解锁",
    )
