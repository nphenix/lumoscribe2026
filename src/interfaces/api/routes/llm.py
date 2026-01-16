"""LLM 配置 API 路由。"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.application.repositories.llm_capability_repository import (
    LLMCapabilityRepository,
)
from src.application.repositories.llm_model_repository import LLMModelRepository
from src.application.repositories.llm_provider_repository import (
    LLMProviderRepository,
)
from src.application.schemas.llm import (
    LLMCapabilityListResponse,
    LLMCapabilityPatchRequest,
    LLMCapabilityPatchResponse,
    LLMCapabilityResponse,
    LLMModelCreate,
    LLMModelDeleteResponse,
    LLMModelListResponse,
    LLMModelResponse,
    LLMModelUpdate,
    LLMProviderCreate,
    LLMProviderDeleteResponse,
    LLMProviderListResponse,
    LLMProviderResponse,
    LLMProviderUpdate,
)
from src.application.services.llm_capability_service import LLMCapabilityService
from src.application.services.llm_model_service import LLMModelService
from src.application.services.llm_provider_service import LLMProviderService
from src.interfaces.api.deps import get_db


router = APIRouter()


def _parse_config(config_json: str | None) -> dict | None:
    if not config_json:
        return None
    try:
        return json.loads(config_json)
    except json.JSONDecodeError:
        return None


def get_provider_service(db: Session = Depends(get_db)) -> LLMProviderService:
    """获取 Provider 服务实例。"""
    provider_repo = LLMProviderRepository(db)
    model_repo = LLMModelRepository(db)
    return LLMProviderService(provider_repo, model_repo)


def get_model_service(db: Session = Depends(get_db)) -> LLMModelService:
    """获取模型服务实例。"""
    model_repo = LLMModelRepository(db)
    provider_repo = LLMProviderRepository(db)
    capability_repo = LLMCapabilityRepository(db)
    return LLMModelService(model_repo, provider_repo, capability_repo)


def get_capability_service(db: Session = Depends(get_db)) -> LLMCapabilityService:
    """获取能力映射服务实例。"""
    capability_repo = LLMCapabilityRepository(db)
    model_repo = LLMModelRepository(db)
    return LLMCapabilityService(capability_repo, model_repo)


@router.get(
    "/llm/providers",
    response_model=LLMProviderListResponse,
    summary="列出 Provider",
    description="获取 LLM Provider 列表，支持过滤。",
)
def list_providers(
    provider_type: Annotated[str | None, Query(max_length=64)] = None,
    enabled: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: LLMProviderService = Depends(get_provider_service),
):
    """列出 Provider。"""
    items, total = service.list_providers(
        provider_type=provider_type,
        enabled=enabled,
        limit=limit,
        offset=offset,
    )
    return LLMProviderListResponse(
        items=[
            LLMProviderResponse(
                id=item.id,
                name=item.name,
                provider_type=item.provider_type,
                base_url=item.base_url,
                api_key_env=item.api_key_env,
                config=_parse_config(item.config_json),
                enabled=item.enabled,
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


@router.post(
    "/llm/providers",
    response_model=LLMProviderResponse,
    status_code=201,
    summary="创建 Provider",
    description="创建新的 LLM Provider。",
)
def create_provider(
    payload: LLMProviderCreate,
    service: LLMProviderService = Depends(get_provider_service),
):
    """创建 Provider。"""
    provider = service.create_provider(
        name=payload.name,
        provider_type=payload.provider_type,
        base_url=payload.base_url,
        api_key_env=payload.api_key_env,
        config=payload.config,
        enabled=payload.enabled,
        description=payload.description,
    )
    return LLMProviderResponse(
        id=provider.id,
        name=provider.name,
        provider_type=provider.provider_type,
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        config=_parse_config(provider.config_json),
        enabled=provider.enabled,
        description=provider.description,
        created_at=provider.created_at,
        updated_at=provider.updated_at,
    )


@router.patch(
    "/llm/providers/{provider_id}",
    response_model=LLMProviderResponse,
    summary="更新 Provider",
    description="更新 LLM Provider 配置。",
)
def update_provider(
    provider_id: str,
    payload: LLMProviderUpdate,
    service: LLMProviderService = Depends(get_provider_service),
):
    """更新 Provider。"""
    provider = service.update_provider(
        provider_id=provider_id,
        name=payload.name,
        provider_type=payload.provider_type,
        base_url=payload.base_url,
        api_key_env=payload.api_key_env,
        config=payload.config,
        enabled=payload.enabled,
        description=payload.description,
    )
    if provider is None:
        raise HTTPException(status_code=404, detail="Provider 不存在")
    return LLMProviderResponse(
        id=provider.id,
        name=provider.name,
        provider_type=provider.provider_type,
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        config=_parse_config(provider.config_json),
        enabled=provider.enabled,
        description=provider.description,
        created_at=provider.created_at,
        updated_at=provider.updated_at,
    )


@router.delete(
    "/llm/providers/{provider_id}",
    response_model=LLMProviderDeleteResponse,
    summary="删除 Provider",
    description="删除 LLM Provider。",
)
def delete_provider(
    provider_id: str,
    service: LLMProviderService = Depends(get_provider_service),
):
    """删除 Provider。"""
    success = service.delete_provider(provider_id)
    if not success:
        raise HTTPException(status_code=404, detail="Provider 不存在")
    return LLMProviderDeleteResponse(id=provider_id, message="Provider 已删除")


@router.get(
    "/llm/models",
    response_model=LLMModelListResponse,
    summary="列出模型",
    description="获取 LLM 模型列表，支持过滤。",
)
def list_models(
    provider_id: Annotated[str | None, Query(max_length=36)] = None,
    model_kind: Annotated[str | None, Query(max_length=64)] = None,
    enabled: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: LLMModelService = Depends(get_model_service),
):
    """列出模型。"""
    items, total = service.list_models(
        provider_id=provider_id,
        model_kind=model_kind,
        enabled=enabled,
        limit=limit,
        offset=offset,
    )
    return LLMModelListResponse(
        items=[
            LLMModelResponse(
                id=item.id,
                provider_id=item.provider_id,
                name=item.name,
                model_kind=item.model_kind,
                config=_parse_config(item.config_json),
                enabled=item.enabled,
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


@router.post(
    "/llm/models",
    response_model=LLMModelResponse,
    status_code=201,
    summary="创建模型",
    description="创建新的 LLM 模型。",
)
def create_model(
    payload: LLMModelCreate,
    service: LLMModelService = Depends(get_model_service),
):
    """创建模型。"""
    model = service.create_model(
        provider_id=payload.provider_id,
        name=payload.name,
        model_kind=payload.model_kind,
        config=payload.config,
        enabled=payload.enabled,
        description=payload.description,
    )
    return LLMModelResponse(
        id=model.id,
        provider_id=model.provider_id,
        name=model.name,
        model_kind=model.model_kind,
        config=_parse_config(model.config_json),
        enabled=model.enabled,
        description=model.description,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


@router.patch(
    "/llm/models/{model_id}",
    response_model=LLMModelResponse,
    summary="更新模型",
    description="更新 LLM 模型配置。",
)
def update_model(
    model_id: str,
    payload: LLMModelUpdate,
    service: LLMModelService = Depends(get_model_service),
):
    """更新模型。"""
    model = service.update_model(
        model_id=model_id,
        name=payload.name,
        model_kind=payload.model_kind,
        config=payload.config,
        enabled=payload.enabled,
        description=payload.description,
    )
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    return LLMModelResponse(
        id=model.id,
        provider_id=model.provider_id,
        name=model.name,
        model_kind=model.model_kind,
        config=_parse_config(model.config_json),
        enabled=model.enabled,
        description=model.description,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


@router.delete(
    "/llm/models/{model_id}",
    response_model=LLMModelDeleteResponse,
    summary="删除模型",
    description="删除 LLM 模型。",
)
def delete_model(
    model_id: str,
    service: LLMModelService = Depends(get_model_service),
):
    """删除模型。"""
    success = service.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="模型不存在")
    return LLMModelDeleteResponse(id=model_id, message="模型已删除")


@router.get(
    "/llm/capabilities",
    response_model=LLMCapabilityListResponse,
    summary="列出能力映射",
    description="获取能力映射列表。",
)
def list_capabilities(
    capability: Annotated[str | None, Query(max_length=128)] = None,
    model_id: Annotated[str | None, Query(max_length=36)] = None,
    enabled: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: LLMCapabilityService = Depends(get_capability_service),
):
    """列出能力映射。"""
    items, total = service.list_capabilities(
        capability=capability,
        model_id=model_id,
        enabled=enabled,
        limit=limit,
        offset=offset,
    )
    return LLMCapabilityListResponse(
        items=[
            LLMCapabilityResponse(
                id=item.id,
                capability=item.capability,
                model_id=item.model_id,
                priority=item.priority,
                enabled=item.enabled,
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


@router.patch(
    "/llm/capabilities",
    response_model=LLMCapabilityPatchResponse,
    summary="批量更新能力映射",
    description="批量 upsert 能力映射并支持禁用。",
)
def patch_capabilities(
    payload: LLMCapabilityPatchRequest,
    service: LLMCapabilityService = Depends(get_capability_service),
):
    """批量更新能力映射。"""
    if not payload.items:
        raise HTTPException(status_code=400, detail="items 不能为空")
    results = []
    for item in payload.items:
        results.append(
            service.upsert_capability(
                capability_id=item.id,
                capability=item.capability,
                model_id=item.model_id,
                priority=item.priority,
                enabled=item.enabled,
                description=item.description,
            )
        )
    return LLMCapabilityPatchResponse(
        items=[
            LLMCapabilityResponse(
                id=item.id,
                capability=item.capability,
                model_id=item.model_id,
                priority=item.priority,
                enabled=item.enabled,
                description=item.description,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in results
        ]
    )
