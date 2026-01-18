"""LLM 配置 API 路由。"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.application.repositories.llm_capability_repository import (
    LLMCapabilityRepository,
)
from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
from src.application.repositories.llm_provider_repository import (
    LLMProviderRepository,
)
from src.application.schemas.llm import (
    LLMCapabilityListResponse,
    LLMCapabilityPatchRequest,
    LLMCapabilityPatchResponse,
    LLMCapabilityResponse,
    LLMCallSiteCreate,
    LLMCallSiteListResponse,
    LLMCallSiteResponse,
    LLMCallSiteUpdate,
    LLMProviderCreate,
    LLMProviderDeleteResponse,
    LLMProviderListResponse,
    LLMProviderResponse,
    LLMProviderUpdate,
)
from src.application.services.llm_capability_service import LLMCapabilityService
from src.application.services.llm_call_site_service import LLMCallSiteService
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
    return LLMProviderService(provider_repo)


def get_capability_service(db: Session = Depends(get_db)) -> LLMCapabilityService:
    """获取能力映射服务实例。"""
    capability_repo = LLMCapabilityRepository(db)
    provider_repo = LLMProviderRepository(db)
    return LLMCapabilityService(capability_repo, provider_repo)


def get_callsite_service(db: Session = Depends(get_db)) -> LLMCallSiteService:
    """获取 CallSite 服务实例。"""
    callsite_repo = LLMCallSiteRepository(db)
    provider_repo = LLMProviderRepository(db)
    return LLMCallSiteService(callsite_repo, provider_repo)


@router.get(
    "/llm/providers",
    response_model=LLMProviderListResponse,
    summary="列出 Provider",
    description="获取 LLM Provider 列表，支持过滤。",
)
def list_providers(
    provider_type: Annotated[str | None, Query(max_length=64)] = None,
    enabled: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1)] = 20,
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
                key=item.key,
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
        key=payload.key,
        name=payload.name,
        provider_type=payload.provider_type,
        base_url=payload.base_url,
        api_key=payload.api_key,
        api_key_env=payload.api_key_env,
        config=payload.config,
        enabled=payload.enabled,
        description=payload.description,
    )
    return LLMProviderResponse(
        id=provider.id,
        key=provider.key,
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
        key=payload.key,
        name=payload.name,
        provider_type=payload.provider_type,
        base_url=payload.base_url,
        api_key=payload.api_key,
        api_key_env=payload.api_key_env,
        config=payload.config,
        enabled=payload.enabled,
        description=payload.description,
    )
    if provider is None:
        raise HTTPException(status_code=404, detail="Provider 不存在")
    return LLMProviderResponse(
        id=provider.id,
        key=provider.key,
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
    "/llm/capabilities",
    response_model=LLMCapabilityListResponse,
    summary="列出能力映射",
    description="获取能力映射列表。",
)
def list_capabilities(
    capability: Annotated[str | None, Query(max_length=128)] = None,
    provider_id: Annotated[str | None, Query(max_length=36)] = None,
    enabled: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: LLMCapabilityService = Depends(get_capability_service),
):
    """列出能力映射。"""
    items, total = service.list_capabilities(
        capability=capability,
        provider_id=provider_id,
        enabled=enabled,
        limit=limit,
        offset=offset,
    )
    return LLMCapabilityListResponse(
        items=[
            LLMCapabilityResponse(
                id=item.id,
                capability=item.capability,
                provider_id=item.provider_id,
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
                provider_id=item.provider_id,
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
                provider_id=item.provider_id,
                priority=item.priority,
                enabled=item.enabled,
                description=item.description,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in results
        ]
    )


@router.get(
    "/llm/call-sites",
    response_model=LLMCallSiteListResponse,
    summary="列出 CallSite",
    description="获取 LLM 调用点列表，支持过滤。",
)
def list_call_sites(
    key: Annotated[str | None, Query(max_length=256)] = None,
    expected_model_kind: Annotated[str | None, Query(max_length=64)] = None,
    enabled: Annotated[bool | None, Query()] = None,
    bound: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: LLMCallSiteService = Depends(get_callsite_service),
):
    items, total = service.list_call_sites(
        key=key,
        expected_model_kind=expected_model_kind,
        enabled=enabled,
        bound=bound,
        limit=limit,
        offset=offset,
    )
    return LLMCallSiteListResponse(
        items=[
            LLMCallSiteResponse(
                id=item.id,
                key=item.key,
                expected_model_kind=item.expected_model_kind,
                provider_id=item.provider_id,
                config=_parse_config(item.config_json),
                prompt_scope=item.prompt_scope,
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
    "/llm/call-sites",
    response_model=LLMCallSiteResponse,
    status_code=201,
    summary="创建 CallSite",
    description="创建新的 LLM 调用点（通常由 seed 自动创建）。",
)
def create_call_site(
    payload: LLMCallSiteCreate,
    service: LLMCallSiteService = Depends(get_callsite_service),
):
    item = service.create_call_site(
        key=payload.key,
        expected_model_kind=payload.expected_model_kind,
        provider_id=payload.provider_id,
        config=payload.config,
        prompt_scope=payload.prompt_scope,
        enabled=payload.enabled,
        description=payload.description,
    )
    return LLMCallSiteResponse(
        id=item.id,
        key=item.key,
        expected_model_kind=item.expected_model_kind,
        provider_id=item.provider_id,
        config=_parse_config(item.config_json),
        prompt_scope=item.prompt_scope,
        enabled=item.enabled,
        description=item.description,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


@router.patch(
    "/llm/call-sites/{callsite_id}",
    response_model=LLMCallSiteResponse,
    summary="更新 CallSite",
    description="更新 LLM 调用点配置（绑定模型/参数覆盖/prompt_scope）。",
)
def update_call_site(
    callsite_id: str,
    payload: LLMCallSiteUpdate,
    service: LLMCallSiteService = Depends(get_callsite_service),
):
    item = service.update_call_site(
        callsite_id=callsite_id,
        provider_id=payload.provider_id,
        config=payload.config,
        prompt_scope=payload.prompt_scope,
        enabled=payload.enabled,
        description=payload.description,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="CallSite 不存在")
    return LLMCallSiteResponse(
        id=item.id,
        key=item.key,
        expected_model_kind=item.expected_model_kind,
        provider_id=item.provider_id,
        config=_parse_config(item.config_json),
        prompt_scope=item.prompt_scope,
        enabled=item.enabled,
        description=item.description,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )
