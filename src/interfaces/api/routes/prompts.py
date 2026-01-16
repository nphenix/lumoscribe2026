"""提示词 API 路由。"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.application.repositories.prompt_repository import PromptRepository
from src.application.schemas.prompt import (
    PromptCreate,
    PromptDeleteResponse,
    PromptListResponse,
    PromptResponse,
    PromptUpdate,
)
from src.application.services.prompt_service import PromptService
from src.interfaces.api.deps import get_db


router = APIRouter()


def _parse_messages(messages_json: str | None) -> list[dict] | None:
    if not messages_json:
        return None
    try:
        return json.loads(messages_json)
    except json.JSONDecodeError:
        return None


def get_prompt_service(db: Session = Depends(get_db)) -> PromptService:
    """获取提示词服务实例。"""
    repository = PromptRepository(db)
    return PromptService(repository)


@router.get(
    "/prompts",
    response_model=PromptListResponse,
    summary="列出提示词",
    description="获取提示词列表，支持 scope/active/format 过滤。",
)
def list_prompts(
    scope: Annotated[str | None, Query(max_length=128)] = None,
    active: Annotated[bool | None, Query()] = None,
    format: Annotated[str | None, Query(max_length=32)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: PromptService = Depends(get_prompt_service),
):
    """列出提示词。"""
    items, total = service.list_prompts(
        scope=scope,
        active=active,
        format=format,
        limit=limit,
        offset=offset,
    )
    return PromptListResponse(
        items=[
            PromptResponse(
                id=item.id,
                scope=item.scope,
                format=item.format,
                content=item.content,
                messages=_parse_messages(item.messages_json),
                version=item.version,
                active=item.active,
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
    "/prompts",
    response_model=PromptResponse,
    status_code=201,
    summary="创建提示词",
    description="创建新的提示词版本。",
)
def create_prompt(
    payload: PromptCreate,
    service: PromptService = Depends(get_prompt_service),
):
    """创建提示词。"""
    prompt = service.create_prompt(
        scope=payload.scope,
        format=payload.format,
        content=payload.content,
        messages=[msg.model_dump() for msg in payload.messages]
        if payload.messages
        else None,
        active=payload.active,
        description=payload.description,
    )
    return PromptResponse(
        id=prompt.id,
        scope=prompt.scope,
        format=prompt.format,
        content=prompt.content,
        messages=_parse_messages(prompt.messages_json),
        version=prompt.version,
        active=prompt.active,
        description=prompt.description,
        created_at=prompt.created_at,
        updated_at=prompt.updated_at,
    )


@router.patch(
    "/prompts/{prompt_id}",
    response_model=PromptResponse,
    summary="更新提示词",
    description="更新提示词的激活状态或说明。",
)
def update_prompt(
    prompt_id: str,
    payload: PromptUpdate,
    service: PromptService = Depends(get_prompt_service),
):
    """更新提示词。"""
    prompt = service.update_prompt(
        prompt_id=prompt_id,
        active=payload.active,
        description=payload.description,
    )
    if prompt is None:
        raise HTTPException(status_code=404, detail="提示词不存在")
    return PromptResponse(
        id=prompt.id,
        scope=prompt.scope,
        format=prompt.format,
        content=prompt.content,
        messages=_parse_messages(prompt.messages_json),
        version=prompt.version,
        active=prompt.active,
        description=prompt.description,
        created_at=prompt.created_at,
        updated_at=prompt.updated_at,
    )


@router.delete(
    "/prompts/{prompt_id}",
    response_model=PromptDeleteResponse,
    summary="删除提示词",
    description="删除提示词。",
)
def delete_prompt(
    prompt_id: str,
    service: PromptService = Depends(get_prompt_service),
):
    """删除提示词。"""
    success = service.delete_prompt(prompt_id)
    if not success:
        raise HTTPException(status_code=404, detail="提示词不存在")
    return PromptDeleteResponse(id=prompt_id, message="提示词已删除")
