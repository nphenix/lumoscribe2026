"""提示词 API 路由。"""

from __future__ import annotations

import difflib
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.application.repositories.prompt_repository import PromptRepository
from src.application.schemas.prompt import (
    PromptCreate,
    PromptDeleteResponse,
    PromptDiffResponse,
    PromptListResponse,
    PromptResponse,
    PromptScopeListResponse,
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


def _prompt_to_lines(prompt) -> list[str]:
    if prompt is None:
        return [""]
    fmt = str(getattr(prompt, "format", "") or "").strip().lower()
    if fmt == "chat":
        msgs = _parse_messages(getattr(prompt, "messages_json", None)) or []
        try:
            rendered = json.dumps(msgs, ensure_ascii=False, indent=2)
        except Exception:
            rendered = str(msgs)
        return rendered.splitlines()
    content = str(getattr(prompt, "content", "") or "")
    return content.splitlines()


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


@router.get(
    "/prompts/scopes",
    response_model=PromptScopeListResponse,
    summary="列出提示词 scope",
    description="按 scope 聚合列出提示词版本概览。",
)
def list_prompt_scopes(
    scope: Annotated[str | None, Query(max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: Session = Depends(get_db),
):
    repo = PromptRepository(db)
    items = repo.list_scopes(scope=scope, limit=limit, offset=offset)
    total = repo.count_scopes(scope=scope)
    return PromptScopeListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/prompts/diff",
    response_model=PromptDiffResponse,
    summary="提示词版本差异",
    description="对比两个提示词版本，返回 unified diff 文本。",
)
def diff_prompts(
    from_id: Annotated[str, Query(min_length=1, max_length=36)],
    to_id: Annotated[str, Query(min_length=1, max_length=36)],
    db: Session = Depends(get_db),
):
    repo = PromptRepository(db)
    p1 = repo.get_by_id(from_id)
    p2 = repo.get_by_id(to_id)
    if p1 is None:
        raise HTTPException(status_code=404, detail="from_id 对应提示词不存在")
    if p2 is None:
        raise HTTPException(status_code=404, detail="to_id 对应提示词不存在")
    scope = p1.scope if p1.scope == p2.scope else None
    from_name = f"{p1.scope}@v{p1.version}"
    to_name = f"{p2.scope}@v{p2.version}"
    diff = "\n".join(
        difflib.unified_diff(
            _prompt_to_lines(p1),
            _prompt_to_lines(p2),
            fromfile=from_name,
            tofile=to_name,
            lineterm="",
        )
    )
    return PromptDiffResponse(from_id=from_id, to_id=to_id, scope=scope, diff=diff)


@router.get(
    "/prompts/{prompt_id}",
    response_model=PromptResponse,
    summary="获取提示词",
    description="根据提示词 ID 获取单条提示词详情。",
)
def get_prompt(
    prompt_id: str,
    db: Session = Depends(get_db),
):
    repo = PromptRepository(db)
    prompt = repo.get_by_id(prompt_id)
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
