"""LLM CallSite 服务层。"""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi import HTTPException

from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.domain.entities.llm_call_site import LLMCallSite


class LLMCallSiteService:
    """LLM CallSite 服务类。"""

    # 注意：ocr 目前用于 MinerU 等 OCR/解析服务的“配置占位”，运行时不一定通过 LLMRuntimeService 构建
    ALLOWED_MODEL_KINDS = {"chat", "embedding", "rerank", "multimodal", "ocr"}

    def __init__(
        self,
        repository: LLMCallSiteRepository,
        provider_repository: LLMProviderRepository,
    ):
        self.repository = repository
        self.provider_repository = provider_repository

    def list_call_sites(
        self,
        *,
        key: str | None = None,
        expected_model_kind: str | None = None,
        enabled: bool | None = None,
        bound: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[LLMCallSite], int]:
        items = self.repository.list(
            key=key,
            expected_model_kind=expected_model_kind,
            enabled=enabled,
            bound=bound,
            limit=limit,
            offset=offset,
        )
        total = self.repository.count(
            key=key,
            expected_model_kind=expected_model_kind,
            enabled=enabled,
            bound=bound,
        )
        return items, total

    def create_call_site(
        self,
        *,
        key: str,
        expected_model_kind: str,
        provider_id: str | None,
        config: dict | None,
        prompt_scope: str | None,
        max_concurrency: int | None,
        enabled: bool,
        description: str | None,
    ) -> LLMCallSite:
        key = (key or "").strip()
        if not key:
            raise HTTPException(status_code=400, detail="key 不能为空")

        expected_model_kind = (expected_model_kind or "").strip()
        if expected_model_kind not in self.ALLOWED_MODEL_KINDS:
            raise HTTPException(
                status_code=400,
                detail=f"expected_model_kind 非法，允许值: {sorted(self.ALLOWED_MODEL_KINDS)}",
            )

        existing = self.repository.get_by_key(key)
        if existing is not None:
            raise HTTPException(status_code=409, detail="CallSite key 已存在")

        if provider_id is not None:
            provider = self.provider_repository.get_by_id(provider_id)
            if provider is None:
                raise HTTPException(status_code=404, detail="绑定的 Provider 不存在")
            if not provider.enabled:
                raise HTTPException(status_code=400, detail="绑定的 Provider 已禁用")

        config_json = None
        if config is not None:
            try:
                config_json = json.dumps(config, ensure_ascii=False)
            except TypeError:
                raise HTTPException(status_code=400, detail="config 必须为可 JSON 序列化对象")

        callsite = LLMCallSite(
            id=str(uuid4()),
            key=key,
            expected_model_kind=expected_model_kind,
            provider_id=provider_id,
            config_json=config_json,
            prompt_scope=prompt_scope,
            max_concurrency=max_concurrency,
            enabled=enabled,
            description=description,
        )
        return self.repository.create(callsite)

    def update_call_site(
        self,
        *,
        callsite_id: str,
        provider_id: str | None,
        config: dict | None,
        prompt_scope: str | None,
        max_concurrency: int | None,
        enabled: bool | None,
        description: str | None,
    ) -> LLMCallSite | None:
        target = self.repository.get_by_id(callsite_id)
        if target is None:
            return None

        if provider_id is not None:
            provider = self.provider_repository.get_by_id(provider_id)
            if provider is None:
                raise HTTPException(status_code=404, detail="绑定的 Provider 不存在")
            if not provider.enabled:
                raise HTTPException(status_code=400, detail="绑定的 Provider 已禁用")
            target.provider_id = provider_id

        if config is not None:
            try:
                target.config_json = json.dumps(config, ensure_ascii=False)
            except TypeError:
                raise HTTPException(status_code=400, detail="config 必须为可 JSON 序列化对象")

        if prompt_scope is not None:
            target.prompt_scope = prompt_scope

        if max_concurrency is not None:
            target.max_concurrency = max_concurrency

        if enabled is not None:
            target.enabled = enabled

        if description is not None:
            target.description = description

        return self.repository.update(target)

