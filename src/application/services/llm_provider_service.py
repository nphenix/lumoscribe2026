"""LLM Provider 服务层。"""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi import HTTPException

from src.application.repositories.llm_model_repository import LLMModelRepository
from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.domain.entities.llm_provider import LLMProvider


class LLMProviderService:
    """LLM Provider 服务类。"""

    def __init__(
        self,
        repository: LLMProviderRepository,
        model_repository: LLMModelRepository,
    ):
        self.repository = repository
        self.model_repository = model_repository

    def _serialize_config(self, config: dict | None) -> str | None:
        if config is None:
            return None
        try:
            return json.dumps(config, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"config 不是有效 JSON: {exc}") from exc

    def create_provider(
        self,
        name: str,
        provider_type: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        config: dict | None = None,
        enabled: bool = True,
        description: str | None = None,
    ) -> LLMProvider:
        """创建 Provider。"""
        if not name.strip():
            raise HTTPException(status_code=400, detail="Provider 名称不能为空")
        if not provider_type.strip():
            raise HTTPException(status_code=400, detail="Provider 类型不能为空")

        existing = self.repository.get_by_name(name)
        if existing is not None:
            raise HTTPException(status_code=409, detail="Provider 名称已存在")

        provider = LLMProvider(
            id=str(uuid4()),
            name=name,
            provider_type=provider_type,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            config_json=self._serialize_config(config),
            enabled=enabled,
            description=description,
        )
        return self.repository.create(provider)

    def list_providers(
        self,
        provider_type: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[LLMProvider], int]:
        """列出 Provider。"""
        items = self.repository.list(
            provider_type=provider_type,
            enabled=enabled,
            limit=limit,
            offset=offset,
        )
        total = self.repository.count(
            provider_type=provider_type,
            enabled=enabled,
        )
        return items, total

    def update_provider(
        self,
        provider_id: str,
        name: str | None = None,
        provider_type: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        config: dict | None = None,
        enabled: bool | None = None,
        description: str | None = None,
    ) -> LLMProvider | None:
        """更新 Provider。"""
        provider = self.repository.get_by_id(provider_id)
        if provider is None:
            return None

        if name is not None:
            if not name.strip():
                raise HTTPException(status_code=400, detail="Provider 名称不能为空")
            existing = self.repository.get_by_name(name)
            if existing is not None and existing.id != provider_id:
                raise HTTPException(status_code=409, detail="Provider 名称已存在")
            provider.name = name

        if provider_type is not None:
            if not provider_type.strip():
                raise HTTPException(status_code=400, detail="Provider 类型不能为空")
            provider.provider_type = provider_type

        if base_url is not None:
            provider.base_url = base_url

        if api_key is not None:
            provider.api_key = api_key

        if api_key_env is not None:
            provider.api_key_env = api_key_env

        if config is not None:
            provider.config_json = self._serialize_config(config)

        if enabled is not None:
            provider.enabled = enabled

        if description is not None:
            provider.description = description

        return self.repository.update(provider)

    def delete_provider(self, provider_id: str) -> bool:
        """删除 Provider。"""
        model_count = self.model_repository.count(provider_id=provider_id)
        if model_count > 0:
            raise HTTPException(status_code=409, detail="Provider 关联模型未清理")
        return self.repository.delete(provider_id)
