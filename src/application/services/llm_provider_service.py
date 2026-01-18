"""LLM Provider 服务层。"""

from __future__ import annotations

import json
import re
from uuid import uuid4

from fastapi import HTTPException

from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.domain.entities.llm_provider import LLMProvider


class LLMProviderService:
    """LLM Provider 服务类。"""

    def __init__(
        self,
        repository: LLMProviderRepository,
    ):
        self.repository = repository

    def _serialize_config(self, config: dict | None) -> str | None:
        if config is None:
            return None
        try:
            return json.dumps(config, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"config 不是有效 JSON: {exc}") from exc

    def _slugify(self, raw: str) -> str:
        s = raw.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    def _generate_unique_key(self, name: str, provider_type: str) -> str:
        base = self._slugify(name)
        if not base:
            base = self._slugify(provider_type) or "provider"
        candidate = base[:64]
        # 保证唯一
        if self.repository.get_by_key(candidate) is None:
            return candidate
        for _ in range(10):
            suffix = uuid4().hex[:8]
            candidate = f"{base[: max(1, 64 - 9)]}-{suffix}"
            if self.repository.get_by_key(candidate) is None:
                return candidate
        # 极端冲突：直接用 uuid
        return uuid4().hex

    def create_provider(
        self,
        key: str | None,
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

        key_value = (key or "").strip()
        if not key_value:
            key_value = self._generate_unique_key(name=name, provider_type=provider_type)
        else:
            existing_by_key = self.repository.get_by_key(key_value)
            if existing_by_key is not None:
                raise HTTPException(status_code=409, detail="Provider 标识已存在")

        existing = self.repository.get_by_name(name)
        if existing is not None:
            raise HTTPException(status_code=409, detail="Provider 名称已存在")

        provider = LLMProvider(
            id=str(uuid4()),
            key=key_value,
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
        key: str | None = None,
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

        if key is not None:
            key_value = key.strip()
            # 兼容前端：空字符串表示“不修改”
            if key_value:
                existing_by_key = self.repository.get_by_key(key_value)
                if existing_by_key is not None and existing_by_key.id != provider_id:
                    raise HTTPException(status_code=409, detail="Provider 标识已存在")
                provider.key = key_value

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
        """删除 Provider。
        
        注意：删除前需要确保没有 CallSite 或 Capability 绑定到此 Provider。
        """
        # TODO: 可以添加检查，确保没有 CallSite 或 Capability 绑定到此 Provider
        return self.repository.delete(provider_id)
