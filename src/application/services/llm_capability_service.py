"""LLM 能力映射服务层。"""

from __future__ import annotations

from uuid import uuid4

from fastapi import HTTPException

from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.domain.entities.llm_capability import LLMCapability


class LLMCapabilityService:
    """LLM 能力映射服务类。"""

    def __init__(
        self,
        repository: LLMCapabilityRepository,
        provider_repository: LLMProviderRepository,
    ):
        self.repository = repository
        self.provider_repository = provider_repository

    def list_capabilities(
        self,
        capability: str | None = None,
        provider_id: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[LLMCapability], int]:
        """列出能力映射。"""
        items = self.repository.list(
            capability=capability,
            provider_id=provider_id,
            enabled=enabled,
            limit=limit,
            offset=offset,
        )
        total = self.repository.count(
            capability=capability,
            provider_id=provider_id,
            enabled=enabled,
        )
        return items, total

    def upsert_capability(
        self,
        *,
        capability_id: str | None,
        capability: str,
        provider_id: str,
        priority: int,
        enabled: bool,
        description: str | None,
    ) -> LLMCapability:
        """创建或更新能力映射。"""
        if not capability.strip():
            raise HTTPException(status_code=400, detail="能力名称不能为空")
        if priority < 0:
            raise HTTPException(status_code=400, detail="priority 不能为负数")

        provider = self.provider_repository.get_by_id(provider_id)
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider 不存在")
        if not provider.enabled:
            raise HTTPException(status_code=400, detail="Provider 已禁用")

        target = None
        if capability_id is not None:
            target = self.repository.get_by_id(capability_id)
            if target is None:
                raise HTTPException(status_code=404, detail="能力映射不存在")

        if target is None:
            target = self.repository.get_by_capability_provider(capability, provider_id)

        if target is None:
            new_capability = LLMCapability(
                id=str(uuid4()),
                capability=capability,
                provider_id=provider_id,
                priority=priority,
                enabled=enabled,
                description=description,
            )
            return self.repository.create(new_capability)

        target.capability = capability
        target.provider_id = provider_id
        target.priority = priority
        target.enabled = enabled
        target.description = description
        return self.repository.update(target)

    def list_active_for_capability(self, capability: str) -> list[LLMCapability]:
        """列出指定 capability 的启用映射。"""
        return self.repository.list(
            capability=capability,
            enabled=True,
            limit=100,
            offset=0,
        )
