"""LLM 模型服务层。"""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi import HTTPException

from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
from src.application.repositories.llm_model_repository import LLMModelRepository
from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.domain.entities.llm_model import LLMModel


class LLMModelService:
    """LLM 模型服务类。"""

    def __init__(
        self,
        repository: LLMModelRepository,
        provider_repository: LLMProviderRepository,
        capability_repository: LLMCapabilityRepository,
    ):
        self.repository = repository
        self.provider_repository = provider_repository
        self.capability_repository = capability_repository

    def _serialize_config(self, config: dict | None) -> str | None:
        if config is None:
            return None
        try:
            return json.dumps(config, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"config 不是有效 JSON: {exc}") from exc

    def create_model(
        self,
        provider_id: str,
        name: str,
        model_kind: str,
        config: dict | None = None,
        enabled: bool = True,
        description: str | None = None,
    ) -> LLMModel:
        """创建模型。"""
        provider = self.provider_repository.get_by_id(provider_id)
        if provider is None:
            raise HTTPException(status_code=404, detail="Provider 不存在")
        if not name.strip():
            raise HTTPException(status_code=400, detail="模型名称不能为空")
        if not model_kind.strip():
            raise HTTPException(status_code=400, detail="模型类型不能为空")

        model = LLMModel(
            id=str(uuid4()),
            provider_id=provider_id,
            name=name,
            model_kind=model_kind,
            config_json=self._serialize_config(config),
            enabled=enabled,
            description=description,
        )
        return self.repository.create(model)

    def list_models(
        self,
        provider_id: str | None = None,
        model_kind: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[LLMModel], int]:
        """列出模型。"""
        items = self.repository.list(
            provider_id=provider_id,
            model_kind=model_kind,
            enabled=enabled,
            limit=limit,
            offset=offset,
        )
        total = self.repository.count(
            provider_id=provider_id,
            model_kind=model_kind,
            enabled=enabled,
        )
        return items, total

    def update_model(
        self,
        model_id: str,
        name: str | None = None,
        model_kind: str | None = None,
        config: dict | None = None,
        enabled: bool | None = None,
        description: str | None = None,
    ) -> LLMModel | None:
        """更新模型。"""
        model = self.repository.get_by_id(model_id)
        if model is None:
            return None

        if name is not None:
            if not name.strip():
                raise HTTPException(status_code=400, detail="模型名称不能为空")
            model.name = name

        if model_kind is not None:
            if not model_kind.strip():
                raise HTTPException(status_code=400, detail="模型类型不能为空")
            model.model_kind = model_kind

        if config is not None:
            model.config_json = self._serialize_config(config)

        if enabled is not None:
            model.enabled = enabled

        if description is not None:
            model.description = description

        return self.repository.update(model)

    def delete_model(self, model_id: str) -> bool:
        """删除模型。"""
        capability_count = self.capability_repository.count(model_id=model_id)
        if capability_count > 0:
            raise HTTPException(status_code=409, detail="模型仍被能力映射引用")
        return self.repository.delete(model_id)
