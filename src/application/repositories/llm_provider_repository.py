"""LLM Provider 仓储层。"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.domain.entities.llm_provider import LLMProvider


class LLMProviderRepository:
    """LLM Provider 仓储类。"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, provider: LLMProvider) -> LLMProvider:
        """创建 Provider。"""
        self.db.add(provider)
        self.db.commit()
        self.db.refresh(provider)
        return provider

    def get_by_id(self, provider_id: str) -> LLMProvider | None:
        """根据ID获取 Provider。"""
        return self.db.query(LLMProvider).filter(LLMProvider.id == provider_id).first()

    def get_by_name(self, name: str) -> LLMProvider | None:
        """根据名称获取 Provider。"""
        return self.db.query(LLMProvider).filter(LLMProvider.name == name).first()

    def get_by_key(self, key: str) -> LLMProvider | None:
        """根据 key 获取 Provider。"""
        return self.db.query(LLMProvider).filter(LLMProvider.key == key).first()

    def list(
        self,
        provider_type: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LLMProvider]:
        """列出 Provider。"""
        query = self.db.query(LLMProvider)

        if provider_type is not None:
            query = query.filter(LLMProvider.provider_type == provider_type)

        if enabled is not None:
            query = query.filter(LLMProvider.enabled == enabled)

        return query.order_by(LLMProvider.created_at.desc()).offset(offset).limit(limit).all()

    def count(
        self,
        provider_type: str | None = None,
        enabled: bool | None = None,
    ) -> int:
        """统计 Provider 数量。"""
        query = self.db.query(LLMProvider)

        if provider_type is not None:
            query = query.filter(LLMProvider.provider_type == provider_type)

        if enabled is not None:
            query = query.filter(LLMProvider.enabled == enabled)

        return query.count()

    def update(self, provider: LLMProvider) -> LLMProvider:
        """更新 Provider。"""
        self.db.commit()
        self.db.refresh(provider)
        return provider

    def delete(self, provider_id: str) -> bool:
        """删除 Provider。"""
        provider = self.get_by_id(provider_id)
        if provider is None:
            return False

        self.db.delete(provider)
        self.db.commit()
        return True
