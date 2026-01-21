"""LLM 能力映射仓储层。"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.domain.entities.llm_capability import LLMCapability


class LLMCapabilityRepository:
    """LLM 能力映射仓储类。"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, capability: LLMCapability) -> LLMCapability:
        """创建能力映射。"""
        self.db.add(capability)
        self.db.commit()
        self.db.refresh(capability)
        return capability

    def get_by_id(self, capability_id: str) -> LLMCapability | None:
        """根据ID获取能力映射。"""
        return (
            self.db.query(LLMCapability)
            .filter(LLMCapability.id == capability_id)
            .first()
        )

    def get_by_capability_provider(
        self, capability: str, provider_id: str
    ) -> LLMCapability | None:
        """根据 capability 与 provider 获取映射。"""
        return (
            self.db.query(LLMCapability)
            .filter(
                LLMCapability.capability == capability,
                LLMCapability.provider_id == provider_id,
            )
            .first()
        )

    def list(
        self,
        capability: str | None = None,
        provider_id: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LLMCapability]:
        """列出能力映射。"""
        query = self.db.query(LLMCapability)

        if capability is not None:
            query = query.filter(LLMCapability.capability == capability)

        if provider_id is not None:
            query = query.filter(LLMCapability.provider_id == provider_id)

        if enabled is not None:
            query = query.filter(LLMCapability.enabled == enabled)

        return (
            query.order_by(LLMCapability.priority.asc(), LLMCapability.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def count(
        self,
        capability: str | None = None,
        provider_id: str | None = None,
        enabled: bool | None = None,
    ) -> int:
        """统计能力映射数量。"""
        query = self.db.query(LLMCapability)

        if capability is not None:
            query = query.filter(LLMCapability.capability == capability)

        if provider_id is not None:
            query = query.filter(LLMCapability.provider_id == provider_id)

        if enabled is not None:
            query = query.filter(LLMCapability.enabled == enabled)

        return query.count()

    def update(self, capability: LLMCapability) -> LLMCapability:
        """更新能力映射。"""
        self.db.commit()
        self.db.refresh(capability)
        return capability

    def delete(self, capability_id: str) -> bool:
        """删除能力映射。"""
        capability = self.get_by_id(capability_id)
        if capability is None:
            return False

        self.db.delete(capability)
        self.db.commit()
        return True
