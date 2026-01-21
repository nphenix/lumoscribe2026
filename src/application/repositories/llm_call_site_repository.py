"""LLM CallSite 仓储层。"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.domain.entities.llm_call_site import LLMCallSite


class LLMCallSiteRepository:
    """LLM CallSite 仓储类。"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, callsite: LLMCallSite) -> LLMCallSite:
        """创建 CallSite。"""
        self.db.add(callsite)
        self.db.commit()
        self.db.refresh(callsite)
        return callsite

    def get_by_id(self, callsite_id: str) -> LLMCallSite | None:
        """根据ID获取 CallSite。"""
        return self.db.query(LLMCallSite).filter(LLMCallSite.id == callsite_id).first()

    def get_by_key(self, key: str) -> LLMCallSite | None:
        """根据 key 获取 CallSite。"""
        return self.db.query(LLMCallSite).filter(LLMCallSite.key == key).first()

    def list(
        self,
        *,
        key: str | None = None,
        expected_model_kind: str | None = None,
        enabled: bool | None = None,
        bound: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LLMCallSite]:
        """列出 CallSite。"""
        query = self.db.query(LLMCallSite)

        if key is not None:
            # 支持模糊查询：便于后台搜索
            query = query.filter(LLMCallSite.key.contains(key))

        if expected_model_kind is not None:
            query = query.filter(LLMCallSite.expected_model_kind == expected_model_kind)

        if enabled is not None:
            query = query.filter(LLMCallSite.enabled == enabled)

        if bound is True:
            query = query.filter(LLMCallSite.provider_id.is_not(None))
        elif bound is False:
            query = query.filter(LLMCallSite.provider_id.is_(None))

        return query.order_by(LLMCallSite.updated_at.desc()).offset(offset).limit(limit).all()

    def count(
        self,
        *,
        key: str | None = None,
        expected_model_kind: str | None = None,
        enabled: bool | None = None,
        bound: bool | None = None,
    ) -> int:
        """统计 CallSite 数量。"""
        query = self.db.query(LLMCallSite)

        if key is not None:
            query = query.filter(LLMCallSite.key.contains(key))

        if expected_model_kind is not None:
            query = query.filter(LLMCallSite.expected_model_kind == expected_model_kind)

        if enabled is not None:
            query = query.filter(LLMCallSite.enabled == enabled)

        if bound is True:
            query = query.filter(LLMCallSite.provider_id.is_not(None))
        elif bound is False:
            query = query.filter(LLMCallSite.provider_id.is_(None))

        return query.count()

    def update(self, callsite: LLMCallSite) -> LLMCallSite:
        """更新 CallSite。"""
        self.db.commit()
        self.db.refresh(callsite)
        return callsite

    def delete(self, callsite_id: str) -> bool:
        """删除 CallSite。"""
        callsite = self.get_by_id(callsite_id)
        if callsite is None:
            return False
        self.db.delete(callsite)
        self.db.commit()
        return True

