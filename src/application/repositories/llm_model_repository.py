"""LLM 模型仓储层。"""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.domain.entities.llm_model import LLMModel


class LLMModelRepository:
    """LLM 模型仓储类。"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, model: LLMModel) -> LLMModel:
        """创建模型。"""
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model

    def get_by_id(self, model_id: str) -> LLMModel | None:
        """根据ID获取模型。"""
        return self.db.query(LLMModel).filter(LLMModel.id == model_id).first()

    def list(
        self,
        provider_id: str | None = None,
        model_kind: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LLMModel]:
        """列出模型。"""
        query = self.db.query(LLMModel)

        if provider_id is not None:
            query = query.filter(LLMModel.provider_id == provider_id)

        if model_kind is not None:
            query = query.filter(LLMModel.model_kind == model_kind)

        if enabled is not None:
            query = query.filter(LLMModel.enabled == enabled)

        return query.order_by(LLMModel.created_at.desc()).offset(offset).limit(limit).all()

    def count(
        self,
        provider_id: str | None = None,
        model_kind: str | None = None,
        enabled: bool | None = None,
    ) -> int:
        """统计模型数量。"""
        query = self.db.query(LLMModel)

        if provider_id is not None:
            query = query.filter(LLMModel.provider_id == provider_id)

        if model_kind is not None:
            query = query.filter(LLMModel.model_kind == model_kind)

        if enabled is not None:
            query = query.filter(LLMModel.enabled == enabled)

        return query.count()

    def update(self, model: LLMModel) -> LLMModel:
        """更新模型。"""
        self.db.commit()
        self.db.refresh(model)
        return model

    def delete(self, model_id: str) -> bool:
        """删除模型。"""
        model = self.get_by_id(model_id)
        if model is None:
            return False

        self.db.delete(model)
        self.db.commit()
        return True
