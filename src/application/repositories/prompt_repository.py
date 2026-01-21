"""提示词仓储层。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from src.domain.entities.prompt import Prompt


class PromptRepository:
    """提示词仓储类。"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, prompt: Prompt) -> Prompt:
        """创建提示词。"""
        self.db.add(prompt)
        self.db.commit()
        self.db.refresh(prompt)
        return prompt

    def get_by_id(self, prompt_id: str) -> Prompt | None:
        """根据ID获取提示词。"""
        return self.db.query(Prompt).filter(Prompt.id == prompt_id).first()

    def list(
        self,
        scope: str | None = None,
        active: bool | None = None,
        format: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Prompt]:
        """列出提示词。"""
        query = self.db.query(Prompt)

        if scope is not None:
            query = query.filter(Prompt.scope == scope)

        if active is not None:
            query = query.filter(Prompt.active == active)

        if format is not None:
            query = query.filter(Prompt.format == format)

        return (
            query.order_by(Prompt.scope.asc(), Prompt.version.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def count(
        self,
        scope: str | None = None,
        active: bool | None = None,
        format: str | None = None,
    ) -> int:
        """统计提示词数量。"""
        query = self.db.query(Prompt)

        if scope is not None:
            query = query.filter(Prompt.scope == scope)

        if active is not None:
            query = query.filter(Prompt.active == active)

        if format is not None:
            query = query.filter(Prompt.format == format)

        return query.count()

    def update(self, prompt: Prompt) -> Prompt:
        """更新提示词。"""
        prompt.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(prompt)
        return prompt

    def delete(self, prompt_id: str) -> bool:
        """删除提示词。"""
        prompt = self.get_by_id(prompt_id)
        if prompt is None:
            return False

        self.db.delete(prompt)
        self.db.commit()
        return True

    def get_latest_version(self, scope: str) -> int:
        """获取指定 scope 的最新版本号。"""
        prompt = (
            self.db.query(Prompt)
            .filter(Prompt.scope == scope)
            .order_by(Prompt.version.desc())
            .first()
        )
        return prompt.version if prompt else 0

    def deactivate_scope(self, scope: str) -> int:
        """将指定 scope 下的提示词全部设为非激活。"""
        rows = (
            self.db.query(Prompt)
            .filter(Prompt.scope == scope, Prompt.active.is_(True))
            .all()
        )
        for item in rows:
            item.active = False
            item.updated_at = datetime.utcnow()
        if rows:
            self.db.commit()
        return len(rows)
