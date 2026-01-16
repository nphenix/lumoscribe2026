"""模板仓储层。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from src.domain.entities.template import Template, TemplateType

if TYPE_CHECKING:
    pass


class TemplateRepository:
    """模板仓储类。"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, template: Template) -> Template:
        """创建模板。"""
        self.db.add(template)
        self.db.commit()
        self.db.refresh(template)
        return template

    def get_by_id(self, template_id: str) -> Template | None:
        """根据ID获取模板。"""
        return self.db.query(Template).filter(Template.id == template_id).first()

    def list(
        self,
        workspace_id: str | None = None,
        template_type: TemplateType | None = None,
        locked: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Template]:
        """列出模板。"""
        query = self.db.query(Template)

        if workspace_id is not None:
            query = query.filter(Template.workspace_id == workspace_id)

        if template_type is not None:
            query = query.filter(Template.type == template_type)

        if locked is not None:
            query = query.filter(Template.locked == locked)

        return query.order_by(Template.created_at.desc()).offset(offset).limit(limit).all()

    def count(
        self,
        workspace_id: str | None = None,
        template_type: TemplateType | None = None,
        locked: bool | None = None,
    ) -> int:
        """统计模板数量。"""
        query = self.db.query(Template)

        if workspace_id is not None:
            query = query.filter(Template.workspace_id == workspace_id)

        if template_type is not None:
            query = query.filter(Template.type == template_type)

        if locked is not None:
            query = query.filter(Template.locked == locked)

        return query.count()

    def update(self, template: Template) -> Template:
        """更新模板。"""
        template.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(template)
        return template

    def update_locked(
        self,
        template_id: str,
        locked: bool,
    ) -> Template | None:
        """更新模板锁定状态。"""
        template = self.get_by_id(template_id)
        if template is None:
            return None

        template.locked = locked
        template.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(template)
        return template

    def delete(self, template_id: str) -> bool:
        """删除模板。"""
        template = self.get_by_id(template_id)
        if template is None:
            return False

        self.db.delete(template)
        self.db.commit()
        return True

    def get_latest_version(
        self,
        workspace_id: str,
        original_filename: str,
        template_type: TemplateType,
    ) -> int:
        """获取指定文件的最新版本号。"""
        template = (
            self.db.query(Template)
            .filter(
                Template.workspace_id == workspace_id,
                Template.original_filename == original_filename,
                Template.type == template_type,
            )
            .order_by(Template.version.desc())
            .first()
        )
        return template.version if template else 0
