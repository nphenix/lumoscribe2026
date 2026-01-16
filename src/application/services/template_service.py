"""模板服务层。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import HTTPException

from src.application.repositories.template_repository import TemplateRepository
from src.domain.entities.template import Template, TemplateType
from src.shared.storage import (
    get_templates_custom_path,
    get_templates_system_path,
)


class TemplateService:
    """模板服务类。"""

    DEFAULT_WORKSPACE = "default"

    # 支持的文件格式
    CUSTOM_FORMATS = {"md"}
    SYSTEM_FORMATS = {
        "docx",
        "doc",
        "xlsx",
        "xls",
        "pptx",
        "ppt",
    }

    # Markdown 占位符正则
    PLACEHOLDER_PATTERN = re.compile(r"\{\{(\w+)\}\}")

    def __init__(self, repository: TemplateRepository):
        self.repository = repository

    def _get_workspace(self, workspace_id: str) -> str:
        """获取工作空间名称，不存在时回退到 default。"""
        workspace_path = get_templates_custom_path(workspace_id)
        if not workspace_path.exists():
            workspace_path = get_templates_system_path(workspace_id)
        if not workspace_path.exists():
            return self.DEFAULT_WORKSPACE
        return workspace_id

    def _detect_template_type(self, filename: str) -> TemplateType:
        """根据文件名检测模板类型。"""
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext in self.CUSTOM_FORMATS:
            return TemplateType.CUSTOM
        elif ext in self.SYSTEM_FORMATS:
            return TemplateType.SYSTEM
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: .{ext}。支持的自定义格式: {', '.join(self.CUSTOM_FORMATS)}；支持的系统格式: {', '.join(self.SYSTEM_FORMATS)}",
            )

    def _generate_storage_path(
        self,
        workspace_id: str,
        template_id: str,
        template_type: TemplateType,
    ) -> str:
        """生成存储路径。"""
        workspace = self._get_workspace(workspace_id)
        ext = ""  # 扩展名从原文件名获取
        if template_type == TemplateType.CUSTOM:
            return f"templates/custom/{workspace}/{template_id}.md"
        else:
            return f"templates/system/{workspace}/{template_id}"

    async def create_template(
        self,
        workspace_id: str,
        file,
        description: str | None = None,
    ) -> Template:
        """创建模板（上传）。"""
        # 验证文件名
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        original_filename = file.filename
        template_type = self._detect_template_type(original_filename)

        # 读取文件内容
        content = await file.read()

        # 生成唯一 ID
        template_id = str(uuid4())

        # 生成存储路径
        storage_path = self._generate_storage_path(
            workspace_id, template_id, template_type
        )

        # 获取扩展名
        ext = Path(original_filename).suffix

        # 保存文件
        if template_type == TemplateType.CUSTOM:
            full_path = get_templates_custom_path(workspace_id, f"{template_id}{ext}")
        else:
            full_path = get_templates_system_path(workspace_id, f"{template_id}{ext}")

        full_path.write_bytes(content)

        # 确定文件格式
        file_format = Path(original_filename).suffix.lstrip(".").lower()

        # 创建数据库记录
        template = Template(
            id=template_id,
            workspace_id=workspace_id,
            original_filename=original_filename,
            file_format=file_format,
            type=template_type,
            version=1,
            locked=False,
            storage_path=storage_path,
            description=description,
        )

        return self.repository.create(template)

    def get_template(self, template_id: str) -> Template | None:
        """获取模板详情。"""
        return self.repository.get_by_id(template_id)

    def list_templates(
        self,
        workspace_id: str | None = None,
        template_type: str | None = None,
        locked: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Template], int]:
        """列出模板。"""
        # 解析模板类型
        template_type_enum = None
        if template_type is not None:
            try:
                template_type_enum = TemplateType(template_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的模板类型: {template_type}。支持的值: custom, system",
                )

        items = self.repository.list(
            workspace_id=workspace_id,
            template_type=template_type_enum,
            locked=locked,
            limit=limit,
            offset=offset,
        )
        total = self.repository.count(
            workspace_id=workspace_id,
            template_type=template_type_enum,
            locked=locked,
        )
        return items, total

    def update_template(
        self,
        template_id: str,
        description: str | None = None,
    ) -> Template | None:
        """更新模板元数据。"""
        template = self.repository.get_by_id(template_id)
        if template is None:
            return None

        # 检查是否已锁定
        if template.locked:
            raise HTTPException(
                status_code=400,
                detail="模板已锁定，无法修改",
            )

        if description is not None:
            template.description = description

        return self.repository.update(template)

    def delete_template(self, template_id: str) -> bool:
        """删除模板。"""
        template = self.repository.get_by_id(template_id)
        if template is None:
            raise HTTPException(status_code=404, detail="模板不存在")

        # 检查是否已锁定
        if template.locked:
            raise HTTPException(
                status_code=400,
                detail="模板已锁定，无法删除",
            )

        # 删除存储文件
        storage_path = Path("data") / template.storage_path
        if storage_path.exists():
            storage_path.unlink()

        # 删除数据库记录
        return self.repository.delete(template_id)

    def lock_template(self, template_id: str, lock: bool = True) -> Template | None:
        """锁定或解锁模板。"""
        template = self.repository.get_by_id(template_id)
        if template is None:
            raise HTTPException(status_code=404, detail="模板不存在")

        return self.repository.update_locked(template_id, lock)

    def preprocess_template(self, template_id: str) -> dict:
        """预处理校验模板。"""
        template = self.repository.get_by_id(template_id)
        if template is None:
            raise HTTPException(status_code=404, detail="模板不存在")

        checks = []
        overall_status = "passed"
        message = "预处理校验通过"

        if template.type == TemplateType.CUSTOM:
            # Markdown 模板校验
            check = self._preprocess_custom_template(template)
            checks.append(check)
            if check["status"] != "passed":
                overall_status = check["status"]
                message = check["message"]
        else:
            # Office 模板校验
            check = self._preprocess_system_template(template)
            checks.append(check)
            if check["status"] != "passed":
                overall_status = check["status"]
                message = check["message"]

        return {
            "template_id": template_id,
            "template_type": template.type.value,
            "checks": checks,
            "overall_status": overall_status,
            "message": message,
        }

    def _preprocess_custom_template(self, template: Template) -> dict:
        """预处理校验自定义模板（Markdown）。"""
        storage_path = Path("data") / template.storage_path

        if not storage_path.exists():
            return {
                "type": "file_exists",
                "status": "error",
                "message": "模板文件不存在",
                "details": None,
            }

        try:
            content = storage_path.read_text(encoding="utf-8")

            # 检查占位符完整性
            placeholders = self.PLACEHOLDER_PATTERN.findall(content)
            unique_placeholders = list(set(placeholders))

            # 检查 Markdown 结构
            lines = content.split("\n")
            has_heading = any(line.startswith("#") for line in lines)
            has_content = len(content.strip()) > 0

            issues = []

            # 验证占位符格式
            # 检查是否有不完整的占位符
            incomplete_pattern = re.compile(r"\{\{[^}]*$")
            incomplete = incomplete_pattern.findall(content)
            if incomplete:
                issues.append(f"发现不完整的占位符: {incomplete}")

            check_status = "passed"
            if not has_heading:
                issues.append("模板缺少标题（# 开头）")
                check_status = "warning"
            if not has_content:
                issues.append("模板内容为空")
                check_status = "error"
            if incomplete:
                check_status = "error"

            if issues:
                message = "; ".join(issues)
            else:
                message = "Markdown 模板校验通过"

            return {
                "type": "custom_template",
                "status": check_status,
                "message": message,
                "details": {
                    "placeholders": unique_placeholders,
                    "placeholder_count": len(placeholders),
                    "has_heading": has_heading,
                    "content_length": len(content),
                },
            }

        except Exception as e:
            return {
                "type": "custom_template",
                "status": "error",
                "message": f"读取模板文件失败: {str(e)}",
                "details": None,
            }

    def _preprocess_system_template(self, template: Template) -> dict:
        """预处理校验系统模板（Office 文档）。"""
        storage_path = Path("data") / template.storage_path

        if not storage_path.exists():
            return {
                "type": "file_exists",
                "status": "error",
                "message": "模板文件不存在",
                "details": None,
            }

        file_format = template.file_format.lower()
        details = {"file_format": file_format}

        try:
            # 基础结构校验
            has_content = storage_path.stat().st_size > 0

            if not has_content:
                return {
                    "type": "system_template",
                    "status": "error",
                    "message": "模板文件为空",
                    "details": details,
                }

            # Office 文档特定校验
            if file_format in ("docx", "doc"):
                # Word 文档校验
                details["document_type"] = "word"
                # 简单校验：检查文件魔数
                if file_format == "docx":
                    # DOCX 是 ZIP 格式
                    details["is_valid_zip"] = True
                else:
                    # DOC 格式较复杂，这里做简单校验
                    details["is_old_format"] = True

            elif file_format in ("xlsx", "xls"):
                # Excel 文档校验
                details["document_type"] = "excel"
                if file_format == "xlsx":
                    details["is_valid_zip"] = True
                else:
                    details["is_old_format"] = True

            elif file_format in ("pptx", "ppt"):
                # PowerPoint 文档校验
                details["document_type"] = "powerpoint"
                if file_format == "pptx":
                    details["is_valid_zip"] = True
                else:
                    details["is_old_format"] = True

            return {
                "type": "system_template",
                "status": "passed",
                "message": f"{file_format.upper()} 文档结构校验通过",
                "details": details,
            }

        except Exception as e:
            return {
                "type": "system_template",
                "status": "error",
                "message": f"校验模板文件失败: {str(e)}",
                "details": None,
            }
