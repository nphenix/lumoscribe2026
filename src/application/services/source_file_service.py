"""源文件服务层。"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from src.application.repositories.source_file_repository import SourceFileRepository
from src.domain.entities.source_file import SourceFile, SourceFileStatus
from src.shared.errors import AppError
from src.shared.storage import get_sources_archive_path, get_sources_path


class SourceFileService:
    """源文件服务类。"""

    DEFAULT_WORKSPACE = "default"

    def __init__(self, repository: SourceFileRepository):
        self.repository = repository

    def _get_workspace(self, workspace_id: str) -> str:
        """获取工作空间名称，不存在时回退到 default。"""
        # workspace_id 本身作为目录名使用
        workspace_path = get_sources_path(workspace_id)
        if not workspace_path.exists():
            # 回退到 default
            return self.DEFAULT_WORKSPACE
        return workspace_id

    def _generate_storage_path(self, workspace_id: str, file_id: str) -> str:
        """生成存储路径。"""
        workspace = self._get_workspace(workspace_id)
        return f"sources/{workspace}/{file_id}.pdf"

    def _generate_archive_storage_path(self, workspace_id: str, file_id: str) -> str:
        """生成归档存储路径。"""
        workspace = self._get_workspace(workspace_id)
        return f"sources-archive/{workspace}/{file_id}.pdf.archived"

    def _calculate_file_hash(self, content: bytes) -> str:
        """计算文件 SHA256 哈希。"""
        return hashlib.sha256(content).hexdigest()

    async def create_source_file(
        self,
        workspace_id: str,
        file: UploadFile,
        description: str | None = None,
    ) -> SourceFile:
        """创建源文件（上传）。"""
        # 读取文件内容
        content = await file.read()
        file_hash = self._calculate_file_hash(content)

        # 检查文件是否已存在
        existing = self.repository.get_by_hash(file_hash)
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail="文件已存在",
            )

        # 生成唯一 ID
        source_id = str(uuid4())

        # 生成存储路径
        storage_path = self._generate_storage_path(workspace_id, source_id)

        # 保存文件
        full_path = get_sources_path(workspace_id, f"{source_id}.pdf")
        full_path.write_bytes(content)

        # 创建数据库记录
        source_file = SourceFile(
            id=source_id,
            workspace_id=workspace_id,
            original_filename=file.filename or "unknown.pdf",
            file_hash=file_hash,
            file_size=len(content),
            storage_path=storage_path,
            status=SourceFileStatus.ACTIVE,
            description=description,
        )

        return self.repository.create(source_file)

    def get_source_file(self, source_id: str) -> SourceFile | None:
        """获取源文件详情。"""
        return self.repository.get_by_id(source_id)

    def list_source_files(
        self,
        workspace_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[SourceFile], int]:
        """列出源文件。"""
        # 解析状态
        status_enum = None
        if status is not None:
            try:
                status_enum = SourceFileStatus(status)
            except ValueError:
                raise AppError(
                    code="invalid_status",
                    message=f"无效的状态值: {status}",
                    status_code=400,
                )

        items = self.repository.list(
            workspace_id=workspace_id,
            status=status_enum,
            limit=limit,
            offset=offset,
        )
        total = self.repository.count(
            workspace_id=workspace_id,
            status=status_enum,
        )
        return items, total

    def update_source_file(
        self,
        source_id: str,
        description: str | None = None,
    ) -> SourceFile | None:
        """更新源文件元数据。"""
        source_file = self.repository.get_by_id(source_id)
        if source_file is None:
            return None

        if description is not None:
            source_file.description = description

        return self.repository.update(source_file)

    def archive_source_file(self, source_id: str) -> SourceFile:
        """归档源文件。"""
        source_file = self.repository.get_by_id(source_id)
        if source_file is None:
            raise HTTPException(status_code=404, detail="源文件不存在")

        if source_file.status == SourceFileStatus.ARCHIVED:
            raise HTTPException(status_code=400, detail="源文件已归档")

        # 计算归档路径
        archive_storage_path = self._generate_archive_storage_path(
            source_file.workspace_id, source_id
        )

        # 移动文件
        source_path = Path("data") / source_file.storage_path
        archive_path = get_sources_archive_path(
            source_file.workspace_id, f"{source_id}.pdf.archived"
        )

        if source_path.exists():
            shutil.move(str(source_path), str(archive_path))

        # 更新数据库
        return self.repository.update_status(
            source_id=source_id,
            status=SourceFileStatus.ARCHIVED,
            archived_at=datetime.utcnow(),
        )

    def unarchive_source_file(self, source_id: str) -> SourceFile:
        """取消归档源文件。"""
        source_file = self.repository.get_by_id(source_id)
        if source_file is None:
            raise HTTPException(status_code=404, detail="源文件不存在")

        if source_file.status == SourceFileStatus.ACTIVE:
            raise HTTPException(status_code=400, detail="源文件未归档")

        # 计算原始路径
        original_storage_path = self._generate_storage_path(
            source_file.workspace_id, source_id
        )

        # 移动文件回原位置
        archive_path = get_sources_archive_path(
            source_file.workspace_id, f"{source_id}.pdf.archived"
        )
        original_path = get_sources_path(source_file.workspace_id, f"{source_id}.pdf")

        if archive_path.exists():
            shutil.move(str(archive_path), str(original_path))

        # 更新数据库
        self.repository.update_storage_path(source_id, original_storage_path)
        return self.repository.update_status(
            source_id=source_id,
            status=SourceFileStatus.ACTIVE,
            archived_at=None,
        )

    def delete_source_file(self, source_id: str) -> bool:
        """删除源文件。"""
        source_file = self.repository.get_by_id(source_id)
        if source_file is None:
            raise HTTPException(status_code=404, detail="源文件不存在")

        # 删除存储文件
        storage_path = Path("data") / source_file.storage_path
        if storage_path.exists():
            storage_path.unlink()

        # 如果是归档文件，删除归档文件
        if source_file.status == SourceFileStatus.ARCHIVED:
            archive_path = get_sources_archive_path(
                source_file.workspace_id, f"{source_id}.pdf.archived"
            )
            if archive_path.exists():
                archive_path.unlink()

        # 删除数据库记录
        return self.repository.delete(source_id)
