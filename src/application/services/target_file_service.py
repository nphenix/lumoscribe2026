"""目标文件服务层。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from src.application.repositories.target_file_repository import TargetFileRepository
from src.domain.entities.target_file import TargetFile
from src.shared.storage import get_targets_path


class TargetFileService:
    """目标文件服务类。"""

    DEFAULT_WORKSPACE = "default"

    def __init__(self, repository: TargetFileRepository):
        self.repository = repository

    def _get_workspace(self, workspace_id: str) -> str:
        """获取工作空间名称，不存在时回退到 default。"""
        workspace_path = get_targets_path(workspace_id)
        if not workspace_path.exists():
            return self.DEFAULT_WORKSPACE
        return workspace_id

    def _parse_storage_path(self, storage_path: str) -> tuple[str, str]:
        """解析存储路径，提取工作空间和文件名。

        Args:
            storage_path: 存储路径 (如 "targets/workspace1/{id}.html")

        Returns:
            (workspace_name, target_id)
        """
        # storage_path 格式: targets/{workspace}/{target_id}.html
        parts = storage_path.split("/")
        if len(parts) >= 3:
            workspace = parts[1]
            filename = parts[-1]
            # 去除扩展名得到 target_id
            target_id = filename.rsplit(".", 1)[0]
            return workspace, target_id
        return self.DEFAULT_WORKSPACE, ""

    def get_target_file(self, target_id: str) -> TargetFile | None:
        """获取目标文件详情。"""
        return self.repository.get_by_id(target_id)

    def list_target_files(
        self,
        workspace_id: str | None = None,
        template_id: str | None = None,
        kb_id: str | None = None,
        job_id: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[TargetFile], int]:
        """列出目标文件。"""
        items = self.repository.list(
            workspace_id=workspace_id,
            template_id=template_id,
            kb_id=kb_id,
            job_id=job_id,
            created_after=created_after,
            created_before=created_before,
            limit=limit,
            offset=offset,
        )
        total = self.repository.count(
            workspace_id=workspace_id,
            template_id=template_id,
            kb_id=kb_id,
            job_id=job_id,
            created_after=created_after,
            created_before=created_before,
        )
        return items, total

    def download_target_file(self, target_id: str) -> StreamingResponse:
        """下载目标文件。

        Args:
            target_id: 目标文件ID

        Returns:
            StreamingResponse: 文件流响应

        Raises:
            HTTPException: 文件不存在时返回 404
        """
        target_file = self.repository.get_by_id(target_id)
        if target_file is None:
            raise HTTPException(status_code=404, detail="目标文件不存在")

        # 构建完整文件路径
        # storage_path 格式: targets/{workspace}/{target_id}.html
        file_path = Path("data") / target_file.storage_path

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="目标文件不存在")

        # 使用 output_filename 作为下载文件名
        filename = target_file.output_filename

        def iter_file() -> AsyncGenerator[bytes, None]:
            """迭代文件内容。"""
            yield file_path.read_bytes()

        return StreamingResponse(
            iter_file(),
            media_type="text/html",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(file_path.stat().st_size),
            },
        )

    def get_file_stream(self, target_id: str) -> tuple[StreamingResponse, str]:
        """获取文件流和文件名（用于内联查看）。

        Args:
            target_id: 目标文件ID

        Returns:
            (StreamingResponse, filename)

        Raises:
            HTTPException: 文件不存在时返回 404
        """
        target_file = self.repository.get_by_id(target_id)
        if target_file is None:
            raise HTTPException(status_code=404, detail="目标文件不存在")

        file_path = Path("data") / target_file.storage_path

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="目标文件不存在")

        filename = target_file.output_filename

        def iter_file() -> AsyncGenerator[bytes, None]:
            yield file_path.read_bytes()

        response = StreamingResponse(
            iter_file(),
            media_type="text/html",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Content-Length": str(file_path.stat().st_size),
            },
        )
        return response, filename

    def delete_target_file(self, target_id: str) -> bool:
        """删除目标文件。"""
        target_file = self.repository.get_by_id(target_id)
        if target_file is None:
            return False

        # 删除存储文件
        storage_path = Path("data") / target_file.storage_path
        if storage_path.exists():
            storage_path.unlink()

        # 删除数据库记录
        return self.repository.delete(target_id)
