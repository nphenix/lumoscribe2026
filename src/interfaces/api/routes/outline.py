"""大纲 API 路由。

提供大纲润色和保存功能。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.application.repositories.prompt_repository import PromptRepository
from src.application.schemas.outline import (
    OutlinePolishRequest,
    OutlinePolishResponse,
    OutlineSaveRequest,
    OutlineSaveResponse,
)
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.application.services.outline_polish.outline_polish_service import (
    OutlinePolishService,
    OutlinePolishServiceError,
)
from src.application.services.outline_polish.schema import OutlinePolishInput
from src.interfaces.api.deps import get_db
from src.shared.logging import logger

router = APIRouter()


def get_prompt_repository(db: Session = Depends(get_db)) -> PromptRepository:
    """获取提示词仓库实例。"""
    return PromptRepository(db)


def get_llm_callsite_repository(db: Session = Depends(get_db)) -> LLMCallSiteRepository:
    """获取 LLM 调用点仓库实例。"""
    return LLMCallSiteRepository(db)


def get_llm_runtime_service(db: Session = Depends(get_db)) -> LLMRuntimeService:
    """获取 LLM 运行时服务实例。"""
    return LLMRuntimeService(
        provider_repository=LLMProviderRepository(db),
        capability_repository=None,  # 不需要 capability 仓库
        callsite_repository=LLMCallSiteRepository(db),
        prompt_repository=PromptRepository(db),
    )


def get_outline_polish_service(
    prompt_repo: PromptRepository = Depends(get_prompt_repository),
    callsite_repo: LLMCallSiteRepository = Depends(get_llm_callsite_repository),
    llm_runtime: LLMRuntimeService = Depends(get_llm_runtime_service),
) -> OutlinePolishService:
    """获取大纲润色服务实例。"""
    return OutlinePolishService(
        prompt_service=prompt_repo,
        llm_call_site_repository=callsite_repo,
        llm_runtime_service=llm_runtime,
    )


def _safe_filename(name: str) -> str:
    """生成安全的文件名（不含路径分隔符和非法字符）。"""
    s = (name or "").strip()
    if not s:
        return "outline"
    # Windows 文件名非法字符：<>:"/\|?*
    bad = '<>:"/\\|?*'
    out = []
    for ch in s:
        out.append("_" if ch in bad else ch)
    # 去掉尾部空格/点
    return "".join(out).strip().strip(".") or "outline"


def _get_drafts_dir() -> Path:
    """获取 drafts 目录路径，确保存在。"""
    drafts_dir = Path("data") / "Templates" / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    return drafts_dir


@router.post(
    "/outline/polish",
    response_model=OutlinePolishResponse,
    summary="大纲润色",
    description="接收用户输入的大纲文本，调用 LLM 执行润色，返回润色后的大纲。",
)
async def polish_outline(
    req: OutlinePolishRequest,
    service: OutlinePolishService = Depends(get_outline_polish_service),
) -> OutlinePolishResponse:
    """润色大纲。

    Args:
        req: 大纲润色请求
        service: 大纲润色服务

    Returns:
        润色后的大纲
    """
    try:
        # 构建输入数据
        input_data = OutlinePolishInput(outline=req.outline)

        # 调用服务执行润色
        result = await service.polish_outline(input_data)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"大纲润色失败: {result.error}",
            )

        if result.output is None:
            raise HTTPException(
                status_code=500,
                detail="大纲润色返回空结果",
            )

        return OutlinePolishResponse(
            polished_outline=result.output.polished_outline,
        )

    except HTTPException:
        raise
    except OutlinePolishServiceError as e:
        logger.error(f"大纲润色服务错误: {e}")
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=e.message,
        )
    except Exception as e:
        logger.exception(f"大纲润色失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"大纲润色失败: {str(e)}",
        )


@router.post(
    "/outline/save",
    response_model=OutlineSaveResponse,
    summary="保存大纲",
    description="将大纲内容保存到 data/Templates/drafts/ 目录。",
)
async def save_outline(req: OutlineSaveRequest) -> OutlineSaveResponse:
    """保存大纲到文件。

    Args:
        req: 大纲保存请求

    Returns:
        保存后的文件路径
    """
    try:
        # 生成安全文件名
        safe_name = _safe_filename(req.filename)
        if not safe_name.endswith(".md"):
            safe_name = f"{safe_name}.md"

        # 获取 drafts 目录并创建（如果不存在）
        drafts_dir = _get_drafts_dir()

        # 构建完整路径
        file_path = drafts_dir / safe_name

        # 保存文件（UTF-8 编码）
        file_path.write_text(req.outline, encoding="utf-8")

        logger.info(f"大纲已保存: {file_path.as_posix()}")

        return OutlineSaveResponse(
            file_path=file_path.as_posix(),
        )

    except Exception as e:
        logger.exception(f"保存大纲失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"保存大纲失败: {str(e)}",
        )
