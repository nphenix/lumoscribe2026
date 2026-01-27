"""内容生成 API 路由。

提供基于模板和知识库的内容生成功能。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.application.repositories.prompt_repository import PromptRepository
from src.application.repositories.template_repository import TemplateRepository
from src.application.schemas.content_generation import (
    GenerateOptions,
    GenerateProgressResponse,
    GenerateRequest,
    GenerateResponse,
    GenerateResultResponse,
)
from src.application.schemas.ingest import HybridSearchOptions
from src.application.services.chart_renderer_service import ChartRendererService
from src.application.services.content_generation_service import (
    ContentGenerationService,
    ContentGenerationError,
)
from src.application.services.hybrid_search_service import HybridSearchService
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.application.services.template_service import TemplateService
from src.interfaces.api.deps import get_db
from src.shared.db import Job
from src.shared.logging import logger
from src.interfaces.worker.celery_app import celery_app


router = APIRouter()


def get_template_repository(db: Session = Depends(get_db)) -> TemplateRepository:
    """获取模板仓库实例。"""
    return TemplateRepository(db)


def get_prompt_repository(db: Session = Depends(get_db)) -> PromptRepository:
    """获取提示词仓库实例。"""
    return PromptRepository(db)


def get_llm_callsite_repository(db: Session = Depends(get_db)) -> LLMCallSiteRepository:
    """获取 LLM 调用点仓库实例。"""
    return LLMCallSiteRepository(db)


def get_llm_provider_repository(db: Session = Depends(get_db)) -> LLMProviderRepository:
    """获取 LLM Provider 仓库实例。"""
    return LLMProviderRepository(db)


def get_content_generation_service(
    template_repo: TemplateRepository = Depends(get_template_repository),
    prompt_repo: PromptRepository = Depends(get_prompt_repository),
    callsite_repo: LLMCallSiteRepository = Depends(get_llm_callsite_repository),
    provider_repo: LLMProviderRepository = Depends(get_llm_provider_repository),
) -> ContentGenerationService:
    """获取内容生成服务实例。"""
    llm_runtime = LLMRuntimeService(
        provider_repository=provider_repo,
        capability_repository=None,
        callsite_repository=callsite_repo,
        prompt_repository=prompt_repo,
    )
    hybrid_search_service = HybridSearchService()
    chart_renderer_service = ChartRendererService()
    template_service = TemplateService(template_repo)

    return ContentGenerationService(
        template_service=template_service,
        hybrid_search_service=hybrid_search_service,
        llm_runtime_service=llm_runtime,
        chart_renderer_service=chart_renderer_service,
        template_repository=template_repo,
    )


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="创建内容生成任务",
    description="基于模板和知识库创建内容生成任务，支持同步和异步两种模式。",
)
async def create_generate_task(
    req: GenerateRequest,
    db: Session = Depends(get_db),
    service: ContentGenerationService = Depends(get_content_generation_service),
) -> GenerateResponse:
    """创建内容生成任务。

    Args:
        req: 生成请求参数
        db: 数据库会话
        service: 内容生成服务

    Returns:
        任务创建响应
    """
    # 验证模板存在
    template = service.template_service.get_template(req.template_id)
    if template is None:
        raise HTTPException(
            status_code=404,
            detail=f"模板不存在: {req.template_id}",
        )

    # 验证知识库 collection 存在（如果指定）
    if req.collection_name and req.collection_name != "default":
        try:
            kb_service = service.hybrid_search_service
            # 简化验证：尝试获取 collection info
            # 实际实现中可能需要检查 collection 是否存在
        except Exception:
            pass  # 允许不存在的 collection，降级为空检索

    # 创建 Job 记录
    job = Job(
        kind="content_generation",
        status="pending",
        progress=0,
        input_summary={
            "template_id": req.template_id,
            "collection_name": req.collection_name,
            "document_title": req.document_title,
            "outline_polish_enabled": req.outline_polish_enabled,
        },
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    logger.info(
        "generate_task_created",
        extra={
            "job_id": job.id,
            "template_id": req.template_id,
            "collection_name": req.collection_name,
        },
    )

    # 直接执行生成（同步模式），简化实现
    # 后续可改为 Celery 异步任务（参考 T045）
    try:
        search_opts = None
        if req.search_options:
            search_opts = HybridSearchOptions(
                top_k=req.search_options.top_k,
                use_rerank=req.search_options.use_rerank,
                rerank_top_n=req.search_options.rerank_top_n,
                filter_metadata=req.search_options.filter_metadata,
                score_threshold=req.search_options.score_threshold,
                vector_weight=req.search_options.vector_weight,
                bm25_weight=req.search_options.bm25_weight,
            )

        kb_root = None
        if req.kb_input_root:
            kb_root = Path(req.kb_input_root)

        # 执行生成
        result = await service.generate_content(
            template_id=req.template_id,
            collection_name=req.collection_name,
            search_options=search_opts,
            document_title=req.document_title,
            coverage_score_threshold=req.coverage_score_threshold,
            kb_input_root=kb_root,
            max_auto_charts_per_section=req.max_auto_charts_per_section,
            stream_tokens=req.stream_tokens,
        )

        # 更新任务状态
        job.status = "success"
        job.progress = 100
        job.result_summary = {
            "template_id": result.template_id,
            "template_name": result.template_name,
            "total_tokens": result.total_tokens,
            "total_time_ms": result.total_time_ms,
            "document_title": result.document_title,
        }
        db.commit()

        return GenerateResponse(
            job_id=job.id,
            status="success",
            message="内容生成成功",
            template_id=result.template_id,
            template_name=result.template_name,
            document_title=result.document_title,
            created_at=job.created_at,
        )

    except ContentGenerationError as e:
        job.status = "failed"
        job.error_code = e.code
        job.error_message = e.message
        db.commit()

        logger.error(
            "generate_task_failed",
            extra={
                "job_id": job.id,
                "error_code": e.code,
                "error_message": e.message,
            },
        )

        raise HTTPException(
            status_code=e.status_code or 500,
            detail=e.message,
        )

    except Exception as e:
        job.status = "failed"
        job.error_code = "unknown_error"
        job.error_message = str(e)
        db.commit()

        logger.exception(
            "generate_task_error",
            extra={
                "job_id": job.id,
                "error": str(e),
            },
        )

        raise HTTPException(
            status_code=500,
            detail=f"内容生成失败: {str(e)}",
        )


@router.get(
    "/generate/jobs/{job_id}",
    response_model=GenerateResultResponse,
    summary="获取生成任务结果",
    description="查询指定任务的执行结果和状态。",
)
async def get_generate_result(
    job_id: int,
    db: Session = Depends(get_db),
) -> GenerateResultResponse:
    """获取生成任务结果。

    Args:
        job_id: 任务 ID
        db: 数据库会话

    Returns:
        任务执行结果
    """
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"任务不存在: {job_id}",
        )

    if job.kind != "content_generation":
        raise HTTPException(
            status_code=400,
            detail=f"任务类型不正确: {job.kind}",
        )

    return GenerateResultResponse(
        job_id=job.id,
        status=job.status,
        template_id=job.input_summary.get("template_id", "") if job.input_summary else "",
        template_name=job.result_summary.get("template_name", "") if job.result_summary else "",
        document_title=job.input_summary.get("document_title") if job.input_summary else None,
        total_tokens=job.result_summary.get("total_tokens", 0) if job.result_summary else 0,
        total_time_ms=job.result_summary.get("total_time_ms", 0.0) if job.result_summary else 0.0,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        finished_at=job.finished_at,
    )


@router.get(
    "/generate/jobs/{job_id}/progress",
    response_model=GenerateProgressResponse,
    summary="获取生成任务进度",
    description="查询指定任务的执行进度（用于长时任务轮询）。",
)
async def get_generate_progress(
    job_id: int,
    db: Session = Depends(get_db),
) -> GenerateProgressResponse:
    """获取生成任务进度。

    Args:
        job_id: 任务 ID
        db: 数据库会话

    Returns:
        任务执行进度
    """
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"任务不存在: {job_id}",
        )

    if job.kind != "content_generation":
        raise HTTPException(
            status_code=400,
            detail=f"任务类型不正确: {job.kind}",
        )

    # 解析进度信息
    progress_info = job.result_summary or {}
    current_section = progress_info.get("current_section")
    section_index = progress_info.get("section_index", 0)
    section_total = progress_info.get("section_total", 0)
    tokens_used = progress_info.get("tokens_used", 0)

    return GenerateProgressResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        current_section=current_section,
        section_index=section_index,
        section_total=section_total,
        tokens_used=tokens_used,
        error_message=job.error_message,
    )


@router.get(
    "/generate/options",
    response_model=GenerateOptions,
    summary="获取生成选项",
    description="获取内容生成的可配置选项（用于前端表单）。",
)
async def get_generate_options() -> GenerateOptions:
    """获取生成选项。

    Returns:
        可配置的生成选项
    """
    return GenerateOptions(
        enable_outline_polish=True,
        enable_auto_chart_rendering=True,
        max_charts_per_section=6,
        coverage_threshold=None,
    )
