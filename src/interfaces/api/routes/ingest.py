"""摄入 API 路由。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.application.repositories.source_file_repository import SourceFileRepository
from src.application.schemas.ingest import (
    IngestJobListResponse,
    IngestJobResponse,
    IngestProgressResponse,
    IngestTriggerRequest,
    IngestTriggerResponse,
)
from src.application.schemas.mineru import MinerUIngestOptions
from src.domain.entities.source_file import SourceFile, SourceFileStatus
from src.interfaces.api.deps import get_db
from src.interfaces.worker.celery_app import celery_app
from src.interfaces.worker.ingest_tasks import process_documents
from src.shared.db import Job


router = APIRouter()


def get_source_file_repository(db: Session = Depends(get_db)) -> SourceFileRepository:
    """获取源文件仓储实例。"""
    return SourceFileRepository(db)


def _serialize_job(job: Job, input_summary: dict[str, Any] | None = None) -> IngestJobResponse:
    """序列化 Job 实体为响应模型。"""
    return IngestJobResponse(
        id=job.id,
        job_type=job.kind,
        status=job.status,
        progress=job.progress or 0,
        celery_task_id=job.celery_task_id,
        input_summary=input_summary,
        result_summary=job.result_summary if hasattr(job, "result_summary") else None,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _parse_ingest_options(options: MinerUIngestOptions | None) -> dict[str, Any] | None:
    """解析摄入选项为字典。"""
    if options is None:
        return None
    return {
        "enable_formula": options.enable_formula,
        "enable_table": options.enable_table,
        "language": options.language.value if options.language else None,
        "model_version": options.model_version.value if options.model_version else None,
        "extra_formats": options.extra_formats,
        "callback_url": options.callback_url,
    }


@router.post(
    "/ingest/trigger",
    response_model=IngestTriggerResponse,
    summary="触发文档摄入任务",
    description="根据条件触发 MinerU 文档摄入任务，返回任务 ID 用于查询状态。",
)
def trigger_ingest_task(
    payload: IngestTriggerRequest,
    db: Session = Depends(get_db),
):
    """触发文档摄入任务。

    支持三种触发模式：
    1. 指定 source_file_ids：仅处理指定的源文件
    2. 指定 workspace_id：处理该工作空间下所有 ACTIVE 状态的源文件
    3. 都不指定：处理所有工作空间下 ACTIVE 状态的源文件

    任务异步执行，可通过返回的 job_id 查询进度。
    """
    repo = get_source_file_repository(db)

    # 1. 查询符合条件的源文件
    source_files: list[SourceFile]
    if payload.source_file_ids:
        # 指定文件 ID 列表
        source_files = []
        for file_id in payload.source_file_ids:
            sf = repo.get_by_id(file_id)
            if sf is not None and sf.status == SourceFileStatus.ACTIVE:
                source_files.append(sf)
        if not source_files:
            raise HTTPException(
                status_code=400,
                detail="指定的文件 ID 列表中没有有效的 ACTIVE 状态文件",
            )
    elif payload.workspace_id:
        # 指定工作空间
        source_files = repo.list(
            workspace_id=payload.workspace_id,
            status=SourceFileStatus.ACTIVE,
            limit=1000,
        )
        if not source_files:
            raise HTTPException(
                status_code=400,
                detail=f"工作空间 {payload.workspace_id} 中没有 ACTIVE 状态的源文件",
            )
    else:
        # 所有工作空间
        source_files = repo.list(status=SourceFileStatus.ACTIVE, limit=1000)
        if not source_files:
            raise HTTPException(
                status_code=400,
                detail="系统中没有 ACTIVE 状态的源文件",
            )

    # 2. 构建输入摘要
    input_summary = {
        "file_count": len(source_files),
        "workspace_id": payload.workspace_id,
        "source_file_ids": [sf.id for sf in source_files],
        "options": _parse_ingest_options(payload.options),
    }

    # 3. 创建 Job 记录
    job = Job(
        kind="ingest",
        status="pending",
        progress=0,
        input_summary=input_summary,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # 4. 执行任务（在 Eager 模式下直接调用，不走 Celery 代理）
    task_args = {
        "job_id": job.id,
        "source_file_ids": [sf.id for sf in source_files],
        "options": _parse_ingest_options(payload.options),
    }
    
    # 手动构造一个假的 AsyncResult 对象，避免 apply_async 的 broker 连接检查
    # 注意：在 Eager 模式下，我们直接调用任务函数
    try:
        # 直接调用任务函数（这会同步执行）
        process_documents(**task_args)
        # 任务成功执行，ID 设为 job.id 或占位符
        task_id = f"eager-{job.id}"
    except Exception as e:
        # 即使失败，也记录 ID
        task_id = f"eager-{job.id}-failed"
        # 异常已由任务内部捕获并更新 Job 状态，这里可以不抛出，或者根据需求处理
        # process_documents 内部有 try-catch，一般不会抛出异常到这里
        pass

    job.celery_task_id = task_id
    db.commit()

    # 5. 如果要求同步等待结果
    if payload.wait_for_completion:
        try:
            async_result.get(timeout=300)  # 最多等待 5 分钟
            db.refresh(job)
        except Exception as e:
            db.refresh(job)
            pass  # 任务状态已由 worker 更新

    return IngestTriggerResponse(
        job_id=job.id,
        status=job.status,
        message="任务已创建并入队执行" if not payload.wait_for_completion else "任务已完成",
        input_summary=input_summary,
        created_at=job.created_at,
    )


@router.post(
    "/ingest/cleaning/trigger",
    response_model=IngestTriggerResponse,
    summary="触发文档清洗任务",
    description="对已完成 MinerU 处理的文件进行清洗（规则过滤 + LLM 智能清洗）。",
)
def trigger_cleaning_task(
    db: Session = Depends(get_db),
):
    """触发文档清洗任务。

    自动查找所有状态为 MINERU_COMPLETED 的源文件并开始清洗。
    """
    from src.interfaces.worker.ingest_tasks import process_cleaning

    repo = get_source_file_repository(db)

    # 1. 查询所有待清洗的文件 (MINERU_COMPLETED)
    source_files = repo.list(status=SourceFileStatus.MINERU_COMPLETED, limit=1000)
    
    if not source_files:
        raise HTTPException(
            status_code=400,
            detail="没有状态为 MINERU_COMPLETED 的文件可供清洗",
        )

    # 2. 构建输入摘要
    input_summary = {
        "file_count": len(source_files),
        "source_file_ids": [sf.id for sf in source_files],
        "stage": "document_cleaning"
    }

    # 3. 创建 Job 记录
    job = Job(
        kind="ingest_cleaning",  # 使用区分于 ingest 的 job 类型
        status="pending",
        progress=0,
        input_summary=input_summary,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # 4. 执行任务（Eager 模式）
    task_args = {
        "job_id": job.id,
        "source_file_ids": [sf.id for sf in source_files],
        # 可以传递默认清洗选项
        "options": {
            "remove_ads": True,
            "remove_noise": True,
            "remove_duplicates": True,
            "preserve_structure": True
        }
    }

    try:
        process_cleaning(**task_args)
        task_id = f"eager-{job.id}"
    except Exception as e:
        task_id = f"eager-{job.id}-failed"
        pass

    job.celery_task_id = task_id
    db.commit()

    return IngestTriggerResponse(
        job_id=job.id,
        status=job.status,
        message="清洗任务已创建并开始执行",
        input_summary=input_summary,
        created_at=job.created_at,
    )


@router.get(
    "/ingest/jobs",
    response_model=IngestJobListResponse,
    summary="列出摄入任务",
    description="分页列出所有摄入任务。",
)
def list_ingest_jobs(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """列出摄入任务。"""
    from sqlalchemy import select, func

    # 筛选 ingest 类型的任务
    stmt = (
        select(Job)
        .where(Job.kind == "ingest")
        .order_by(Job.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    jobs = db.execute(stmt).scalars().all()

    count_stmt = select(func.count()).select_from(Job).where(Job.kind == "ingest")
    total = db.execute(count_stmt).scalar() or 0

    items = [_serialize_job(job, job.input_summary) for job in jobs]
    return IngestJobListResponse(items=items, total=total)


@router.get(
    "/ingest/jobs/{job_id}",
    response_model=IngestJobResponse,
    summary="获取摄入任务详情",
    description="获取指定任务的详细信息。",
)
def get_ingest_job(job_id: int, db: Session = Depends(get_db)):
    """获取摄入任务详情。"""
    job = db.get(Job, job_id)
    if job is None or job.kind != "ingest":
        raise HTTPException(status_code=404, detail="任务不存在")

    return _serialize_job(job, job.input_summary)

@router.get(
    "/ingest/cleaning/jobs",
    response_model=IngestJobListResponse,
    summary="列出清洗任务",
    description="分页列出所有文档清洗任务。",
)
def list_cleaning_jobs(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """列出清洗任务。"""
    from sqlalchemy import select, func

    stmt = (
        select(Job)
        .where(Job.kind == "ingest_cleaning")
        .order_by(Job.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    jobs = db.execute(stmt).scalars().all()

    count_stmt = select(func.count()).select_from(Job).where(Job.kind == "ingest_cleaning")
    total = db.execute(count_stmt).scalar() or 0

    items = [_serialize_job(job, job.input_summary) for job in jobs]
    return IngestJobListResponse(items=items, total=total)

@router.get(
    "/ingest/cleaning/jobs/{job_id}",
    response_model=IngestJobResponse,
    summary="获取清洗任务详情",
    description="获取指定清洗任务的详细信息。",
)
def get_cleaning_job(job_id: int, db: Session = Depends(get_db)):
    """获取清洗任务详情。"""
    job = db.get(Job, job_id)
    if job is None or job.kind != "ingest_cleaning":
        raise HTTPException(status_code=404, detail="任务不存在")
    return _serialize_job(job, job.input_summary)

@router.get(
    "/ingest/jobs/{job_id}/progress",
    response_model=IngestProgressResponse,
    summary="获取摄入任务进度",
    description="获取任务执行进度和结果摘要。",
)
def get_ingest_progress(job_id: int, db: Session = Depends(get_db)):
    """获取任务进度。"""
    job = db.get(Job, job_id)
    if job is None or job.kind != "ingest":
        raise HTTPException(status_code=404, detail="任务不存在")

    input_summary = job.input_summary or {}
    result_summary = job.result_summary if hasattr(job, "result_summary") else {}

    # 计算进度信息
    total_count = input_summary.get("file_count", 0)
    processed_count = result_summary.get("processed_count", 0)
    success_count = result_summary.get("success_count", 0)
    failed_count = result_summary.get("failed_count", 0)

    # 解析结果 URL
    result_urls = result_summary.get("result_urls", [])

    # 解析错误信息
    errors = result_summary.get("errors", [])
    error_message = "; ".join(errors) if errors else None

    return IngestProgressResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress or 0,
        processed_count=processed_count,
        total_count=total_count,
        success_count=success_count,
        failed_count=failed_count,
        current_file=result_summary.get("current_file"),
        result_urls=result_urls if result_urls else None,
        error_message=error_message,
    )
