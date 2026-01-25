"""图转 JSON API 路由（T094）。

说明：
- 仅负责触发 T094（cleaned_doc/images -> pic_to_json/chart_json）。
- 进度日志：每张图片都会在控制台输出一条 structured log（见 worker 任务实现）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.application.repositories.source_file_repository import SourceFileRepository
from src.application.schemas.ingest import IngestJobListResponse, IngestJobResponse, IngestTriggerResponse
from src.domain.entities.source_file import SourceFileStatus
from src.interfaces.api.deps import get_db
from src.interfaces.worker.ingest_tasks import process_chart_json
from src.shared.db import Job

router = APIRouter()


class ChartJsonTriggerRequest(BaseModel):
    """触发图转 JSON（T094）请求。"""

    workspace_id: str | None = Field(
        default=None,
        max_length=36,
        description="工作空间 ID（可选）。不传时使用 source_file_ids 或默认筛选。",
    )
    source_file_ids: list[str] | None = Field(
        default=None,
        description="源文件 ID 列表（可选）。传入则仅处理这些文件。",
    )
    # 运行参数（全部为可选，默认值在 worker 侧兜底）
    concurrency: int | None = Field(
        default=None,
        ge=1,
        le=8,
        description="并发数（按 source_file 并发，默认 2）。",
    )
    max_images: int | None = Field(
        default=None,
        ge=1,
        description="每个 source_file 最多处理多少张图片（为空则全量）。",
    )
    resume: bool = Field(default=False, description="是否启用断点续跑（复用 run_report/state）。")
    strict: bool = Field(default=True, description="严格模式：必须存在激活 Prompt，否则报错。")
    timeout_seconds: int | None = Field(default=300, ge=10, le=3600, description="单图超时时间（秒）。")
    wait_for_completion: bool = Field(
        default=False,
        description="是否同步等待任务完成（当前为 Eager 模式，会在本进程同步执行）。",
    )


def _serialize_job(job: Job) -> IngestJobResponse:
    return IngestJobResponse(
        id=job.id,
        job_type=job.kind,
        status=job.status,
        progress=job.progress or 0,
        celery_task_id=job.celery_task_id,
        input_summary=job.input_summary,
        result_summary=job.result_summary,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.post(
    "/chart-json/trigger",
    response_model=IngestTriggerResponse,
    summary="触发图转 JSON（T094）",
    description="对已完成清洗（CLEANING_COMPLETED）的文件执行 T094 图转 JSON，并输出每张图片的处理日志。",
)
def trigger_chart_json_task(
    payload: ChartJsonTriggerRequest | None = Body(default=None),
    db: Session = Depends(get_db),
):
    # 允许前端不传任何参数（空 body），按默认逻辑执行
    payload = payload or ChartJsonTriggerRequest()
    repo = SourceFileRepository(db)

    # 1) 选择要处理的源文件
    if payload.source_file_ids:
        source_files = []
        for file_id in payload.source_file_ids:
            sf = repo.get_by_id(file_id)
            if sf is not None:
                source_files.append(sf)
        if not source_files:
            raise HTTPException(status_code=400, detail="指定的文件 ID 列表中没有有效文件")
    else:
        # 默认：只跑已清洗完成的文件（避免误跑尚未产出 cleaned_doc 的文件）
        source_files = repo.list(
            workspace_id=payload.workspace_id,
            status=SourceFileStatus.CLEANING_COMPLETED,
            limit=1000,
        )
        if not source_files:
            raise HTTPException(
                status_code=400,
                detail="没有状态为 CLEANING_COMPLETED 的文件可供图转 JSON",
            )

    source_file_ids = [sf.id for sf in source_files]

    # 2) 创建 Job
    input_summary: dict[str, Any] = {
        "file_count": len(source_file_ids),
        "workspace_id": payload.workspace_id,
        "source_file_ids": source_file_ids,
        "stage": "t094_pic_to_json",
        "options": {
            "concurrency": payload.concurrency,
            "max_images": payload.max_images,
            "resume": payload.resume,
            "strict": payload.strict,
            "timeout_seconds": payload.timeout_seconds,
        },
    }
    job = Job(kind="chart_json", status="pending", progress=0, input_summary=input_summary)
    db.add(job)
    db.commit()
    db.refresh(job)

    # 3) 执行任务（Eager 模式：直接在当前进程执行）
    options = input_summary.get("options") or {}
    try:
        process_chart_json(job_id=job.id, source_file_ids=source_file_ids, options=options)
        task_id = f"eager-{job.id}"
    except Exception:
        # 任务内部会尽量写入 failed；这里不强行抛错给前端，避免前端误判“接口不可用”
        task_id = f"eager-{job.id}-failed"
    job.celery_task_id = task_id
    db.commit()

    # 返回尽量最新状态
    db.refresh(job)
    return IngestTriggerResponse(
        job_id=job.id,
        status=job.status,
        message="图转 JSON 任务已创建并开始执行" if not payload.wait_for_completion else "图转 JSON 任务已完成",
        input_summary=input_summary,
        created_at=job.created_at,
    )


@router.get(
    "/chart-json/jobs",
    response_model=IngestJobListResponse,
    summary="列出图转 JSON 任务",
    description="分页列出所有图转 JSON（T094）任务。",
)
def list_chart_json_jobs(limit: int = 20, offset: int = 0, db: Session = Depends(get_db)):
    stmt = select(Job).where(Job.kind == "chart_json").order_by(Job.created_at.desc()).limit(limit).offset(offset)
    jobs = db.execute(stmt).scalars().all()

    count_stmt = select(func.count()).select_from(Job).where(Job.kind == "chart_json")
    total = db.execute(count_stmt).scalar() or 0

    return IngestJobListResponse(items=[_serialize_job(j) for j in jobs], total=total)


@router.get(
    "/chart-json/jobs/{job_id}",
    response_model=IngestJobResponse,
    summary="获取图转 JSON 任务详情",
)
def get_chart_json_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None or job.kind != "chart_json":
        raise HTTPException(status_code=404, detail="任务不存在")
    return _serialize_job(job)

