from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.interfaces.api.deps import get_db
from src.interfaces.worker.celery_app import celery_app
from src.shared.db import Job


router = APIRouter()


class CreateJobRequest(BaseModel):
    kind: str = Field(min_length=1, max_length=64)


class JobResponse(BaseModel):
    id: int
    kind: str
    status: str
    progress: int
    celery_task_id: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobListResponse(BaseModel):
    items: list[JobResponse]
    total: int


@router.post("/jobs", response_model=JobResponse)
def create_job(payload: CreateJobRequest, db: Session = Depends(get_db)):
    job = Job(kind=payload.kind, status="pending", progress=0)
    db.add(job)
    db.commit()
    db.refresh(job)

    async_result = celery_app.send_task("jobs.execute_placeholder", args=[job.id])
    job.celery_task_id = async_result.id
    db.commit()
    db.refresh(job)

    return JobResponse(
        id=job.id,
        kind=job.kind,
        status=job.status,
        progress=job.progress,
        celery_task_id=job.celery_task_id,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    from sqlalchemy import select, func

    # Query items
    stmt = select(Job).order_by(Job.created_at.desc()).limit(limit).offset(offset)
    jobs = db.execute(stmt).scalars().all()

    # Query total count
    count_stmt = select(func.count()).select_from(Job)
    total = db.execute(count_stmt).scalar() or 0

    items = [
        JobResponse(
            id=job.id,
            kind=job.kind,
            status=job.status,
            progress=job.progress,
            celery_task_id=job.celery_task_id,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )
        for job in jobs
    ]

    return JobListResponse(items=items, total=total)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        # 暂用 404 http 错误码（统一错误码体系后再替换为 AppError）
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="job not found")
    return JobResponse(
        id=job.id,
        kind=job.kind,
        status=job.status,
        progress=job.progress,
        celery_task_id=job.celery_task_id,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )
