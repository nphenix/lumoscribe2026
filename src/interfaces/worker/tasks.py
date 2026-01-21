from __future__ import annotations

from datetime import datetime
from time import sleep

from celery import shared_task
from sqlalchemy.orm import Session

from src.shared.config import get_settings
from src.shared.db import Job, make_engine, make_session_factory
from src.shared.logging import configure_logging, get_logger


log = get_logger(__name__)


def _get_db() -> Session:
    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    session_factory = make_session_factory(engine)
    return session_factory()


@shared_task(name="jobs.execute_placeholder")
def execute_placeholder(job_id: int) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    db = _get_db()
    try:
        job = db.get(Job, job_id)
        if job is None:
            log.warning("job_not_found", extra={"job_id": job_id})
            return

        job.status = "running"
        job.started_at = datetime.utcnow()
        job.progress = 5
        db.commit()

        # 占位执行：模拟耗时任务
        for p in (25, 50, 75, 100):
            sleep(0.2)
            job.progress = p
            db.commit()

        job.status = "succeeded"
        job.finished_at = datetime.utcnow()
        db.commit()
    except Exception as exc:  # noqa: BLE001
        log.exception("job_failed", extra={"job_id": job_id})
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "failed"
            job.error_code = "worker_error"
            job.error_message = str(exc)
            job.finished_at = datetime.utcnow()
            db.commit()
        raise
    finally:
        db.close()
