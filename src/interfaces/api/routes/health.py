from __future__ import annotations

import redis
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.interfaces.api.deps import get_db
from src.interfaces.worker.celery_app import celery_app
from src.shared.config import get_settings


router = APIRouter()


@router.get("/health")
def health(db: Session = Depends(get_db)):
    # 1. Check Database
    db_status = False
    try:
        db.execute(text("SELECT 1"))
        db_status = True
    except Exception:
        pass

    # 2. Check Redis
    redis_status = False
    settings = get_settings()
    try:
        r = redis.from_url(settings.redis_url, socket_connect_timeout=1)
        r.ping()
        redis_status = True
        r.close()
    except Exception:
        pass

    # 3. Check Worker (Celery)
    worker_status = False
    active_workers = 0
    if redis_status:
        try:
            # ping workers with short timeout
            workers = celery_app.control.ping(timeout=0.5)
            if workers:
                worker_status = True
                active_workers = len(workers)
        except Exception:
            pass

    return {
        "status": "ok" if (db_status and redis_status) else "error",
        "version": "0.1.0",
        "components": {
            "db": db_status,
            "redis": redis_status,
            "worker": worker_status,
        },
        "info": {
            "db": {
                "type": "SQLite",
                "path": str(settings.sqlite_path),
                "description": "核心业务数据存储（任务、文件记录、配置）",
            },
            "worker": {
                "active_count": active_workers,
                "description": "异步任务处理节点（OCR、文档清洗、RAG索引、LLM生成）",
            },
        },
    }
