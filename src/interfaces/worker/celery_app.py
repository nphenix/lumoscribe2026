from __future__ import annotations

from celery import Celery

from src.shared.config import get_settings


def make_celery() -> Celery:
    settings = get_settings()
    celery_app = Celery(
        "lumoscribe2026",
        # 强制使用 memory 作为 broker 和 backend，彻底绕过 Redis
        broker="memory://",
        backend="cache+memory://",
        include=["src.interfaces.worker.tasks", "src.interfaces.worker.ingest_tasks"],
    )
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # 强制开启 Eager 模式，无需启动 Worker 进程，直接在当前进程执行任务
        task_always_eager=True,
        task_eager_propagates=True,  # 异常向上冒泡
    )
    return celery_app


celery_app = make_celery()
