from __future__ import annotations

from celery import Celery

from src.shared.config import get_settings


def make_celery() -> Celery:
    settings = get_settings()
    celery_app = Celery(
        "lumoscribe2026",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["src.interfaces.worker.tasks"],
    )
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
    )
    return celery_app


celery_app = make_celery()
