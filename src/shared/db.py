from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Integer, String, Text, JSON, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

if TYPE_CHECKING:
    # 避免循环导入，仅用于类型检查
    pass


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类。"""

    @classmethod
    def register_entity(cls, entity_cls) -> None:
        """手动注册实体到 metadata（解决循环导入问题）。"""
        cls.metadata.add(entity_cls)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)  # ingest/generate/...
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    celery_task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    input_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)


def make_engine(sqlite_path: Path):
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite+pysqlite:///{sqlite_path}", future=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
    _apply_lightweight_sqlite_migrations(engine)


def _apply_lightweight_sqlite_migrations(engine) -> None:
    """轻量 SQLite 迁移：为已有表补齐新列（避免开发期频繁删库）。

    说明：
    - 本项目未引入 Alembic，因此仅做“向后兼容”的 add column。
    - 仅对 SQLite 生效。
    """
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)

    # llm_providers.api_key（明文）
    if "llm_providers" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("llm_providers")}
        if "api_key" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE llm_providers ADD COLUMN api_key TEXT"))
        if "max_concurrency" not in cols:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE llm_providers ADD COLUMN max_concurrency INTEGER")
                )

    # jobs.input_summary / result_summary
    if "jobs" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("jobs")}
        with engine.begin() as conn:
            if "input_summary" not in cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN input_summary JSON"))
            if "result_summary" not in cols:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN result_summary JSON"))

    if "llm_call_sites" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("llm_call_sites")}
        if "max_concurrency" not in cols:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE llm_call_sites ADD COLUMN max_concurrency INTEGER")
                )
