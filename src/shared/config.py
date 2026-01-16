from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LUMO_", extra="ignore")

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    storage_root: Path = Path(".runtime/storage")
    sqlite_path: Path = Path(".runtime/sqlite/lumoscribe.db")

    redis_url: str = "redis://127.0.0.1:6379/0"

    log_level: str = "INFO"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_for_tests() -> None:
    """仅用于测试：清空配置缓存，便于使用 monkeypatch 设置环境变量。"""
    global _settings
    _settings = None
