from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LUMO_", extra="ignore")

    api_host: str = "127.0.0.1"
    api_port: int = 7901

    storage_root: Path = Path(".runtime/storage")
    sqlite_path: Path = Path(".runtime/sqlite/lumoscribe.db")

    redis_url: str = "redis://127.0.0.1:6379/0"

    log_level: str = "INFO"

    # ============== 业务默认参数（面向真实环境，不是测试专用） ==============
    # 白皮书生成（T096/whitepaper）默认检索参数：前端不传时使用
    whitepaper_top_k: int = 50
    whitepaper_rerank_top_n: int = 50
    whitepaper_polish_outline: bool = False

    # LLM 配置
    llm_openai_api_key: str = ""
    llm_openai_base_url: str = ""
    llm_ollama_base_url: str = "http://localhost:11434"
    llm_flagembedding_host: str = "http://localhost:7904"
    llm_huggingface_cache: str = "./models/huggingface"

    # MinerU 配置（在线服务）
    mineru_api_url: str = "https://mineru.net"
    mineru_api_key: str = ""
    mineru_model_version: str = "pipeline"
    mineru_timeout: int = 60
    mineru_upload_timeout: int = 300
    mineru_max_retries: int = 3


_settings: Settings | None = None


def _load_dotenv_into_environ(dotenv_path: Path = Path(".env")) -> None:
    """轻量加载 `.env` 到 os.environ（不覆盖已存在的环境变量）。

    说明：
    - 我们的 LLM 运行时会通过 `os.getenv(provider.api_key_env)` 获取密钥；
      仅靠 pydantic-settings 的 env_file 无法满足这一点（它不会写入 os.environ）。
    - 因此这里实现一个最小 `.env` 解析器，满足本项目需求。
    """

    # 允许测试或部署环境显式禁用
    if os.getenv("LUMO_DISABLE_DOTENV") == "1":
        return

    if not dotenv_path.exists():
        return

    try:
        raw = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return

    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue

        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        # 去掉两侧引号
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _load_dotenv_into_environ()
        _settings = Settings()
    return _settings


def reset_settings_for_tests() -> None:
    """仅用于测试：清空配置缓存，便于使用 monkeypatch 设置环境变量。"""
    global _settings
    _settings = None
