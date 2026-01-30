from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import reset_settings_for_tests

# 导入所有实体以注册到 Base.metadata（解决循环导入问题）
from src.domain.entities import (
    SourceFile,
    Template,
    IntermediateArtifact,
    TargetFile,
    LLMProvider,
    LLMCapability,
    LLMCallSite,
    Prompt,
)


def pytest_configure(config) -> None:
    """将 pytest 的 basetemp 优先指向 F:\\temp（如可用）。

    说明：
    - `tmp_path`/`tmp_path_factory` 会使用 basetemp 作为临时目录根；
    - 如果用户已显式传入 `--basetemp`，则不覆盖；
    - 如果 F 盘不存在/不可写，则回退为 pytest 默认行为（系统临时目录）。
    """

    # 若用户已显式设置（例如命令行 --basetemp 或 PYTEST_ADDOPTS），则不覆盖
    basetemp = getattr(config.option, "basetemp", None)
    if basetemp:
        return

    preferred_root = Path(r"F:\temp")
    preferred = preferred_root / "pytest"

    try:
        preferred.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 回退到 pytest 默认 basetemp（通常是系统临时目录）
        return

    config.option.basetemp = str(preferred)


@pytest.fixture()
async def api_client(tmp_path: Path):
    """全局 API 客户端 Fixture，为每个测试用例提供隔离的运行时环境（SQLite + Storage）。"""
    # 为测试隔离运行时目录与 sqlite 文件
    runtime_dir = tmp_path / "runtime"
    storage_root = runtime_dir / "storage"
    sqlite_path = runtime_dir / "sqlite" / "test.db"

    # 配置环境变量
    os.environ["LUMO_STORAGE_ROOT"] = str(storage_root)
    os.environ["LUMO_SQLITE_PATH"] = str(sqlite_path)
    # 禁用加载 .env 文件，防止本地配置干扰测试
    os.environ["LUMO_DISABLE_DOTENV"] = "1"
    os.environ["LUMO_API_MODE"] = "full"
    
    # 重置 settings 单例以应用新的环境变量
    reset_settings_for_tests()

    # 初始化数据库并播种数据
    _init_test_database(sqlite_path)

    from src.interfaces.api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _init_test_database(sqlite_path: Path):
    """初始化测试数据库并播种必要的 LLM 调用点配置。"""
    from pathlib import Path
    
    # 确保目录存在
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    
    from src.shared.db import Base, make_engine, make_session_factory
    from src.shared.constants.llm_callsites import DEFAULT_CALLSITES
    from src.shared.constants.prompts import DEFAULT_PROMPTS
    from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
    from src.application.repositories.prompt_repository import PromptRepository
    from src.application.services.prompt_service import PromptService
    from src.domain.entities.llm_call_site import LLMCallSite
    from uuid import uuid4
    
    # 创建引擎和表
    engine = make_engine(sqlite_path)
    Base.metadata.create_all(engine)
    
    session_factory = make_session_factory(engine)
    
    # 播种 LLM 调用点
    with session_factory() as session:
        callsite_repo = LLMCallSiteRepository(session)
        
        for key, data in DEFAULT_CALLSITES.items():
            existing = callsite_repo.get_by_key(key)
            if existing is None:
                print(f"[TEST SEED] Creating callsite: {key}")
                callsite = LLMCallSite(
                    id=str(uuid4()),
                    key=key,
                    expected_model_kind=data.get("expected_model_kind"),
                    provider_id=None,  # 需要在后台配置
                    config_json=None,
                    prompt_scope=data.get("prompt_scope"),
                    enabled=True,
                    description=data.get("description"),
                )
                callsite_repo.create(callsite)
        
        session.commit()
    
    # 播种提示词
    with session_factory() as session:
        prompt_repo = PromptRepository(session)
        prompt_service = PromptService(prompt_repo)
        
        for scope, data in DEFAULT_PROMPTS.items():
            latest = prompt_repo.get_latest_version(scope)
            if latest == 0:
                print(f"[TEST SEED] Creating prompt: {scope}")
                prompt_service.create_prompt(
                    scope=scope,
                    format=data["format"],
                    content=data["content"],
                    messages=None,
                    active=True,
                    description=data["description"]
                )
        
        session.commit()
    
    print(f"[TEST SEED] Database initialized: {sqlite_path}")
