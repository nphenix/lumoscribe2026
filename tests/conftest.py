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
    
    # 重置 settings 单例以应用新的环境变量
    reset_settings_for_tests()

    from src.interfaces.api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
