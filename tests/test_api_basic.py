from __future__ import annotations

from pathlib import Path

import pytest
import httpx

from src.shared.config import reset_settings_for_tests


@pytest.fixture()
async def api_client(tmp_path: Path):
    # 为测试隔离运行时目录与 sqlite 文件
    runtime_dir = tmp_path / "runtime"
    storage_root = runtime_dir / "storage"
    sqlite_path = runtime_dir / "sqlite" / "test.db"

    import os

    os.environ["LUMO_STORAGE_ROOT"] = str(storage_root)
    os.environ["LUMO_SQLITE_PATH"] = str(sqlite_path)
    os.environ["LUMO_DISABLE_DOTENV"] = "1"
    reset_settings_for_tests()

    from src.interfaces.api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.anyio
async def test_health(api_client: httpx.AsyncClient):
    resp = await api_client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_validation_error_format(api_client: httpx.AsyncClient):
    resp = await api_client.post("/v1/jobs", json={})
    assert resp.status_code == 422

    body = resp.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["request_id"]
    assert resp.headers.get("X-Request-ID")
