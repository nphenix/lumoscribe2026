from __future__ import annotations

import pytest
import httpx


@pytest.mark.anyio
async def test_health(api_client: httpx.AsyncClient):
    resp = await api_client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["status"] in ["ok", "error"]
    assert "version" in data
    assert "components" in data
    assert "info" in data
    
    components = data["components"]
    assert "db" in components
    assert "redis" in components
    assert "worker" in components
    
    info = data["info"]
    assert "db" in info
    assert "worker" in info
    assert "type" in info["db"]
    assert "path" in info["db"]
    assert "description" in info["db"]
    assert "description" in info["worker"]
    
    # In test environment with SQLite, DB should be connected
    assert components["db"] is True


@pytest.mark.anyio
async def test_validation_error_format(api_client: httpx.AsyncClient):
    resp = await api_client.post("/v1/jobs", json={})
    assert resp.status_code == 422

    body = resp.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["request_id"]
    assert resp.headers.get("X-Request-ID")
