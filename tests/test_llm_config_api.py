from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.application.repositories.prompt_repository import PromptRepository
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.shared.config import reset_settings_for_tests
from src.shared.db import init_db, make_engine, make_session_factory


@pytest.mark.anyio
async def test_llm_provider_crud_and_concurrency_fields(api_client: httpx.AsyncClient):
    resp = await api_client.post(
        "/v1/llm/providers",
        json={
            "key": "openai-local",
            "name": "openai-local",
            "provider_type": "openai_compatible",
            "base_url": "http://localhost:7907/v1",
            "api_key_env": "OPENAI_API_KEY",
            "max_concurrency": 3,
            "enabled": True,
        },
    )
    assert resp.status_code == 201
    provider = resp.json()
    assert provider["max_concurrency"] == 3

    resp = await api_client.patch(
        f"/v1/llm/providers/{provider['id']}",
        json={"max_concurrency": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["max_concurrency"] == 5

    resp = await api_client.get("/v1/llm/providers", params={"limit": 50, "offset": 0})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any(p["id"] == provider["id"] and p["max_concurrency"] == 5 for p in items)


@pytest.mark.anyio
async def test_llm_callsite_crud_and_concurrency_fields_and_effective_policy(
    api_client: httpx.AsyncClient,
):
    resp = await api_client.post(
        "/v1/llm/providers",
        json={
            "key": "openai-local",
            "name": "openai-local",
            "provider_type": "openai_compatible",
            "base_url": "http://localhost:7907/v1",
            "api_key_env": "OPENAI_API_KEY",
            "enabled": True,
        },
    )
    assert resp.status_code == 201
    provider_id = resp.json()["id"]

    resp = await api_client.post(
        "/v1/llm/call-sites",
        json={
            "key": "tests:callsite:chat",
            "expected_model_kind": "chat",
            "provider_id": provider_id,
            "prompt_scope": "tests:callsite:chat",
            "max_concurrency": 2,
            "enabled": True,
            "description": "test callsite",
        },
    )
    assert resp.status_code == 201
    callsite = resp.json()
    assert callsite["max_concurrency"] == 2
    callsite_id = callsite["id"]

    resp = await api_client.get("/v1/llm/call-sites", params={"key": "tests:callsite:chat"})
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    assert resp.json()["items"][0]["id"] == callsite_id
    assert resp.json()["items"][0]["max_concurrency"] == 2

    resp = await api_client.patch(
        f"/v1/llm/call-sites/{callsite_id}",
        json={"max_concurrency": 4},
    )
    assert resp.status_code == 200
    assert resp.json()["max_concurrency"] == 4

    sqlite_path = Path(os.environ["LUMO_SQLITE_PATH"])
    engine = make_engine(sqlite_path)
    init_db(engine)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        runtime = LLMRuntimeService(
            provider_repository=LLMProviderRepository(session),
            capability_repository=LLMCapabilityRepository(session),
            callsite_repository=LLMCallSiteRepository(session),
            prompt_repository=PromptRepository(session),
        )
        resolved = runtime.resolve_effective_concurrency("tests:callsite:chat")
        assert resolved["callsite_limit"] == 4
        assert resolved["default_limit"] == 4
        assert resolved["effective"] == 4


@pytest.mark.anyio
async def test_prompt_scopes_diff_and_rollback(api_client: httpx.AsyncClient):
    resp = await api_client.post(
        "/v1/prompts",
        json={
            "scope": "tests:prompt",
            "format": "text",
            "content": "v1: hello {name}",
            "active": True,
            "description": "v1",
        },
    )
    assert resp.status_code == 201
    p1 = resp.json()
    assert p1["version"] == 1

    resp = await api_client.post(
        "/v1/prompts",
        json={
            "scope": "tests:prompt",
            "format": "text",
            "content": "v2: hello {name}!",
            "active": True,
            "description": "v2",
        },
    )
    assert resp.status_code == 201
    p2 = resp.json()
    assert p2["version"] == 2

    resp = await api_client.get("/v1/prompts/scopes", params={"scope": "tests:prompt"})
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    scope = resp.json()["items"][0]
    assert scope["scope"] == "tests:prompt"
    assert scope["latest_version"] == 2
    assert scope["active_version"] == 2
    assert scope["versions"] == 2

    resp = await api_client.get("/v1/prompts/diff", params={"from_id": p1["id"], "to_id": p2["id"]})
    assert resp.status_code == 200
    diff = resp.json()["diff"]
    assert "-v1: hello {name}" in diff
    assert "+v2: hello {name}!" in diff

    resp = await api_client.patch(f"/v1/prompts/{p1['id']}", json={"active": True})
    assert resp.status_code == 200
    assert resp.json()["active"] is True

    resp = await api_client.get("/v1/prompts", params={"scope": "tests:prompt", "active": True})
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    assert resp.json()["items"][0]["version"] == 1


def test_llm_runtime_concurrency_resolution_errors(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    sqlite_path = runtime_dir / "sqlite" / "test.db"
    storage_root = runtime_dir / "storage"

    os.environ["LUMO_STORAGE_ROOT"] = str(storage_root)
    os.environ["LUMO_SQLITE_PATH"] = str(sqlite_path)
    os.environ["LUMO_DISABLE_DOTENV"] = "1"
    reset_settings_for_tests()

    engine = make_engine(sqlite_path)
    init_db(engine)
    session_factory = make_session_factory(engine)

    from uuid import uuid4

    from src.domain.entities.llm_call_site import LLMCallSite
    from src.shared.errors import AppError

    with session_factory() as session:
        runtime = LLMRuntimeService(
            provider_repository=LLMProviderRepository(session),
            capability_repository=LLMCapabilityRepository(session),
            callsite_repository=LLMCallSiteRepository(session),
            prompt_repository=PromptRepository(session),
        )

        with pytest.raises(AppError) as excinfo:
            runtime.resolve_effective_concurrency("missing:callsite")
        assert excinfo.value.code == "llm_callsite_not_found"

        cs = LLMCallSite(
            id=str(uuid4()),
            key="tests:unbound",
            expected_model_kind="chat",
            provider_id=None,
            config_json=None,
            prompt_scope=None,
            enabled=True,
            description=None,
            max_concurrency=None,
        )
        session.add(cs)
        session.commit()

        with pytest.raises(AppError) as excinfo:
            runtime.resolve_effective_concurrency("tests:unbound")
        assert excinfo.value.code == "llm_callsite_unbound"
