from __future__ import annotations

import httpx
import pytest
import os
from pathlib import Path

from src.shared.config import reset_settings_for_tests
from src.shared.db import init_db, make_engine, make_session_factory


@pytest.mark.anyio
async def test_llm_provider_model_capability_flow(api_client: httpx.AsyncClient):
    # 创建 Provider
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

    # 创建模型
    resp = await api_client.post(
        "/v1/llm/models",
        json={
            "provider_id": provider_id,
            "name": "gpt-4o-mini",
            "model_kind": "chat",
            "config": {"temperature": 0.2},
            "enabled": True,
        },
    )
    assert resp.status_code == 201
    model_id = resp.json()["id"]

    # 创建能力映射
    resp = await api_client.patch(
        "/v1/llm/capabilities",
        json={
            "items": [
                {
                    "capability": "doc_clean",
                    "model_id": model_id,
                    "priority": 10,
                    "enabled": True,
                    "description": "清洗能力",
                }
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.json()["items"][0]["capability"] == "doc_clean"

    # 删除 Provider 应被阻止
    resp = await api_client.delete(f"/v1/llm/providers/{provider_id}")
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_llm_provider_type_validation_and_provider_config(api_client: httpx.AsyncClient):
    # provider_type 非法（已移除 Cohere）
    resp = await api_client.post(
        "/v1/llm/providers",
        json={
            "key": "cohere-test",
            "name": "cohere-test",
            "provider_type": "cohere",
            "base_url": "https://api.cohere.com",
            "enabled": True,
        },
    )
    assert resp.status_code == 422

    # openai_compatible 支持 provider.config 写入通用参数
    resp = await api_client.post(
        "/v1/llm/providers",
        json={
            "key": "openai-with-config",
            "name": "openai-with-config",
            "provider_type": "openai_compatible",
            "base_url": "http://localhost:7907/v1",
            "api_key_env": "OPENAI_API_KEY",
            "config": {
                "temperature": 0.1,
                "max_tokens": 123,
                "timeout_seconds": 7,
                "stream": False,
            },
            "enabled": True,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["config"]["temperature"] == 0.1
    assert data["config"]["max_tokens"] == 123
    assert data["config"]["timeout_seconds"] == 7
    assert data["config"]["stream"] is False


@pytest.mark.anyio
async def test_llm_callsite_flow(api_client: httpx.AsyncClient):
    # 创建 Provider
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

    # 创建 chat 模型
    resp = await api_client.post(
        "/v1/llm/models",
        json={
            "provider_id": provider_id,
            "name": "gpt-4o-mini",
            "model_kind": "chat",
            "config": {"temperature": 0.2},
            "enabled": True,
        },
    )
    assert resp.status_code == 201
    model_id = resp.json()["id"]

    # 创建 CallSite（绑定模型 + 覆盖参数）
    resp = await api_client.post(
        "/v1/llm/call-sites",
        json={
            "key": "tests:callsite:chat",
            "expected_model_kind": "chat",
            "model_id": model_id,
            "config": {"temperature": 0.1, "max_tokens": 123},
            "prompt_scope": "tests:callsite:chat",
            "enabled": True,
            "description": "doc cleaning callsite",
        },
    )
    assert resp.status_code == 201
    callsite = resp.json()
    assert callsite["key"] == "tests:callsite:chat"
    assert callsite["model_id"] == model_id
    assert callsite["config"]["temperature"] == 0.1
    callsite_id = callsite["id"]

    # 列表查询
    resp = await api_client.get(
        "/v1/llm/call-sites",
        params={"key": "tests:callsite:chat"},
    )
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    assert resp.json()["items"][0]["id"] == callsite_id

    # 更新：禁用 + 修改参数覆盖
    resp = await api_client.patch(
        f"/v1/llm/call-sites/{callsite_id}",
        json={"enabled": False, "config": {"temperature": 0.3}},
    )
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False
    assert resp.json()["config"]["temperature"] == 0.3

    # 校验：expected_model_kind 非法
    resp = await api_client.post(
        "/v1/llm/call-sites",
        json={
            "key": "invalid:kind",
            "expected_model_kind": "invalid_kind",
        },
    )
    assert resp.status_code == 422 or resp.status_code == 400

    # 校验：model_kind 与 expected_model_kind 不一致
    resp = await api_client.post(
        "/v1/llm/call-sites",
        json={
            "key": "mismatch:kind",
            "expected_model_kind": "embedding",
            "model_id": model_id,
        },
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_prompt_versioning_and_chat(api_client: httpx.AsyncClient):
    # 创建 text 提示词 v1
    resp = await api_client.post(
        "/v1/prompts",
        json={
            "scope": "long_doc_generate",
            "format": "text",
            "content": "生成长文：{topic}",
            "active": True,
        },
    )
    assert resp.status_code == 201
    assert resp.json()["version"] == 1

    # 创建 text 提示词 v2，并激活
    resp = await api_client.post(
        "/v1/prompts",
        json={
            "scope": "long_doc_generate",
            "format": "text",
            "content": "生成长文 v2：{topic}",
            "active": True,
        },
    )
    assert resp.status_code == 201
    assert resp.json()["version"] == 2

    # scope 内仅一个 active
    resp = await api_client.get(
        "/v1/prompts",
        params={"scope": "long_doc_generate", "active": True},
    )
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    assert resp.json()["items"][0]["version"] == 2

    # 创建 chat 提示词
    resp = await api_client.post(
        "/v1/prompts",
        json={
            "scope": "doc_clean",
            "format": "chat",
            "messages": [
                {"role": "system", "content": "你是清洗助手"},
                {"role": "user", "content": "请清洗：{text}"},
            ],
            "active": True,
        },
    )
    assert resp.status_code == 201
    assert resp.json()["format"] == "chat"
    assert len(resp.json()["messages"]) == 2


def test_llm_runtime_callsite_errors(tmp_path: Path):
    # 独立 DB 环境
    runtime_dir = tmp_path / "runtime"
    sqlite_path = runtime_dir / "sqlite" / "test.db"
    storage_root = runtime_dir / "storage"

    os.environ["LUMO_STORAGE_ROOT"] = str(storage_root)
    os.environ["LUMO_SQLITE_PATH"] = str(sqlite_path)
    os.environ["LUMO_DISABLE_DOTENV"] = "1"
    reset_settings_for_tests()

    engine = make_engine(sqlite_path)
    # 导入实体以注册到 Base.metadata（确保 init_db 创建对应数据表）
    from src.domain.entities import (  # noqa: F401
        LLMCallSite,
        LLMCapability,
        LLMModel,
        LLMProvider,
        Prompt,
    )
    init_db(engine)
    session_factory = make_session_factory(engine)

    from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
    from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
    from src.application.repositories.llm_model_repository import LLMModelRepository
    from src.application.repositories.llm_provider_repository import LLMProviderRepository
    from src.application.repositories.prompt_repository import PromptRepository
    from src.application.services.llm_runtime_service import LLMRuntimeService
    from src.domain.entities.llm_call_site import LLMCallSite
    from src.shared.errors import AppError

    with session_factory() as session:
        runtime = LLMRuntimeService(
            provider_repository=LLMProviderRepository(session),
            model_repository=LLMModelRepository(session),
            capability_repository=LLMCapabilityRepository(session),
            callsite_repository=LLMCallSiteRepository(session),
            prompt_repository=PromptRepository(session),
        )

        # not found
        with pytest.raises(AppError) as excinfo:
            runtime.get_model_for_callsite("missing:callsite")
        assert excinfo.value.code == "llm_callsite_not_found"

        # unbound
        cs = LLMCallSite(
            id="00000000-0000-0000-0000-000000000001",
            key="doc_cleaning:clean_text",
            expected_model_kind="chat",
            model_id=None,
            config_json=None,
            prompt_scope=None,
            enabled=True,
            description=None,
        )
        session.add(cs)
        session.commit()

        with pytest.raises(AppError) as excinfo:
            runtime.get_model_for_callsite("doc_cleaning:clean_text")
        assert excinfo.value.code == "llm_callsite_unbound"
