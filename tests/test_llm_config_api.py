from __future__ import annotations

import httpx
import pytest


@pytest.mark.anyio
async def test_llm_provider_model_capability_flow(api_client: httpx.AsyncClient):
    # 创建 Provider
    resp = await api_client.post(
        "/v1/llm/providers",
        json={
            "name": "openai-local",
            "provider_type": "openai_compatible",
            "base_url": "http://localhost:8001/v1",
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
