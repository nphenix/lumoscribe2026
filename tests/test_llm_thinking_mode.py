from __future__ import annotations

import json

import pytest

import src.application.services.llm_runtime_service as lrs
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.domain.entities.llm_provider import LLMProvider


class _FakeChatOpenAI:
    last_params: dict | None = None

    def __init__(self, **params):
        type(self).last_params = params


def _make_provider(*, model: str, config: dict | None = None) -> LLMProvider:
    cfg = {"model": model, **(config or {})}
    return LLMProvider(
        id="00000000-0000-0000-0000-000000000001",
        key="test-provider",
        name=model,
        provider_type="openai_compatible",
        base_url="http://localhost:8001/v1",
        api_key="sk-test",
        api_key_env=None,
        config_json=json.dumps(cfg, ensure_ascii=True),
        enabled=True,
        description=None,
    )


def test_thinking_default_off_for_minimax_m21_injects_reasoning_split(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(lrs, "ChatOpenAI", _FakeChatOpenAI)
    runtime = LLMRuntimeService(  # type: ignore[arg-type]
        provider_repository=None,
        capability_repository=None,
        callsite_repository=None,
        prompt_repository=None,
    )

    provider = _make_provider(model="MiniMax-M2.1")
    runtime._build_chat_model(provider, callsite_config={})

    params = _FakeChatOpenAI.last_params or {}
    assert params["model"] == "MiniMax-M2.1"
    assert params.get("extra_body", {}).get("reasoning_split") is True


def test_thinking_enabled_for_minimax_m21_does_not_inject_reasoning_split(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(lrs, "ChatOpenAI", _FakeChatOpenAI)
    runtime = LLMRuntimeService(  # type: ignore[arg-type]
        provider_repository=None,
        capability_repository=None,
        callsite_repository=None,
        prompt_repository=None,
    )

    provider = _make_provider(model="MiniMax-M2.1", config={"thinking_enabled": True})
    runtime._build_chat_model(provider, callsite_config={})

    params = _FakeChatOpenAI.last_params or {}
    # 不应注入 reasoning_split
    assert "extra_body" not in params or "reasoning_split" not in (params.get("extra_body") or {})


def test_reasoning_split_not_sent_for_non_m21_even_if_configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(lrs, "ChatOpenAI", _FakeChatOpenAI)
    runtime = LLMRuntimeService(  # type: ignore[arg-type]
        provider_repository=None,
        capability_repository=None,
        callsite_repository=None,
        prompt_repository=None,
    )

    provider = _make_provider(
        model="gpt-4o-mini",
        config={"extra_body": {"reasoning_split": True}},
    )
    runtime._build_chat_model(provider, callsite_config={})

    params = _FakeChatOpenAI.last_params or {}
    assert params["model"] == "gpt-4o-mini"
    assert "extra_body" not in params or "reasoning_split" not in (params.get("extra_body") or {})


def test_force_streaming_overrides_provider_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(lrs, "ChatOpenAI", _FakeChatOpenAI)
    runtime = LLMRuntimeService(  # type: ignore[arg-type]
        provider_repository=None,
        capability_repository=None,
        callsite_repository=None,
        prompt_repository=None,
    )

    provider = _make_provider(model="gpt-4o-mini", config={"streaming": False})
    runtime._build_chat_model(provider, callsite_config={}, force_streaming=True)
    params = _FakeChatOpenAI.last_params or {}
    assert params.get("streaming") is True

