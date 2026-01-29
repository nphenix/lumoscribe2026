"""LLM 运行时封装服务。

支持多种模型类型：
- chat: 对话模型 (ChatOpenAI, ChatOllama)
- embedding: 向量模型 (FlagEmbedding, OpenAI Embeddings)
- rerank: 重排序模型 (FlagEmbedding Reranker)
- multimodal: 多模态模型 (支持图像输入)
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, Union, runtime_checkable

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.outputs import LLMResult
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.application.repositories.prompt_repository import PromptRepository
from src.domain.entities.llm_capability import LLMCapability
from src.domain.entities.llm_call_site import LLMCallSite
from src.domain.entities.llm_provider import LLMProvider
from src.domain.entities.prompt import Prompt
from src.shared.errors import AppError
from src.shared.logging import logger


# ============== 统一模型接口协议 ==============

@runtime_checkable
class ChatProtocol(Protocol):
    """对话模型协议。"""

    def invoke(
        self,
        input: LanguageModelInput,
        config: dict | None = None,
    ) -> Any: ...

    def stream(
        self,
        input: LanguageModelInput,
        config: dict | None = None,
    ): ...


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """向量模型协议。"""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class RerankProtocol(Protocol):
    """重排序模型协议。"""

    def rerank(
        self, documents: list[str], query: str, top_n: int | None = None
    ) -> list[dict]: ...


# ============== FlagEmbedding Reranker 封装 ==============

class FlagEmbeddingReranker(LLM):
    """FlagEmbedding Reranker 封装。

    用于文档重排序，支持 BAAI/bge-reranker 系列模型。
    """

    model_name: str
    host: str = "http://localhost:8000"
    use_gpu: bool = False
    batch_size: int = 64
    max_length: int = 512

    @property
    def _llm_type(self) -> str:
        return "flagembedding_reranker"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("Reranker 不支持 _call 方法，请使用 rerank 方法")

    def rerank(
        self, documents: list[str], query: str, top_n: int | None = None
    ) -> list[dict]:
        """执行重排序。

        Args:
            documents: 待排序的文档列表
            query: 查询文本
            top_n: 返回前 N 个结果

        Returns:
            重排序结果列表，包含 index, text, score
        """
        import requests

        url = f"{self.host}/rerank"
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "max_length": self.max_length,
        }

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("results", [])

    def embed_query(self, text: str) -> list[float]:
        """获取查询文本的向量表示。"""
        import requests

        url = f"{self.host}/embed"
        payload = {"model": self.model_name, "text": text}

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("embedding", [])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """获取文档列表的向量表示。"""
        import requests

        url = f"{self.host}/embed"
        payload = {"model": self.model_name, "texts": texts}

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("embeddings", [])


# ============== FlagEmbedding 本地 Reranker 封装 ==============

class FlagEmbeddingLocalReranker:
    """FlagEmbedding 本地 Reranker 封装（无需远程服务）。

    依赖：FlagEmbedding（已在项目依赖中）。
    """

    def __init__(
        self,
        model_name: str,
        use_fp16: bool = True,
        device: str = "cpu",
        trust_remote_code: bool = True,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise AppError(
                code="flagembedding_not_installed",
                message="请安装 FlagEmbedding: pip install FlagEmbedding",
                status_code=400,
            ) from exc

        # FlagReranker 支持 device/use_fp16/trust_remote_code
        self._model = FlagReranker(
            self.model_name,
            use_fp16=self.use_fp16,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
        )

    def rerank(
        self, documents: list[str], query: str, top_n: int | None = None
    ) -> list[dict]:
        if self._model is None:
            self._load()

        pairs = [[query, doc] for doc in documents]
        # FlagReranker.compute_score 返回 list[float]（或 numpy array）
        scores = self._model.compute_score(pairs, batch_size=self.batch_size)
        try:
            scores_list = list(scores)
        except TypeError:
            scores_list = [float(scores)]

        results = [
            {"index": i, "text": documents[i], "score": float(scores_list[i])}
            for i in range(len(documents))
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        if top_n is not None:
            return results[: max(0, int(top_n))]
        return results


# ============== FlagEmbedding Embeddings 封装 ==============

class FlagEmbeddingEmbeddings:
    """FlagEmbedding Embeddings 封装。

    用于文本向量化，支持 BAAI/bge 系列模型。
    支持本地部署或远程 API 调用，支持 GPU 加速。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh",
        host: str | None = None,
        use_fp16: bool = True,
        device: str = "cpu",
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.host = host
        self.use_fp16 = use_fp16
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._model = None

    def _load_local_model(self):
        """加载本地 FlagEmbedding 模型。

        GPU 使用说明：
        - device = "cuda" 或 "cuda:0" 使用 NVIDIA GPU
        - device = "mps" 使用 Apple Silicon GPU
        - device = "cpu" 使用 CPU
        """
        try:
            from FlagEmbedding import FlagModel

            # 转换 device 字符串为 devices 列表
            devices = [self.device] if self.device else ["cpu"]

            self._model = FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                devices=devices,
                trust_remote_code=self.trust_remote_code,
            )
        except ImportError:
            raise AppError(
                code="flagembedding_not_installed",
                message="请安装 FlagEmbedding: pip install FlagEmbedding",
                status_code=400,
            )

    def embed_query(self, text: str) -> list[float]:
        """获取查询文本的向量表示。"""
        if self.host:
            return self._embed_via_api(text)
        if self._model is None:
            self._load_local_model()
        vecs = self.embed_documents([text])
        return vecs[0] if vecs else []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """获取文档列表的向量表示。"""
        if self.host:
            return self._embed_batch_via_api(texts)
        if self._model is None:
            self._load_local_model()
        if not texts:
            return []

        # FlagEmbedding/Transformers 版本差异较大：此处避免传 normalize_embeddings，
        # 统一在封装层做 L2 normalize，保证跨版本稳定。
        try:
            embeddings = self._model.encode(texts, batch_size=32)
        except TypeError:
            embeddings = self._model.encode(texts)

        try:
            mat = embeddings.tolist()
        except Exception:
            mat = [list(x) for x in embeddings]

        def _l2_normalize(v: list[float]) -> list[float]:
            import math

            s = 0.0
            for x in v:
                try:
                    fx = float(x)
                except Exception:
                    fx = 0.0
                s += fx * fx
            if s <= 0.0:
                return [0.0 for _ in v]
            inv = 1.0 / math.sqrt(s)
            return [float(x) * inv for x in v]

        return [_l2_normalize(v) for v in mat]

    def _embed_via_api(self, text: str) -> list[float]:
        """通过 API 获取文本向量。"""
        import requests

        url = f"{self.host}/embed"
        payload = {"model": self.model_name, "text": text}

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("embedding", [])

    def _embed_batch_via_api(self, texts: list[str]) -> list[list[float]]:
        """通过 API 批量获取文本向量。"""
        import requests

        url = f"{self.host}/embed"
        payload = {"model": self.model_name, "texts": texts}

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("embeddings", [])


# ============== 统一 LLM 运行时应 ==============

class LLMRuntimeService:
    """LLM 运行时服务。

    按 capability 选择模型并构建 LangChain Runnable。
    支持 chat、embedding、rerank、multimodal 四种模型类型。
    """

    ROLE_MAP = {"system": "system", "user": "human", "assistant": "ai"}

    # 模型类型常量
    MODEL_KIND_CHAT = "chat"
    MODEL_KIND_EMBEDDING = "embedding"
    MODEL_KIND_RERANK = "rerank"
    MODEL_KIND_MULTIMODAL = "multimodal"

    @staticmethod
    def _parse_bool_maybe(v: Any) -> bool | None:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        if not s:
            return None
        if s in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if s in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
        return None

    @staticmethod
    def _is_minimax_m21(model_name: str | None) -> bool:
        """判断是否为 MiniMax-M2.1 家族模型（包括 lightning 变体）。"""
        if not model_name:
            return False
        s = str(model_name).strip().lower()
        return "minimax-m2.1" in s

    @staticmethod
    def _normalize_device(device: Any) -> str:
        """规范化 device 配置，提升中台配置容错。

        说明：部分用户习惯写 "gpu"/"cuda0" 等，但 PyTorch 只接受 "cuda"/"cuda:0"/"cpu" 等。
        """
        if device is None:
            return "cpu"
        s = str(device).strip().lower()
        if not s:
            return "cpu"
        if s in {"gpu", "cuda", "nvidia"}:
            return "cuda"
        if s in {"gpu:0", "cuda0", "cuda:0"}:
            return "cuda:0"
        if s.startswith("gpu:"):
            # gpu:1 -> cuda:1
            return "cuda:" + s.split("gpu:", 1)[1]
        return s

    def _apply_thinking_mode_for_openai_compatible(
        self,
        *,
        provider: LLMProvider,
        model_name: str | None,
        extra_body: Any,
        thinking_enabled: bool | None,
    ) -> dict[str, Any] | None:
        """按“思考模式开关”调整 extra_body（当前仅对 MiniMax-M2.1 生效）。

        约定：
        - thinking_enabled=True：允许思考输出（不注入 reasoning_split）
        - thinking_enabled=False：隐藏思考输出（对 M2.1 注入 reasoning_split=True，使 thinking 不混入 content）
        - thinking_enabled=None：对 M2.1 默认按 False 处理
        """
        eb: dict[str, Any] = {}
        if isinstance(extra_body, dict):
            eb = dict(extra_body)
        elif extra_body is not None:
            # 容错：extra_body 类型不正确时忽略，避免影响主流程
            logger.warning(
                "invalid extra_body type, ignored",
                extra={
                    "provider_id": getattr(provider, "id", None),
                    "provider_key": getattr(provider, "key", None),
                    "model_name": model_name,
                    "extra_body_type": type(extra_body).__name__,
                },
            )
            eb = {}

        is_m21 = self._is_minimax_m21(model_name)
        if not is_m21:
            # 仅对 M2.1 生效：避免误把 reasoning_split 透传到其他 OpenAI-compatible 服务导致 400
            if "reasoning_split" in eb:
                eb.pop("reasoning_split", None)
            return eb or None

        # M2.1 默认关闭思考输出（避免 <think> 混入 content）
        if thinking_enabled is None:
            thinking_enabled = False

        if thinking_enabled:
            eb.pop("reasoning_split", None)
        else:
            eb["reasoning_split"] = True

        return eb or None

    def _sanitize_openai_compatible_model_kwargs(
        self,
        model_kwargs: dict[str, Any],
        *,
        provider: LLMProvider,
        model_name: str | None,
    ) -> dict[str, Any]:
        """移除明显不兼容 OpenAI-compatible 的参数，避免 400。

        背景：我们历史上给 Ollama 会传 `format=json` 等字段；当 CallSite/Provider 切到
        OpenAI-compatible（例如火山方舟 Doubao）时，这些字段可能被透传到请求体导致失败。
        """
        if not model_kwargs:
            return model_kwargs

        # 仅移除“确定是 Ollama-only”的字段；不要误伤 OpenAI 支持字段（如 stop/response_format 等）
        disallowed = {
            "format",
            "ollama_model",
            "keep_alive",
            "num_ctx",
            "num_gpu",
            "top_k",
            "tfs_z",
            "repeat_last_n",
            "repeat_penalty",
            "mirostat",
            "mirostat_tau",
            "mirostat_eta",
            "disable_streaming",
        }

        cleaned = dict(model_kwargs)
        removed: dict[str, Any] = {}
        for k in list(disallowed):
            if k in cleaned:
                removed[k] = cleaned.pop(k, None)

        if removed:
            try:
                logger.warning(
                    "removed unsupported openai_compatible model_kwargs",
                    extra={
                        "provider_id": getattr(provider, "id", None),
                        "provider_key": getattr(provider, "key", None),
                        "provider_type": getattr(provider, "provider_type", None),
                        "model_name": model_name,
                        "removed_keys": sorted(list(removed.keys())),
                    },
                )
            except Exception:
                pass

        return cleaned

    def __init__(
        self,
        provider_repository: LLMProviderRepository,
        capability_repository: LLMCapabilityRepository,
        callsite_repository: LLMCallSiteRepository,
        prompt_repository: PromptRepository,
    ):
        self.provider_repository = provider_repository
        self.capability_repository = capability_repository
        self.callsite_repository = callsite_repository
        self.prompt_repository = prompt_repository

    # ============== 统一构建接口 ==============

    def build_runnable_for_callsite(self, callsite_key: str, *, force_streaming: bool | None = None):
        """构建指定调用点的 runnable。

        Args:
            callsite_key: 调用点 key（建议 module:action）
            force_streaming: 当 callsite 为 chat 模型时，强制开启/关闭 token 流式（不修改数据库配置）。

        Returns:
            LangChain Runnable 对象
        """
        provider, callsite = self._resolve_callsite_provider(callsite_key)
        callsite_config = self._parse_json(callsite.config_json)
        model_kind = callsite.expected_model_kind

        if model_kind == self.MODEL_KIND_CHAT:
            scope = callsite.prompt_scope or callsite.key
            prompt = self._get_active_prompt(scope)
            llm = self._build_chat_model(provider, callsite_config, force_streaming=force_streaming)
            prompt_template = self._build_prompt(prompt)
            return prompt_template | llm | StrOutputParser()

        elif model_kind == self.MODEL_KIND_EMBEDDING:
            return self._build_embedding_model(provider, callsite_config)

        elif model_kind == self.MODEL_KIND_RERANK:
            return self._build_rerank_model(provider, callsite_config)

        elif model_kind == self.MODEL_KIND_MULTIMODAL:
            scope = callsite.prompt_scope or callsite.key
            prompt = self._get_active_prompt(scope)
            llm = self._build_multimodal_model(provider, callsite_config)
            prompt_template = self._build_prompt(prompt)
            return prompt_template | llm

        else:
            raise AppError(
                code="unsupported_model_kind",
                message=f"不支持的模型类型: {model_kind}",
                status_code=400,
                details={"callsite_key": callsite_key},
            )

    def build_runnable_for_capability(self, capability: str):
        """构建指定能力的 runnable。

        Args:
            capability: 能力名称 (如 'inference', 'embedding', 'rerank')

        Returns:
            LangChain Runnable 对象
        """
        # 兼容：如果存在同名 callsite，优先走 callsite 配置
        callsite = self.callsite_repository.get_by_key(capability)
        if callsite is not None:
            return self.build_runnable_for_callsite(capability)

        # 从 capability 映射获取 provider，但需要确定模型类型
        # 由于 capability 不再绑定 model，我们需要从第一个可用的 provider 获取
        # 或者要求 capability 必须通过 callsite 配置
        raise AppError(
            code="capability_requires_callsite",
            message=f"能力 '{capability}' 必须通过 callsite 配置，请创建对应的调用点",
            status_code=400,
        )

    def get_model_for_capability(
        self, capability: str
    ) -> Union[ChatProtocol, EmbeddingProtocol, RerankProtocol]:
        """获取指定能力的模型实例。

        Args:
            capability: 能力名称

        Returns:
            模型实例（根据类型不同返回不同协议）
        """
        # 兼容：如果存在同名 callsite，优先走 callsite 配置
        callsite = self.callsite_repository.get_by_key(capability)
        if callsite is not None:
            return self.get_model_for_callsite(capability)

        # 能力（capability）不再直接绑定模型：必须通过 CallSite 配置
        raise AppError(
            code="capability_requires_callsite",
            message=f"能力 '{capability}' 必须通过 callsite 配置，请创建对应的调用点",
            status_code=400,
        )

    def get_model_for_callsite(
        self, callsite_key: str
    ) -> Union[ChatProtocol, EmbeddingProtocol, RerankProtocol]:
        """获取指定调用点的模型实例。"""
        provider, callsite = self._resolve_callsite_provider(callsite_key)
        callsite_config = self._parse_json(callsite.config_json)
        model_kind = callsite.expected_model_kind

        if model_kind == self.MODEL_KIND_CHAT:
            return self._build_chat_model(provider, callsite_config)

        elif model_kind == self.MODEL_KIND_EMBEDDING:
            return self._build_embedding_model(provider, callsite_config)

        elif model_kind == self.MODEL_KIND_RERANK:
            return self._build_rerank_model(provider, callsite_config)

        elif model_kind == self.MODEL_KIND_MULTIMODAL:
            return self._build_multimodal_model(provider, callsite_config)

        else:
            raise AppError(
                code="unsupported_model_kind",
                message=f"不支持的模型类型: {model_kind}",
                status_code=400,
                details={"callsite_key": callsite_key},
            )

    # ============== 模型构建方法 ==============

    def _build_chat_model(
        self,
        provider: LLMProvider,
        callsite_config: dict | None = None,
        *,
        force_streaming: bool | None = None,
    ) -> ChatProtocol:
        """构建对话模型。"""
        provider_config = self._parse_json(provider.config_json)
        config = {**provider_config, **(callsite_config or {})}
        
        # 模型名称从 Provider config 中获取，如果没有则使用 Provider name
        default_model_name = config.get("model") or config.get("model_name") or provider.name

        extra_body = config.pop("extra_body", None)
        model_kwargs = config.pop("model_kwargs", None)
        # 常用参数（Provider/Model/CallSite 均可覆盖）
        # - OpenAI Compatible: temperature/max_tokens/timeout/streaming/model
        # - Ollama: temperature/max_tokens(timeout)/model(ollama_model)
        temperature = config.pop("temperature", None)
        max_tokens = config.pop("max_tokens", None)
        timeout_seconds = config.pop("timeout_seconds", None)
        if timeout_seconds is None:
            # 兼容其他常见命名
            timeout_seconds = config.pop("timeout", None)
        streaming = config.pop("stream", None)
        if streaming is None:
            streaming = config.pop("streaming", None)
        # 运行期强制开关（仅影响本次构建，不回写 DB）
        if force_streaming is not None:
            streaming = bool(force_streaming)
        model_override = config.pop("model", None)
        ollama_model = config.pop("ollama_model", None)  # 仅对 provider_type=ollama 生效
        # 中台“思考模式”开关（当前仅对 MiniMax-M2.1 生效）
        thinking_enabled = self._parse_bool_maybe(config.pop("thinking_enabled", None))
        if thinking_enabled is None:
            # 兼容未来可能的命名：thinking_mode=on/off
            thinking_enabled = self._parse_bool_maybe(config.pop("thinking_mode", None))

        if config:
            model_kwargs = {**(model_kwargs or {}), **config}

        if provider.provider_type == "openai_compatible":
            api_key = self._resolve_api_key(provider)
            model_name = model_override or default_model_name
            extra_body = self._apply_thinking_mode_for_openai_compatible(
                provider=provider,
                model_name=model_name,
                extra_body=extra_body,
                thinking_enabled=thinking_enabled,
            )
            if model_kwargs:
                model_kwargs = self._sanitize_openai_compatible_model_kwargs(
                    model_kwargs, provider=provider, model_name=model_name
                )
            params: dict[str, Any] = {
                "model": model_name,
                "api_key": api_key,
            }
            if provider.base_url:
                params["base_url"] = provider.base_url
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if timeout_seconds is not None:
                # langchain-openai: ChatOpenAI(timeout=...)
                params["timeout"] = timeout_seconds
            if streaming is not None:
                # langchain-openai: ChatOpenAI(streaming=...)
                if isinstance(streaming, str):
                    params["streaming"] = streaming.strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "y",
                        "on",
                    }
                else:
                    params["streaming"] = bool(streaming)
            if model_kwargs:
                params["model_kwargs"] = model_kwargs
            if extra_body:
                params["extra_body"] = extra_body
            return ChatOpenAI(**params)

        if provider.provider_type == "ollama":
            model_name = ollama_model or model_override or default_model_name
            params: dict[str, Any] = {"model": model_name}
            if provider.base_url:
                params["base_url"] = provider.base_url
            # 官方 langchain-ollama ChatOllama 支持 format="json" 等参数
            # 这里从 provider_config + callsite_config 做“显式透传”，避免 config_json 无效
            if temperature is not None:
                params["temperature"] = temperature
            if timeout_seconds is not None:
                params["timeout"] = timeout_seconds

            # 从 config 中透传常用 Ollama 参数（优先级：callsite > provider）
            passthrough_keys = [
                "format",  # "" | "json"
                "keep_alive",
                "num_ctx",
                "num_gpu",
                "top_k",
                "top_p",
                "tfs_z",
                "repeat_last_n",
                "repeat_penalty",
                "seed",
                "mirostat",
                "mirostat_tau",
                "mirostat_eta",
                "disable_streaming",
                "stop",
            ]
            for k in passthrough_keys:
                v = config.pop(k, None)
                if v is not None:
                    params[k] = v

            # 兼容用户把这些字段放在 model_kwargs 里
            if model_kwargs:
                for k in passthrough_keys:
                    if k in model_kwargs and k not in params and model_kwargs[k] is not None:
                        params[k] = model_kwargs[k]

            return ChatOllama(**params)

        # llama.cpp: 已移除 langchain-community 依赖，暂不支持该 provider_type
        if provider.provider_type == "llamacpp":
            raise AppError(
                code="provider_not_supported",
                message="llamacpp provider 已移除（为迁移到 LangChain 官方拆分包并移除 langchain-community）。请改用 ollama/openai_compatible，或在后续单独引入 llama.cpp 官方集成包后再启用。",
                status_code=400,
            )

        raise AppError(
            code="provider_not_supported",
            message=f"不支持的 provider_type: {provider.provider_type}",
            status_code=400,
        )

    def _build_embedding_model(
        self,
        provider: LLMProvider,
        callsite_config: dict | None = None,
    ) -> EmbeddingProtocol:
        """构建向量模型。

        支持:
        - FlagEmbedding 本地/远程模型
        - OpenAI Embeddings
        - HuggingFace Embeddings
        """
        provider_config = self._parse_json(provider.config_json)
        config = {**provider_config, **(callsite_config or {})}
        
        # 模型名称/路径从 Provider config 中获取，优先级：embedding_model_path > model > model_name > provider.name
        model_name = (
            config.get("embedding_model_path")
            or config.get("model")
            or config.get("model_name")
            or provider.name
        )

        # FlagEmbedding
        if provider.provider_type == "flagembedding":
            # 默认本地模式；只有显式声明 remote 时才走远程 API
            use_remote = config.pop("remote", None)
            host = config.pop("host", None)
            if use_remote is None:
                use_remote = bool(host)
            if use_remote and not host:
                host = provider.base_url
            use_fp16 = config.pop("use_fp16", True)
            device = self._normalize_device(config.pop("device", "cpu"))
            # embedding_dimension 用于元数据记录，不影响模型加载
            embedding_dimension = config.pop("embedding_dimension", None)

            if host:
                # 远程 API 模式
                return FlagEmbeddingEmbeddings(
                    model_name=model_name,
                    host=host,
                    use_fp16=use_fp16,
                    device=device,
                )
            else:
                # 本地模式
                return FlagEmbeddingEmbeddings(
                    model_name=model_name,
                    use_fp16=use_fp16,
                    device=device,
                )

        # OpenAI Embeddings
        if provider.provider_type == "openai_compatible":
            api_key = self._resolve_api_key(provider)
            from langchain_openai import OpenAIEmbeddings

            params = {"model": model_name, "api_key": api_key}
            if provider.base_url:
                params["base_url"] = provider.base_url
            return OpenAIEmbeddings(**params)

        # HuggingFace Embeddings
        if provider.provider_type == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError as exc:
                raise AppError(
                    code="huggingface_not_installed",
                    message="请安装 langchain-huggingface 以使用 HuggingFace Embeddings",
                    status_code=400,
                ) from exc
            cache_folder = config.pop("cache_folder", None)
            return HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=cache_folder,
                **config,
            )

        # llama.cpp embeddings: 已移除 langchain-community 依赖，暂不支持该 provider_type
        if provider.provider_type == "llamacpp":
            raise AppError(
                code="provider_not_supported",
                message="llamacpp embedding provider 已移除（为迁移到 LangChain 官方拆分包并移除 langchain-community）。请改用 openai_compatible/flagembedding/huggingface。",
                status_code=400,
            )

        raise AppError(
            code="provider_not_supported",
            message=f"embedding 不支持的 provider_type: {provider.provider_type}",
            status_code=400,
        )

    def _build_rerank_model(
        self,
        provider: LLMProvider,
        callsite_config: dict | None = None,
    ) -> RerankProtocol:
        """构建重排序模型。

        支持:
        - FlagEmbedding Reranker（默认本地；可选远程 API）
        """
        provider_config = self._parse_json(provider.config_json)
        config = {**provider_config, **(callsite_config or {})}
        
        # 模型名称/路径从 Provider config 中获取，优先级：rerank_model_path > model > model_name > provider.name
        model_name = (
            config.get("rerank_model_path")
            or config.get("model")
            or config.get("model_name")
            or provider.name
        )
        # rerank_top_k 用于控制返回结果数量，可在调用时使用
        rerank_top_k = config.pop("rerank_top_k", None)

        # FlagEmbedding Reranker（默认本地）
        if provider.provider_type == "flagembedding":
            use_remote = config.pop("remote", None)
            host = config.pop("host", None)
            if use_remote is None:
                use_remote = bool(host)

            # 远程模式：兼容历史实现（FlagEmbedding API Server）
            if use_remote:
                if not host:
                    host = provider.base_url or "http://localhost:8000"
                use_gpu = config.pop("use_gpu", False)
                batch_size = config.pop("batch_size", 64)
                max_length = config.pop("max_length", 512)

                return FlagEmbeddingReranker(
                    model_name=model_name,
                    host=host,
                    use_gpu=use_gpu,
                    batch_size=batch_size,
                    max_length=max_length,
                )

            # 本地模式：直接加载 reranker 模型
            use_fp16 = config.pop("use_fp16", True)
            device = self._normalize_device(config.pop("device", "cpu"))
            trust_remote_code = config.pop("trust_remote_code", True)
            batch_size = config.pop("batch_size", 64)
            return FlagEmbeddingLocalReranker(
                model_name=model_name,
                use_fp16=use_fp16,
                device=device,
                trust_remote_code=trust_remote_code,
                batch_size=batch_size,
            )

        raise AppError(
            code="provider_not_supported",
            message=f"rerank 不支持的 provider_type: {provider.provider_type}",
            status_code=400,
        )

    def _build_multimodal_model(
        self,
        provider: LLMProvider,
        callsite_config: dict | None = None,
    ) -> ChatProtocol:
        """构建多模态模型。

        支持:
        - GPT-4V (OpenAI)
        - 其他支持图像的 OpenAI 兼容模型
        """
        # 多模态：openai_compatible 与 ollama 均可复用 chat 构建（通过 content blocks 传 image_url）
        if provider.provider_type in {"openai_compatible", "ollama"}:
            return self._build_chat_model(provider, callsite_config)

        raise AppError(
            code="provider_not_supported",
            message=f"multimodal 不支持的 provider_type: {provider.provider_type}",
            status_code=400,
        )

    async def unload_model_if_ollama(self, callsite_key: str) -> bool:
        """如果指定调用点是 Ollama，则卸载其模型。

        Args:
            callsite_key: 调用点 key

        Returns:
            bool: 是否执行了卸载操作
        """
        try:
            provider, callsite = self._resolve_callsite_provider(callsite_key)
            if provider.provider_type != "ollama":
                return False

            callsite_config = self._parse_json(callsite.config_json)
            provider_config = self._parse_json(provider.config_json)
            config = {**provider_config, **(callsite_config or {})}
            
            model_name = (
                config.get("model")
                or config.get("model_name")
                or config.get("ollama_model")
                or provider.name
            )
            base_url = provider.base_url or "http://localhost:11434"

            # 构造卸载请求：keep_alive=0
            import httpx
            
            # 兼容 base_url 可能带 /v1 或不带的情况
            # Ollama API 通常是 POST /api/generate
            api_url = base_url.rstrip("/")
            if api_url.endswith("/v1"):
                api_url = api_url[:-3]
            
            url = f"{api_url}/api/generate"
            payload = {"model": model_name, "keep_alive": 0}

            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=10.0)
                # 404 可能意味着模型未加载或 API 路径不对，忽略
                if resp.status_code not in (200, 404):
                    logger.warning(f"卸载 Ollama 模型失败: {model_name} {resp.status_code} {resp.text}")
                    return False
            
            logger.info(f"已卸载 Ollama 模型: {model_name}")
            return True

        except Exception as e:
            logger.warning(f"尝试卸载 Ollama 模型时出错: {e}")
            return False

    def _build_prompt(self, prompt: Prompt) -> ChatPromptTemplate:
        """构建 Prompt 模板。"""
        if prompt.format == "text":
            content = prompt.content or ""
            return ChatPromptTemplate.from_messages([("human", content)])

        messages = self._parse_messages(prompt.messages_json)
        if not messages:
            raise AppError(
                code="prompt_invalid",
                message=f"提示词 messages 为空: {prompt.scope}",
                status_code=400,
            )
        mapped = []
        for msg in messages:
            role = self.ROLE_MAP.get(msg.get("role"))
            if role is None:
                raise AppError(
                    code="prompt_invalid",
                    message=f"提示词角色非法: {msg.get('role')}",
                    status_code=400,
                )
            mapped.append((role, msg.get("content", "")))
        return ChatPromptTemplate.from_messages(mapped)

    # ============== 辅助方法 ==============

    def _resolve_callsite_provider(
        self, callsite_key: str
    ) -> tuple[LLMProvider, LLMCallSite]:
        """解析 callsite -> provider。"""
        callsite = self.callsite_repository.get_by_key(callsite_key)
        if callsite is None:
            raise AppError(
                code="llm_callsite_not_found",
                message=f"未注册的调用点: {callsite_key}",
                status_code=404,
                details={"callsite_key": callsite_key},
            )
        if not callsite.enabled:
            raise AppError(
                code="llm_callsite_disabled",
                message=f"调用点已禁用: {callsite_key}",
                status_code=400,
                details={"callsite_key": callsite_key},
            )
        if not callsite.provider_id:
            raise AppError(
                code="llm_callsite_unbound",
                message=f"调用点未绑定 Provider: {callsite_key}",
                status_code=409,
                details={"callsite_key": callsite_key},
            )

        provider = self.provider_repository.get_by_id(callsite.provider_id)
        if provider is None:
            raise AppError(
                code="llm_callsite_provider_not_found",
                message="调用点绑定的 Provider 不存在",
                status_code=404,
                details={"callsite_key": callsite_key, "provider_id": callsite.provider_id},
            )
        if not provider.enabled:
            raise AppError(
                code="llm_callsite_provider_disabled",
                message="调用点绑定的 Provider 已禁用",
                status_code=400,
                details={"callsite_key": callsite_key, "provider_id": provider.id},
            )

        return provider, callsite

    def _resolve_provider_from_capability(
        self, capability: str
    ) -> tuple[LLMProvider, LLMCapability]:
        """解析 capability -> provider 映射。"""
        mappings = self.capability_repository.list(
            capability=capability,
            enabled=True,
            limit=100,
            offset=0,
        )
        for mapping in mappings:
            provider = self.provider_repository.get_by_id(mapping.provider_id)
            if provider is None or not provider.enabled:
                continue
            return provider, mapping

        raise AppError(
            code="llm_capability_not_found",
            message=f"未找到可用能力映射: {capability}",
            status_code=404,
        )

    def _get_active_prompt(self, scope: str) -> Prompt:
        """获取激活的提示词。"""
        prompt = self.prompt_repository.get_active_prompt(scope)
        if prompt is None:
            raise AppError(
                code="prompt_not_found",
                message=f"未找到激活提示词: {scope}",
                status_code=404,
            )
        return prompt

    def _resolve_api_key(self, provider: LLMProvider) -> str | None:
        """解析 API Key。"""
        # 优先使用 DB 中的明文（若配置）
        if provider.api_key:
            return provider.api_key
        if not provider.api_key_env:
            return None
        api_key = os.getenv(provider.api_key_env)
        if not api_key:
            raise AppError(
                code="llm_api_key_missing",
                message=f"缺少环境变量: {provider.api_key_env}",
                status_code=400,
            )
        return api_key

    def _parse_json(self, raw: str | None) -> dict:
        """解析 JSON 配置。"""
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AppError(
                code="llm_config_invalid",
                message="配置 JSON 解析失败",
                status_code=400,
                details={"error": str(exc)},
            ) from exc

    def _parse_messages(self, raw: str | None) -> list[dict]:
        """解析 messages JSON。"""
        if not raw:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            return []
