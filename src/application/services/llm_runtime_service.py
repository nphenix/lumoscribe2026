"""LLM 运行时封装服务。

支持多种模型类型：
- chat: 对话模型 (ChatOpenAI, ChatOllama, LlamaCpp)
- embedding: 向量模型 (FlagEmbedding, OpenAI Embeddings, LlamaCpp)
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
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings

from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
from src.application.repositories.llm_model_repository import LLMModelRepository
from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.application.repositories.prompt_repository import PromptRepository
from src.domain.entities.llm_capability import LLMCapability
from src.domain.entities.llm_call_site import LLMCallSite
from src.domain.entities.llm_model import LLMModel
from src.domain.entities.llm_provider import LLMProvider
from src.domain.entities.prompt import Prompt
from src.shared.errors import AppError


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
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """获取文档列表的向量表示。"""
        if self.host:
            return self._embed_batch_via_api(texts)
        if self._model is None:
            self._load_local_model()
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32
        )
        return embeddings.tolist()

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

    def __init__(
        self,
        provider_repository: LLMProviderRepository,
        model_repository: LLMModelRepository,
        capability_repository: LLMCapabilityRepository,
        callsite_repository: LLMCallSiteRepository,
        prompt_repository: PromptRepository,
    ):
        self.provider_repository = provider_repository
        self.model_repository = model_repository
        self.capability_repository = capability_repository
        self.callsite_repository = callsite_repository
        self.prompt_repository = prompt_repository

    # ============== 统一构建接口 ==============

    def build_runnable_for_callsite(self, callsite_key: str):
        """构建指定调用点的 runnable。

        Args:
            callsite_key: 调用点 key（建议 module:action）

        Returns:
            LangChain Runnable 对象
        """
        provider, model, callsite = self._resolve_callsite_provider_model(callsite_key)
        callsite_config = self._parse_json(callsite.config_json)
        model_kind = model.model_kind

        if model_kind == self.MODEL_KIND_CHAT:
            scope = callsite.prompt_scope or callsite.key
            prompt = self._get_active_prompt(scope)
            llm = self._build_chat_model(provider, model, callsite_config)
            prompt_template = self._build_prompt(prompt)
            return prompt_template | llm | StrOutputParser()

        elif model_kind == self.MODEL_KIND_EMBEDDING:
            return self._build_embedding_model(provider, model, callsite_config)

        elif model_kind == self.MODEL_KIND_RERANK:
            return self._build_rerank_model(provider, model, callsite_config)

        elif model_kind == self.MODEL_KIND_MULTIMODAL:
            scope = callsite.prompt_scope or callsite.key
            prompt = self._get_active_prompt(scope)
            llm = self._build_multimodal_model(provider, model, callsite_config)
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

        provider, model, _mapping = self._resolve_provider_model(capability)
        model_kind = model.model_kind

        if model_kind == self.MODEL_KIND_CHAT:
            prompt = self._get_active_prompt(capability)
            llm = self._build_chat_model(provider, model)
            prompt_template = self._build_prompt(prompt)
            return prompt_template | llm | StrOutputParser()

        elif model_kind == self.MODEL_KIND_EMBEDDING:
            return self._build_embedding_model(provider, model)

        elif model_kind == self.MODEL_KIND_RERANK:
            return self._build_rerank_model(provider, model)

        elif model_kind == self.MODEL_KIND_MULTIMODAL:
            prompt = self._get_active_prompt(capability)
            llm = self._build_multimodal_model(provider, model)
            prompt_template = self._build_prompt(prompt)
            return prompt_template | llm

        else:
            raise AppError(
                code="unsupported_model_kind",
                message=f"不支持的模型类型: {model_kind}",
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

        provider, model, _mapping = self._resolve_provider_model(capability)
        model_kind = model.model_kind

        if model_kind == self.MODEL_KIND_CHAT:
            return self._build_chat_model(provider, model)

        elif model_kind == self.MODEL_KIND_EMBEDDING:
            return self._build_embedding_model(provider, model)

        elif model_kind == self.MODEL_KIND_RERANK:
            return self._build_rerank_model(provider, model)

        elif model_kind == self.MODEL_KIND_MULTIMODAL:
            return self._build_multimodal_model(provider, model)

        else:
            raise AppError(
                code="unsupported_model_kind",
                message=f"不支持的模型类型: {model_kind}",
                status_code=400,
            )

    def get_model_for_callsite(
        self, callsite_key: str
    ) -> Union[ChatProtocol, EmbeddingProtocol, RerankProtocol]:
        """获取指定调用点的模型实例。"""
        provider, model, callsite = self._resolve_callsite_provider_model(callsite_key)
        callsite_config = self._parse_json(callsite.config_json)
        model_kind = model.model_kind

        if model_kind == self.MODEL_KIND_CHAT:
            return self._build_chat_model(provider, model, callsite_config)

        elif model_kind == self.MODEL_KIND_EMBEDDING:
            return self._build_embedding_model(provider, model, callsite_config)

        elif model_kind == self.MODEL_KIND_RERANK:
            return self._build_rerank_model(provider, model, callsite_config)

        elif model_kind == self.MODEL_KIND_MULTIMODAL:
            return self._build_multimodal_model(provider, model, callsite_config)

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
        model: LLMModel,
        callsite_config: dict | None = None,
    ) -> ChatProtocol:
        """构建对话模型。"""
        provider_config = self._parse_json(provider.config_json)
        model_config = self._parse_json(model.config_json)
        config = {**provider_config, **model_config, **(callsite_config or {})}

        extra_body = config.pop("extra_body", None)
        model_kwargs = config.pop("model_kwargs", None)
        temperature = config.pop("temperature", None)
        max_tokens = config.pop("max_tokens", None)

        if config:
            model_kwargs = {**(model_kwargs or {}), **config}

        if provider.provider_type == "openai_compatible":
            api_key = self._resolve_api_key(provider)
            params: dict[str, Any] = {
                "model": model.name,
                "api_key": api_key,
            }
            if provider.base_url:
                params["base_url"] = provider.base_url
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if model_kwargs:
                params["model_kwargs"] = model_kwargs
            if extra_body:
                params["extra_body"] = extra_body
            return ChatOpenAI(**params)

        if provider.provider_type == "ollama":
            params = {"model": model.name}
            if provider.base_url:
                params["base_url"] = provider.base_url
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["num_predict"] = max_tokens
            if model_kwargs:
                params.update(model_kwargs)
            return ChatOllama(**params)

        # LlamaCpp (本地 GGUF 模型，支持 GPU)
        if provider.provider_type == "llamacpp":
            model_path = model.name  # model 字段存储模型文件路径
            params: dict[str, Any] = {
                "model_path": model_path,
                "n_gpu_layers": config.pop("n_gpu_layers", -1),  # -1 表示全部层到 GPU
                "n_batch": config.pop("n_batch", 512),
            }
            # 从 config 中读取额外参数
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["n_ctx"] = max_tokens
            if model_kwargs:
                params.update(model_kwargs)
            return LlamaCpp(**params)

        raise AppError(
            code="provider_not_supported",
            message=f"不支持的 provider_type: {provider.provider_type}",
            status_code=400,
        )

    def _build_embedding_model(
        self,
        provider: LLMProvider,
        model: LLMModel,
        callsite_config: dict | None = None,
    ) -> EmbeddingProtocol:
        """构建向量模型。

        支持:
        - FlagEmbedding 本地/远程模型
        - OpenAI Embeddings
        - HuggingFace Embeddings
        """
        provider_config = self._parse_json(provider.config_json)
        model_config = self._parse_json(model.config_json)
        config = {**provider_config, **model_config, **(callsite_config or {})}

        model_name = model.name

        # FlagEmbedding
        if provider.provider_type == "flagembedding":
            host = provider.base_url or config.pop("host", None)
            use_fp16 = config.pop("use_fp16", True)
            device = config.pop("device", "cpu")

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

        # LlamaCpp Embeddings (本地 GGUF 模型，支持 GPU)
        if provider.provider_type == "llamacpp":
            model_path = model.name  # model 字段存储模型文件路径
            return LlamaCppEmbeddings(
                model_path=model_path,
                n_gpu_layers=config.pop("n_gpu_layers", -1),  # -1 表示全部层到 GPU
                **config,
            )

        raise AppError(
            code="provider_not_supported",
            message=f"embedding 不支持的 provider_type: {provider.provider_type}",
            status_code=400,
        )

    def _build_rerank_model(
        self,
        provider: LLMProvider,
        model: LLMModel,
        callsite_config: dict | None = None,
    ) -> RerankProtocol:
        """构建重排序模型。

        支持:
        - FlagEmbedding Reranker 远程 API
        - Cohere Rerank
        """
        provider_config = self._parse_json(provider.config_json)
        model_config = self._parse_json(model.config_json)
        config = {**provider_config, **model_config, **(callsite_config or {})}

        model_name = model.name

        # FlagEmbedding Reranker (远程 API)
        if provider.provider_type == "flagembedding":
            host = provider.base_url or config.pop("host", "http://localhost:8000")
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

        # Cohere Rerank
        if provider.provider_type == "cohere":
            api_key = self._resolve_api_key(provider)
            from langchain_cohere import CohereRerank

            return CohereRerank(
                model=model_name,
                api_key=api_key,
                **config,
            )

        raise AppError(
            code="provider_not_supported",
            message=f"rerank 不支持的 provider_type: {provider.provider_type}",
            status_code=400,
        )

    def _build_multimodal_model(
        self,
        provider: LLMProvider,
        model: LLMModel,
        callsite_config: dict | None = None,
    ) -> ChatProtocol:
        """构建多模态模型。

        支持:
        - GPT-4V (OpenAI)
        - 其他支持图像的 OpenAI 兼容模型
        """
        # 多模态模型目前复用 chat 模型接口
        # 通过 config 中的 stop_actions 等参数区分
        return self._build_chat_model(provider, model, callsite_config)

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

    def _resolve_callsite_provider_model(
        self, callsite_key: str
    ) -> tuple[LLMProvider, LLMModel, LLMCallSite]:
        """解析 callsite -> provider/model。"""
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
        if not callsite.model_id:
            raise AppError(
                code="llm_callsite_unbound",
                message=f"调用点未绑定模型: {callsite_key}",
                status_code=409,
                details={"callsite_key": callsite_key},
            )

        model = self.model_repository.get_by_id(callsite.model_id)
        if model is None:
            raise AppError(
                code="llm_callsite_model_not_found",
                message="调用点绑定的模型不存在",
                status_code=404,
                details={"callsite_key": callsite_key, "model_id": callsite.model_id},
            )
        if not model.enabled:
            raise AppError(
                code="llm_callsite_model_disabled",
                message="调用点绑定的模型已禁用",
                status_code=400,
                details={"callsite_key": callsite_key, "model_id": model.id},
            )

        provider = self.provider_repository.get_by_id(model.provider_id)
        if provider is None:
            raise AppError(
                code="llm_callsite_provider_not_found",
                message="调用点绑定的 Provider 不存在",
                status_code=404,
                details={"callsite_key": callsite_key, "provider_id": model.provider_id},
            )
        if not provider.enabled:
            raise AppError(
                code="llm_callsite_provider_disabled",
                message="调用点绑定的 Provider 已禁用",
                status_code=400,
                details={"callsite_key": callsite_key, "provider_id": provider.id},
            )

        if model.model_kind != callsite.expected_model_kind:
            raise AppError(
                code="llm_callsite_kind_mismatch",
                message="调用点期望类型与模型类型不一致",
                status_code=400,
                details={
                    "callsite_key": callsite_key,
                    "expected_model_kind": callsite.expected_model_kind,
                    "model_kind": model.model_kind,
                },
            )

        return provider, model, callsite

    def _resolve_provider_model(
        self, capability: str
    ) -> tuple[LLMProvider, LLMModel, LLMCapability]:
        """解析 provider、model、capability 映射。"""
        mappings = self.capability_repository.list(
            capability=capability,
            enabled=True,
            limit=100,
            offset=0,
        )
        for mapping in mappings:
            model = self.model_repository.get_by_id(mapping.model_id)
            if model is None or not model.enabled:
                continue
            provider = self.provider_repository.get_by_id(model.provider_id)
            if provider is None or not provider.enabled:
                continue
            return provider, model, mapping

        raise AppError(
            code="llm_capability_not_found",
            message=f"未找到可用能力映射: {capability}",
            status_code=404,
        )

    def _get_active_prompt(self, scope: str) -> Prompt:
        """获取激活的提示词。"""
        items = self.prompt_repository.list(scope=scope, active=True, limit=1, offset=0)
        if not items:
            raise AppError(
                code="prompt_not_found",
                message=f"未找到激活提示词: {scope}",
                status_code=404,
            )
        return items[0]

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
