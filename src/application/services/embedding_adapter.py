"""将注入的 embedding 模型适配为 LlamaIndex BaseEmbedding。"""

from __future__ import annotations

from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from src.shared.errors import AppError


class InjectedEmbeddingAdapter(BaseEmbedding):
    embed_dim: int = Field(..., ge=1)
    model_name: str = Field(default="injected")
    _model: Any = PrivateAttr()

    def __init__(self, *, model: Any, embed_dim: int, model_name: str | None = None, **kwargs: Any):
        super().__init__(embed_dim=embed_dim, model_name=model_name or "injected", **kwargs)
        self._model = model

    @classmethod
    def class_name(cls) -> str:
        return "InjectedEmbeddingAdapter"

    def _get_query_embedding(self, query: str) -> list[float]:
        m = self._model
        if hasattr(m, "embed_query"):
            return list(m.embed_query(query))
        return list(m.get_text_embedding(query))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        m = self._model
        if hasattr(m, "embed_query"):
            return list(m.embed_query(text))
        return list(m.get_text_embedding(text))

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        m = self._model
        if hasattr(m, "embed_documents"):
            return [list(x) for x in m.embed_documents(texts)]
        return [self._get_text_embedding(t) for t in texts]


def to_llamaindex_embedding(model: Any) -> BaseEmbedding:
    if isinstance(model, BaseEmbedding):
        return model

    try:
        test = None
        if hasattr(model, "get_text_embedding"):
            test = model.get_text_embedding("test")
        elif hasattr(model, "embed_query"):
            test = model.embed_query("test")
        if not test:
            raise RuntimeError("embedding_model returned empty embedding for probe")
        dim = len(list(test))
    except Exception as e:
        raise AppError(
            code="embed_dimension_infer_failed",
            message=f"无法推断 embedding 维度: {e}",
            status_code=500,
        ) from e

    return InjectedEmbeddingAdapter(model=model, embed_dim=dim)
