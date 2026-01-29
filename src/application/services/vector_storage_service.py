"""向量存储服务（T034）。

使用 ChromaDB 实现向量存储，支持：
- 批量向量写入
- 元数据索引
- 集合管理
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.application.schemas.ingest import (
    KBChunk,
    VectorCollectionInfo,
)
from src.application.services.embedding_adapter import to_llamaindex_embedding
from src.shared.config import get_settings
from src.shared.errors import AppError


class VectorStorageError(AppError):
    """向量存储错误。"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "vector_storage_error",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
        )


class VectorStorageService:
    """使用 ChromaDB 的向量存储服务。"""

    # 批量操作大小
    BATCH_SIZE = 100
    _LI_PERSIST_SUBDIR = "llamaindex"

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_kwargs: dict[str, Any] | None = None,
        embedding_model: Any | None = None,
    ):
        """初始化向量存储服务。

        Args:
            persist_directory: ChromaDB 持久化目录
            collection_kwargs: 额外集合参数
        """
        settings = get_settings()
        self.persist_directory = Path(
            persist_directory or settings.storage_root / "chroma"
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._client: chromadb.PersistentClient | None = None
        self._collection_kwargs = collection_kwargs or {}
        self._embedding_model = embedding_model
        self._li_index_cache: dict[str, VectorStoreIndex] = {}

    def _get_client(self) -> chromadb.PersistentClient:
        """获取 ChromaDB 客户端。"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        return self._client

    def _get_embed_model(self):
        """获取嵌入模型（优先使用注入的 T023 embedding callsite 模型）。"""
        if self._embedding_model is not None:
            return self._embedding_model

        # 兼容旧行为：未注入 embedding 时回退到 HuggingFaceEmbedding
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            self._embedding_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-large-zh-v1.5",
            )
            return self._embedding_model
        except ImportError as e:
            raise VectorStorageError(
                message=f"无法导入嵌入模型: {e}",
                code="embed_model_import_error",
            ) from e

    def _get_li_embed_model(self) -> BaseEmbedding:
        model = self._get_embed_model()
        try:
            return to_llamaindex_embedding(model)
        except AppError as e:
            raise VectorStorageError(
                message=str(getattr(e, "message", None) or str(e)),
                code="embed_dimension_infer_failed",
            ) from e

    def _li_persist_dir(self, collection_name: str) -> Path:
        settings = get_settings()
        root = Path(settings.storage_root) / self._LI_PERSIST_SUBDIR
        return root / (collection_name or "default")

    def _to_li_filters(self, filter_metadata: dict[str, Any] | None) -> MetadataFilters | None:
        if not filter_metadata:
            return None
        filters: list[MetadataFilter] = []
        for key, value in filter_metadata.items():
            if isinstance(value, dict):
                if "$eq" in value:
                    filters.append(
                        MetadataFilter(key=key, operator=FilterOperator.EQ, value=value["$eq"])
                    )
                elif "$ne" in value:
                    filters.append(
                        MetadataFilter(key=key, operator=FilterOperator.NE, value=value["$ne"])
                    )
                elif "$gt" in value:
                    filters.append(
                        MetadataFilter(key=key, operator=FilterOperator.GT, value=value["$gt"])
                    )
                elif "$gte" in value:
                    filters.append(
                        MetadataFilter(key=key, operator=FilterOperator.GTE, value=value["$gte"])
                    )
                elif "$lt" in value:
                    filters.append(
                        MetadataFilter(key=key, operator=FilterOperator.LT, value=value["$lt"])
                    )
                elif "$lte" in value:
                    filters.append(
                        MetadataFilter(key=key, operator=FilterOperator.LTE, value=value["$lte"])
                    )
                else:
                    filters.append(
                        MetadataFilter(key=key, operator=FilterOperator.EQ, value=value)
                    )
            else:
                filters.append(MetadataFilter(key=key, operator=FilterOperator.EQ, value=value))
        if not filters:
            return None
        return MetadataFilters(filters=filters)

    def _flatten_metadata_for_li(self, meta: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in (meta or {}).items():
            if v is None or isinstance(v, (str, int, float)):
                out[str(k)] = v
                continue
            try:
                out[str(k)] = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                out[str(k)] = str(v)
        return out

    def _load_or_create_li_index(self, collection_name: str) -> VectorStoreIndex:
        key = str(collection_name or "default")
        cached = self._li_index_cache.get(key)
        if cached is not None:
            return cached
        embed_model = self._get_li_embed_model()
        collection = self._ensure_collection(key, embedding_model=embed_model)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        persist_dir = self._li_persist_dir(key)
        if persist_dir.exists():
            storage_context = StorageContext.from_defaults(
                persist_dir=str(persist_dir),
                vector_store=vector_store,
            )
            try:
                index = load_index_from_storage(storage_context, embed_model=embed_model)
            except Exception:
                index = VectorStoreIndex(
                    [],
                    storage_context=storage_context,
                    embed_model=embed_model,
                    store_nodes_override=True,
                )
        else:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                [],
                storage_context=storage_context,
                embed_model=embed_model,
                store_nodes_override=True,
            )
        self._li_index_cache[key] = index
        return index

    def get_llamaindex_index(self, collection_name: str) -> VectorStoreIndex:
        return self._load_or_create_li_index(collection_name)

    def to_llamaindex_filters(self, filter_metadata: dict[str, Any] | None) -> MetadataFilters | None:
        return self._to_li_filters(filter_metadata)

    def _ensure_collection(
        self,
        collection_name: str,
        embedding_model=None,
    ) -> chromadb.Collection:
        """确保集合存在。

        Args:
            collection_name: 集合名称
            embedding_model: 嵌入模型

        Returns:
            集合
        """
        client = self._get_client()

        try:
            collection = client.get_collection(name=collection_name)
        except NotFoundError:
            # 创建集合
            if embedding_model is None:
                embedding_model = self._get_embed_model()

            # 获取嵌入维度（兼容 llama_index / LangChain embedding 协议）
            try:
                test_embedding = embedding_model.get_text_embedding("test")
            except Exception:
                test_embedding = embedding_model.embed_query("test")
            dimension = len(test_embedding)

            collection = client.create_collection(
                name=collection_name,
                metadata={
                    "dimension": dimension,
                    "created_at": datetime.now().isoformat(),
                    **self._collection_kwargs,
                },
            )

        return collection

    async def _generate_embedding(self, text: str) -> list[float]:
        """生成文本的嵌入向量。

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        model = self._get_embed_model()
        loop = asyncio.get_running_loop()

        def _do() -> list[float]:
            # 优先 LangChain EmbeddingProtocol
            if hasattr(model, "embed_query"):
                return list(model.embed_query(text))
            return list(model.get_text_embedding(text))

        return await loop.run_in_executor(None, _do)

    async def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量生成嵌入向量（优先走 embed_documents）。"""
        if not texts:
            return []
        model = self._get_embed_model()
        loop = asyncio.get_running_loop()

        def _do() -> list[list[float]]:
            if hasattr(model, "embed_documents"):
                return [list(x) for x in model.embed_documents(texts)]
            # 兼容 llama_index embedding（没有批量 API）
            return [list(model.get_text_embedding(t)) for t in texts]

        return await loop.run_in_executor(None, _do)

    async def upsert_vectors(
        self,
        chunks: list[KBChunk],
        collection_name: str,
        skip_embedding: bool = False,
        docstore_only_nodes: list[TextNode] | None = None,
    ) -> int:
        """批量写入向量。

        Args:
            chunks: chunks 列表
            collection_name: 集合名称
            skip_embedding: 如果已有嵌入则跳过生成

        Returns:
            写入的向量数量

        Raises:
            VectorStorageError: 写入失败
        """
        if not chunks:
            return 0

        try:
            now = datetime.now().isoformat()
            index = self._load_or_create_li_index(collection_name)
            loop = asyncio.get_running_loop()

            if docstore_only_nodes:
                def _do_docstore() -> None:
                    index.docstore.add_documents(docstore_only_nodes)

                await loop.run_in_executor(None, _do_docstore)
                index.storage_context.persist(persist_dir=str(self._li_persist_dir(collection_name)))

            for i in range(0, len(chunks), self.BATCH_SIZE):
                batch = chunks[i : i + self.BATCH_SIZE]
                nodes: list[TextNode] = []
                for c in batch:
                    meta = {
                        **(c.metadata or {}),
                        "source_file_id": c.source_file_id,
                        "chunk_type": c.chunk_type.value,
                        "chunk_index": c.chunk_index,
                        "created_at": now,
                    }
                    meta = self._flatten_metadata_for_li(meta)
                    relationships: dict | None = {}
                    parent_id = meta.get("parent_node_id") or meta.get("parent_id")
                    if isinstance(parent_id, str) and parent_id.strip():
                        relationships = {NodeRelationship.PARENT: RelatedNodeInfo(node_id=parent_id.strip())}
                    node = TextNode(
                        id_=c.chunk_id,
                        text=c.content,
                        metadata=meta,
                        relationships=relationships,
                        embedding=(list(c.embedding) if (skip_embedding and c.embedding is not None) else None),
                    )
                    nodes.append(node)

                def _do() -> None:
                    index.insert_nodes(nodes)

                await loop.run_in_executor(None, _do)
                index.storage_context.persist(persist_dir=str(self._li_persist_dir(collection_name)))

            return len(chunks)

        except Exception as e:
            raise VectorStorageError(
                message=f"写入向量失败: {e}",
                code="upsert_failed",
                details={
                    "collection_name": collection_name,
                    "chunk_count": len(chunks),
                },
            ) from e

    async def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        include_scores: bool = True,
    ) -> list[dict[str, Any]]:
        """向量检索。

        Args:
            query: 搜索查询
            collection_name: 集合名称
            top_k: 返回结果数量
            filter_metadata: 元数据过滤
            include_scores: 包含相似度分数

        Returns:
            搜索结果
        """
        try:
            index = self._load_or_create_li_index(collection_name)
            li_filters = self._to_li_filters(filter_metadata)
            retriever = index.as_retriever(similarity_top_k=top_k, filters=li_filters)
            loop = asyncio.get_running_loop()

            def _do():
                return retriever.retrieve(query)

            nodes = await loop.run_in_executor(None, _do)
            out: list[dict[str, Any]] = []
            for nws in nodes:
                n = nws.node
                item = {
                    "id": n.node_id,
                    "document": n.get_content(),
                    "metadata": dict(n.metadata or {}),
                }
                if include_scores:
                    item["score"] = float(nws.score or 0.0)
                out.append(item)
            return out

        except NotFoundError:
            # 生成阶段（如白皮书生成）允许“无知识库”场景：此处降级为空结果，
            # 由上层逻辑决定是否让 LLM 纯补全。
            return []
        except Exception as e:
            raise VectorStorageError(
                message=f"搜索失败: {e}",
                code="search_failed",
                details={"collection_name": collection_name},
            ) from e

    def _build_where_clause(
        self,
        filter_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """从过滤元数据构建 ChromaDB where 子句。

        Args:
            filter_metadata: 过滤条件

        Returns:
            ChromaDB where 子句
        """
        where_clause = {}

        for key, value in filter_metadata.items():
            if isinstance(value, dict):
                # 处理操作符如 $gt, $lt, $eq 等
                operator_map = {
                    "$gt": "$gt",
                    "$gte": "$gte",
                    "$lt": "$lt",
                    "$lte": "$lte",
                    "$eq": "$eq",
                    "$ne": "$ne",
                }
                operators = {k: v for k, v in value.items() if k in operator_map}
                if operators:
                    where_clause[key] = operators
                else:
                    where_clause[key] = value
            else:
                where_clause[key] = value

        return where_clause

    async def get_collection_info(
        self,
        collection_name: str,
    ) -> VectorCollectionInfo | None:
        """获取集合信息。

        Args:
            collection_name: 集合名称

        Returns:
            集合信息或 None
        """
        try:
            client = self._get_client()
            collection = client.get_collection(name=collection_name)

            count = collection.count()
            metadata = collection.metadata or {}

            return VectorCollectionInfo(
                name=collection_name,
                chunk_count=count,
                embedding_model=metadata.get("embedding_model", ""),
                dimension=metadata.get("dimension", 0),
                created_at=metadata.get("created_at"),
                updated_at=metadata.get("updated_at"),
            )

        except NotFoundError:
            return None

    async def delete_collection(
        self,
        collection_name: str,
    ) -> bool:
        """删除集合。

        Args:
            collection_name: 集合名称

        Returns:
            是否删除成功
        """
        try:
            client = self._get_client()
            client.delete_collection(name=collection_name)
            self._li_index_cache.pop(str(collection_name or "default"), None)
            persist_dir = self._li_persist_dir(collection_name)
            if persist_dir.exists():
                import shutil

                shutil.rmtree(persist_dir, ignore_errors=True)
            return True

        except NotFoundError:
            return False
        except Exception as e:
            raise VectorStorageError(
                message=f"删除集合失败: {e}",
                code="delete_collection_failed",
            ) from e

    async def delete_by_file_id(
        self,
        collection_name: str,
        source_file_id: str,
    ) -> int:
        """按源文件 ID 删除向量。

        Args:
            collection_name: 集合名称
            source_file_id: 源文件 ID

        Returns:
            删除的向量数量
        """
        try:
            client = self._get_client()
            collection = client.get_collection(name=collection_name)

            # 查找并删除
            results = collection.get(
                where={"source_file_id": source_file_id},
                include=["ids"],
            )

            if results["ids"]:
                collection.delete(ids=results["ids"])
                return len(results["ids"])

            return 0

        except NotFoundError:
            return 0
        except Exception as e:
            raise VectorStorageError(
                message=f"按文件 ID 删除失败: {e}",
                code="delete_failed",
            ) from e

    async def list_collections(self) -> list[str]:
        """列出所有集合。

        Returns:
            集合名称列表
        """
        client = self._get_client()
        collections = client.list_collections()
        return [c.name for c in collections]

    async def count_vectors(
        self,
        collection_name: str,
    ) -> int:
        """统计集合中的向量数量。

        Args:
            collection_name: 集合名称

        Returns:
            向量数量
        """
        try:
            client = self._get_client()
            collection = client.get_collection(name=collection_name)
            return collection.count()

        except NotFoundError:
            return 0

    async def reset(self) -> bool:
        """重置所有集合。

        Returns:
            是否重置成功
        """
        try:
            client = self._get_client()
            client.reset()
            return True

        except Exception as e:
            raise VectorStorageError(
                message=f"重置失败: {e}",
                code="reset_failed",
            ) from e

    def close(self) -> None:
        """关闭客户端连接。"""
        if self._client is not None:
            self._client = None
