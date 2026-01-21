"""向量存储服务（T034）。

使用 ChromaDB 实现向量存储，支持：
- 批量向量写入
- 元数据索引
- 集合管理
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.application.schemas.ingest import (
    KBChunk,
    VectorCollectionInfo,
)
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

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_kwargs: dict[str, Any] | None = None,
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
        self._embed_model = None

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
        """获取嵌入模型。"""
        if self._embed_model is None:
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                settings = get_settings()
                self._embed_model = HuggingFaceEmbedding(
                    model_name="BAAI/bge-large-zh-v1.5",
                )
            except ImportError as e:
                raise VectorStorageError(
                    message=f"无法导入嵌入模型: {e}",
                    code="embed_model_import_error",
                )
        return self._embed_model

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
        except chromadb.NotFoundError:
            # 创建集合
            if embedding_model is None:
                embedding_model = self._get_embed_model()

            # 获取嵌入维度
            test_embedding = embedding_model.get_text_embedding("test")
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
        # 在执行器中运行以避免阻塞
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: model.get_text_embedding(text),
        )

    async def upsert_vectors(
        self,
        chunks: list[KBChunk],
        collection_name: str,
        skip_embedding: bool = False,
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
            # 确保集合存在
            collection = self._ensure_collection(collection_name)

            # 准备数据
            ids = []
            documents = []
            metadatas = []
            embeddings = []

            for chunk in chunks:
                chunk_id = chunk.chunk_id
                content = chunk.content

                # 使用现有嵌入或生成新嵌入
                if chunk.embedding is not None and not skip_embedding:
                    embedding = chunk.embedding
                else:
                    # 检查 chunk 是否已有嵌入
                    if chunk.embedding is not None:
                        embedding = chunk.embedding
                    else:
                        # 生成嵌入
                        embedding = await self._generate_embedding(content)

                ids.append(chunk_id)
                documents.append(content)
                metadatas.append(
                    {
                        **chunk.metadata,
                        "source_file_id": chunk.source_file_id,
                        "chunk_type": chunk.chunk_type.value,
                        "chunk_index": chunk.chunk_index,
                        "created_at": datetime.now().isoformat(),
                    }
                )
                embeddings.append(embedding)

            # 批量写入
            total_upserted = 0
            for i in range(0, len(ids), self.BATCH_SIZE):
                batch_ids = ids[i : i + self.BATCH_SIZE]
                batch_docs = documents[i : i + self.BATCH_SIZE]
                batch_metas = metadatas[i : i + self.BATCH_SIZE]
                batch_embeds = embeddings[i : i + self.BATCH_SIZE]

                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=batch_embeds,
                )
                total_upserted += len(batch_ids)

            return total_upserted

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
            client = self._get_client()
            collection = client.get_collection(name=collection_name)

            # 生成查询嵌入
            query_embedding = await self._generate_embedding(query)

            # 构建过滤条件
            where_clause = None
            if filter_metadata:
                where_clause = self._build_where_clause(filter_metadata)

            # 搜索
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"] if include_scores else ["documents", "metadatas"],
            )

            # 格式化结果
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for idx, chunk_id in enumerate(results["ids"][0]):
                    result = {
                        "id": chunk_id,
                        "document": results["documents"][0][idx] if results["documents"] else "",
                        "metadata": results["metadatas"][0][idx] if results["metadatas"] else {},
                    }

                    if include_scores and results.get("distances"):
                        # 将距离转换为相似度分数
                        distance = results["distances"][0][idx]
                        score = 1.0 / (1.0 + distance)
                        result["score"] = score

                    formatted_results.append(result)

            return formatted_results

        except chromadb.NotFoundError:
            raise VectorStorageError(
                message=f"集合不存在: {collection_name}",
                code="collection_not_found",
                status_code=404,
            )
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

        except chromadb.NotFoundError:
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
            return True

        except chromadb.NotFoundError:
            return False
        except Exception as e:
            raise VectorStorageError(
                message=f"删除集合失败: {e}",
                code="delete_collection_failed",
            ) from e

    async def delete_by_file_id(
        self,
        collection_name: str,
        source_file_id: int,
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

        except chromadb.NotFoundError:
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

        except chromadb.NotFoundError:
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
