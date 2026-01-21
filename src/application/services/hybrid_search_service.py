"""混合检索服务（T035）。

结合 BM25 和向量检索，使用 RRF（Reciprocal Rank Fusion）融合结果。
支持可选的重排序。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.application.schemas.ingest import (
    HybridSearchOptions,
    HybridSearchResponse,
    KBChunk,
    SearchMetrics,
    SearchResult,
    SearchStrategy,
)
from src.application.services.vector_storage_service import VectorStorageService
from src.shared.config import get_settings
from src.shared.errors import AppError


class HybridSearchError(AppError):
    """混合检索错误。"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        code: str = "hybrid_search_error",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code=code,
            details=details,
        )


class HybridSearchService:
    """混合检索服务，结合 BM25 和向量检索。"""

    # RRF 常数
    RRF_K = 60

    def __init__(
        self,
        vector_service: VectorStorageService | None = None,
        collection_name: str = "default",
    ):
        """初始化混合检索服务。

        Args:
            vector_service: 向量存储服务
            collection_name: 默认集合名称
        """
        self.vector_service = vector_service or VectorStorageService()
        self.default_collection = collection_name
        self._reranker = None
        self._bm25_index = None

    def _get_reranker(self):
        """获取重排序模型。"""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
            except ImportError as e:
                raise HybridSearchError(
                    message=f"无法导入重排序模型: {e}",
                    code="reranker_import_error",
                )
        return self._reranker

    async def _rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 5,
    ) -> list[SearchResult]:
        """对搜索结果进行重排序。

        Args:
            query: 搜索查询
            results: 搜索结果列表
            top_n: 返回结果数量

        Returns:
            重排序后的结果
        """
        if not results:
            return []

        try:
            reranker = self._get_reranker()

            # 为 cross-encoder 准备 pairs
            pairs = [(query, result.content) for result in results]

            # 获取分数
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: reranker.predict(pairs),
            )

            # 添加重排序分数
            for idx, result in enumerate(results):
                result.metadata["rerank_score"] = float(scores[idx])
                # 合并原始分数和重排序分数
                result.metadata["combined_score"] = (
                    result.score * 0.3 + float(scores[idx]) * 0.7
                )

            # 按综合分数排序
            results.sort(
                key=lambda x: x.metadata.get("combined_score", x.score),
                reverse=True,
            )

            return results[:top_n]

        except Exception as e:
            # 如果重排序失败，返回原始结果
            return results[:top_n]

    def _rrf_fusion(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> list[SearchResult]:
        """使用 RRF（Reciprocal Rank Fusion）融合结果。

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25 检索结果
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重

        Returns:
            融合后的结果
        """
        # 创建排名映射
        vector_rank = {r.chunk_id: rank for rank, r in enumerate(vector_results)}
        bm25_rank = {r.chunk_id: rank for rank, r in enumerate(bm25_results)}

        # 合并结果
        all_results = {}
        for result in vector_results + bm25_results:
            if result.chunk_id not in all_results:
                all_results[result.chunk_id] = result

        # 计算 RRF 分数
        for chunk_id, result in all_results.items():
            vector_pos = vector_rank.get(chunk_id, float("inf"))
            bm25_pos = bm25_rank.get(chunk_id, float("inf"))

            # RRF 分数
            vector_rrf = 1.0 / (self.RRF_K + vector_pos) if vector_pos != float("inf") else 0
            bm25_rrf = 1.0 / (self.RRF_K + bm25_pos) if bm25_pos != float("inf") else 0

            # 加权融合
            result.score = vector_weight * vector_rrf + bm25_weight * bm25_rrf
            result.metadata["vector_rank"] = vector_pos if vector_pos != float("inf") else None
            result.metadata["bm25_rank"] = bm25_pos if bm25_pos != float("inf") else None
            result.metadata["vector_rrf"] = vector_rrf
            result.metadata["bm25_rrf"] = bm25_rrf

        # 按融合分数排序
        fused_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return fused_results

    async def _search_vector(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """执行向量检索。

        Args:
            query: 搜索查询
            collection_name: 集合名称
            top_k: 返回结果数量
            filter_metadata: 元数据过滤

        Returns:
            向量检索结果
        """
        raw_results = await self.vector_service.search(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        results = []
        for idx, raw in enumerate(raw_results):
            results.append(
                SearchResult(
                    chunk_id=raw["id"],
                    content=raw["document"],
                    score=raw.get("score", 0.0),
                    search_type=SearchStrategy.VECTOR,
                    source_file_id=raw["metadata"].get("source_file_id", 0),
                    metadata=raw["metadata"],
                    rank=idx,
                )
            )

        return results

    async def _search_bm25(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """执行 BM25 检索。

        Args:
            query: 搜索查询
            collection_name: 集合名称
            top_k: 返回结果数量
            filter_metadata: 元数据过滤

        Returns:
            BM25 检索结果
        """
        try:
            from llama_index.core import Settings
            from llama_index.core.schema import Document
            from llama_index.retrievers.bm25 import BM25Retriever

            # 获取集合中的文档
            collection_info = await self.vector_service.get_collection_info(collection_name)
            if collection_info is None or collection_info.chunk_count == 0:
                return []

            # 从集合中获取所有文档
            client = self.vector_service._get_client()
            collection = client.get_collection(name=collection_name)

            # 拉取文档（限制在合理批次内）
            max_fetch = min(10000, collection.count())
            results = collection.get(limit=max_fetch, include=["documents", "metadatas", "ids"])

            if not results["ids"]:
                return []

            # 为 BM25 创建文档
            documents = [
                Document(text=doc, metadata=meta)
                for doc, meta in zip(results["documents"], results["metadatas"])
            ]

            # 构建 BM25 检索器
            bm25_retriever = BM25Retriever.from_documents(
                documents,
                similarity_top_k=top_k * 2,  # 为融合多拉取一些
            )

            # 检索
            nodes = bm25_retriever.retrieve(query)

            # 构建结果
            id_to_metadata = dict(zip(results["ids"], results["metadatas"]))
            search_results = []

            for idx, node in enumerate(nodes):
                chunk_id = node.node_id
                metadata = id_to_metadata.get(chunk_id, {})

                # 标准化 BM25 分数
                score = node.get_score()
                # BM25 分数可能为负，标准化到 0-1
                normalized_score = 1.0 / (1.0 + abs(score))

                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        content=node.get_content(),
                        score=normalized_score,
                        search_type=SearchStrategy.BM25,
                        source_file_id=metadata.get("source_file_id", 0),
                        metadata={
                            **metadata,
                            "bm25_score": score,
                        },
                        rank=idx,
                    )
                )

            return search_results

        except ImportError as e:
            # BM25 不可用，返回空结果
            return []
        except Exception as e:
            # 记录错误但不失败
            return []

    async def search(
        self,
        query: str,
        collection_name: str | None = None,
        options: HybridSearchOptions | None = None,
    ) -> HybridSearchResponse:
        """执行混合检索。

        Args:
            query: 搜索查询
            collection_name: 集合名称（未提供则使用默认）
            options: 搜索选项

        Returns:
            混合检索响应
        """
        start_time = time.time()

        options = options or HybridSearchOptions()
        collection = collection_name or self.default_collection

        # 准备过滤条件
        filter_metadata = options.filter_metadata

        # 计算每种检索方法的结果数
        effective_top_k = options.top_k * 3  # 为融合多拉取一些

        # 并行检索
        vector_task = self._search_vector(
            query=query,
            collection_name=collection,
            top_k=effective_top_k,
            filter_metadata=filter_metadata,
        )
        bm25_task = self._search_bm25(
            query=query,
            collection_name=collection,
            top_k=effective_top_k,
            filter_metadata=filter_metadata,
        )

        vector_results, bm25_results = await asyncio.gather(
            vector_task,
            bm25_task,
        )

        # 计算指标
        metrics = SearchMetrics(
            total_results=0,
            vector_results=len(vector_results),
            bm25_results=len(bm25_results),
            rerank_applied=options.use_rerank,
        )

        # 融合结果
        if vector_results or bm25_results:
            fused_results = self._rrf_fusion(
                vector_results,
                bm25_results,
                vector_weight=options.vector_weight,
                bm25_weight=options.bm25_weight,
            )
        else:
            fused_results = []

        # 如果请求重排序，应用重排序
        if options.use_rerank and fused_results:
            rerank_top_n = options.rerank_top_n or options.top_k
            fused_results = await self._rerank_results(
                query,
                fused_results,
                top_n=rerank_top_n,
            )

        # 更新排名并按阈值过滤
        final_results = []
        for rank, result in enumerate(fused_results):
            result.rank = rank

            # 应用分数阈值
            if options.score_threshold is None or result.score >= options.score_threshold:
                final_results.append(result)

            if len(final_results) >= options.top_k:
                break

        metrics.total_results = len(final_results)
        metrics.query_time_ms = (time.time() - start_time) * 1000

        return HybridSearchResponse(
            query=query,
            results=final_results,
            metrics=metrics,
            total_time_ms=(time.time() - start_time) * 1000,
        )

    def build_index(
        self,
        chunks: list[KBChunk],
        collection_name: str | None = None,
        recreate: bool = False,
    ) -> dict[str, Any]:
        """从 chunks 构建混合索引。

        Args:
            chunks: chunks 列表
            collection_name: 集合名称
            recreate: 是否重建集合

        Returns:
            构建结果
        """
        import time

        start_time = time.time()
        collection = collection_name or self.default_collection

        try:
            # 检查集合是否存在
            loop = asyncio.get_event_loop()
            existing_info = loop.run_until_complete(
                self.vector_service.get_collection_info(collection)
            )

            if existing_info and not recreate:
                # 写入到现有集合
                indexed_count = loop.run_until_complete(
                    self.vector_service.upsert_vectors(chunks, collection)
                )
            else:
                # 如果需要则删除并重建
                if existing_info:
                    loop.run_until_complete(
                        self.vector_service.delete_collection(collection)
                    )

                # 创建新索引
                indexed_count = loop.run_until_complete(
                    self.vector_service.upsert_vectors(chunks, collection)
                )

            duration = time.time() - start_time

            # 获取最终集合信息
            final_info = loop.run_until_complete(
                self.vector_service.get_collection_info(collection)
            )

            return {
                "success": True,
                "collection_name": collection,
                "chunks_indexed": len(chunks),
                "vectors_stored": indexed_count,
                "duration_seconds": duration,
                "collection_info": final_info.model_dump() if final_info else None,
            }

        except Exception as e:
            return {
                "success": False,
                "collection_name": collection,
                "error": str(e),
                "duration_seconds": time.time() - start_time,
            }

    async def build_index_async(
        self,
        chunks: list[KBChunk],
        collection_name: str | None = None,
        recreate: bool = False,
        progress_callback=None,
    ) -> dict[str, Any]:
        """异步构建混合索引并跟踪进度。

        Args:
            chunks: chunks 列表
            collection_name: 集合名称
            recreate: 是否重建集合
            progress_callback: 进度回调函数

        Returns:
            构建结果
        """
        import time

        start_time = time.time()
        collection = collection_name or self.default_collection

        try:
            # 检查集合是否存在
            existing_info = await self.vector_service.get_collection_info(collection)

            if existing_info and not recreate:
                indexed_count = await self.vector_service.upsert_vectors(chunks, collection)
            else:
                if existing_info:
                    await self.vector_service.delete_collection(collection)

                indexed_count = await self.vector_service.upsert_vectors(chunks, collection)

            duration = time.time() - start_time

            final_info = await self.vector_service.get_collection_info(collection)

            return {
                "success": True,
                "collection_name": collection,
                "chunks_indexed": len(chunks),
                "vectors_stored": indexed_count,
                "duration_seconds": duration,
                "collection_info": final_info.model_dump() if final_info else None,
            }

        except Exception as e:
            return {
                "success": False,
                "collection_name": collection,
                "error": str(e),
                "duration_seconds": time.time() - start_time,
            }

    async def delete_by_query(
        self,
        query: str,
        collection_name: str | None = None,
    ) -> int:
        """删除匹配查询的 chunks。

        Args:
            query: 搜索查询以查找要删除的 chunks
            collection_name: 集合名称

        Returns:
            删除的 chunks 数量
        """
        collection = collection_name or self.default_collection

        # 搜索匹配的 chunks
        results = await self._search_vector(
            query=query,
            collection_name=collection,
            top_k=1000,
        )

        if not results:
            return 0

        # 获取向量服务并删除
        chunk_ids = [r.chunk_id for r in results]
        client = self.vector_service._get_client()
        collection_obj = client.get_collection(name=collection)
        collection_obj.delete(ids=chunk_ids)

        return len(chunk_ids)

    async def get_stats(self, collection_name: str | None = None) -> dict[str, Any]:
        """获取检索统计信息。

        Args:
            collection_name: 集合名称

        Returns:
            统计信息
        """
        collection = collection_name or self.default_collection

        info = await self.vector_service.get_collection_info(collection)

        return {
            "collection_name": collection,
            "chunk_count": info.chunk_count if info else 0,
            "dimension": info.dimension if info else 0,
            "embedding_model": info.embedding_model if info else "",
            "created_at": info.created_at.isoformat() if info and info.created_at else None,
        }

    def close(self) -> None:
        """关闭服务。"""
        if self.vector_service:
            self.vector_service.close()
