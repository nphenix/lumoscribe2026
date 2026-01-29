"""混合检索服务（T035）。

结合 BM25 和向量检索，使用 RRF（Reciprocal Rank Fusion）融合结果。
支持可选的重排序。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms.mock import MockLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeRelationship, NodeWithScore, QueryBundle, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from src.application.schemas.ingest import (
    HybridSearchOptions,
    HybridSearchResponse,
    KBChunk,
    SearchMetrics,
    SearchResult,
    SearchStrategy,
)
from src.application.services.vector_storage_service import VectorStorageService
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
        reranker: Any | None = None,
        bm25_index_storage_paths: list[str] | None = None,
    ):
        """初始化混合检索服务。

        Args:
            vector_service: 向量存储服务
            collection_name: 默认集合名称
            bm25_index_storage_paths: 多个 BM25 索引路径列表
        """
        self.vector_service = vector_service or VectorStorageService()
        self.default_collection = collection_name
        self._reranker = reranker
        self._bm25_index_storage_paths = bm25_index_storage_paths or []
        self._bm25_indices: list[Any] = []
        self._bm25_index_used = False

    def _get_bm25_indices(self):
        """获取所有预建 BM25 索引（惰性加载并缓存）。"""
        if not self._bm25_index_storage_paths:
            return []
        if self._bm25_indices:
            return self._bm25_indices

        from src.application.services.bm25_index_service import BM25Index

        loaded = []
        for path in self._bm25_index_storage_paths:
            try:
                normalized = path.replace("\\", "/").strip()
                idx_path = Path("data") / normalized
                if idx_path.exists():
                    idx = BM25Index.load(idx_path)
                    loaded.append(idx)
            except Exception:
                continue

        self._bm25_indices = loaded
        return loaded

    def _get_reranker(self):
        """获取重排序模型（应由 T023 LLMRuntimeService 注入）。"""
        if self._reranker is None:
            raise HybridSearchError(
                message="未配置重排序模型（请在中台配置 callsite: hybrid_search:rerank，并确保 provider_id 已绑定且 enabled=true）",
                code="reranker_not_configured",
                status_code=400,
            )
        return self._reranker
 
    def _promote_to_parent(self, *, index, candidates: list[NodeWithScore]) -> list[NodeWithScore]:
        out: list[NodeWithScore] = []
        for nws in candidates:
            if not isinstance(nws, NodeWithScore) or not isinstance(nws.node, TextNode):
                continue
            meta = dict(getattr(nws.node, "metadata", {}) or {})
            if meta.get("chunk_type") == "chart":
                continue
            parent_id = None
            rels = getattr(nws.node, "relationships", None) or {}
            try:
                parent = rels.get(NodeRelationship.PARENT)
            except Exception:
                parent = None
            if parent is not None:
                try:
                    parent_id = getattr(parent, "node_id", None)
                except Exception:
                    parent_id = None
            if isinstance(parent_id, str) and parent_id.strip():
                try:
                    pnode = index.docstore.get_node(parent_id.strip())
                    out.append(NodeWithScore(node=pnode, score=nws.score))
                    continue
                except Exception:
                    pass
            out.append(nws)
        return out
 
    def _aggregate_by_node_id(self, candidates: list[NodeWithScore]) -> list[NodeWithScore]:
        agg: dict[str, dict[str, Any]] = {}
        for nws in candidates:
            node = getattr(nws, "node", None)
            nid = str(getattr(node, "node_id", "") or "").strip()
            if not nid:
                continue
            try:
                score = float(getattr(nws, "score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            cur = agg.get(nid)
            if cur is None:
                agg[nid] = {"node": node, "max": score, "sum": score, "hits": 1}
                continue
            cur["hits"] = int(cur.get("hits", 1)) + 1
            cur["sum"] = float(cur.get("sum", 0.0)) + score
            if score > float(cur.get("max", 0.0)):
                cur["max"] = score
        out: list[NodeWithScore] = []
        for nid, info in agg.items():
            node = info.get("node")
            if node is None:
                continue
            meta = dict(getattr(node, "metadata", {}) or {})
            meta["retrieval_hit_count"] = int(info.get("hits", 1))
            meta["retrieval_score_max"] = float(info.get("max", 0.0))
            meta["retrieval_score_sum"] = float(info.get("sum", 0.0))
            try:
                node.metadata = meta
            except Exception:
                pass
            out.append(NodeWithScore(node=node, score=float(info.get("max", 0.0))))
        return out
 
    def _dedup_by_group(
        self,
        candidates: list[NodeWithScore],
        *,
        max_citation_sources: int = 5,
    ) -> list[NodeWithScore]:
        rep_by_key: dict[str, Any] = {}
        seen_order: list[str] = []
        seen_sources_by_key: dict[str, set[str]] = {}
        citation_by_key: dict[str, list[dict[str, Any]]] = {}

        def _extract_source(meta: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
            src_file_id = str(meta.get("source_file_id", "") or "").strip()
            doc_rel = str(meta.get("doc_rel_path", "") or "").strip()
            doc_title = str(meta.get("doc_title", "") or "").strip()
            original_filename = str(meta.get("original_filename", "") or "").strip()
            key = doc_rel or src_file_id
            if not key:
                return None
            return (
                key,
                {
                    "source_file_id": src_file_id,
                    "doc_rel_path": doc_rel,
                    "doc_title": doc_title,
                    "original_filename": original_filename,
                },
            )

        for nws in candidates:
            node = getattr(nws, "node", None)
            if node is None:
                continue
            meta = dict(getattr(node, "metadata", {}) or {})
            gid = str(meta.get("dedup_group_id") or "").strip()
            nid = str(getattr(node, "node_id", "") or "").strip()
            key = gid or nid
            if not key:
                continue
            rep = rep_by_key.get(key)
            if rep is None:
                rep_by_key[key] = nws
                seen_order.append(key)
                seen_sources_by_key[key] = set()
                citation_by_key[key] = []

            ext = _extract_source(meta)
            if ext is not None:
                src_key, src_obj = ext
                if src_key not in seen_sources_by_key[key]:
                    if len(citation_by_key[key]) < max(0, int(max_citation_sources)):
                        citation_by_key[key].append(src_obj)
                    seen_sources_by_key[key].add(src_key)

        out: list[NodeWithScore] = []
        for key in seen_order:
            nws = rep_by_key.get(key)
            if nws is None:
                continue
            node = getattr(nws, "node", None)
            if node is None:
                continue
            meta = dict(getattr(node, "metadata", {}) or {})
            citations = citation_by_key.get(key) or []
            if citations:
                meta["citation_sources"] = citations
            try:
                node.metadata = meta
            except Exception:
                pass
            out.append(nws)
        return out
 
    def _apply_recall_bonus(self, candidates: list[NodeWithScore]) -> list[NodeWithScore]:
        boosted: list[NodeWithScore] = []
        for nws in candidates:
            node = nws.node
            meta = dict(getattr(node, "metadata", {}) or {})
            bonus = 0.0
            chart_ids = meta.get("chart_ids")
            if isinstance(chart_ids, list) and chart_ids:
                bonus += 0.03
            elif isinstance(chart_ids, str) and chart_ids.strip() and chart_ids.strip() not in {"[]", "{}"}:
                bonus += 0.03
            try:
                txt = node.get_content() or ""
            except Exception:
                txt = ""
            if "images/" in txt or "[图表]" in txt:
                bonus += 0.02
            try:
                hits = int(meta.get("retrieval_hit_count", 1) or 1)
            except Exception:
                hits = 1
            if hits >= 2:
                bonus += min(0.05, 0.01 * float(hits - 1))
            if bonus:
                boosted.append(NodeWithScore(node=node, score=float(nws.score or 0.0) + bonus))
            else:
                boosted.append(nws)
        return boosted
 
    def _parse_chart_ids(self, meta: dict[str, Any]) -> list[str]:
        raw = meta.get("chart_ids")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                raw = parsed
            except Exception:
                raw = [x.strip() for x in raw.split(",") if x.strip()]
        if isinstance(raw, list):
            out: list[str] = []
            for x in raw:
                s = str(x or "").strip()
                if s and s not in out:
                    out.append(s)
            return out
        return []
 
    def _expand_chart_children(
        self,
        *,
        index,
        parents: list[NodeWithScore],
        max_total_charts: int,
        max_charts_per_parent: int,
    ) -> tuple[list[NodeWithScore], int]:
        expanded: list[NodeWithScore] = []
        seen: set[str] = set()
        charts_added = 0
        for nws in parents:
            node = getattr(nws, "node", None)
            nid = str(getattr(node, "node_id", "") or "").strip()
            if nid and nid not in seen:
                seen.add(nid)
                expanded.append(nws)
            meta = dict(getattr(node, "metadata", {}) or {})
            chart_ids = self._parse_chart_ids(meta)
            if not chart_ids:
                continue
            per = 0
            for cid in chart_ids:
                if charts_added >= max(0, int(max_total_charts)):
                    break
                if per >= max(0, int(max_charts_per_parent)):
                    break
                try:
                    cnode = index.docstore.get_node(str(cid))
                except Exception:
                    continue
                cnid = str(getattr(cnode, "node_id", "") or "").strip()
                if not cnid or cnid in seen:
                    continue
                seen.add(cnid)
                expanded.append(NodeWithScore(node=cnode, score=nws.score))
                charts_added += 1
                per += 1
        return expanded, charts_added

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
            docs = [r.content for r in results]
            loop = asyncio.get_running_loop()

            def _do():
                return reranker.rerank(documents=docs, query=query, top_n=top_n)

            raw = await loop.run_in_executor(None, _do)
            if not isinstance(raw, list) or not raw:
                return results[:top_n]

            reranked: list[SearchResult] = []
            used: set[int] = set()
            for item in raw:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                if idx is None:
                    continue
                try:
                    i = int(idx)
                except Exception:
                    continue
                if i < 0 or i >= len(results) or i in used:
                    continue
                used.add(i)
                r = results[i]
                try:
                    r.metadata["rerank_score"] = float(item.get("score", 0.0))
                except Exception:
                    r.metadata["rerank_score"] = 0.0
                reranked.append(r)

            if not reranked:
                return results[:top_n]
            return reranked[:top_n]
        except Exception:
            # 重排序失败：严格降级为返回前 top_n（保留融合排序）
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
                    source_file_id=str(raw["metadata"].get("source_file_id", "")),
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
        # 优先使用"预建 BM25 索引"（支持多个索引）
        indices = self._get_bm25_indices()
        if indices:
            self._bm25_index_used = True
            all_results: list[SearchResult] = []
            # 从所有索引收集结果
            for idx in indices:
                raw = idx.search(query=query, top_k=top_k * 2, filter_metadata=filter_metadata)
                for item in raw:
                    all_results.append(
                        SearchResult(
                            chunk_id=item["chunk_id"],
                            content=item["content"],
                            score=float(item.get("bm25_score_norm", 0.0)),
                            search_type=SearchStrategy.BM25,
                            source_file_id=str(item.get("source_file_id", "")),
                            metadata={
                                **(item.get("metadata") or {}),
                                "bm25_score": float(item.get("bm25_score", 0.0)),
                                "bm25_score_norm": float(item.get("bm25_score_norm", 0.0)),
                                "bm25_index_used": True,
                            },
                            rank=0,
                        )
                    )
            # 根据 chunk_id 去重，保留分数最高的结果
            seen: dict[str, SearchResult] = {}
            for r in all_results:
                if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
                    seen[r.chunk_id] = r
            # 按分数排序并返回 top_k
            unique_results = sorted(seen.values(), key=lambda x: x.score, reverse=True)
            for i, r in enumerate(unique_results):
                r.rank = i
            return unique_results[:top_k]

        # 兼容降级：如果没有预建索引，再走旧逻辑（查询时动态构建）
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
                        source_file_id=str(metadata.get("source_file_id", "")),
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
        self._bm25_index_used = False

        try:
            index = self.vector_service.get_llamaindex_index(collection)
        except Exception:
            metrics = SearchMetrics(
                total_results=0,
                vector_results=0,
                bm25_results=0,
                bm25_index_used=False,
                rerank_applied=False,
            )
            metrics.query_time_ms = (time.time() - start_time) * 1000
            return HybridSearchResponse(
                query=query,
                results=[],
                metrics=metrics,
                total_time_ms=(time.time() - start_time) * 1000,
            )

        li_filters = self.vector_service.to_llamaindex_filters(options.filter_metadata)
        effective_top_k = max(1, int(options.top_k) * 3)

        base_vector_retriever = index.as_retriever(similarity_top_k=effective_top_k, filters=li_filters)
        vector_retriever = AutoMergingRetriever(
            base_vector_retriever,
            index.storage_context,
            simple_ratio_thresh=0.25,
            verbose=False,
        )
        indices = self._get_bm25_indices()
        if indices:
            self._bm25_index_used = True

            class _PrebuiltBM25Retriever(BaseRetriever):
                def __init__(self):
                    super().__init__(callback_manager=None, verbose=False)

                def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
                    q = str(getattr(query_bundle, "query_str", "") or "")
                    if not q.strip():
                        return []

                    all_items: list[dict[str, Any]] = []
                    for idx in indices:
                        try:
                            raw = idx.search(
                                query=q,
                                top_k=effective_top_k * 2,
                                filter_metadata=options.filter_metadata,
                            )
                            if isinstance(raw, list):
                                all_items.extend([x for x in raw if isinstance(x, dict)])
                        except Exception:
                            continue

                    if not all_items:
                        return []

                    best: dict[str, dict[str, Any]] = {}
                    for it in all_items:
                        cid = str(it.get("chunk_id", "") or "").strip()
                        if not cid:
                            continue
                        prev = best.get(cid)
                        try:
                            score = float(it.get("bm25_score_norm", 0.0))
                        except Exception:
                            score = 0.0
                        if prev is None:
                            best[cid] = {**it, "bm25_score_norm": score}
                            continue
                        try:
                            prev_score = float(prev.get("bm25_score_norm", 0.0))
                        except Exception:
                            prev_score = 0.0
                        if score > prev_score:
                            best[cid] = {**it, "bm25_score_norm": score}

                    ranked = sorted(best.values(), key=lambda x: float(x.get("bm25_score_norm", 0.0) or 0.0), reverse=True)
                    out: list[NodeWithScore] = []
                    for it in ranked[:effective_top_k]:
                        cid = str(it.get("chunk_id", "") or "").strip()
                        if not cid:
                            continue
                        try:
                            node = index.docstore.get_node(cid)
                        except Exception:
                            continue
                        try:
                            score = float(it.get("bm25_score_norm", 0.0))
                        except Exception:
                            score = 0.0
                        out.append(NodeWithScore(node=node, score=score))
                    return out

            bm25_retriever = _PrebuiltBM25Retriever()
        else:
            bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=effective_top_k)
            self._bm25_index_used = True

        fusion = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=effective_top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
            llm=MockLLM(),
            verbose=False,
        )

        candidates = await fusion.aretrieve(query)
        if not isinstance(candidates, list):
            candidates = []

        primary_candidates = self._promote_to_parent(index=index, candidates=candidates)
        primary_candidates = self._aggregate_by_node_id(primary_candidates)
        primary_candidates = self._apply_recall_bonus(primary_candidates)
        primary_candidates = sorted(primary_candidates, key=lambda x: float(getattr(x, "score", 0.0) or 0.0), reverse=True)
        primary_candidates = self._dedup_by_group(primary_candidates)

        rerank_applied = False
        parent_top_k = max(1, int(options.top_k))
        primary_nodes: list[NodeWithScore] = primary_candidates[:parent_top_k]

        if options.use_rerank and primary_candidates:
            try:
                reranker = self._get_reranker()
                pool_n = int(options.rerank_top_n or effective_top_k)
                pool_n = max(parent_top_k, min(pool_n, len(primary_candidates)))
                pool = primary_candidates[:pool_n]
                post = _FlagEmbeddingRerankPostprocessor(reranker=reranker, top_n=pool_n)
                loop = asyncio.get_running_loop()

                def _do_rerank():
                    return post.postprocess_nodes(pool, query_bundle=QueryBundle(query))

                reranked_pool = await loop.run_in_executor(None, _do_rerank)
                reranked_pool = reranked_pool if isinstance(reranked_pool, list) else []

                used: set[str] = set()
                merged: list[NodeWithScore] = []
                for nws in reranked_pool:
                    if not isinstance(nws, NodeWithScore):
                        continue
                    nid = str(getattr(nws.node, "node_id", "") or "")
                    if not nid or nid in used:
                        continue
                    used.add(nid)
                    merged.append(nws)
                for nws in pool:
                    nid = str(getattr(nws.node, "node_id", "") or "")
                    if nid and nid not in used:
                        used.add(nid)
                        merged.append(nws)
                for nws in primary_candidates[pool_n:]:
                    nid = str(getattr(nws.node, "node_id", "") or "")
                    if nid and nid not in used:
                        used.add(nid)
                        merged.append(nws)
                    if len(merged) >= parent_top_k:
                        break

                primary_nodes = merged[:parent_top_k]
                rerank_applied = True
            except Exception:
                primary_nodes = primary_candidates[:parent_top_k]

        max_total_charts = max(0, int(getattr(options, "max_total_charts", 0) or 0))
        if max_total_charts <= 0:
            max_total_charts = min(12, max(3, int(options.top_k)))
        max_charts_per_parent = max(0, int(getattr(options, "max_charts_per_parent", 0) or 0))
        if max_charts_per_parent <= 0:
            max_charts_per_parent = 2
        expanded, charts_added = self._expand_chart_children(
            index=index,
            parents=primary_nodes,
            max_total_charts=max_total_charts,
            max_charts_per_parent=max_charts_per_parent,
        )

        results: list[SearchResult] = []
        added: set[str] = set()
        added_groups: set[str] = set()
        primary_seen = 0
        for nws in expanded:
            node = getattr(nws, "node", None)
            if node is None:
                continue
            nid = str(getattr(node, "node_id", "") or "").strip()
            if not nid or nid in added:
                continue
            added.add(nid)
            meta = dict(getattr(node, "metadata", {}) or {})
            gid = str(meta.get("dedup_group_id") or "").strip()
            if gid and gid in added_groups:
                continue
            if gid:
                added_groups.add(gid)
            is_chart = meta.get("chunk_type") == "chart"
            if not is_chart:
                if primary_seen >= int(options.top_k):
                    break
                primary_seen += 1
            score = float(getattr(nws, "score", 0.0) or 0.0)
            if options.score_threshold is not None and not is_chart:
                if score < float(options.score_threshold):
                    continue
            results.append(
                SearchResult(
                    chunk_id=nid,
                    content=node.get_content(),
                    score=score,
                    search_type=SearchStrategy.HYBRID,
                    source_file_id=str(meta.get("source_file_id", "")),
                    metadata=meta,
                    rank=len(results),
                )
            )

        metrics = SearchMetrics(
            total_results=len(results),
            vector_results=len(candidates),
            bm25_results=len(candidates),
            bm25_index_used=self._bm25_index_used,
            rerank_applied=rerank_applied,
        )
        metrics.query_time_ms = (time.time() - start_time) * 1000
        return HybridSearchResponse(
            query=query,
            results=results,
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


class _FlagEmbeddingRerankPostprocessor(BaseNodePostprocessor):
    top_n: int = Field(default=5, ge=1, le=2000)
    reranker: Any = Field(exclude=True)

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return []
        if query_bundle is None:
            return nodes[: self.top_n]
        docs = [n.node.get_content() for n in nodes]
        raw = self.reranker.rerank(documents=docs, query=query_bundle.query_str, top_n=self.top_n)
        if not isinstance(raw, list) or not raw:
            return nodes[: self.top_n]
        out: list[NodeWithScore] = []
        used: set[int] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if idx is None:
                continue
            try:
                i = int(idx)
            except Exception:
                continue
            if i < 0 or i >= len(nodes) or i in used:
                continue
            used.add(i)
            nws = nodes[i]
            try:
                score = float(item.get("score", 0.0))
            except Exception:
                score = 0.0
            meta = dict(getattr(nws.node, "metadata", {}) or {})
            meta["rerank_score"] = score
            nws.node.metadata = meta
            out.append(NodeWithScore(node=nws.node, score=score))
            if len(out) >= int(self.top_n):
                break
        return out or nodes[: self.top_n]
