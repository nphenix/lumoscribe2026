## 目标
- 把当前“向量检索（Chroma）+ 融合（RRF）+ rerank（FlagEmbedding）”迁移到 LlamaIndex 的检索链路。
- 落地 Path B：引入 LlamaIndex 的“结构化层级检索/递归检索（Recursive Retrieval）”，实现“命中父节点时自动带出关联图表子节点”。
- 保持现有对外接口（HybridSearchResponse/SearchResult、白皮书生成 assemble_context_for_section 等）尽量不变，内部实现切换为 LlamaIndex。

## 现状基线（需要改造的点）
- Chroma 检索目前由自研 [VectorStorageService](file:///f:/lumoscribe2026/src/application/services/vector_storage_service.py) 直接调用 `chromadb` 完成；不是 LlamaIndex VectorStoreIndex。
- rerank 目前在 [HybridSearchService](file:///f:/lumoscribe2026/src/application/services/hybrid_search_service.py) 里调用注入的 `reranker.rerank(...)`；不是 LlamaIndex NodePostprocessor。
- LlamaIndex 目前主要用于 chunking（SentenceSplitter/SemanticSplitterNodeParser），没有在 query-time 用它的 Retriever/QueryEngine/RecursiveRetriever。

## 设计总览（落地 Path B 的“递归”方式）
- **Chroma 走 LlamaIndex**：使用 `ChromaVectorStore + StorageContext + VectorStoreIndex` 来写入/查询（LlamaIndex 官方集成）
  - 参考：Chroma Vector Store 示例（metadata filter、StorageContext、VectorStoreIndex）https://docs.llamaindex.ai/en/stable/examples/vector_stores/chroma_metadata_filter/
- **融合检索走 LlamaIndex**：使用 `QueryFusionRetriever`（mode=reciprocal_rerank）组合 `vector_retriever + bm25_retriever`，替换当前自研 RRF。
  - 参考：Reciprocal Rerank Fusion Retriever https://developers.llamaindex.ai/python/examples/retrievers/reciprocal_rerank_fusion/
- **rerank 走 LlamaIndex**：实现一个自定义 `BaseNodePostprocessor`，内部复用现有 `RerankProtocol.rerank`（FlagEmbedding/远程 API），将返回列表重排为 NodeWithScore。
  - 参考：LlamaIndex Node Postprocessor 模块 https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/
- **父子图表“递归带出”走 LlamaIndex**：在索引中引入 `IndexNode` 作为“父节点代理”，其 `obj` 指向一个“子检索器”（只检索该父节点关联的图表子节点）；然后用 `RecursiveRetriever` 在 query-time 自动执行子检索器并把子节点结果带回。
  - 参考：RecursiveRetriever 的“IndexNode 绑定下游 query engine/retriever”思路 https://docs.llamaindex.ai/en/stable/examples/query_engine/pdf_tables/recursive_retriever/

## 关键数据结构与链接策略
- **节点类型**
  - Parent 节点：正文 chunk（段落/小节级），TextNode(id_=chunk_id, text=content, metadata=...)
  - Child 节点：图表 chunk（chart_json 语义），TextNode(id_=chart_chunk_id, text=semantic_text, metadata 中含 chart_id/doc_rel_path/source_file_id...)
- **父子链接（建库时生成）**
  - 在 Parent metadata 写入 `chart_ids: list[str]`（或 `child_chart_node_ids`），来源：
    - 解析 chunk 内的 `[Chart: <id>]`、或 `images/<stem>`、或注入的 `[图表] <name>`，并回查 `doc_dir/chart_json/*.json` 得到 `_chart_id`。
  - 子节点 metadata 保留 `chart_id`（已存在）并可补 `doc_rel_path/source_file_id` 便于过滤。
- **递归执行方式**
  - 为每个 Parent 构造一个 `IndexNode(text=<父节点文本>, obj=<ChartChildrenRetriever(chart_ids)>)`
  - 向量索引/融合检索检索的是这些 IndexNode；一旦命中，`RecursiveRetriever` 会执行其 obj，把子图表节点一并取回。

## 具体实施步骤（不含任何提交）
1) **把向量存储写入/查询迁移到 LlamaIndex-Chroma**
   - 在 [VectorStorageService](file:///f:/lumoscribe2026/src/application/services/vector_storage_service.py) 内部改为：
     - 构造 `ChromaVectorStore`（persist_dir 复用现有 `.runtime/storage/chroma` 或引入可配置子目录，避免破坏老集合）。
     - 用 `VectorStoreIndex(nodes, storage_context=...)` 进行写入；由 LlamaIndex embed_model 生成 embedding。
   - 处理 collection 生命周期（create/delete/recreate）映射到 Chroma。

2) **在 KnowledgeBaseService.build 中产出 LlamaIndex Nodes 与父子映射**
   - 在 [knowledge_base_service.py](file:///f:/lumoscribe2026/src/application/services/knowledge_base_service.py) 的建库流程：
     - 继续复用现有 chunking + chart_chunks 产出（KBChunk）。
     - 新增一步：从 KBChunk 构造 LlamaIndex `TextNode` 列表。
     - 基于 doc_dir/chart_json + chunk 内容解析，给 parent chunks 写入 `chart_ids` 列表。
     - 产出 parent IndexNodes（带子检索器引用）+ chart child TextNodes。
     - 写入 LlamaIndex index（向量存储落在 Chroma）。

3) **把 HybridSearchService 替换为 LlamaIndex 检索管线**
   - 在 [hybrid_search_service.py](file:///f:/lumoscribe2026/src/application/services/hybrid_search_service.py)：
     - 构建 `vector_retriever = index.as_retriever(similarity_top_k=...)`
     - 构建 `bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=...)`
     - 用 `QueryFusionRetriever([vector_retriever, bm25_retriever], mode="reciprocal_rerank", ...)` 替代现有 RRF。
     - 用 `RecursiveRetriever` 包一层，使得命中 parent IndexNode 时自动执行子检索器取回图表节点。
     - 将返回的 `NodeWithScore` 转换为现有 `SearchResult`（保留 chunk_id/node_id、metadata、score、rank 等）。

4) **把 rerank 迁移为 LlamaIndex NodePostprocessor**
   - 实现 `FlagEmbeddingRerankPostprocessor(BaseNodePostprocessor)`：
     - 输入 nodes + query_bundle，调用现有 `reranker.rerank(documents, query, top_n)`
     - 输出按 rerank 分数排序的 NodeWithScore 列表（必要时把 rerank_score 写入 metadata）
   - 在 QueryEngine/检索阶段把该 postprocessor 接入（而不是当前在 HybridSearchService 里手工排序）。
   - 参考：SentenceTransformerRerank 的用法（接口形态一致）https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/SentenceTransformerRerank/

5) **验证与回归（以可运行为准）**
   - 新增/调整单测：
     - 构造一个 parent node + chart child node 的最小索引，验证：检索命中 parent 时结果集中包含 child chart node。
     - 验证 rerank postprocessor 确实改变顺序。
   - 端到端：跑现有 `test_knowledge_base.py` 的核心链路（建库→检索）与 `test_content_generation.py` 的“自动图表带出”相关断言（只要测试环境允许）。

## 迁移策略（避免一次性破坏）
- 先引入“LlamaIndex 模式”并在服务内部加开关（默认开/默认关可按你偏好）：
  - 允许同一个 API 继续工作。
  - 旧 collection 不动；新 collection 用前缀/独立目录，验证无误后再切换/清理。

## 风险与应对
- **索引持久化与 docstore**：RecursiveRetriever 需要能访问 node_dict/child nodes；方案采用“IndexNode 携带子检索器 + child nodes 在同一 ChromaIndex 可过滤检索”来避免依赖内存 node_dict。
- **父子映射质量**：若 parent→chart_ids 构建不稳，会导致“带错图/漏图”。会先从已存在的 `[Chart: <id>]` 与 images stem 精确映射开始，弱匹配仅作为补充且加阈值。
- **性能**：RecursiveRetriever 触发子检索会额外发起查询；将对子检索 top_k、去重、同一 chart_id 只取一次做约束。

## 交付物
- 代码改造（上述 3 个核心服务文件为主，必要时极少量新增模块文件承载 LlamaIndex 适配器/后处理器）。
- 单测/最小回归用例。

如果你确认这个方案，我会退出计划模式并开始按上述步骤逐项改造代码与验证。