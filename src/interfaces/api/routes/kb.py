"""Knowledge Base API 路由（T095）。

说明：
- router_admin：建库/管理（可独立部署）
- router_query：查询（hybrid + rerank，可独立部署）

通过环境变量 LUMO_API_MODE 在 app 中选择性挂载：
- full / kb_admin / kb_query
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.application.repositories.intermediate_artifact_repository import (
    IntermediateArtifactRepository,
)
from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.application.repositories.prompt_repository import PromptRepository
from src.application.repositories.source_file_repository import SourceFileRepository
from src.application.schemas.ingest import (
    ChunkingConfig,
    ChunkingOptions,
    ChunkingStrategy,
    HybridSearchResponse,
)
from src.application.services.chunking_service import DocumentChunkingService
from src.application.services.hybrid_search_service import HybridSearchService
from src.application.services.knowledge_base_service import (
    KnowledgeBaseBuildOptions,
    KnowledgeBaseService,
    _parse_json_maybe,
    resolve_all_bm25_index_storage_paths,
)
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.application.services.vector_storage_service import VectorStorageService
from src.domain.entities.intermediate_artifact import IntermediateArtifact, IntermediateType
from src.interfaces.api.deps import get_db
from src.shared.config import get_settings
from src.shared.db import make_engine, make_session_factory
from src.shared.logging import get_logger


router_admin = APIRouter()
router_query = APIRouter()
log = get_logger(__name__)


class KBBuildRequest(BaseModel):
    input_root: str = Field(
        ...,
        min_length=1,
        description="输入根目录（例如 data/intermediates/{source_file_id}/pic_to_json）",
    )
    collection_name: str = Field(default="default", description="Chroma collection 名称")
    recreate: bool = Field(default=False, description="是否重建 collection（会先删除再写入）")
    max_docs: int | None = Field(default=None, ge=1, description="最多处理多少个 md（为空则全量）")
    workspace_id: str = Field(default="default", description="工作空间 ID（用于写入 intermediate_artifacts）")
    chunk_size: int | None = Field(default=None, ge=100, le=8192)
    chunk_overlap: int | None = Field(default=None, ge=0, le=512)


class KBBuildResponse(BaseModel):
    success: bool
    collection_name: str
    docs_indexed: int
    chunks_indexed: int
    chart_snippets_injected: int
    artifact_id: str | None = None
    artifact_storage_path: str | None = None
    collection_info: dict[str, Any] | None = None


class KBBuildJobRequest(KBBuildRequest):
    progress_update_every_n_docs: int = Field(
        default=1,
        ge=1,
        le=50,
        description="进度更新频率（每 N 篇文档更新一次，默认为 1）",
    )


class KBBuildJobResponse(BaseModel):
    build_id: str = Field(..., description="建库任务 ID（等同于 artifact_id）")
    status: str = Field(..., description="queued/running/succeeded/failed")
    stage: str | None = Field(None, description="阶段标识（用于观测）")
    progress_percent: int | None = Field(None, ge=0, le=100)
    artifact_id: str = Field(..., description="对应 intermediate_artifacts.id")
    artifact_storage_path: str = Field(..., description="对应 intermediate_artifacts.storage_path")


class KBBuildStatusResponse(BaseModel):
    build_id: str
    status: str | None = None
    stage: str | None = None
    progress_percent: int | None = None
    docs_total: int | None = None
    docs_indexed: int | None = None
    chunks_indexed: int | None = None
    chart_snippets_injected: int | None = None
    started_at: str | None = None
    updated_at: str | None = None
    error: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    artifact_id: str
    artifact_storage_path: str


class KBQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="查询文本")
    collection_name: str = Field(default="default", description="collection 名称")
    workspace_id: str = Field(default="default", description="工作空间 ID（用于解析 BM25 索引路径）")
    top_k: int = Field(default=10, ge=1, le=50)
    rerank_top_n: int | None = Field(default=None, ge=1, le=50)
    filter_metadata: dict[str, Any] | None = Field(default=None, description="Chroma 元数据过滤条件")


def _make_llm_runtime(session: Session) -> LLMRuntimeService:
    return LLMRuntimeService(
        provider_repository=LLMProviderRepository(session),
        capability_repository=LLMCapabilityRepository(session),
        callsite_repository=LLMCallSiteRepository(session),
        prompt_repository=PromptRepository(session),
    )


@router_admin.post(
    "/kb/build",
    response_model=KBBuildResponse,
    summary="构建知识库（T095）",
    description="基于输入目录（例如 data/intermediates/{source_file_id}/pic_to_json）完成切块与向量入库，并落盘 kb_chunks 工件用于中台观测。",
)
async def build_kb(
    req: KBBuildRequest,
    db: Session = Depends(get_db),
):
    log.info(
        "t095.kb_build.requested",
        extra={
            "collection_name": req.collection_name,
            "input_root": req.input_root,
            "recreate": req.recreate,
            "max_docs": req.max_docs,
            "workspace_id": req.workspace_id,
        },
    )
    llm_runtime = _make_llm_runtime(db)
    embedder = llm_runtime.get_model_for_callsite("vector_storage:embed_text")

    vector_service = VectorStorageService(embedding_model=embedder)
    chunking_config = ChunkingConfig(
        chunk_size=req.chunk_size or ChunkingConfig().chunk_size,
        chunk_overlap=req.chunk_overlap or ChunkingConfig().chunk_overlap,
    )
    chunking_service = DocumentChunkingService(config=chunking_config)

    kb_service = KnowledgeBaseService(
        chunking_service=chunking_service,
        vector_service=vector_service,
        # build 端口不依赖 rerank（查询端口独立注入）
        hybrid_search_service=HybridSearchService(vector_service=vector_service, reranker=None),
        source_file_repository=SourceFileRepository(db),
        artifact_repository=IntermediateArtifactRepository(db),
    )

    result = await kb_service.build_from_t094_output(
        options=KnowledgeBaseBuildOptions(
            input_root=Path(req.input_root),
            collection_name=req.collection_name,
            recreate=req.recreate,
            max_docs=req.max_docs,
            chunking=ChunkingOptions(
                strategy=ChunkingStrategy.STRUCTURE_AWARE,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            ),
        ),
        workspace_id=req.workspace_id,
    )
    log.info(
        "t095.kb_build.completed",
        extra={
            "collection_name": req.collection_name,
            "success": bool(result.get("success")),
            "docs_indexed": result.get("docs_indexed"),
            "chunks_indexed": result.get("chunks_indexed"),
            "artifact_id": result.get("artifact_id"),
            "artifact_storage_path": result.get("artifact_storage_path"),
        },
    )
    return KBBuildResponse(**result)


def _run_kb_build_in_thread(*, build_id: str, artifact_storage_path: str, req: KBBuildJobRequest) -> None:
    """后台线程：执行全量/增量建库，并持续写入 intermediate_artifacts.extra_metadata。"""
    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    session_factory = make_session_factory(engine)

    db = session_factory()
    try:
        log.info(
            "t095.kb_build.thread_started",
            extra={
                "build_id": build_id,
                "collection_name": req.collection_name,
                "input_root": req.input_root,
                "artifact_storage_path": artifact_storage_path,
            },
        )
        artifact_repo = IntermediateArtifactRepository(db)

        # 标记：开始加载模型
        artifact_repo.update_extra_metadata(
            build_id,
            extra_metadata=json.dumps(
                {
                    "task": "T095",
                    "build_id": build_id,
                    "collection_name": req.collection_name,
                    "input_root": req.input_root,
                    "artifact_storage_path": artifact_storage_path,
                    "status": "running",
                    "stage": "loading_models",
                    "progress_percent": 0,
                    "updated_at": datetime.now().isoformat(),
                },
                ensure_ascii=False,
            ),
        )

        llm_runtime = _make_llm_runtime(db)
        embedder = llm_runtime.get_model_for_callsite("vector_storage:embed_text")

        vector_service = VectorStorageService(embedding_model=embedder)
        chunking_config = ChunkingConfig(
            chunk_size=req.chunk_size or ChunkingConfig().chunk_size,
            chunk_overlap=req.chunk_overlap or ChunkingConfig().chunk_overlap,
        )
        chunking_service = DocumentChunkingService(config=chunking_config)

        kb_service = KnowledgeBaseService(
            chunking_service=chunking_service,
            vector_service=vector_service,
            # build 端口不依赖 rerank（查询端口独立注入）
            hybrid_search_service=HybridSearchService(vector_service=vector_service, reranker=None),
            source_file_repository=SourceFileRepository(db),
            artifact_repository=artifact_repo,
        )

        asyncio.run(
            kb_service.build_from_t094_output(
                options=KnowledgeBaseBuildOptions(
                    input_root=Path(req.input_root),
                    collection_name=req.collection_name,
                    recreate=req.recreate,
                    max_docs=req.max_docs,
                    chunking=ChunkingOptions(
                        strategy=ChunkingStrategy.STRUCTURE_AWARE,
                        chunk_size=req.chunk_size,
                        chunk_overlap=req.chunk_overlap,
                    ),
                    build_id=build_id,
                    artifact_storage_path=artifact_storage_path,
                    progress_update_every_n_docs=req.progress_update_every_n_docs,
                ),
                workspace_id=req.workspace_id,
            )
        )
        log.info(
            "t095.kb_build.thread_finished",
            extra={
                "build_id": build_id,
                "collection_name": req.collection_name,
                "artifact_storage_path": artifact_storage_path,
            },
        )
    except Exception as exc:  # noqa: BLE001
        log.exception(
            "t095.kb_build.thread_failed",
            extra={
                "build_id": build_id,
                "collection_name": req.collection_name,
                "artifact_storage_path": artifact_storage_path,
            },
        )
        # 尽量写入失败态，便于观测（避免线程静默失败）
        try:
            repo = IntermediateArtifactRepository(db)
            current = repo.get_by_id(build_id)
            meta = _parse_json_maybe(current.extra_metadata) if current else None
            payload = meta or {}
            payload.update(
                {
                    "task": "T095",
                    "build_id": build_id,
                    "collection_name": req.collection_name,
                    "input_root": req.input_root,
                    "artifact_storage_path": artifact_storage_path,
                    "status": "failed",
                    "stage": "failed",
                    "progress_percent": payload.get("progress_percent"),
                    "updated_at": datetime.now().isoformat(),
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }
            )
            repo.update_extra_metadata(build_id, extra_metadata=json.dumps(payload, ensure_ascii=False))
        except Exception:
            # 失败态写入也不应影响进程
            pass
        return
    finally:
        db.close()


@router_admin.post(
    "/kb/builds",
    response_model=KBBuildJobResponse,
    summary="启动建库任务（后台执行）",
    description="用于第一次全量建设：立即返回 build_id，并在后台执行建库；可通过 GET /v1/kb/builds/{build_id} 轮询进度。",
)
def start_kb_build_job(
    req: KBBuildJobRequest,
    db: Session = Depends(get_db),
):
    build_id = str(uuid4())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_storage_path = f"intermediates/kb_chunks/{req.collection_name}/{ts}_{build_id}.json"
    log.info(
        "t095.kb_build.job_queued",
        extra={
            "build_id": build_id,
            "collection_name": req.collection_name,
            "input_root": req.input_root,
            "artifact_storage_path": artifact_storage_path,
            "recreate": req.recreate,
            "max_docs": req.max_docs,
            "workspace_id": req.workspace_id,
        },
    )

    # 先写入 DB（保证 build_id 可立即查询）
    initial = {
        "task": "T095",
        "build_id": build_id,
        "collection_name": req.collection_name,
        "input_root": req.input_root,
        "artifact_storage_path": artifact_storage_path,
        "status": "queued",
        "stage": "queued",
        "progress_percent": 0,
        "docs_total": None,
        "docs_indexed": 0,
        "chunks_indexed": 0,
        "chart_snippets_injected": 0,
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    # 同步落盘一份初始报告（方便排查）
    out_path = Path("data") / artifact_storage_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8")

    artifact_repo = IntermediateArtifactRepository(db)
    artifact_repo.create(
        IntermediateArtifact(
            id=build_id,
            workspace_id=req.workspace_id,
            source_id=None,
            type=IntermediateType.KB_CHUNKS,
            storage_path=artifact_storage_path,
            deletable=True,
            extra_metadata=json.dumps(initial, ensure_ascii=False),
        )
    )

    t = threading.Thread(
        target=_run_kb_build_in_thread,
        kwargs={"build_id": build_id, "artifact_storage_path": artifact_storage_path, "req": req},
        daemon=True,
    )
    t.start()

    return KBBuildJobResponse(
        build_id=build_id,
        status="queued",
        stage="queued",
        progress_percent=0,
        artifact_id=build_id,
        artifact_storage_path=artifact_storage_path,
    )


@router_admin.get(
    "/kb/builds/{build_id}",
    response_model=KBBuildStatusResponse,
    summary="查询建库任务进度",
    description="轮询获取建库进度与阶段信息（数据来自 intermediate_artifacts.extra_metadata）。",
)
def get_kb_build_status(
    build_id: str,
    db: Session = Depends(get_db),
):
    repo = IntermediateArtifactRepository(db)
    artifact = repo.get_by_id(build_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="build not found")

    meta = _parse_json_maybe(artifact.extra_metadata) or {}
    return KBBuildStatusResponse(
        build_id=build_id,
        status=meta.get("status"),
        stage=meta.get("stage"),
        progress_percent=meta.get("progress_percent"),
        docs_total=meta.get("docs_total"),
        docs_indexed=meta.get("docs_indexed"),
        chunks_indexed=meta.get("chunks_indexed"),
        chart_snippets_injected=meta.get("chart_snippets_injected"),
        started_at=meta.get("started_at"),
        updated_at=meta.get("updated_at"),
        error=meta.get("error"),
        result=meta.get("result"),
        artifact_id=artifact.id,
        artifact_storage_path=artifact.storage_path,
    )


@router_admin.get(
    "/kb/collections",
    summary="列出知识库 collections",
    description="用于中台管理：列出当前 Chroma 所有 collections。",
)
async def list_kb_collections(db: Session = Depends(get_db)) -> dict[str, Any]:
    vector_service = VectorStorageService()
    names = await vector_service.list_collections()
    return {"items": names, "count": len(names)}


@router_admin.get(
    "/kb/collections/{collection_name}",
    summary="查看 collection 统计信息",
)
async def get_kb_collection_stats(
    collection_name: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    vector_service = VectorStorageService()
    info = await vector_service.get_collection_info(collection_name)
    return {"collection": collection_name, "info": info.model_dump() if info else None}


@router_admin.delete(
    "/kb/collections/{collection_name}",
    summary="删除 collection",
)
async def delete_kb_collection(
    collection_name: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    vector_service = VectorStorageService()
    ok = await vector_service.delete_collection(collection_name)
    return {"collection": collection_name, "deleted": ok}


@router_query.post(
    "/kb/query",
    response_model=HybridSearchResponse,
    summary="知识库查询（hybrid + rerank）",
    description="固定使用 hybrid 检索 + rerank（通过 T023 callsite 注入）。",
)
async def query_kb(
    req: KBQueryRequest,
    db: Session = Depends(get_db),
):
    llm_runtime = _make_llm_runtime(db)
    embedder = llm_runtime.get_model_for_callsite("vector_storage:embed_text")
    reranker = llm_runtime.get_model_for_callsite("hybrid_search:rerank")

    vector_service = VectorStorageService(embedding_model=embedder)
    bm25_paths = resolve_all_bm25_index_storage_paths(
        db=db,
        collection_name=req.collection_name,
        workspace_id=req.workspace_id,
    )
    hybrid_service = HybridSearchService(
        vector_service=vector_service,
        collection_name=req.collection_name,
        reranker=reranker,
        bm25_index_storage_paths=bm25_paths,
    )
    kb_service = KnowledgeBaseService(
        chunking_service=DocumentChunkingService(config=ChunkingConfig()),
        vector_service=vector_service,
        hybrid_search_service=hybrid_service,
        source_file_repository=SourceFileRepository(db),
        artifact_repository=None,
    )
    return await kb_service.query_hybrid_rerank(
        query=req.query,
        collection_name=req.collection_name,
        top_k=req.top_k,
        filter_metadata=req.filter_metadata,
        rerank_top_n=req.rerank_top_n,
    )

