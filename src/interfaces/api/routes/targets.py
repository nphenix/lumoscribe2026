"""目标文件 API 路由。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.application.repositories.target_file_repository import TargetFileRepository
from src.application.repositories.template_repository import TemplateRepository
from src.application.schemas.target_file import (
    DeleteTargetFileResponse,
    TargetFileListResponse,
    TargetFileResponse,
)
from src.application.services.chart_renderer_service import ChartRendererService
from src.application.services.content_generation_service import ContentGenerationService
from src.application.services.hybrid_search_service import HybridSearchService
from src.application.services.knowledge_base_service import (
    resolve_latest_bm25_index_storage_path,
    resolve_latest_kb_input_root,
)
from src.application.services.llm_runtime_service import LLMRuntimeService
from src.application.services.template_service import TemplateService
from src.application.services.target_file_service import TargetFileService
from src.application.services.vector_storage_service import VectorStorageService
from src.application.repositories.intermediate_artifact_repository import IntermediateArtifactRepository
from src.domain.entities.intermediate_artifact import IntermediateType
from src.domain.entities.target_file import TargetFile
from src.domain.entities.template import Template, TemplateType
from src.interfaces.api.deps import get_db
from src.shared.request_id import get_request_id
from src.application.schemas.ingest import HybridSearchOptions
from src.shared.config import get_settings

# LLM runtime dependencies (repositories)
from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
from src.application.repositories.llm_capability_repository import LLMCapabilityRepository
from src.application.repositories.llm_provider_repository import LLMProviderRepository
from src.application.repositories.prompt_repository import PromptRepository

# 生成白皮书只关心“检索+生成”，不依赖建库流程的中间工件


router = APIRouter()


def get_target_file_service(db: Session = Depends(get_db)) -> TargetFileService:
    """获取目标文件服务实例。"""
    repository = TargetFileRepository(db)
    return TargetFileService(repository)

def _make_llm_runtime(db: Session) -> LLMRuntimeService:
    return LLMRuntimeService(
        provider_repository=LLMProviderRepository(db),
        capability_repository=LLMCapabilityRepository(db),
        callsite_repository=LLMCallSiteRepository(db),
        prompt_repository=PromptRepository(db),
    )


def _safe_filename(name: str) -> str:
    """生成 Windows/HTTP 友好的文件名（不含路径分隔符和非法字符）。"""
    s = (name or "").strip()
    if not s:
        return "whitepaper"
    # Windows 文件名非法字符：<>:\"/\\|?*
    bad = '<>:"/\\|?*'
    out = []
    for ch in s:
        out.append("_" if ch in bad else ch)
    # 去掉尾部空格/点（Windows 不允许）
    return "".join(out).strip().strip(".") or "whitepaper"


def _extract_md_title(md: str) -> str | None:
    for line in (md or "").replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip() or None
        if s:
            break
    return None


def _is_success_status(v: object) -> bool:
    """兼容不同版本的 KB build 状态字段（以事实为准，保持向后兼容）。"""
    s = str(v or "").strip().lower()
    return s in {"completed", "succeeded", "success", "ok", "done"}


def _read_text_best_effort(path: Path) -> str:
    """尽量读取文本（兼容 Windows 常见编码）。

    说明：正式环境要求 UTF-8，但用户本地可能用 GBK/GB18030 保存 drafts。
    """
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    # 最后兜底：不抛错，避免接口直接 500
    return path.read_text(encoding="utf-8", errors="replace")


class WhitepaperDraftItem(BaseModel):
    filename: str = Field(..., description="drafts 目录下的文件名（不含路径）")
    title: str | None = Field(None, description="从 # 标题解析出的文档标题")
    size_bytes: int = Field(..., ge=0)
    updated_at: str | None = Field(None, description="ISO 时间戳（文件 mtime）")


class WhitepaperDraftListResponse(BaseModel):
    items: list[WhitepaperDraftItem]
    total: int


class WhitepaperGenerateRequest(BaseModel):
    workspace_id: str = Field(default="default", max_length=36)
    # 注意：当前 target_files.kb_id 字段为 String(36)；这里收紧避免写库失败
    collection_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=36,
        description=(
            "知识库 collection 名称。建议由前端显式传递。"
            "为空则：若本机仅存在 1 个 collection 自动使用该 collection；"
            "若存在多个则返回 400 并列出可选项；若不存在任何 collection 则使用 default（并自动降级为纯 LLM 补全）。"
        ),
    )
    outline_filename: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description=(
            "drafts 下的 .md 文件名。建议由前端显式传递。"
            "为空则自动使用 drafts 目录下最新修改的 .md（按 mtime 选择）。"
        ),
    )
    polish_outline: bool | None = Field(
        default=None,
        description="是否先调用 LLM 润色大纲（保持结构/层级）。为空则使用服务端默认（可用 LUMO_T096_POLISH_OUTLINE 配置）。",
    )

    # 检索参数（hybrid + rerank）
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="召回候选数量（为空则使用服务端默认配置）",
    )
    rerank_top_n: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="重排序后返回数量（为空则默认等于 top_k 或服务端默认配置）",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="覆盖判定阈值（用于标记哪些子章节由 LLM 补全），不影响最终生成",
    )


def _resolve_generation_defaults(req: WhitepaperGenerateRequest) -> tuple[bool, int, int]:
    """将请求参数与服务端默认值合并，降低前端集成成本。"""
    settings = get_settings()
    # 默认：尽量多召回，利于多文档覆盖；上限受 schema 约束 (<=100)
    top_k = req.top_k if req.top_k is not None else int(getattr(settings, "whitepaper_top_k", 50))
    rerank_top_n = (
        req.rerank_top_n
        if req.rerank_top_n is not None
        else int(getattr(settings, "whitepaper_rerank_top_n", top_k))
    )
    # 合理约束：rerank_top_n 不应大于 top_k
    if rerank_top_n > top_k:
        rerank_top_n = top_k

    polish_outline = (
        req.polish_outline
        if req.polish_outline is not None
        else bool(getattr(settings, "whitepaper_polish_outline", False))
    )
    return polish_outline, top_k, rerank_top_n


def _list_draft_md_files() -> list[Path]:
    drafts_dir = Path("data") / "Templates" / "drafts"
    if not drafts_dir.exists():
        raise HTTPException(status_code=404, detail=f"drafts 目录不存在: {drafts_dir.as_posix()}")
    return sorted(drafts_dir.glob("*.md"))


def _resolve_outline(req: WhitepaperGenerateRequest) -> tuple[str, str]:
    """解析大纲：前端可传；未传则取最新 drafts md（确定性规则）。"""
    filename = (req.outline_filename or "").strip() or ""

    drafts = _list_draft_md_files()
    drafts_dir = Path("data") / "Templates" / "drafts"

    if not filename:
        if not drafts:
            raise HTTPException(status_code=404, detail=f"drafts 目录为空: {drafts_dir.as_posix()}")
        # 取最新修改的 .md（mtime 最大）
        try:
            p = max(drafts, key=lambda x: x.stat().st_mtime)
        except Exception:
            # 兜底：按文件名排序取最后一个（确定性）
            p = sorted(drafts, key=lambda x: x.name)[-1]
        return p.name, _read_text_best_effort(p)

    safe = Path(filename).name
    if safe != filename or ".." in Path(filename).parts:
        raise HTTPException(status_code=400, detail="outline_filename 非法（仅允许文件名）")

    outline_path = drafts_dir / safe
    if outline_path.suffix.lower() != ".md":
        raise HTTPException(status_code=400, detail="仅支持 .md 大纲文件")
    if not outline_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"outline 文件不存在: {outline_path.as_posix()}",
                "candidates": [p.name for p in drafts],
            },
        )
    return safe, _read_text_best_effort(outline_path)


async def _resolve_collection_name(req: WhitepaperGenerateRequest, db: Session) -> str:
    """解析 collection（确定性，基于 DB 事实）。

    规则（不做“猜你要哪个”的启发式）：
    1) 请求显式传入
    2) DB 最近一次生成的 target_files.kb_id（同 workspace）
    3) DB 最近一次成功建库（intermediate_artifacts: kb_chunks, status=completed）的 collection_name
    4) Chroma 若仅存在 1 个 collection，则使用它
    5) Chroma 无 collection：回退 default（生成阶段将自然降级为纯 LLM 补全）
    6) 否则 400 返回候选列表
    """
    name = (req.collection_name or "").strip() or ""
    if name:
        return name

    # 2) 最近一次生成使用的 kb_id（DB 事实）
    try:
        latest = (
            db.query(TargetFile)
            .filter(
                TargetFile.workspace_id == req.workspace_id,
                TargetFile.kb_id.isnot(None),
            )
            .order_by(TargetFile.created_at.desc())
            .first()
        )
        if latest is not None:
            kb_id = (latest.kb_id or "").strip()
            if kb_id:
                return kb_id
    except Exception:
        pass

    # 3) 最近一次成功建库的 kb_chunks 产物（DB 事实）
    try:
        repo = IntermediateArtifactRepository(db)
        items = repo.list(
            workspace_id=req.workspace_id,
            artifact_type=IntermediateType.KB_CHUNKS,
            source_id=None,
            limit=50,
            offset=0,
        )
        for it in items:
            meta_raw = (it.extra_metadata or "").strip()
            if not meta_raw:
                continue
            try:
                meta = json.loads(meta_raw)
            except Exception:
                continue
            if not isinstance(meta, dict):
                continue
            if not _is_success_status(meta.get("status")):
                continue
            cn = str(meta.get("collection_name") or "").strip()
            if cn:
                return cn
        # 没有 completed，也不盲目选择 running/error（避免“生成时绑定到不完整 KB”）
    except Exception:
        pass

    try:
        vector_service = VectorStorageService()
        names = await vector_service.list_collections()
    except Exception:
        names = []

    if not names:
        return "default"
    if len(names) == 1:
        return names[0]
    raise HTTPException(
        status_code=400,
        detail={
            "message": "无法从数据库确定 collection_name，且 Chroma collections 不唯一；请显式指定或先调用 /kb/collections 获取候选",
            "candidates": names,
        },
    )


class WhitepaperGenerateResponse(BaseModel):
    request_id: str | None = None
    target_id: str
    template_id: str
    storage_path: str
    output_filename: str
    document_title: str
    collection_name: str
    coverage: dict
    html_length: int
    generated_at: str


@router.get(
    "/targets/whitepaper/drafts",
    response_model=WhitepaperDraftListResponse,
    summary="列出白皮书大纲 drafts",
    description="读取 data/Templates/drafts 下的 Markdown 大纲文件（正式环境也复用此目录）。",
)
def list_whitepaper_drafts(db: Session = Depends(get_db)):
    drafts_dir = Path("data") / "Templates" / "drafts"
    if not drafts_dir.exists():
        raise HTTPException(status_code=404, detail=f"drafts 目录不存在: {drafts_dir.as_posix()}")

    items: list[WhitepaperDraftItem] = []
    for p in sorted(drafts_dir.glob("*.md")):
        text = _read_text_best_effort(p)
        title = _extract_md_title(text)
        try:
            st = p.stat()
            mtime = datetime.fromtimestamp(st.st_mtime).isoformat()
            size = int(st.st_size)
        except Exception:
            mtime = None
            size = 0
        items.append(
            WhitepaperDraftItem(
                filename=p.name,
                title=title,
                size_bytes=size,
                updated_at=mtime,
            )
        )

    return WhitepaperDraftListResponse(items=items, total=len(items))


@router.post(
    "/targets/whitepaper/generate",
    response_model=WhitepaperGenerateResponse,
    summary="按白皮书大纲生成单 HTML",
    description="按章节/子章节逐段 RAG 召回，覆盖不足处由 LLM 补全，输出单 HTML 并写入 target_files。",
)
async def generate_whitepaper(
    req: WhitepaperGenerateRequest,
    db: Session = Depends(get_db),
):
    polish_outline, top_k, rerank_top_n = _resolve_generation_defaults(req)
    # 0) 解析 collection / outline（只在“唯一”时自动选择；否则返回候选列表，避免猜测）
    collection_name = await _resolve_collection_name(req, db)
    filename, outline_text = _resolve_outline(req)

    # 2) 确保 drafts 大纲已注册为 Template（只写 DB，不复制文件）
    storage_rel = f"Templates/drafts/{filename}".replace("\\", "/")
    template = (
        db.query(Template)
        .filter(Template.workspace_id == req.workspace_id, Template.storage_path == storage_rel)
        .first()
    )
    if template is None:
        from uuid import uuid4

        template = Template(
            id=str(uuid4()),
            workspace_id=req.workspace_id,
            original_filename=filename,
            file_format="md",
            type=TemplateType.CUSTOM,
            version=1,
            locked=True,
            storage_path=storage_rel,
            description="whitepaper outline draft (data/Templates/drafts)",
        )
        db.add(template)
        db.commit()
        db.refresh(template)

    # 3) 组装服务依赖（真实 LLM + 真实 KB）
    llm_runtime = _make_llm_runtime(db)
    embedder = llm_runtime.get_model_for_callsite("vector_storage:embed_text")
    reranker = llm_runtime.get_model_for_callsite("hybrid_search:rerank")

    vector_service = VectorStorageService(embedding_model=embedder)
    bm25_path = resolve_latest_bm25_index_storage_path(
        db=db,
        collection_name=collection_name,
        workspace_id=req.workspace_id,
    )
    hybrid_service = HybridSearchService(
        vector_service=vector_service,
        collection_name=collection_name,
        reranker=reranker,
        bm25_index_storage_path=bm25_path,
    )

    template_repo = TemplateRepository(db)
    template_service = TemplateService(template_repo)
    chart_renderer = ChartRendererService()
    content_service = ContentGenerationService(
        template_service=template_service,
        hybrid_search_service=hybrid_service,
        llm_runtime_service=llm_runtime,
        chart_renderer_service=chart_renderer,
        template_repository=template_repo,
    )

    # 4) 可选：先润色大纲（保持格式/层级）
    polished_outline = outline_text
    if polish_outline:
        polished_outline = content_service.polish_outline(outline_text)

    document_title = _extract_md_title(polished_outline) or _extract_md_title(outline_text) or filename

    # 5) 检索参数（hybrid + rerank）
    search_options = HybridSearchOptions(
        top_k=top_k,
        use_rerank=True,
        rerank_top_n=rerank_top_n,
    )

    # 5) 生成内容（按章/子章召回，覆盖不足由 LLM 补全）
    kb_input_root_raw = resolve_latest_kb_input_root(
        db=db,
        collection_name=collection_name,
        workspace_id=req.workspace_id,
    )
    kb_input_root = Path(kb_input_root_raw) if kb_input_root_raw else None
    result = await content_service.generate_content(
        template_id=template.id,
        collection_name=collection_name,
        chart_configs=None,
        search_options=search_options,
        template_content_override=polished_outline,
        document_title=document_title,
        coverage_score_threshold=req.score_threshold,
        kb_input_root=kb_input_root,
    )

    # 6) 落盘 HTML 并写入 target_files
    from uuid import uuid4

    target_id = str(uuid4())
    safe_title = _safe_filename(document_title)
    output_filename = f"{safe_title}.html"

    storage_path = f"targets/{req.workspace_id}/{target_id}.html"
    out_path = Path("data") / storage_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.html_content, encoding="utf-8")

    target = TargetFile(
        id=target_id,
        workspace_id=req.workspace_id,
        template_id=template.id,
        kb_id=collection_name,
        job_id=None,
        output_filename=output_filename,
        storage_path=storage_path,
        description=f"whitepaper generated from drafts/{filename} (collection={collection_name})",
    )
    db.add(target)
    db.commit()

    coverage = {
        "template_id": template.id,
        "outline_filename": filename,
        "sections": [
            {
                "section_id": s.section_id,
                "title": s.title,
                "coverage": s.coverage,
                "sources": s.sources,
                "rendered_charts": list(s.rendered_charts.keys()),
                "tokens_used": s.tokens_used,
                "generation_time_ms": s.generation_time_ms,
            }
            for s in result.sections
        ],
        "total_tokens": result.total_tokens,
        "total_time_ms": result.total_time_ms,
    }

    return WhitepaperGenerateResponse(
        request_id=get_request_id(),
        target_id=target_id,
        template_id=template.id,
        storage_path=storage_path,
        output_filename=output_filename,
        document_title=document_title,
        collection_name=collection_name,
        coverage=coverage,
        html_length=len(result.html_content or ""),
        generated_at=result.generated_at.isoformat(),
    )


@router.post(
    "/targets/whitepaper/generate_stream",
    summary="按白皮书大纲生成单 HTML（token 级流式）",
    description="以 SSE 返回事件流：按章开始/召回进度/LLM token/按章完成，最终事件返回与 generate 相同结构的结果。",
)
async def generate_whitepaper_stream(
    req: WhitepaperGenerateRequest,
    db: Session = Depends(get_db),
):
    polish_outline, top_k, rerank_top_n = _resolve_generation_defaults(req)
    # 说明：此接口会输出 token 级别流式（LLM 生成阶段），并同时输出“进度/召回”事件便于观测。
    import asyncio
    from uuid import uuid4

    rid = get_request_id()
    queue: asyncio.Queue[dict] = asyncio.Queue()

    async def _push(evt: dict) -> None:
        evt.setdefault("request_id", rid)
        await queue.put(evt)

    # 0) 解析 collection / outline（只在“唯一”时自动选择；否则返回候选列表，避免猜测）
    collection_name = await _resolve_collection_name(req, db)
    filename, outline_text = _resolve_outline(req)

    # 2) 确保 drafts 大纲已注册为 Template（只写 DB，不复制文件）
    storage_rel = f"Templates/drafts/{filename}".replace("\\", "/")
    template = (
        db.query(Template)
        .filter(Template.workspace_id == req.workspace_id, Template.storage_path == storage_rel)
        .first()
    )
    if template is None:
        template = Template(
            id=str(uuid4()),
            workspace_id=req.workspace_id,
            original_filename=filename,
            file_format="md",
            type=TemplateType.CUSTOM,
            version=1,
            locked=True,
            storage_path=storage_rel,
            description="whitepaper outline draft (data/Templates/drafts)",
        )
        db.add(template)
        db.commit()
        db.refresh(template)

    # 3) 组装服务依赖（真实 LLM + 真实 KB）
    llm_runtime = _make_llm_runtime(db)
    embedder = llm_runtime.get_model_for_callsite("vector_storage:embed_text")
    reranker = llm_runtime.get_model_for_callsite("hybrid_search:rerank")

    vector_service = VectorStorageService(embedding_model=embedder)
    bm25_path = resolve_latest_bm25_index_storage_path(
        db=db,
        collection_name=collection_name,
        workspace_id=req.workspace_id,
    )
    hybrid_service = HybridSearchService(
        vector_service=vector_service,
        collection_name=collection_name,
        reranker=reranker,
        bm25_index_storage_path=bm25_path,
    )

    template_repo = TemplateRepository(db)
    template_service = TemplateService(template_repo)
    chart_renderer = ChartRendererService()
    content_service = ContentGenerationService(
        template_service=template_service,
        hybrid_search_service=hybrid_service,
        llm_runtime_service=llm_runtime,
        chart_renderer_service=chart_renderer,
        template_repository=template_repo,
    )

    polished_outline = outline_text
    if polish_outline:
        polished_outline = content_service.polish_outline(outline_text)

    document_title = _extract_md_title(polished_outline) or _extract_md_title(outline_text) or filename

    # 检索参数（hybrid + rerank）
    search_options = HybridSearchOptions(
        top_k=top_k,
        use_rerank=True,
        rerank_top_n=rerank_top_n,
    )

    async def _run() -> None:
        try:
            await _push({"type": "stage", "stage": "start"})
            await _push(
                {
                    "type": "stage",
                    "stage": "retrieval_meta",
                    "collection_name": collection_name,
                }
            )

            kb_input_root_raw = resolve_latest_kb_input_root(
                db=db,
                collection_name=collection_name,
                workspace_id=req.workspace_id,
            )
            kb_input_root = Path(kb_input_root_raw) if kb_input_root_raw else None

            result = await content_service.generate_content(
                template_id=template.id,
                collection_name=collection_name,
                chart_configs=None,
                search_options=search_options,
                template_content_override=polished_outline,
                document_title=document_title,
                coverage_score_threshold=req.score_threshold,
                kb_input_root=kb_input_root,
                on_event=_push,
                # token 级别流式：让 service 在 LLM 生成阶段推送 token 事件
                stream_tokens=True,
            )

            # 落盘 HTML 并写入 target_files
            target_id = str(uuid4())
            safe_title = _safe_filename(document_title)
            output_filename = f"{safe_title}.html"
            storage_path = f"targets/{req.workspace_id}/{target_id}.html"
            out_path = Path("data") / storage_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result.html_content, encoding="utf-8")

            target = TargetFile(
                id=target_id,
                workspace_id=req.workspace_id,
                template_id=template.id,
                kb_id=collection_name,
                job_id=None,
                output_filename=output_filename,
                storage_path=storage_path,
                description=f"whitepaper generated from drafts/{filename} (collection={collection_name})",
            )
            db.add(target)
            db.commit()

            coverage = {
                "template_id": template.id,
                "outline_filename": filename,
                "sections": [
                    {
                        "section_id": s.section_id,
                        "title": s.title,
                        "coverage": s.coverage,
                        "sources": s.sources,
                        "rendered_charts": list(s.rendered_charts.keys()),
                        "tokens_used": s.tokens_used,
                        "generation_time_ms": s.generation_time_ms,
                    }
                    for s in result.sections
                ],
                "total_tokens": result.total_tokens,
                "total_time_ms": result.total_time_ms,
            }

            await _push(
                {
                    "type": "final",
                    "payload": {
                        "request_id": rid,
                        "target_id": target_id,
                        "template_id": template.id,
                        "storage_path": storage_path,
                        "output_filename": output_filename,
                        "document_title": document_title,
                        "collection_name": collection_name,
                        "coverage": coverage,
                        "html_length": len(result.html_content or ""),
                        "generated_at": result.generated_at.isoformat(),
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            await _push(
                {
                    "type": "error",
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }
            )
        finally:
            await queue.put({"type": "end", "request_id": rid})

    async def _event_stream():
        task = asyncio.create_task(_run())
        try:
            while True:
                evt = await queue.get()
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n".encode("utf-8")
                if evt.get("type") == "end":
                    break
        finally:
            task.cancel()

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 避免反向代理/中间层缓冲 SSE（即便当前未使用 Nginx，也不影响本地调试）
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/targets",
    response_model=TargetFileListResponse,
    summary="列出目标文件",
    description="获取目标文件列表，支持按模板、任务、工作空间和时间过滤。",
)
def list_target_files(
    workspace_id: Annotated[str | None, Query(max_length=36, description="工作空间ID")] = None,
    template_id: Annotated[str | None, Query(max_length=36, description="模板ID")] = None,
    kb_id: Annotated[str | None, Query(max_length=36, description="知识库ID")] = None,
    job_id: Annotated[str | None, Query(max_length=36, description="生成任务ID")] = None,
    created_after: Annotated[datetime | None, Query(description="创建时间上限")] = None,
    created_before: Annotated[datetime | None, Query(description="创建时间下限")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: TargetFileService = Depends(get_target_file_service),
):
    """列出目标文件。"""
    items, total = service.list_target_files(
        workspace_id=workspace_id,
        template_id=template_id,
        kb_id=kb_id,
        job_id=job_id,
        created_after=created_after,
        created_before=created_before,
        limit=limit,
        offset=offset,
    )

    return TargetFileListResponse(
        items=[
            {
                "id": item.id,
                "workspace_id": item.workspace_id,
                "template_id": item.template_id,
                "kb_id": item.kb_id,
                "job_id": item.job_id,
                "output_filename": item.output_filename,
                "storage_path": item.storage_path,
                "description": item.description,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in items
        ],
        total=total,
    )


@router.get(
    "/targets/{target_id}",
    response_model=TargetFileResponse,
    summary="获取目标文件详情",
    description="获取指定目标文件的详细信息，包括关联的模板和知识库信息。",
)
def get_target_file(
    target_id: str,
    service: TargetFileService = Depends(get_target_file_service),
):
    """获取目标文件详情。"""
    target_file = service.get_target_file(target_id)
    if target_file is None:
        raise HTTPException(status_code=404, detail="目标文件不存在")

    return TargetFileResponse(
        id=target_file.id,
        workspace_id=target_file.workspace_id,
        template_id=target_file.template_id,
        kb_id=target_file.kb_id,
        job_id=target_file.job_id,
        output_filename=target_file.output_filename,
        storage_path=target_file.storage_path,
        description=target_file.description,
        created_at=target_file.created_at,
        updated_at=target_file.updated_at,
        template_name=None,  # TODO: 关联查询模板名称
        kb_name=None,  # TODO: 关联查询知识库名称
    )


@router.get(
    "/targets/{target_id}/download",
    summary="下载目标文件",
    description="下载目标 HTML 文件，支持自定义输出文件名。",
)
def download_target_file(
    target_id: str,
    filename: Annotated[str | None, Query(description="自定义输出文件名")] = None,
    service: TargetFileService = Depends(get_target_file_service),
):
    """下载目标文件。"""
    target_file = service.get_target_file(target_id)
    if target_file is None:
        raise HTTPException(status_code=404, detail="目标文件不存在")

    from pathlib import Path

    file_path = Path("data") / target_file.storage_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="目标文件不存在")

    # 使用自定义文件名或默认 output_filename
    output_filename = filename or target_file.output_filename

    def iter_file():
        yield file_path.read_bytes()

    return StreamingResponse(
        iter_file(),
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="{output_filename}"',
            "Content-Length": str(file_path.stat().st_size),
        },
    )


@router.get(
    "/targets/{target_id}/view",
    summary="查看目标文件",
    description="内联查看目标 HTML 文件（直接在浏览器中打开）。",
)
def view_target_file(
    target_id: str,
    service: TargetFileService = Depends(get_target_file_service),
):
    """查看目标文件。"""
    target_file = service.get_target_file(target_id)
    if target_file is None:
        raise HTTPException(status_code=404, detail="目标文件不存在")

    from pathlib import Path

    file_path = Path("data") / target_file.storage_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="目标文件不存在")

    def iter_file():
        yield file_path.read_bytes()

    return StreamingResponse(
        iter_file(),
        media_type="text/html",
        headers={
            "Content-Disposition": f'inline; filename="{target_file.output_filename}"',
            "Content-Length": str(file_path.stat().st_size),
        },
    )


@router.delete(
    "/targets/{target_id}",
    response_model=DeleteTargetFileResponse,
    summary="删除目标文件",
    description="删除目标文件及其关联的存储文件。",
)
def delete_target_file(
    target_id: str,
    service: TargetFileService = Depends(get_target_file_service),
):
    """删除目标文件。"""
    success = service.delete_target_file(target_id)
    if not success:
        raise HTTPException(status_code=404, detail="目标文件不存在")

    return DeleteTargetFileResponse(
        id=target_id,
        message="目标文件已删除",
    )
