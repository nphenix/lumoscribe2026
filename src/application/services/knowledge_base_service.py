"""知识库构建与查询编排服务（T095）。

设计目标：
- 核心逻辑在服务端，测试脚本仅调用与验收
- 输入基于 T094 的真实产物目录（data/intermediates/{source_file_id}/pic_to_json）
- chunk 元数据可追溯：source_file_id(UUID)/original_filename/doc_title/doc_rel_path/chart_id/chart_name
- 模型依赖通过 T023 注入：embedding/rerank（不在此处硬编码）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from src.application.repositories.intermediate_artifact_repository import (
    IntermediateArtifactRepository,
)
from src.application.repositories.source_file_repository import SourceFileRepository
from src.application.schemas.ingest import (
    ChunkingConfig,
    ChunkingOptions,
    ChunkingStrategy,
    HybridSearchOptions,
    HybridSearchResponse,
    KBChunk,
)
from src.application.services.chunking_service import DocumentChunkingService
from src.application.services.hybrid_search_service import HybridSearchService
from src.application.services.vector_storage_service import VectorStorageService
from src.domain.entities.intermediate_artifact import IntermediateArtifact, IntermediateType
from src.shared.errors import AppError
from src.shared.logging import get_logger


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _normalize_rel_posix(p: str) -> str:
    s = (p or "").strip().strip('"').strip("'")
    s = s.replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    # 去掉 query/hash
    s = s.split("#", 1)[0].split("?", 1)[0].strip()
    return s


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        # 进度/报告可能包含 datetime 等对象；这里统一做宽松序列化，避免建库在 finalizing 阶段失败
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _compact_progress_for_db(payload: dict[str, Any]) -> dict[str, Any]:
    """压缩进度元数据，避免超过 extra_metadata 字段长度。"""
    keys = {
        "task",
        "build_id",
        "collection_name",
        "input_root",
        "status",
        "stage",
        "progress_percent",
        "docs_total",
        "docs_indexed",
        "chunks_indexed",
        "chart_snippets_injected",
        "artifact_storage_path",
        "bm25_index_storage_path",
        "started_at",
        "updated_at",
        "error",
        "result",
    }
    compact: dict[str, Any] = {k: payload.get(k) for k in keys if k in payload}

    # 进一步裁剪 error/result，防止过长
    err = compact.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        if isinstance(msg, str) and len(msg) > 400:
            err["message"] = msg[:400] + "…"
        compact["error"] = err

    res = compact.get("result")
    if isinstance(res, dict):
        # result 里可能包含 collection_info，保持最小必要信息
        allow = {"docs_indexed", "chunks_indexed", "chart_snippets_injected", "collection_name"}
        compact["result"] = {k: res.get(k) for k in allow if k in res}

    return compact


def _parse_json_maybe(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        v = json.loads(raw)
    except Exception:
        return None
    return v if isinstance(v, dict) else None


def resolve_all_bm25_index_storage_paths(
    *,
    db: Session,
    collection_name: str,
    workspace_id: str = "default",
    limit: int = 50,
) -> list[str]:
    """解析所有 kb_chunks 工件中的 bm25 索引路径（相对 data/）。

    用途：
    - T095：查询端口加载全部预建 BM25 索引
    - T096：白皮书生成（RAG）复用同一套索引解析逻辑

    Returns:
        索引路径列表（去重，按时间排序）
    """
    repo = IntermediateArtifactRepository(db)
    items = repo.list(
        workspace_id=workspace_id,
        artifact_type=IntermediateType.KB_CHUNKS,
        source_id=None,
        limit=limit,
        offset=0,
    )
    needle = f"intermediates/kb_chunks/{collection_name}/"
    paths: dict[str, str] = {}  # path -> storage_path 去重

    for it in items:
        sp = (it.storage_path or "").replace("\\", "/")
        if needle not in sp:
            continue
        meta = _parse_json_maybe(it.extra_metadata)
        if not meta:
            continue
        p = meta.get("bm25_index_storage_path")
        if isinstance(p, str) and p.strip():
            normalized = p.strip().replace("\\", "/")
            paths[normalized] = sp

    # 返回去重后的路径列表（保持插入顺序）
    return list(dict.fromkeys(paths.keys()))


def resolve_latest_kb_input_root(
    *,
    db: Session,
    collection_name: str,
    workspace_id: str = "default",
    limit: int = 50,
) -> str | None:
    """解析最新 kb_chunks 工件中的 input_root（相对或绝对皆可）。

    用途：
    - T096：白皮书生成需要从 doc_rel_path 反查到 chart_json/images 等真实文件
    """
    def _is_success_status(v: object) -> bool:
        """兼容不同版本的 build 状态字段（以事实为准，保持向后兼容）。"""
        s = str(v or "").strip().lower()
        return s in {"completed", "succeeded", "success", "ok", "done"}

    def _resolve_existing_dir(raw: str) -> Path | None:
        """将 input_root 解析为存在的目录（兼容绝对/相对 data/）。"""
        if not raw:
            return None
        cand = Path(str(raw).strip())
        if cand.exists() and cand.is_dir():
            return cand
        cand2 = Path("data") / str(raw).strip()
        if cand2.exists() and cand2.is_dir():
            return cand2
        return None

    repo = IntermediateArtifactRepository(db)
    items = repo.list(
        workspace_id=workspace_id,
        artifact_type=IntermediateType.KB_CHUNKS,
        source_id=None,
        limit=limit,
        offset=0,
    )
    needle = f"intermediates/kb_chunks/{collection_name}/"
    for it in items:
        sp = (it.storage_path or "").replace("\\", "/")
        if needle not in sp:
            continue
        meta = _parse_json_maybe(it.extra_metadata)
        if not meta:
            continue
        # 只接受“成功完成”的建库工件，避免拿到 running/failed 的不完整元数据
        if not _is_success_status(meta.get("status")) and not _is_success_status(meta.get("stage")):
            continue

        ir = meta.get("input_root")
        if not isinstance(ir, str) or not ir.strip():
            continue

        root_dir = _resolve_existing_dir(ir.strip())
        if root_dir is None:
            continue

        # 必须具备 chart_json 产物，才能支持 T096 图表反查
        chart_dir = root_dir / "chart_json"
        if not chart_dir.exists() or not chart_dir.is_dir():
            continue
        try:
            if not any(chart_dir.glob("*.json")):
                continue
        except Exception:
            continue

        # 保持与历史行为一致：返回 meta 中记录的 raw 字符串
        return ir.strip()
    return None


def _extract_title_from_layout(layout_path: Path) -> str | None:
    if not layout_path.exists():
        return None
    try:
        data = json.loads(_safe_read_text(layout_path))
    except Exception:
        return None

    # 取第一个 title block 的文本
    try:
        pdf_info = data.get("pdf_info") or []
        for page in pdf_info:
            blocks = page.get("preproc_blocks") or []
            for b in blocks:
                if b.get("type") != "title":
                    continue
                lines = b.get("lines") or []
                parts: list[str] = []
                for line in lines:
                    spans = line.get("spans") or []
                    for sp in spans:
                        if sp.get("type") == "text" and isinstance(sp.get("content"), str):
                            parts.append(sp["content"].strip())
                title = " ".join([p for p in parts if p]).strip()
                if title:
                    # 去掉过长标题中的多余空格
                    title = re.sub(r"\s+", " ", title)
                    return title
    except Exception:
        return None
    return None


def _extract_title_from_md(md_text: str) -> str | None:
    if not md_text:
        return None
    for line in md_text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip() or None
        if s:
            break
    return None


def _extract_title_from_content_list(doc_dir: Path) -> str | None:
    # content_list.json 命名并不固定：尝试匹配 *_content_list.json
    files = sorted(doc_dir.glob("*_content_list.json"))
    for p in files:
        try:
            arr = json.loads(_safe_read_text(p))
        except Exception:
            continue
        if not isinstance(arr, list):
            continue
        for item in arr:
            if not isinstance(item, dict):
                continue
            # MinerU 常见：text_level=1 是封面标题
            if item.get("type") == "text" and item.get("text_level") == 1 and isinstance(item.get("text"), str):
                t = re.sub(r"\s+", " ", item["text"].strip())
                if t:
                    return t
    return None


def _pick_doc_title(doc_dir: Path, md_text: str) -> str:
    title = _extract_title_from_layout(doc_dir / "layout.json")
    if title:
        return title
    title = _extract_title_from_md(md_text)
    if title:
        return title
    title = _extract_title_from_content_list(doc_dir)
    if title:
        return title
    return doc_dir.name


def _uuid_parts_from_relpath(rel_path: Path) -> list[str]:
    parts = []
    for p in rel_path.parts:
        if _UUID_RE.match(p):
            parts.append(p)
    return parts


def _validate_and_extract_source_id(root: Path) -> tuple[str, str]:
    """验证输入根目录结构并提取 source_file_id。
    
    Args:
        root: 输入根目录路径
        
    Returns:
        (root_kind, source_file_id)
        
    Raises:
        AppError: 如果路径结构不符合预期
    """
    root_abs = root.resolve()
    
    # 检查是否是 intermediates/{id}/pic_to_json 结构
    if root_abs.name.lower() == "pic_to_json":
        parent = root_abs.parent
        
        # 验证 parent 是有效的 UUID
        if not parent.name or not _UUID_RE.match(parent.name):
            raise AppError(
                code="kb_invalid_path_structure",
                message=(
                    f"pic_to_json 的父目录应为有效的 UUID (source_file_id): {parent}\n"
                    f"期望路径格式: data/intermediates/{{source_file_id}}/pic_to_json/\n"
                    f"实际路径: {root_abs}"
                ),
                status_code=400,
            )
        
        # 验证再上一级是 intermediates
        grandparent = parent.parent
        if grandparent.name.lower() != "intermediates":
            raise AppError(
                code="kb_invalid_path_structure",
                message=(
                    f"路径结构不符合预期: {root_abs}\n"
                    f"期望路径格式: data/intermediates/{{source_file_id}}/pic_to_json/\n"
                    f"实际路径: {root_abs}"
                ),
                status_code=400,
            )
        
        return "pic_to_json", parent.name
    
    # 其他路径结构的处理（向后兼容）
    return "other", "unknown"


def _summarize_chart_json(obj: dict[str, Any]) -> str:
    # 目标：可检索、短、稳定（避免把整段 JSON 塞进 KB）
    chart_name = obj.get("_chart_name") or obj.get("chart_name") or "图表"
    chart_type = obj.get("chart_type") or ""
    desc = obj.get("description") or ""

    legend_keys: list[str] = []
    try:
        cd = obj.get("chart_data")
        if isinstance(cd, list) and cd:
            first = cd[0]
            if isinstance(first, dict) and isinstance(first.get("legend"), dict):
                legend_keys = [str(k) for k in list(first["legend"].keys())[:12]]
    except Exception:
        legend_keys = []

    lines = [f"[图表] {str(chart_name).strip()}"]
    if chart_type:
        lines.append(f"类型: {str(chart_type).strip()}")
    if desc:
        desc_s = str(desc).strip()
        if len(desc_s) > 240:
            desc_s = desc_s[:240] + "…"
        lines.append(f"说明: {desc_s}")
    if legend_keys:
        lines.append(f"系列: {', '.join(legend_keys)}")
    return "\n".join(lines).strip()


def _inject_chart_snippets(*, md_text: str, doc_dir: Path) -> tuple[str, int]:
    """将 chart_json 的语义以短文本形式注入到 Markdown（在图片引用之后）。

    返回：(新文本, 注入数量)
    """
    injected = 0
    out_lines: list[str] = []
    lines = md_text.replace("\r\n", "\n").split("\n")

    img_line_re = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
    for line in lines:
        out_lines.append(line)
        m = img_line_re.search(line)
        if not m:
            continue
        rel = _normalize_rel_posix(m.group(1))
        if not rel.startswith("images/"):
            continue
        stem = Path(rel).stem
        chart_json_path = doc_dir / "chart_json" / f"{stem}.json"
        if not chart_json_path.exists():
            continue
        try:
            obj = json.loads(_safe_read_text(chart_json_path))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("is_chart") is not True:
            continue
        snippet = _summarize_chart_json(obj)
        if not snippet:
            continue
        # 注意：这里**不要**插入空行（\n\n）。
        # chunking_service 的结构切分会按“双换行”拆段；
        # 若插入空行，图像引用与 [图表] 摘要更容易被切到不同 chunk，
        # 导致“召回段落无法准确定位对应图表”。
        out_lines.append(snippet)
        injected += 1

    return "\n".join(out_lines), injected


@dataclass
class KnowledgeBaseBuildOptions:
    input_root: Path
    collection_name: str
    recreate: bool = False
    max_docs: int | None = None
    chunking: ChunkingOptions | None = None
    # 进度观测：允许外部指定 build_id 与 storage_path（便于后台任务先返回 ID）
    build_id: str | None = None
    artifact_storage_path: str | None = None
    progress_update_every_n_docs: int = 1


class KnowledgeBaseService:
    """知识库构建与查询服务（服务端核心）。"""

    def __init__(
        self,
        *,
        chunking_service: DocumentChunkingService,
        vector_service: VectorStorageService,
        hybrid_search_service: HybridSearchService,
        source_file_repository: SourceFileRepository | None = None,
        artifact_repository: IntermediateArtifactRepository | None = None,
    ):
        self.chunking_service = chunking_service
        self.vector_service = vector_service
        self.hybrid_search_service = hybrid_search_service
        self.source_file_repository = source_file_repository
        self.artifact_repository = artifact_repository

    async def build_from_t094_output(
        self,
        *,
        options: KnowledgeBaseBuildOptions,
        workspace_id: str = "default",
    ) -> dict[str, Any]:
        log = get_logger(__name__)
        # 运行期可用：resolve 仅用于文件遍历/存在性校验；对外回写保持相对路径
        root = options.input_root
        root_abs = options.input_root.resolve()
        if not root_abs.exists():
            raise AppError(
                code="kb_input_root_not_found",
                message=f"输入目录不存在: {root}",
                status_code=400,
            )

        build_id = options.build_id or str(uuid4())
        started_at = datetime.now().isoformat()

        # 进度观测工件：复用 intermediate_artifacts（type=kb_chunks）承载 build 过程与最终报告
        artifact_id: str | None = None
        artifact_storage_path: str | None = None
        artifact_out_path: Path | None = None
        last_log_stage: str | None = None
        last_log_pct: int | None = None

        def _progress_update(payload: dict[str, Any]) -> None:
            nonlocal artifact_id, artifact_storage_path, artifact_out_path
            if self.artifact_repository is None:
                return

            if artifact_id is None:
                artifact_id = build_id
            if artifact_storage_path is None:
                # 若外部已传入 storage_path，则严格复用；否则生成默认路径
                artifact_storage_path = options.artifact_storage_path
                if not artifact_storage_path:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    artifact_storage_path = (
                        f"intermediates/kb_chunks/{options.collection_name}/{ts}_{artifact_id}.json"
                    )
            if artifact_out_path is None:
                artifact_out_path = Path("data") / artifact_storage_path

            # 统一回写（确保对外可观测字段一致）
            payload["build_id"] = build_id
            payload["artifact_storage_path"] = artifact_storage_path

            # 确保 DB 记录存在
            repo = self.artifact_repository
            existing = repo.get_by_id(artifact_id)
            if existing is None:
                artifact = IntermediateArtifact(
                    id=artifact_id,
                    workspace_id=workspace_id,
                    source_id=None,
                    type=IntermediateType.KB_CHUNKS,
                    storage_path=artifact_storage_path,
                    deletable=True,
                    extra_metadata=json.dumps(
                        _compact_progress_for_db(payload),
                        ensure_ascii=False,
                    ),
                )
                repo.create(artifact)
            else:
                # 若已存在记录，以 DB 的 storage_path 为准（避免启动端与执行端不一致）
                if existing.storage_path and existing.storage_path != artifact_storage_path:
                    artifact_storage_path = existing.storage_path
                    artifact_out_path = Path("data") / artifact_storage_path
                    payload["artifact_storage_path"] = artifact_storage_path
                # 高频更新只写 extra_metadata；storage_path 不在此处覆盖（由启动侧保证一致）
                repo.update_extra_metadata(
                    artifact_id,
                    extra_metadata=json.dumps(
                        _compact_progress_for_db(payload),
                        ensure_ascii=False,
                    ),
                )

            # 同步落盘（供回放/审计）
            _write_json(artifact_out_path, payload)

            nonlocal last_log_stage, last_log_pct
            stage = payload.get("stage")
            status = payload.get("status")
            pct = payload.get("progress_percent")
            should_log = False
            if isinstance(stage, str) and stage != last_log_stage:
                should_log = True
            if isinstance(pct, int) and pct != last_log_pct:
                should_log = True
            if status in {"failed", "succeeded"}:
                should_log = True
            if should_log:
                last_log_stage = stage if isinstance(stage, str) else last_log_stage
                last_log_pct = pct if isinstance(pct, int) else last_log_pct
                log.info(
                    "t095.kb_build.progress",
                    extra={
                        "build_id": payload.get("build_id"),
                        "collection_name": payload.get("collection_name"),
                        "status": status,
                        "stage": stage,
                        "progress_percent": pct,
                        "docs_total": payload.get("docs_total"),
                        "docs_indexed": payload.get("docs_indexed"),
                        "chunks_indexed": payload.get("chunks_indexed"),
                        "chart_snippets_injected": payload.get("chart_snippets_injected"),
                        "bm25_index_storage_path": payload.get("bm25_index_storage_path"),
                        "artifact_storage_path": payload.get("artifact_storage_path"),
                    },
                )

        # 验证路径结构并提取 source_file_id
        root_kind, source_file_id_from_path = _validate_and_extract_source_id(root)
        
        if root_kind == "pic_to_json":
            # 新规范：pic_to_json 目录下应且仅应包含 1 个 Markdown 主文档（文件名不做约束）
            md_files = sorted([p for p in root_abs.glob("*.md") if p.is_file()])
            if len(md_files) != 1:
                raise AppError(
                    code="kb_invalid_input",
                    message=(
                        f"pic_to_json 目录下应且仅应包含 1 个 md 文件\n"
                        f"输入路径: {root}\n"
                        f"期望路径格式: data/intermediates/{{source_file_id}}/pic_to_json/\n"
                        f"实际找到: {len(md_files)} 个 md 文件\n"
                        f"文件列表: {[p.name for p in md_files]}"
                    ),
                    status_code=400,
                )
        else:
            # 新规范：只从 pic_to_json 目录下的“唯一主 md”建库（cleaned_doc 等路径已废弃）
            md_files = []
            ambiguous: list[dict[str, Any]] = []
            try:
                pic_dirs = [p for p in root_abs.rglob("pic_to_json") if p.is_dir()]
            except Exception:
                pic_dirs = []
            for d in sorted(pic_dirs, key=lambda x: x.as_posix()):
                items = sorted([p for p in d.glob("*.md") if p.is_file()], key=lambda x: x.name)
                if not items:
                    continue
                if len(items) == 1:
                    md_files.append(items[0])
                else:
                    ambiguous.append({"dir": d.as_posix(), "md_files": [p.name for p in items]})
            if ambiguous:
                raise AppError(
                    code="kb_ambiguous_markdown",
                    message=(
                        "检测到多个 pic_to_json 目录包含多个 .md 文件，无法确定主文档。\n"
                        "新规范要求：每个 pic_to_json 目录下必须且只允许 1 个 .md 主文档。"
                    ),
                    status_code=400,
                    details={"ambiguous": ambiguous[:20], "ambiguous_count": len(ambiguous)},
                )
        if options.max_docs and options.max_docs > 0:
            md_files = md_files[: int(options.max_docs)]
        if not md_files:
            raise AppError(
                code="kb_no_documents",
                message=(
                    f"未找到可用的主 Markdown 文档（仅支持新路径：.../pic_to_json/*.md 且每目录唯一）: {root}\n"
                    "请先完成 T094 图转 JSON（确保 data/intermediates/.../pic_to_json 下存在 1 个 .md 主文档）。"
                ),
                status_code=400,
            )

        docs_total = len(md_files)
        try:
            _progress_update(
                {
                    "task": "T095",
                    "build_id": build_id,
                    "collection_name": options.collection_name,
                    "input_root": root.as_posix(),
                    "artifact_storage_path": options.artifact_storage_path,
                    "status": "running",
                    "stage": "scanning_documents",
                    "progress_percent": 1,
                    "docs_total": docs_total,
                    "docs_indexed": 0,
                    "chunks_indexed": 0,
                    "chart_snippets_injected": 0,
                    "started_at": started_at,
                    "updated_at": datetime.now().isoformat(),
                }
            )

            if options.recreate:
                _progress_update(
                    {
                        "task": "T095",
                        "build_id": build_id,
                        "collection_name": options.collection_name,
                        "input_root": root.as_posix(),
                        "artifact_storage_path": artifact_storage_path,
                        "status": "running",
                        "stage": "recreating_collection",
                        "progress_percent": 5,
                        "docs_total": docs_total,
                        "docs_indexed": 0,
                        "chunks_indexed": 0,
                        "chart_snippets_injected": 0,
                        "started_at": started_at,
                        "updated_at": datetime.now().isoformat(),
                    }
                )
                await self.vector_service.delete_collection(options.collection_name)

            total_chunks = 0
            total_docs = 0
            total_chart_injected = 0
            source_ids: set[str] = set()
            all_chunks: list[KBChunk] = []
            bm25_index_storage_path: str | None = None

            for i, md_path in enumerate(md_files, start=1):
                doc_dir = md_path.parent
                rel = md_path.relative_to(root_abs)
                md_text = _safe_read_text(md_path)
                doc_title = _pick_doc_title(doc_dir, md_text)

                if root_kind == "pic_to_json":
                    # 使用已验证的 source_file_id
                    source_file_id = source_file_id_from_path
                    doc_id = md_path.stem
                else:
                    uuid_parts = _uuid_parts_from_relpath(rel)
                    source_file_id = uuid_parts[0] if len(uuid_parts) >= 1 else "unknown"
                    if len(uuid_parts) >= 2:
                        doc_id = uuid_parts[1]
                    elif doc_dir.name.lower() == "pic_to_json" and len(uuid_parts) >= 1:
                        # 典型新结构：intermediates/{source_file_id}/pic_to_json/<main>.md
                        # 此时 doc_dir.name 固定为 pic_to_json；用 source_file_id 作为 doc_id 以保证稳定/唯一
                        doc_id = uuid_parts[0]
                    else:
                        doc_id = doc_dir.name
                source_ids.add(source_file_id)

                original_filename = None
                if self.source_file_repository is not None and _UUID_RE.match(source_file_id):
                    sf = self.source_file_repository.get_by_id(source_file_id)
                    if sf is not None:
                        original_filename = sf.original_filename
                if original_filename:
                    dt = str(doc_title or "").strip()
                    stem = Path(original_filename).stem
                    if (
                        not dt
                        or _UUID_RE.match(dt)
                        or re.fullmatch(r"\d{4}", dt)
                        or re.fullmatch(r"\d+", dt)
                        or dt in {"证券研究报告", "研究报告", "年度报告", "报告"}
                        or len(dt) < 4
                    ):
                        doc_title = stem or doc_title

                augmented, injected = _inject_chart_snippets(md_text=md_text, doc_dir=doc_dir)
                total_chart_injected += injected

                base_meta: dict[str, Any] = {
                    "source_file_id": source_file_id,
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "doc_rel_path": str(rel).replace("\\", "/"),
                    "original_filename": original_filename,
                }

                chunks: list[KBChunk] = await self.chunking_service.chunk_document(
                    text=augmented,
                    metadata=base_meta,
                    options=options.chunking
                    or ChunkingOptions(strategy=ChunkingStrategy.STRUCTURE_AWARE),
                )

                await self.vector_service.upsert_vectors(
                    chunks=chunks,
                    collection_name=options.collection_name,
                )

                total_docs += 1
                total_chunks += len(chunks)
                all_chunks.extend(chunks)

                if options.progress_update_every_n_docs <= 1 or (i % options.progress_update_every_n_docs) == 0:
                    pct = 20 + int(75 * (total_docs / max(docs_total, 1)))
                    _progress_update(
                        {
                            "task": "T095",
                            "build_id": build_id,
                            "collection_name": options.collection_name,
                            "input_root": root.as_posix(),
                            "artifact_storage_path": artifact_storage_path,
                            "status": "running",
                            "stage": "processing_documents",
                            "progress_percent": min(pct, 95),
                            "docs_total": docs_total,
                            "docs_indexed": total_docs,
                            "chunks_indexed": total_chunks,
                            "chart_snippets_injected": total_chart_injected,
                            "bm25_index_storage_path": bm25_index_storage_path,
                            "started_at": started_at,
                            "updated_at": datetime.now().isoformat(),
                        }
                    )

            if all_chunks:
                try:
                    from src.application.services.bm25_index_service import BM25Index

                    _progress_update(
                        {
                            "task": "T095",
                            "build_id": build_id,
                            "collection_name": options.collection_name,
                            "input_root": root.as_posix(),
                            "status": "running",
                            "stage": "building_bm25",
                            "progress_percent": 97,
                            "docs_total": docs_total,
                            "docs_indexed": total_docs,
                            "chunks_indexed": total_chunks,
                            "chart_snippets_injected": total_chart_injected,
                            "bm25_index_storage_path": bm25_index_storage_path,
                            "started_at": started_at,
                            "updated_at": datetime.now().isoformat(),
                        }
                    )

                    if artifact_storage_path and artifact_storage_path.endswith(".json"):
                        bm25_index_storage_path = artifact_storage_path[:-5] + ".bm25.json"
                    else:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        bm25_index_storage_path = (
                            f"intermediates/kb_chunks/{options.collection_name}/{ts}_{build_id}.bm25.json"
                        )

                    idx = BM25Index.build_from_chunks(all_chunks)
                    idx.save(Path("data") / bm25_index_storage_path)
                except Exception as exc:
                    bm25_index_storage_path = None
                    log.warning(
                        "t095.kb_build.bm25_failed",
                        extra={
                            "build_id": build_id,
                            "collection_name": options.collection_name,
                            "error_type": exc.__class__.__name__,
                            "error_message": str(exc),
                        },
                    )

            _progress_update(
                {
                    "task": "T095",
                    "build_id": build_id,
                    "collection_name": options.collection_name,
                    "input_root": root.as_posix(),
                    "artifact_storage_path": artifact_storage_path,
                    "bm25_index_storage_path": bm25_index_storage_path,
                    "status": "running",
                    "stage": "finalizing",
                    "progress_percent": 98,
                    "docs_total": docs_total,
                    "docs_indexed": total_docs,
                    "chunks_indexed": total_chunks,
                    "chart_snippets_injected": total_chart_injected,
                    "started_at": started_at,
                    "updated_at": datetime.now().isoformat(),
                }
            )

            info = await self.vector_service.get_collection_info(options.collection_name)
            result = {
                "success": True,
                "collection_name": options.collection_name,
                "docs_indexed": total_docs,
                "chunks_indexed": total_chunks,
                "chart_snippets_injected": total_chart_injected,
                "artifact_id": artifact_id,
                "artifact_storage_path": artifact_storage_path,
                "bm25_index_storage_path": bm25_index_storage_path,
                "collection_info": info.model_dump() if info else None,
            }

            _progress_update(
                {
                    "task": "T095",
                    "build_id": build_id,
                    "collection_name": options.collection_name,
                    "input_root": root.as_posix(),
                    "artifact_storage_path": artifact_storage_path,
                    "status": "succeeded",
                    "stage": "succeeded",
                    "progress_percent": 100,
                    "docs_total": docs_total,
                    "docs_indexed": total_docs,
                    "chunks_indexed": total_chunks,
                    "chart_snippets_injected": total_chart_injected,
                    "bm25_index_storage_path": bm25_index_storage_path,
                    "source_file_ids": sorted(source_ids),
                    "started_at": started_at,
                    "updated_at": datetime.now().isoformat(),
                    "result": result,
                }
            )
            return result
        except Exception as exc:
            try:
                _progress_update(
                    {
                        "task": "T095",
                        "build_id": build_id,
                        "collection_name": options.collection_name,
                        "input_root": root.as_posix(),
                        "artifact_storage_path": artifact_storage_path,
                        "status": "failed",
                        "stage": "failed",
                        "progress_percent": last_log_pct,
                        "docs_total": docs_total,
                        "docs_indexed": None,
                        "chunks_indexed": None,
                        "chart_snippets_injected": None,
                        "started_at": started_at,
                        "updated_at": datetime.now().isoformat(),
                        "error": {"type": exc.__class__.__name__, "message": str(exc)},
                    }
                )
            except Exception:
                pass
            log.exception(
                "t095.kb_build.failed",
                extra={
                    "build_id": build_id,
                    "collection_name": options.collection_name,
                    "input_root": root.as_posix(),
                    "artifact_storage_path": artifact_storage_path,
                },
            )
            raise

    async def query_hybrid_rerank(
        self,
        *,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
        rerank_top_n: int | None = None,
    ) -> HybridSearchResponse:
        opts = HybridSearchOptions(
            top_k=top_k,
            use_rerank=True,
            rerank_top_n=rerank_top_n or top_k,
            filter_metadata=filter_metadata,
        )
        return await self.hybrid_search_service.search(
            query=query,
            collection_name=collection_name,
            options=opts,
        )

