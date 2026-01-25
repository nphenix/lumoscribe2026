from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from sqlalchemy import delete, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.db import make_engine, make_session_factory
from src.shared.config import get_settings
from src.domain.entities.intermediate_artifact import IntermediateArtifact, IntermediateType


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _safe_unlink(path: Path, *, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        print(f"[DRY-RUN] remove file: {path}")
        return True
    try:
        path.unlink()
        print(f"removed file: {path}")
        return True
    except Exception as exc:
        print(f"failed to remove file: {path}: {exc}")
        return False


def _safe_rmtree(path: Path, *, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        print(f"[DRY-RUN] remove dir:  {path}")
        return True
    try:
        shutil.rmtree(path)
        print(f"removed dir:  {path}")
        return True
    except Exception as exc:
        print(f"failed to remove dir: {path}: {exc}")
        return False


def _iter_source_dirs(workspace: Path, ids: list[str] | None) -> list[Path]:
    if ids:
        return [workspace / sid for sid in ids]
    out: list[Path] = []
    try:
        for sub in sorted(workspace.iterdir()):
            if not sub.is_dir():
                continue
            if _UUID_RE.fullmatch(sub.name):
                out.append(sub)
    except OSError:
        return []
    return out


def clean_pic_to_json(
    *,
    workspace: Path,
    ids: list[str] | None,
    delete_db_charts: bool,
    delete_db_kb_chunks: bool,
    delete_files: bool,
    force: bool,
    dry_run: bool,
) -> None:
    if not workspace.exists():
        print(f"workspace not found: {workspace}")
        return

    # 安全策略：默认 dry-run；只有显式 --force 才会执行删除
    effective_dry_run = True if (dry_run or not force) else False
    if effective_dry_run:
        print("[INFO] running in DRY-RUN mode (no changes will be made). Use --force to execute.")

    removed_files = 0
    removed_dirs = 0

    # 0) workspace 根目录报告（历史兼容：部分脚本会把报告落在 workspace 根）
    if delete_files:
        for name in ("run_report.json", "chart_index.json", "t094_state.jsonl"):
            p = workspace / name
            if _safe_unlink(p, dry_run=effective_dry_run):
                removed_files += 1

    # 1) 文件系统清理：按 source_id 清理中间产物目录
    # - data/intermediates/<id>/pic_to_json/      (T094 workspace snapshot)
    # - data/intermediates/<id>/chart_json/       (DB chart_json artifacts files, if present)
    # - data/intermediates/kb_chunks/...          (DB kb_chunks artifacts files, if present)
    if delete_files:
        source_dirs = _iter_source_dirs(workspace, ids)
        for sub in source_dirs:
            if not sub.exists() or not sub.is_dir():
                continue
            if not _is_within(sub, workspace):
                continue

            # 1.1 删除 pic_to_json 整体目录（比只删 chart_json 更干净；后续会从 cleaned_doc 重新复制）
            pic_to_json = sub / "pic_to_json"
            if pic_to_json.exists() and pic_to_json.is_dir() and pic_to_json.name == "pic_to_json":
                if _safe_rmtree(pic_to_json, dry_run=effective_dry_run):
                    removed_dirs += 1

            # 1.2 删除 DB chart_json 文件目录（T032/T094 可能都会生成）
            chart_json_dir = sub / "chart_json"
            if chart_json_dir.exists() and chart_json_dir.is_dir() and chart_json_dir.name == "chart_json":
                if _safe_rmtree(chart_json_dir, dry_run=effective_dry_run):
                    removed_dirs += 1

        # 1.3 kb_chunks 为全局目录（非按 source_id），若选择回退到 cleaned_doc，则一并清理
        if delete_db_kb_chunks:
            kb_root = workspace / "kb_chunks"
            if kb_root.exists() and kb_root.is_dir() and kb_root.name == "kb_chunks":
                if _safe_rmtree(kb_root, dry_run=effective_dry_run):
                    removed_dirs += 1

    # 2) DB 清理：回退到 cleaned_doc 阶段（删除 chart_json / kb_chunks 工件记录）
    if delete_db_charts or delete_db_kb_chunks:
        settings = get_settings()
        engine = make_engine(settings.sqlite_path)
        session_factory = make_session_factory(engine)
        with session_factory() as session:
            # 2.1 预览将删除的工件（便于审计）
            to_delete_types: set[IntermediateType] = set()
            if delete_db_charts:
                to_delete_types.add(IntermediateType.CHART_JSON)
            if delete_db_kb_chunks:
                to_delete_types.add(IntermediateType.KB_CHUNKS)

            q = select(IntermediateArtifact).where(IntermediateArtifact.type.in_(list(to_delete_types)))
            if ids and delete_db_charts:
                # chart_json 可以按 source_id 过滤；kb_chunks 通常没有 source_id（全局）
                q = q.where(
                    (IntermediateArtifact.type != IntermediateType.CHART_JSON)
                    | (IntermediateArtifact.source_id.in_(ids))
                )
            items = list(session.execute(q).scalars().all())

            print(f"[DB] matched artifacts: {len(items)} types={[t.value for t in to_delete_types]}")
            # 2.2 删除对应的文件（基于 storage_path），并在 dry-run 时只打印
            data_root = (PROJECT_ROOT / "data").resolve()
            for it in items:
                sp = (it.storage_path or "").replace("\\", "/").strip()
                if not sp:
                    continue
                p = (data_root / Path(sp)).resolve()
                if not _is_within(p, data_root):
                    print(f"[WARN] skip deleting path outside data/: {p} (storage_path={sp})")
                    continue
                if _safe_unlink(p, dry_run=effective_dry_run):
                    removed_files += 1

            # 2.3 删除 DB 记录（仅在非 dry-run）
            if effective_dry_run:
                print("[DRY-RUN] skip deleting DB rows")
            else:
                if delete_db_charts:
                    stmt = delete(IntermediateArtifact).where(
                        IntermediateArtifact.type == IntermediateType.CHART_JSON
                    )
                    if ids:
                        stmt = stmt.where(IntermediateArtifact.source_id.in_(ids))
                    session.execute(stmt)
                if delete_db_kb_chunks:
                    session.execute(
                        delete(IntermediateArtifact).where(
                            IntermediateArtifact.type == IntermediateType.KB_CHUNKS
                        )
                    )
                session.commit()
                print("[DB] deleted intermediate_artifacts rows")

    print(f"removed_files={removed_files} removed_dirs={removed_dirs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT / "data" / "intermediates"),
        help="workspace directory (default: data/intermediates)",
    )
    parser.add_argument("--ids", default=None, help="comma separated source file ids")
    parser.add_argument(
        "--delete-db-charts",
        action="store_true",
        help="delete DB IntermediateArtifact rows of type chart_json (default: false)",
    )
    parser.add_argument(
        "--delete-db-kb-chunks",
        action="store_true",
        help="delete DB IntermediateArtifact rows of type kb_chunks (default: false)",
    )
    parser.add_argument(
        "--no-delete-files",
        action="store_true",
        help="do not delete files/directories on disk (default: delete files)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="dry-run (print actions, do not delete); default is dry-run unless --force is set",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="execute deletion (required unless --dry-run)",
    )
    args = parser.parse_args()
    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
    clean_pic_to_json(
        workspace=Path(args.workspace).resolve(),
        ids=ids,
        delete_db_charts=bool(args.delete_db_charts),
        delete_db_kb_chunks=bool(args.delete_db_kb_chunks),
        delete_files=(not bool(args.no_delete_files)),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )
