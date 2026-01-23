#!/usr/bin/env python3
"""T095：知识库状态清理/重置脚本（测试辅助）。

用途：
- 在测试过程中快速“初始化知识库状态”
  - 删除 Chroma collections（按 collection/prefix/all）
  - 可选：删除中间态 kb_chunks 工件（DB 记录 + data/ 下文件）

示例（PowerShell）：
  # 删除所有以 t095_test_ 开头的测试 collection，并清理对应 kb_chunks 工件
  uv run python "scripts/t095-reset-kb.py" --prefix t095_test_ --delete-artifacts --yes

  # 删除指定 collection
  uv run python "scripts/t095-reset-kb.py" --collection default --yes

  # 全量清理（危险操作，需要 --yes）
  uv run python "scripts/t095-reset-kb.py" --all --delete-artifacts --yes
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _match(name: str, *, collection: str | None, prefix: str | None, all_flag: bool) -> bool:
    if all_flag:
        return True
    if collection and name == collection:
        return True
    if prefix and name.startswith(prefix):
        return True
    return False


async def _run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="", help="删除指定 collection（精确匹配）")
    parser.add_argument("--prefix", default="", help="删除以 prefix 开头的 collections（常用于测试）")
    parser.add_argument("--all", action="store_true", help="删除所有 collections（危险）")
    parser.add_argument(
        "--delete-artifacts",
        action="store_true",
        help="同时删除 kb_chunks 中间态工件（DB 记录 + data/ 文件）",
    )
    parser.add_argument(
        "--workspace-id",
        default="",
        help="仅清理指定 workspace_id 的 kb_chunks 记录（默认空=不过滤）",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="确认执行（无该参数时仅打印将要删除的内容）",
    )
    args = parser.parse_args()

    collection = args.collection.strip() or None
    prefix = args.prefix.strip() or None
    all_flag = bool(args.all)

    if not (all_flag or collection or prefix):
        raise SystemExit("请至少指定 --collection / --prefix / --all 之一")

    from src.shared.config import get_settings
    from src.shared.db import init_db, make_engine, make_session_factory
    from src.application.services.vector_storage_service import VectorStorageService
    from src.application.repositories.intermediate_artifact_repository import (
        IntermediateArtifactRepository,
    )
    from src.application.services.intermediate_artifact_service import (
        IntermediateArtifactService,
    )
    from src.domain.entities.intermediate_artifact import IntermediateType

    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    init_db(engine)
    session_factory = make_session_factory(engine)

    vector = VectorStorageService()
    existing = await vector.list_collections()
    targets = [n for n in existing if _match(n, collection=collection, prefix=prefix, all_flag=all_flag)]

    print("[T095] collections existing:", len(existing))
    print("[T095] collections matched :", len(targets))
    for n in targets:
        print(" -", n)

    if args.delete_artifacts:
        # 仅列出（不删除）符合条件的 kb_chunks 工件
        with session_factory() as db:
            repo = IntermediateArtifactRepository(db)
            service = IntermediateArtifactService(repo)
            items = repo.list(
                workspace_id=args.workspace_id or None,
                artifact_type=IntermediateType.KB_CHUNKS,
                source_id=None,
                limit=10000,
                offset=0,
            )

            matched_artifacts = []
            for it in items:
                sp = (it.storage_path or "").replace("\\", "/")
                # 形如：intermediates/kb_chunks/<collection>/<ts>_<uuid>.json
                if any(f"intermediates/kb_chunks/{n}/" in sp for n in targets) or (
                    all_flag and "intermediates/kb_chunks/" in sp
                ):
                    matched_artifacts.append(it)

            print("[T095] kb_chunks artifacts matched:", len(matched_artifacts))
            for it in matched_artifacts[:20]:
                print(" -", it.id, it.storage_path)
            if len(matched_artifacts) > 20:
                print(" ... (truncated)")

            if args.yes:
                for it in matched_artifacts:
                    try:
                        # 先清理 bm25 索引文件（与 kb_chunks 报告同名同目录）
                        try:
                            sp = (it.storage_path or "").replace("\\", "/")
                            if "intermediates/kb_chunks/" in sp and sp.endswith(".json"):
                                bm25 = Path("data") / (sp[:-5] + ".bm25.json")
                                if bm25.exists():
                                    bm25.unlink()
                        except Exception as exc:
                            print("[WARN] delete bm25 index failed:", it.id, exc)
                        service.delete_artifact(it.id)
                    except Exception as exc:
                        print("[WARN] delete_artifact failed:", it.id, exc)

    if not args.yes:
        print("[T095] dry-run done (add --yes to execute).")
        return 0

    # 删除 collections（不存在的会返回 False）
    deleted = 0
    for n in targets:
        ok = await vector.delete_collection(n)
        if ok:
            deleted += 1
    print("[T095] collections deleted:", deleted)
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())

