#!/usr/bin/env python3
"""T095：知识库清理/重置脚本（用于一键获得干净建库环境）。

用途：
- 清空 Chroma collections（默认全量）
- 清理 kb_chunks 中间态工件（默认启用：DB 记录 + data/ 下文件 + bm25 索引文件）

示例：
  # 一键清库（全量清空 collections + kb_chunks 工件）
  python "scripts/t095-reset-kb.py" --yes

  # 交互确认（无 --yes 时会提示输入 YES）
  python "scripts/t095-reset-kb.py"
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
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


def _confirm_or_exit(*, yes: bool) -> None:
    if yes:
        return
    try:
        ans = input("确认执行清库操作？输入 YES 继续：").strip()
    except EOFError:
        raise SystemExit("未检测到交互输入，请使用 --yes 直接确认执行")
    if ans != "YES":
        raise SystemExit("已取消")


async def _run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="", help="仅清理指定 collection（精确匹配）")
    parser.add_argument("--prefix", default="", help="仅清理以 prefix 开头的 collections")
    parser.add_argument("--all", action="store_true", help="清理所有 collections（默认：不传参数时等同于 all）")
    parser.add_argument("--keep-default", action="store_true", help="保留 default collection（默认会清理）")
    parser.add_argument("--keep-artifacts", action="store_true", help="不清理 kb_chunks 工件（默认会清理）")
    parser.add_argument(
        "--workspace-id",
        default="",
        help="仅清理指定 workspace_id 的 kb_chunks 记录（默认空=不过滤）",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="直接确认执行（无该参数时会提示输入 YES）",
    )
    args = parser.parse_args()

    collection = args.collection.strip() or None
    prefix = args.prefix.strip() or None
    all_flag = bool(args.all) or not (collection or prefix)
    keep_default = bool(args.keep_default)
    delete_artifacts = not bool(args.keep_artifacts)

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
    if keep_default:
        targets = [n for n in targets if n != "default"]

    print("[T095] collections existing:", len(existing))
    print("[T095] collections matched :", len(targets))
    for n in targets:
        print(" -", n)

    kb_root = PROJECT_ROOT / "data" / "intermediates" / "kb_chunks"

    if delete_artifacts:
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
                if "intermediates/kb_chunks/" not in sp:
                    continue
                if all_flag:
                    if keep_default and "intermediates/kb_chunks/default/" in sp:
                        continue
                    matched_artifacts.append(it)
                    continue
                if any(f"intermediates/kb_chunks/{n}/" in sp for n in targets):
                    matched_artifacts.append(it)

            print("[T095] kb_chunks artifacts matched:", len(matched_artifacts))
            for it in matched_artifacts[:20]:
                print(" -", it.id, it.storage_path)
            if len(matched_artifacts) > 20:
                print(" ... (truncated)")

            _confirm_or_exit(yes=bool(args.yes))

            if matched_artifacts:
                for it in matched_artifacts:
                    try:
                        service.delete_artifact(it.id)
                    except Exception as exc:
                        print("[WARN] delete_artifact failed:", it.id, exc)

        if kb_root.exists():
            try:
                if all_flag and not keep_default:
                    shutil.rmtree(kb_root)
                elif all_flag and keep_default:
                    for sub in sorted(kb_root.iterdir()):
                        if sub.is_dir() and sub.name != "default":
                            shutil.rmtree(sub)
                else:
                    for n in targets:
                        p = kb_root / n
                        if p.exists() and p.is_dir():
                            shutil.rmtree(p)
            except Exception as exc:
                print("[WARN] delete kb_chunks directories failed:", exc)

    if not delete_artifacts:
        _confirm_or_exit(yes=bool(args.yes))

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

