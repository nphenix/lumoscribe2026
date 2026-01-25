
"""清理文档清洗产生的中间数据。

此脚本用于清理 Document Cleaning 阶段产生的中间产物，包括：
1. 物理文件: data/intermediates/{id}/cleaned_doc/
2. 数据库记录: intermediate_artifacts (type=cleaned_doc)
3. 回退源文件状态: CLEANING_PROCESSING/CLEANING_COMPLETED -> MINERU_COMPLETED
"""

import sys
import shutil
import argparse
from pathlib import Path
from sqlalchemy import delete, select, update

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.db import make_engine, make_session_factory, Job
from src.shared.config import get_settings
from src.domain.entities.source_file import SourceFile, SourceFileStatus
from src.domain.entities.intermediate_artifact import IntermediateArtifact, IntermediateType

def _is_consistent_cleaned_dir(cleaned_dir: Path) -> bool:
    if not cleaned_dir.exists():
        return False
    md_files = list(cleaned_dir.glob("*.md"))
    json_files = [p for p in cleaned_dir.glob("*.json") if p.name not in {"content_list.json", "layout.json"}]
    images_dir = cleaned_dir / "images"
    has_images = images_dir.exists() and any(images_dir.iterdir())
    has_content_list = (cleaned_dir / "content_list.json").exists()
    has_layout = (cleaned_dir / "layout.json").exists()
    return bool(md_files) and bool(json_files) and has_images and has_content_list and has_layout

def clean_cleaned_docs(ids: list[str] | None = None, force: bool = False, limit: int | None = None, order: str = "recent"):
    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    session_factory = make_session_factory(engine)
    
    with session_factory() as session:
        print("开始清理文档清洗相关数据...")
        
        selected_ids: list[str] | None = None
        if ids:
            selected_ids = ids
        elif limit is not None and limit > 0:
            q = select(SourceFile).where(
                SourceFile.status.in_([SourceFileStatus.CLEANING_PROCESSING, SourceFileStatus.CLEANING_COMPLETED])
            )
            if order == "recent":
                q = q.order_by(SourceFile.updated_at.desc())
            else:
                q = q.order_by(SourceFile.updated_at.asc())
            q = q.limit(limit)
            rows = session.execute(q).scalars().all()
            selected_ids = [sf.id for sf in rows]
            print(f"按数量选取 {len(selected_ids)} 个文件进行清理: {', '.join(selected_ids)}")
        
        if selected_ids:
            print(f"按目标 ID 删除 cleaned_doc 产物记录")
            deleted_artifacts = 0
            for sid in selected_ids:
                result = session.execute(
                    delete(IntermediateArtifact).where(
                        IntermediateArtifact.source_id == sid,
                        IntermediateArtifact.type == IntermediateType.CLEANED_DOC,
                    )
                )
                deleted_artifacts += result.rowcount or 0
            print(f"已删除 {deleted_artifacts} 条 cleaned_doc 中间产物记录（目标 ID）")
        else:
            if force:
                result = session.execute(
                    delete(IntermediateArtifact).where(IntermediateArtifact.type == IntermediateType.CLEANED_DOC)
                )
                deleted_count = result.rowcount or 0
                print(f"已删除所有 cleaned_doc 中间产物记录: {deleted_count}")
            else:
                artifacts = session.execute(
                    select(IntermediateArtifact).where(IntermediateArtifact.type == IntermediateType.CLEANED_DOC)
                ).scalars().all()
                deleted_artifacts = 0
                for art in artifacts:
                    storage_path = Path(art.storage_path)
                    if not storage_path.is_absolute():
                        base_path = PROJECT_ROOT / "data" / storage_path
                    else:
                        base_path = storage_path
                    cleaned_dir = base_path.parent
                    if not _is_consistent_cleaned_dir(cleaned_dir):
                        try:
                            session.delete(art)
                            deleted_artifacts += 1
                        except Exception:
                            pass
                print(f"已删除 {deleted_artifacts} 条不完整 cleaned_doc 中间产物记录")
        
        files_to_reset = []
        if selected_ids:
            files_to_reset = session.execute(
                select(SourceFile).where(SourceFile.id.in_(selected_ids))
            ).scalars().all()
        else:
            files_to_reset = session.execute(
                select(SourceFile).where(
                    SourceFile.status.in_([SourceFileStatus.CLEANING_PROCESSING, SourceFileStatus.CLEANING_COMPLETED])
                )
            ).scalars().all()
        
        if files_to_reset:
            print(f"发现 {len(files_to_reset)} 个文件需要回退状态...")
            for sf in files_to_reset:
                cleaned_dir = PROJECT_ROOT / "data" / "intermediates" / sf.id / "cleaned_doc"
                if force or not _is_consistent_cleaned_dir(cleaned_dir):
                    sf.status = SourceFileStatus.MINERU_COMPLETED
                    print(f"  - 回退文件 {sf.id} ({sf.original_filename}) -> MINERU_COMPLETED")
        
        intermediates_dir = PROJECT_ROOT / "data" / "intermediates"
        if intermediates_dir.exists():
            count = 0
            if selected_ids:
                for sid in selected_ids:
                    cleaned_doc_dir = intermediates_dir / sid / "cleaned_doc"
                    if cleaned_doc_dir.exists():
                        try:
                            if force or not _is_consistent_cleaned_dir(cleaned_doc_dir):
                                shutil.rmtree(cleaned_doc_dir)
                                count += 1
                                print(f"  - 删除目录: {cleaned_doc_dir}")
                        except Exception as e:
                            print(f"  - 删除失败 {cleaned_doc_dir}: {e}")
            else:
                for subdir in intermediates_dir.iterdir():
                    if subdir.is_dir():
                        cleaned_doc_dir = subdir / "cleaned_doc"
                        if cleaned_doc_dir.exists():
                            try:
                                if force or not _is_consistent_cleaned_dir(cleaned_doc_dir):
                                    shutil.rmtree(cleaned_doc_dir)
                                    count += 1
                                    print(f"  - 删除目录: {cleaned_doc_dir}")
                            except Exception as e:
                                print(f"  - 删除失败 {cleaned_doc_dir}: {e}")
            print(f"已清理 {count} 个不完整物理目录 (intermediates)")

        old_cleaned_dir = PROJECT_ROOT / "data" / "intermediates_cleaned"
        if old_cleaned_dir.exists():
            try:
                shutil.rmtree(old_cleaned_dir)
                print(f"已删除旧清理目录: {old_cleaned_dir}")
            except Exception as e:
                print(f"删除旧清理目录失败: {e}")
            
        session.commit()
        print("清理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理文档清洗产生的中间数据")
    parser.add_argument("--ids", type=str, help="指定需要回退清理的源文件 ID，逗号分隔", default=None)
    parser.add_argument("--force", action="store_true", help="强制回退状态并删除目录（即使目录完整）")
    parser.add_argument("--limit", type=int, help="限制处理的文件数量（例如 5 表示处理最近/最早 5 个）", default=None)
    parser.add_argument("--order", type=str, choices=["recent", "oldest"], help="选择处理顺序（recent=按更新时间从新到旧，oldest=从旧到新）", default="recent")
    args = parser.parse_args()
    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
    clean_cleaned_docs(ids=ids, force=args.force, limit=args.limit, order=args.order)
