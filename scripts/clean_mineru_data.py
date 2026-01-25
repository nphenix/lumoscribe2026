"""
MinerU 数据清理脚本

此脚本用于清理与 MinerU 相关的数据库记录，包括：
1. source_files: 所有上传的源文件记录
2. jobs: 所有 ingest 类型的任务记录
3. intermediate_artifacts: 类型为 mineru_raw 的中间产物

使用方法：
python scripts/clean_mineru_data.py
"""

import sys
import shutil
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import delete
from src.shared.db import make_engine, make_session_factory, Job
from src.shared.config import get_settings
from src.domain.entities.source_file import SourceFile
from src.domain.entities.intermediate_artifact import IntermediateArtifact, IntermediateType

def clean_mineru_data():
    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    session_factory = make_session_factory(engine)
    
    with session_factory() as session:
        print("开始清理 MinerU 相关数据...")
        
        # 1. 清理源文件 (source_files)
        # 注意：这里我们清理所有源文件，因为目前源文件主要用于 MinerU
        # 如果未来有其他用途，可以根据条件筛选
        deleted_files = session.execute(delete(SourceFile))
        print(f"已删除 {deleted_files.rowcount} 条源文件记录")
        
        # 2. 清理 Ingest 任务 (jobs)
        deleted_jobs = session.execute(
            delete(Job).where(Job.kind == "ingest")
        )
        print(f"已删除 {deleted_jobs.rowcount} 条 Ingest 任务记录")
        
        # 3. 清理 MinerU 中间产物 (intermediate_artifacts)
        deleted_artifacts = session.execute(
            delete(IntermediateArtifact).where(
                IntermediateArtifact.type == IntermediateType.MINERU_RAW
            )
        )
        print(f"已删除 {deleted_artifacts.rowcount} 条 MinerU 中间产物记录")
        
        # 4. 清理物理文件 (data/intermediates)
        project_root = Path(__file__).parent.parent
        intermediates_dir = project_root / "data" / "intermediates"
        
        if intermediates_dir.exists():
            shutil.rmtree(intermediates_dir)
            intermediates_dir.mkdir(parents=True, exist_ok=True)
            print(f"已清空目录: {intermediates_dir}")

        # 5. 清理物理文件 (data/sources)
        sources_dir = project_root / "data" / "sources"
        
        if sources_dir.exists():
            shutil.rmtree(sources_dir)
            sources_dir.mkdir(parents=True, exist_ok=True)
            print(f"已清空目录: {sources_dir}")
        
        session.commit()
        print("清理完成！")

if __name__ == "__main__":
    clean_mineru_data()
