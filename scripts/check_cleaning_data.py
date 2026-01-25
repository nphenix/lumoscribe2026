import sys
from pathlib import Path
from sqlalchemy import select

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from src.shared.db import make_engine, make_session_factory
from src.shared.config import get_settings
from src.domain.entities.source_file import SourceFile, SourceFileStatus

def check_cleaning_status():
    settings = get_settings()
    engine = make_engine(settings.sqlite_path)
    session_factory = make_session_factory(engine)
    
    with session_factory() as session:
        # 查询所有文件状态
        files = session.execute(select(SourceFile)).scalars().all()
        
        print("\n=== 文档清洗状态检查 ===")
        print(f"总文件数: {len(files)}")
        
        status_counts = {}
        completed_files = []
        processing_files = []
        failed_files = [] # 假设停留在 MINERU_COMPLETED 但预期要清洗的视为潜在失败/未开始
        
        for sf in files:
            status_counts[sf.status] = status_counts.get(sf.status, 0) + 1
            
            if sf.status == SourceFileStatus.CLEANING_COMPLETED:
                completed_files.append(sf)
            elif sf.status == SourceFileStatus.CLEANING_PROCESSING:
                processing_files.append(sf)
            elif sf.status == SourceFileStatus.MINERU_COMPLETED:
                failed_files.append(sf)

        print("\n--- 状态统计 ---")
        for status, count in status_counts.items():
            print(f"{status.value}: {count}")

        if completed_files:
            print("\n✅ 已完成清洗的文件:")
            for sf in completed_files:
                print(f"- {sf.original_filename} ({sf.id})")
        else:
            print("\n⚠️ 暂无已完成清洗的文件")

        if processing_files:
            print("\n⏳ 正在清洗中的文件:")
            for sf in processing_files:
                print(f"- {sf.original_filename} ({sf.id})")
        
        if failed_files:
            print("\nzzz 待清洗/已回退的文件 (MINERU_COMPLETED):")
            for sf in failed_files:
                print(f"- {sf.original_filename} ({sf.id})")

if __name__ == "__main__":
    check_cleaning_status()
