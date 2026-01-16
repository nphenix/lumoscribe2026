#!/usr/bin/env python3
"""数据库初始化脚本。

用于创建所有数据表。适用于开发环境和首次部署。

使用方法：
    python scripts/init-db.py
"""

from pathlib import Path

from src.shared.config import get_settings
from src.shared.db import Base, make_engine

# 导入所有实体以注册到 Base.metadata（解决循环导入问题）
from src.domain.entities import (
    SourceFile,
    Template,
    IntermediateArtifact,
    TargetFile,
)


def main() -> None:
    """初始化数据库并创建所有表。"""
    settings = get_settings()

    # 确保目录存在
    db_path = Path(settings.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建引擎
    engine = make_engine(db_path)

    # 创建所有表
    Base.metadata.create_all(engine)

    print(f"✅ 数据库初始化完成: {db_path}")
    print("已创建的表:")
    for table in Base.metadata.tables.values():
        print(f"  - {table.name}")


if __name__ == "__main__":
    main()
