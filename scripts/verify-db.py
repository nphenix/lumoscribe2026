#!/usr/bin/env python3
"""验证数据库表结构。"""

import sys
from pathlib import Path

# 允许从任意工作目录运行脚本：确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import get_settings
from sqlalchemy import create_engine, inspect

settings = get_settings()
engine = create_engine(f"sqlite+pysqlite:///{settings.sqlite_path}")
inspector = inspect(engine)

print("DB file:", settings.sqlite_path)
print()
print("Tables:")
for table_name in inspector.get_table_names():
    print(f"  - {table_name}")
    for col in inspector.get_columns(table_name):
        print(f"      {col['name']}: {col['type']}")
