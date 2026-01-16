#!/usr/bin/env python3
"""验证数据库表结构。"""

from src.shared.config import get_settings
from sqlalchemy import create_engine, inspect

settings = get_settings()
engine = create_engine(f"sqlite+pysqlite:///{settings.sqlite_path}")
inspector = inspect(engine)

print("数据库文件:", settings.sqlite_path)
print()
print("已创建的表:")
for table_name in inspector.get_table_names():
    print(f"  - {table_name}")
    for col in inspector.get_columns(table_name):
        print(f"      {col['name']}: {col['type']}")
