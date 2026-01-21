#!/usr/bin/env python3
"""检查数据库结构和数据。"""

import sqlite3
from pathlib import Path

# 使用正确的数据库路径
DB_PATH = Path(".runtime/sqlite/lumoscribe.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 列出所有表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall()]
print("Tables:", tables)

# 检查 llm_providers 表
if "llm_providers" in tables:
    print("\n=== llm_providers ===")
    cursor.execute("SELECT id, key, name, provider_type, base_url, enabled FROM llm_providers")
    for row in cursor.fetchall():
        print(row)
else:
    print("\nllm_providers 表不存在")

# 检查 prompts 表
if "prompts" in tables:
    print("\n=== prompts ===")
    cursor.execute("SELECT scope, format, active FROM prompts")
    for row in cursor.fetchall():
        print(row)

conn.close()
