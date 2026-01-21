#!/usr/bin/env python3
"""检查 MinerU Provider 完整配置。"""

import sqlite3
from pathlib import Path

DB_PATH = Path(".runtime/sqlite/lumoscribe.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 检查 MinerU Provider 的所有字段
cursor.execute("""
    SELECT id, key, name, provider_type, base_url, api_key, api_key_env, config_json, enabled
    FROM llm_providers
    WHERE key = 'mineru'
""")

row = cursor.fetchone()
if row:
    print("=== MinerU Provider 完整配置 ===")
    print(f"id: {row[0]}")
    print(f"key: {row[1]}")
    print(f"name: {row[2]}")
    print(f"provider_type: {row[3]}")
    print(f"base_url: {row[4]}")
    print(f"api_key: {row[5]}")
    print(f"api_key_env: {row[6]}")
    print(f"config_json: {row[7]}")
    print(f"enabled: {row[8]}")
else:
    print("MinerU Provider 不存在")

conn.close()
