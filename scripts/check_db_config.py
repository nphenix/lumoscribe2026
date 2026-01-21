
import sqlite3
import json
from pathlib import Path

# Use absolute path or relative to CWD
db_path = Path("f:/lumoscribe2026/.runtime/sqlite/lumoscribe.db")

if not db_path.exists():
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

def print_table_config(table_name):
    print(f"\n--- Checking {table_name} ---")
    try:
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        if not rows:
            print(f"No records found in {table_name}")
            return

        for row in rows:
            print(f"ID: {row['id']}")
            if 'name' in row.keys():
                print(f"Name: {row['name']}")
            if 'key' in row.keys():
                print(f"Key: {row['key']}")
            
            # Safe access for row columns
            config_json = row['config_json'] if 'config_json' in row.keys() else None
            
            if config_json:
                try:
                    config = json.loads(config_json)
                    print(f"Config: {json.dumps(config, indent=2, ensure_ascii=False)}")
                    
                    # Check for streaming keys
                    stream_keys = ['stream', 'streaming']
                    found_stream = {k: config.get(k) for k in stream_keys if k in config}
                    if found_stream:
                        print(f"Found streaming config: {found_stream}")
                    else:
                        print("No explicit streaming config found.")
                except json.JSONDecodeError:
                    print(f"Invalid JSON in config_json: {config_json}")
            else:
                print("No config_json")
            print("-" * 20)
    except sqlite3.OperationalError as e:
        print(f"Error querying {table_name}: {e}")

# Check Providers
print_table_config("llm_providers")

# Check Call Sites
print_table_config("llm_call_sites")

conn.close()
