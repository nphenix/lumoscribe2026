import sqlite3
import os

db_path = r'F:\lumoscribe2026\.runtime\sqlite\lumoscribe.db'

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])

# Check llm_models or llm_providers
# Assuming there is a table for models or providers
try:
    cursor.execute("SELECT * FROM llm_models")
    columns = [description[0] for description in cursor.description]
    print("\nColumns in llm_models:", columns)
    rows = cursor.fetchall()
    print("\nRows in llm_models:")
    for row in rows:
        print(row)
except Exception as e:
    print(f"Error querying llm_models: {e}")

try:
    cursor.execute("SELECT * FROM llm_providers")
    columns = [description[0] for description in cursor.description]
    print("\nColumns in llm_providers:", columns)
    rows = cursor.fetchall()
    print("\nRows in llm_providers:")
    for row in rows:
        print(row)
except Exception as e:
    print(f"Error querying llm_providers: {e}")

conn.close()
