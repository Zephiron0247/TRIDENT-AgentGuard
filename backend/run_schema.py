import os
from pathlib import Path
import pyexasol

# Get password from environment variable
pw = os.environ["EXAPW"]

# Project root = parent of backend/
BASE_DIR = Path(__file__).resolve().parent.parent
schema_file = BASE_DIR / "backend" / "schema.sql"

print(f"Reading schema from: {schema_file}")

c = pyexasol.connect(
    dsn="127.0.0.1:8563",
    user="sys",
    password=pw,
    encryption=True,
    websocket_sslopt={"cert_reqs": 0}
)

with open(schema_file, "r", encoding="utf-8") as f:
    sql = f.read()

statements = [s.strip() for s in sql.split(";") if s.strip()]

for stmt in statements:
    print(f"Running: {stmt[:80]}...")
    c.execute(stmt)

print("Schema created successfully!")