"""
tools/cache/db.py
──────────────────
SQLite schema management for the tool result cache.
"""
import json
import sqlite3
from pathlib import Path

CACHE_DIR = Path.home() / ".jarvis" / "cache"
TOOL_CACHE_DB = CACHE_DIR / "tool_cache.db"


def ensure_db() -> None:
    """Create the tool cache SQLite database and indexes if they don't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(TOOL_CACHE_DB))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tool_cache (
            cache_key TEXT PRIMARY KEY,
            tool_name TEXT NOT NULL,
            result BLOB NOT NULL,
            created_at REAL NOT NULL,
            ttl_seconds INTEGER NOT NULL,
            hit_count INTEGER DEFAULT 0
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_cache(tool_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON tool_cache(created_at)")
    conn.commit()
    conn.close()


def make_cache_key(tool_name: str, arguments: dict) -> str:
    """Generate a deterministic cache key from tool name and arguments."""
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True, default=str)}"
