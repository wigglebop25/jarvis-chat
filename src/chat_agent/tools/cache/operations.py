"""
tools/cache/operations.py
──────────────────────────
CRUD operations for the tool result SQLite cache.
"""
import json
import logging
import sqlite3
import time
from typing import Any, Optional

from .db import TOOL_CACHE_DB

logger = logging.getLogger(__name__)


def get_cached_result(cache_key: str, ttl_seconds: int) -> Optional[Any]:
    """Return a cached result if still within TTL, else None."""
    try:
        conn = sqlite3.connect(str(TOOL_CACHE_DB))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT result, created_at FROM tool_cache WHERE cache_key = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        if row:
            result_blob, created_at = row
            if time.time() - created_at < ttl_seconds:
                cursor.execute(
                    "UPDATE tool_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (cache_key,)
                )
                conn.commit()
                conn.close()
                return json.loads(result_blob.decode())
            cursor.execute("DELETE FROM tool_cache WHERE cache_key = ?", (cache_key,))
            conn.commit()
        conn.close()
        return None
    except Exception as e:
        logger.debug(f"Failed to get cached result: {e}")
        return None


def set_cached_result(cache_key: str, tool_name: str, result: Any, ttl_seconds: int) -> bool:
    """Persist a result to the cache. Returns True on success."""
    try:
        result_blob = json.dumps(result, default=str).encode()
        conn = sqlite3.connect(str(TOOL_CACHE_DB))
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO tool_cache "
            "(cache_key, tool_name, result, created_at, ttl_seconds) VALUES (?, ?, ?, ?, ?)",
            (cache_key, tool_name, result_blob, time.time(), ttl_seconds)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.debug(f"Failed to cache result: {e}")
        return False


def clear_tool_cache(tool_name: Optional[str] = None) -> int:
    """Delete cache entries. Pass tool_name to target one tool, or None for all."""
    try:
        conn = sqlite3.connect(str(TOOL_CACHE_DB))
        cursor = conn.cursor()
        if tool_name:
            cursor.execute("DELETE FROM tool_cache WHERE tool_name = ?", (tool_name,))
        else:
            cursor.execute("DELETE FROM tool_cache")
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted > 0:
            logger.info(f"Cleared {deleted} tool cache entries")
        return deleted
    except Exception as e:
        logger.error(f"Failed to clear tool cache: {e}")
        return 0


def get_cache_stats() -> dict[str, Any]:
    """Return per-tool entry count and hit count statistics."""
    try:
        conn = sqlite3.connect(str(TOOL_CACHE_DB))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT tool_name, COUNT(*) as count, SUM(hit_count) as hits "
            "FROM tool_cache GROUP BY tool_name"
        )
        stats = {name: {"entries": count, "hits": hits or 0} for name, count, hits in cursor.fetchall()}
        conn.close()
        return stats
    except Exception as e:
        logger.debug(f"Failed to get cache stats: {e}")
        return {}
