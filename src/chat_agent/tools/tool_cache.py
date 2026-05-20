"""
tools/tool_cache.py
────────────────────
Backward-compatible re-export. Implementation now lives in tools/cache/.
"""
from typing import Any, Optional

from .cache.db import ensure_db as _ensure_tool_cache_db
from .cache.operations import (
    get_cached_result as _get_cached_result,
    set_cached_result as _set_cached_result,
    clear_tool_cache,
    get_cache_stats,
)
from .cache.decorator import cache_with_ttl


def set_tool_cache(key: str, result: Any, ttl_seconds: int = 3600) -> bool:
    """Store a result in the tool cache."""
    _ensure_tool_cache_db()
    return _set_cached_result(key, key, result, ttl_seconds)


def get_tool_cache(key: str) -> Optional[Any]:
    """Retrieve a result from the tool cache."""
    _ensure_tool_cache_db()
    return _get_cached_result(key, 3600)


__all__ = [
    "cache_with_ttl",
    "clear_tool_cache",
    "get_cache_stats",
    "set_tool_cache",
    "get_tool_cache",
]
