"""Smart caching for tool execution results with TTL support."""

import asyncio
import functools
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Use .cache directory for tool cache
CACHE_DIR = Path.home() / ".jarvis" / "cache"
TOOL_CACHE_DB = CACHE_DIR / "tool_cache.db"


def _ensure_tool_cache_db():
    """Ensure tool cache database exists."""
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
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_cache(tool_name)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at ON tool_cache(created_at)
    """)
    
    conn.commit()
    conn.close()


def _make_cache_key(tool_name: str, arguments: dict) -> str:
    """Generate cache key from tool name and arguments."""
    # Sort arguments for consistent hashing
    sorted_args = json.dumps(arguments, sort_keys=True, default=str)
    key = f"{tool_name}:{sorted_args}"
    return key


def _get_cached_result(cache_key: str, ttl_seconds: int) -> Optional[Any]:
    """
    Retrieve cached result if still fresh.
    
    Args:
        cache_key: Cache key
        ttl_seconds: Time-to-live in seconds
        
    Returns:
        Cached result or None if expired/not found
    """
    try:
        conn = sqlite3.connect(str(TOOL_CACHE_DB))
        cursor = conn.cursor()
        
        now = time.time()
        cursor.execute("""
            SELECT result, created_at FROM tool_cache
            WHERE cache_key = ?
        """, (cache_key,))
        
        row = cursor.fetchone()
        if row:
            result_blob, created_at = row
            age_seconds = now - created_at
            
            if age_seconds < ttl_seconds:
                # Still fresh, increment hit count
                cursor.execute("""
                    UPDATE tool_cache SET hit_count = hit_count + 1
                    WHERE cache_key = ?
                """, (cache_key,))
                conn.commit()
                conn.close()
                
                result_str = result_blob.decode()
                return json.loads(result_str)
            else:
                # Expired, delete it
                cursor.execute("DELETE FROM tool_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
        
        conn.close()
        return None
        
    except Exception as e:
        logger.debug(f"Failed to get cached result: {e}")
        return None


def _set_cached_result(cache_key: str, tool_name: str, result: Any, ttl_seconds: int) -> bool:
    """
    Store result in cache.
    
    Args:
        cache_key: Cache key
        tool_name: Tool name for indexing
        result: Result to cache (must be JSON-serializable)
        ttl_seconds: Time-to-live in seconds
        
    Returns:
        True if successful
    """
    try:
        result_str = json.dumps(result, default=str)
        result_blob = result_str.encode()
        
        conn = sqlite3.connect(str(TOOL_CACHE_DB))
        cursor = conn.cursor()
        
        now = time.time()
        cursor.execute("""
            INSERT OR REPLACE INTO tool_cache
            (cache_key, tool_name, result, created_at, ttl_seconds)
            VALUES (?, ?, ?, ?, ?)
        """, (cache_key, tool_name, result_blob, now, ttl_seconds))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.debug(f"Failed to cache result: {e}")
        return False


def cache_with_ttl(minutes: int = 60, skip_cache_for: Optional[list[str]] = None) -> Callable:
    """
    Decorator to cache tool results with TTL.
    
    Args:
        minutes: Time-to-live in minutes
        skip_cache_for: List of argument keys to exclude from cache (e.g., ["deviceId"] for state-changing args)
        
    Returns:
        Decorated function
        
    Example:
        @cache_with_ttl(minutes=60, skip_cache_for=["uri"])
        def playMusic(arguments: dict) -> dict:
            ...
    """
    _ensure_tool_cache_db()
    
    def decorator(func: Callable) -> Callable:
        tool_name = func.__name__
        ttl_seconds = minutes * 60
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Extract arguments (usually passed as first positional arg)
            arguments = args[0] if args else kwargs.get('arguments', {})
            
            # Check if we should skip caching for this call
            skip = False
            if skip_cache_for:
                for key in skip_cache_for:
                    if key in arguments:
                        skip = True
                        break
            
            if not skip:
                cache_key = _make_cache_key(tool_name, arguments)
                cached = _get_cached_result(cache_key, ttl_seconds)
                if cached is not None:
                    logger.debug(f"Cache hit for {tool_name}: {cache_key[:50]}...")
                    return cached
            
            # Not in cache, execute function
            result = await func(*args, **kwargs)
            
            # Cache the result (if not skipped)
            if not skip:
                cache_key = _make_cache_key(tool_name, arguments)
                _set_cached_result(cache_key, tool_name, result, ttl_seconds)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Extract arguments
            arguments = args[0] if args else kwargs.get('arguments', {})
            
            # Check skip list
            skip = False
            if skip_cache_for:
                for key in skip_cache_for:
                    if key in arguments:
                        skip = True
                        break
            
            if not skip:
                cache_key = _make_cache_key(tool_name, arguments)
                cached = _get_cached_result(cache_key, ttl_seconds)
                if cached is not None:
                    logger.debug(f"Cache hit for {tool_name}: {cache_key[:50]}...")
                    return cached
            
            # Not in cache, execute function
            result = func(*args, **kwargs)
            
            # Cache the result
            if not skip:
                cache_key = _make_cache_key(tool_name, arguments)
                _set_cached_result(cache_key, tool_name, result, ttl_seconds)
            
            return result
        
        # Return the appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def clear_tool_cache(tool_name: Optional[str] = None) -> int:
    """
    Clear tool cache (all or for specific tool).
    
    Args:
        tool_name: Optional tool name to filter (clears all if None)
        
    Returns:
        Number of cache entries deleted
    """
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
    """Get cache statistics."""
    try:
        conn = sqlite3.connect(str(TOOL_CACHE_DB))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT tool_name, COUNT(*) as count, SUM(hit_count) as hits
            FROM tool_cache
            GROUP BY tool_name
        """)
        
        stats = {}
        for tool_name, count, hits in cursor.fetchall():
            stats[tool_name] = {'entries': count, 'hits': hits or 0}
        
        conn.close()
        return stats
        
    except Exception as e:
        logger.debug(f"Failed to get cache stats: {e}")
        return {}


def set_tool_cache(key: str, result: Any, ttl_seconds: int = 3600) -> bool:
    """
    Store a result in the tool cache.
    
    Args:
        key: Cache key
        result: Result to cache (must be JSON-serializable)
        ttl_seconds: Time-to-live in seconds (default: 1 hour)
        
    Returns:
        True if successful
    """
    _ensure_tool_cache_db()
    # Treat key as tool_name for simplicity
    return _set_cached_result(key, key, result, ttl_seconds)


def get_tool_cache(key: str) -> Optional[Any]:
    """
    Retrieve a result from the tool cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached result or None if expired/not found
    """
    _ensure_tool_cache_db()
    # Use default TTL of 1 hour for retrieval
    return _get_cached_result(key, 3600)
