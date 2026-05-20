"""
tools/cache/decorator.py
─────────────────────────
cache_with_ttl: decorator that wraps sync and async tool functions
with SQLite-backed TTL caching.
"""
import asyncio
import functools
import logging
from typing import Any, Callable, Optional

from .db import ensure_db, make_cache_key
from .operations import get_cached_result, set_cached_result

logger = logging.getLogger(__name__)


def cache_with_ttl(minutes: int = 60, skip_cache_for: Optional[list[str]] = None) -> Callable:
    """
    Decorator to cache tool results with TTL.

    Args:
        minutes: Time-to-live in minutes.
        skip_cache_for: Argument keys that disable caching for that call
                        (e.g. ["deviceId"] for state-changing args).
    """
    ensure_db()
    ttl_seconds = minutes * 60

    def decorator(func: Callable) -> Callable:
        tool_name = func.__name__

        def _should_skip(arguments: dict) -> bool:
            return bool(skip_cache_for and any(k in arguments for k in skip_cache_for))

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            arguments = args[0] if args else kwargs.get("arguments", {})
            if not _should_skip(arguments):
                key = make_cache_key(tool_name, arguments)
                cached = get_cached_result(key, ttl_seconds)
                if cached is not None:
                    logger.debug(f"Cache hit for {tool_name}: {key[:50]}...")
                    return cached
            result = await func(*args, **kwargs)
            if not _should_skip(arguments):
                set_cached_result(make_cache_key(tool_name, arguments), tool_name, result, ttl_seconds)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            arguments = args[0] if args else kwargs.get("arguments", {})
            if not _should_skip(arguments):
                key = make_cache_key(tool_name, arguments)
                cached = get_cached_result(key, ttl_seconds)
                if cached is not None:
                    logger.debug(f"Cache hit for {tool_name}: {key[:50]}...")
                    return cached
            result = func(*args, **kwargs)
            if not _should_skip(arguments):
                set_cached_result(make_cache_key(tool_name, arguments), tool_name, result, ttl_seconds)
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
