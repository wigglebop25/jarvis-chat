"""Tool execution caching layer that wraps MCP router."""

import asyncio
import logging
from typing import Any, Optional

from .router import MCPRouter
from .models import MCPToolResult

logger = logging.getLogger(__name__)


class CachedMCPRouter:
    """Wraps MCPRouter with caching for read-only tools."""
    
    # Tools to cache (read-only, not state-changing)
    CACHEABLE_TOOLS = {
        'getMyPlaylists': 60,  # TTL in minutes
        'getTopTracks': 6 * 60,
        'getNowPlaying': 1,  # Very short cache
        'checkSpotifyAuth': 24 * 60,
        'getQueue': 5,
        'getRecentlyPlayed': 60,
        'getMyProfile': 24 * 60,
    }
    
    # Tools to NEVER cache (state-changing)
    NON_CACHEABLE = {
        'playMusic',
        'pausePlayback',
        'resumePlayback',
        'skipToNext',
        'skipToPrevious',
        'setRepeat',
        'setShuffle',
    }
    
    def __init__(self, mcp_router: Optional[MCPRouter] = None):
        """Initialize cached router wrapper."""
        self.router = mcp_router or MCPRouter()
        try:
            from ..tools.tool_cache import get_cache_stats
            stats = get_cache_stats()
            if stats:
                logger.info(f"Tool cache initialized: {stats}")
        except Exception as e:
            logger.debug(f"Tool cache not available: {e}")
    
    async def execute_tool(
        self,
        name: str,
        use_cache: bool = True,
        **kwargs,
    ) -> MCPToolResult:
        """
        Execute tool with optional caching.
        
        Args:
            name: Tool name
            use_cache: Whether to use cache for this tool
            **kwargs: Tool arguments
            
        Returns:
            MCPToolResult
        """
        # Check if tool is cacheable
        if not use_cache or name not in self.CACHEABLE_TOOLS:
            return await self.router.execute_tool(name, **kwargs)
        
        # Try cache first
        try:
            from ..tools.tool_cache import _make_cache_key, _get_cached_result
            
            cache_key = _make_cache_key(name, kwargs)
            ttl_seconds = self.CACHEABLE_TOOLS[name] * 60
            
            cached_result = _get_cached_result(cache_key, ttl_seconds)
            if cached_result is not None:
                logger.debug(f"✓ Cache hit for {name}")
                return MCPToolResult(result=cached_result)
            
        except Exception as e:
            logger.debug(f"Cache lookup failed for {name}: {e}")
        
        # Not in cache or cache unavailable, execute tool
        result = await self.router.execute_tool(name, **kwargs)
        
        # Store result in cache if successful
        if not result.is_error:
            try:
                from ..tools.tool_cache import _make_cache_key, _set_cached_result
                
                cache_key = _make_cache_key(name, kwargs)
                ttl_seconds = self.CACHEABLE_TOOLS[name] * 60
                
                _set_cached_result(cache_key, name, result.result, ttl_seconds)
                logger.debug(f"✓ Cached result for {name}")
                
            except Exception as e:
                logger.debug(f"Failed to cache result for {name}: {e}")
        
        return result
    
    def execute_tool_sync(self, name: str, use_cache: bool = True, **kwargs) -> MCPToolResult:
        """Synchronous wrapper."""
        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                return MCPToolResult(
                    error="Cannot execute sync tool call while event loop is running",
                    is_error=True,
                )
        except RuntimeError:
            pass
        
        return asyncio.run(self.execute_tool(name, use_cache=use_cache, **kwargs))
    
    async def route_and_call(self, text: str) -> dict[str, Any]:
        """Delegate to underlying router."""
        return await self.router.route_and_call(text)
    
    async def list_tools(self) -> list[dict[str, Any]]:
        """Delegate to underlying router."""
        return await self.router.list_tools()


# Singleton instance
_cached_router = None


def get_cached_router() -> CachedMCPRouter:
    """Get or create the singleton cached router."""
    global _cached_router
    if _cached_router is None:
        _cached_router = CachedMCPRouter()
    return _cached_router
