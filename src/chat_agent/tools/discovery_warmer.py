"""Tool discovery prewarming."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ToolDiscoveryWarmer:
    """Pre-warms tool definitions on startup."""
    
    def __init__(self, mcp_client):
        """Initialize with MCP client."""
        self.mcp_client = mcp_client
        self.tool_definitions = None
        self.is_warmed = False
    
    async def preheat(self) -> bool:
        """
        Pre-warm tool definitions on startup.
        
        Returns:
            True if successful, False if failed (non-blocking)
        """
        try:
            logger.info("Pre-warming tool discovery...")
            self.tool_definitions = await self._discover_tools()
            self.is_warmed = True
            logger.info(f"Pre-warmed {len(self.tool_definitions) if self.tool_definitions else 0} tools")
            return True
            
        except Exception as e:
            logger.warning(f"Tool preload failed (non-blocking): {e}")
            self.is_warmed = False
            return False
    
    async def get_tools(self, force_refresh: bool = False) -> list[dict]:
        """
        Get tool definitions (cached if available).
        
        Args:
            force_refresh: Force discovery even if cached
            
        Returns:
            List of tool definitions
        """
        if self.tool_definitions and not force_refresh:
            return self.tool_definitions
        
        try:
            self.tool_definitions = await self._discover_tools()
            return self.tool_definitions or []
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            return self.tool_definitions or []
    
    async def _discover_tools(self) -> Optional[list[dict]]:
        """Discover tools from MCP server."""
        try:
            result = await self.mcp_client.call("tools/list", {})
            return result.get("tools", [])
        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")
            raise
