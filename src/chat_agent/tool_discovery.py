"""
Tool discovery and management.

Handles dynamic loading of tool definitions from MCP server
with fallback to bundled schemas.
"""

import logging
from typing import Any, Optional

from .mcp import MCPRouter
from .tools.definitions import normalize_mcp_tool_definitions

logger = logging.getLogger(__name__)


class ToolDiscovery:
    """Manages dynamic tool definition discovery from MCP server."""
    
    def __init__(self, mcp_router: MCPRouter, bundled_tools: list[dict[str, Any]]):
        """
        Initialize tool discovery.
        
        Args:
            mcp_router: MCPRouter instance for tool discovery
            bundled_tools: Fallback bundled tool definitions
        """
        self.mcp_router = mcp_router
        self.bundled_tools = bundled_tools
        self.current_tools = bundled_tools.copy()
        self._loaded = False
        self._error: str | None = None
    
    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get current tool definitions."""
        return self.current_tools
    
    @property
    def is_loaded(self) -> bool:
        """Whether tools have been loaded from MCP."""
        return self._loaded
    
    @property
    def last_error(self) -> Optional[str]:
        """Get last discovery error if any."""
        return self._error
    
    async def refresh_async(self, supports_tools: bool = True) -> bool:
        """
        Refresh tool definitions asynchronously from MCP server.
        
        Args:
            supports_tools: Whether the provider supports tools
        
        Returns:
            True if refreshed, False if skipped
        """
        if not supports_tools:
            return False
        
        if self._loaded:
            return False
        
        try:
            discovered_tools = await self.mcp_router.list_tools()
            normalized_tools = normalize_mcp_tool_definitions(discovered_tools)
            
            if not normalized_tools:
                raise RuntimeError("MCP tools/list returned no valid tool definitions")
            
            tool_defs_changed = normalized_tools != self.current_tools
            self.current_tools = normalized_tools
            self._loaded = True
            self._error = None
            
            logger.info(
                "Loaded %s tool definitions from Rust MCP server.",
                len(normalized_tools),
            )
            
            return tool_defs_changed
        
        except Exception as exc:
            error_text = str(exc)
            if error_text != self._error:
                logger.warning(
                    "MCP tool discovery failed, using bundled schemas: %s",
                    error_text,
                )
            self._error = error_text
            return False
    
    def refresh_sync(self, supports_tools: bool = True) -> bool:
        """
        Refresh tool definitions synchronously from MCP server.
        
        Args:
            supports_tools: Whether the provider supports tools
        
        Returns:
            True if refreshed, False if skipped
        """
        if not supports_tools:
            return False
        
        if self._loaded:
            return False
        
        try:
            discovered_tools = self.mcp_router.list_tools_sync()
            normalized_tools = normalize_mcp_tool_definitions(discovered_tools)
            
            if not normalized_tools:
                raise RuntimeError("MCP tools/list returned no valid tool definitions")
            
            tool_defs_changed = normalized_tools != self.current_tools
            self.current_tools = normalized_tools
            self._loaded = True
            self._error = None
            
            logger.info(
                "Loaded %s tool definitions from Rust MCP server.",
                len(normalized_tools),
            )
            
            return tool_defs_changed
        
        except Exception as exc:
            error_text = str(exc)
            if error_text != self._error:
                logger.warning(
                    "MCP tool discovery failed, using bundled schemas: %s",
                    error_text,
                )
            self._error = error_text
            return False
    
    def reset(self) -> None:
        """Reset tool discovery state."""
        self._loaded = False
        self._error = None
        self.current_tools = self.bundled_tools.copy()
