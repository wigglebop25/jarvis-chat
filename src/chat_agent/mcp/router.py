from typing import Any, Optional

from .client import MCPClient
from .models import MCPToolResult


class MCPRouter:
    """Routes tool execution to MCP server with fallback to direct execution."""

    def __init__(self, mcp_client: Optional[MCPClient] = None, fallback_enabled: bool = True):
        self.mcp_client = mcp_client
        self.fallback_enabled = fallback_enabled

    async def execute_tool(
        self,
        name: str,
        **kwargs,
    ) -> MCPToolResult:
        """
        Execute a tool via MCP server with fallback.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            MCPToolResult with result or error
        """
        if self.mcp_client:
            try:
                result = await self.mcp_client.call(
                    method="tools.execute",
                    params={
                        "name": name,
                        "arguments": kwargs,
                    },
                )
                return MCPToolResult(result=result)
            except Exception as e:
                if not self.fallback_enabled:
                    return MCPToolResult(error=str(e), is_error=True)
                # Fall through to direct execution

        if self.fallback_enabled:
            return await self._execute_direct(name, kwargs)

        return MCPToolResult(
            error="No MCP server available and fallback disabled",
            is_error=True,
        )

    def execute_tool_sync(
        self,
        name: str,
        **kwargs,
    ) -> MCPToolResult:
        """
        Execute a tool synchronously with fallback.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            MCPToolResult with result or error
        """
        if self.fallback_enabled:
            return self._execute_direct_sync(name, kwargs)

        return MCPToolResult(
            error="No MCP server available and fallback disabled",
            is_error=True,
        )

    async def _execute_direct(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """Direct tool execution without MCP server."""
        try:
            if name == "get_system_info":
                import psutil

                return MCPToolResult(
                    result={
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage("/").percent,
                    }
                )

            return MCPToolResult(
                error=f"Unknown tool: {name}",
                is_error=True,
            )
        except Exception as e:
            return MCPToolResult(error=str(e), is_error=True)

    def _execute_direct_sync(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """Synchronous direct tool execution without MCP server."""
        try:
            if name == "get_system_info":
                import psutil

                return MCPToolResult(
                    result={
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage("/").percent,
                    }
                )

            return MCPToolResult(
                error=f"Unknown tool: {name}",
                is_error=True,
            )
        except Exception as e:
            return MCPToolResult(error=str(e), is_error=True)
