import asyncio
from typing import Any, Optional

from .client import MCPClient
from .models import MCPToolResult


class MCPRouter:
    """Routes tool execution to MCP server with optional fallback behavior."""

    def __init__(self, mcp_client: Optional[MCPClient] = None, fallback_enabled: bool = False):
        self.mcp_client = mcp_client or MCPClient()
        self.fallback_enabled = fallback_enabled

    async def execute_tool(
        self,
        name: str,
        **kwargs,
    ) -> MCPToolResult:
        """
        Execute a tool through MCP JSON-RPC and optionally fall back to direct execution.
        """
        if self.mcp_client:
            try:
                result = await self.mcp_client.call(
                    method="tools/call",
                    params={
                        "name": name,
                        "arguments": kwargs,
                    },
                )
                return MCPToolResult(result=result)
            except Exception as exc:
                if not self.fallback_enabled:
                    return MCPToolResult(error=str(exc), is_error=True)

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
        """Synchronous execution wrapper."""
        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                return MCPToolResult(
                    error="Cannot execute sync tool call while event loop is running",
                    is_error=True,
                )
        except RuntimeError:
            pass

        return asyncio.run(self.execute_tool(name, **kwargs))

    async def list_tools(self) -> list[dict[str, Any]]:
        """List tools from MCP server."""
        if not self.mcp_client:
            return []
        try:
            result = await self.mcp_client.call(method="tools/list", params={})
            if isinstance(result, dict):
                tools = result.get("tools", [])
                return tools if isinstance(tools, list) else []
        except Exception:
            return []
        return []

    async def health_check(self) -> bool:
        """Check MCP server health."""
        if not self.mcp_client:
            return False
        return await self.mcp_client.health_check()

    async def _execute_direct(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """All tools must be executed through MCP server - no embedded fallback."""
        return MCPToolResult(
            error=(
                f"Tool '{name}' requires MCP server connection. "
                "Start jarvis-skills MCP server: cd jarvis-skills && python src/jarvis_skills/server.py"
            ),
            is_error=True,
        )
