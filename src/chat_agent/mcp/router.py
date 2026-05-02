import asyncio
from typing import Optional, Any

from .client import MCPClient
from .models import MCPToolResult


class MCPRouter:
    """Routes tool execution to MCP server only (no local fallback)."""

    def __init__(self, mcp_client: Optional[MCPClient] = None):
        self.mcp_client = mcp_client or MCPClient()

    async def execute_tool(
        self,
        name: str,
        **kwargs,
    ) -> MCPToolResult:
        """Execute a tool through MCP JSON-RPC."""
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
            return MCPToolResult(
                error=(
                    f"Rust MCP tool execution failed: {exc}. "
                    "Ensure jarvis-skills Rust MCP server is running."
                ),
                is_error=True,
            )

    async def route_and_call(self, text: str) -> dict[str, Any]:
        """Route intent and optionally execute tool via MCP."""
        try:
            return await self.mcp_client.call(
                method="jarvis/route_and_call",
                params={"text": text},
            )
        except Exception as exc:
            return {
                "intent": "UNKNOWN",
                "confidence": 0.0,
                "tool_name": None,
                "arguments": {},
                "should_execute": False,
                "execution_error": str(exc),
            }

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
        try:
            result = await self.mcp_client.call(method="tools/list", params={})
            if not isinstance(result, dict):
                raise RuntimeError("Invalid tools/list result shape")
            tools = result.get("tools")
            if not isinstance(tools, list):
                raise RuntimeError("tools/list returned invalid tools payload")
            return tools
        except Exception as exc:
            raise RuntimeError(
                f"Rust MCP tool discovery failed: {exc}. "
                "Ensure jarvis-skills Rust MCP server is running and reachable."
            ) from exc

    def list_tools_sync(self) -> list[dict[str, Any]]:
        """Synchronous tools/list wrapper."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            raise RuntimeError("Cannot execute sync tools/list while event loop is running")

        try:
            return asyncio.run(self.list_tools())
        except RuntimeError as exc:
            if "event loop is running" in str(exc).lower():
                raise RuntimeError(
                    "Cannot execute sync tools/list while event loop is running"
                ) from exc
            raise

    async def health_check(self) -> bool:
        """Check MCP server health."""
        return await self.mcp_client.health_check()
