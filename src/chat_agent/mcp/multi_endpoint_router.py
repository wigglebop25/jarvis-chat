"""
Multi-Endpoint MCP Router

Routes tool execution to different MCP servers based on tool category
(system tools vs Spotify tools).
"""

import asyncio
from typing import Optional, Any

from .multi_endpoint_client import MultiEndpointMCPClient
from .models import MCPToolResult


class MultiEndpointMCPRouter:
    """Routes tool execution to appropriate MCP server (system vs Spotify)."""

    def __init__(
        self,
        mcp_client: Optional[MultiEndpointMCPClient] = None,
        system_endpoint: Optional[str] = None,
        spotify_endpoint: Optional[str] = None,
    ):
        """
        Initialize multi-endpoint router.

        Args:
            mcp_client: Custom MultiEndpointMCPClient instance
            system_endpoint: Custom system MCP server endpoint
            spotify_endpoint: Custom Spotify MCP server endpoint
        """
        self.mcp_client = mcp_client or MultiEndpointMCPClient(
            system_endpoint=system_endpoint,
            spotify_endpoint=spotify_endpoint,
        )

    async def execute_tool(
        self,
        name: str,
        **kwargs,
    ) -> MCPToolResult:
        """Execute a tool through the appropriate MCP server."""
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
            # Determine which server failed
            endpoint_hint = "Spotify" if name in self.mcp_client.ENDPOINT_MAP["spotify"] else "Rust"
            return MCPToolResult(
                error=(
                    f"{endpoint_hint} MCP tool execution failed: {exc}. "
                    f"Ensure the {endpoint_hint.lower()} MCP server is running."
                ),
                is_error=True,
            )

    async def route_and_call(self, text: str) -> dict[str, Any]:
        """Route intent and optionally execute tool via system MCP server."""
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
        """List tools from all available MCP servers."""
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
                f"MCP tool discovery failed: {exc}. "
                "Ensure both Rust and Spotify MCP servers are running and reachable."
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

    async def health_check(self) -> dict[str, Any]:
        """Check health of all MCP endpoints."""
        try:
            health = await self.mcp_client.health_check()
            return {
                "status": "ok" if any(health.values()) else "degraded",
                "endpoints": health,
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
            }

    def health_check_sync(self) -> dict[str, Any]:
        """Synchronous health check wrapper."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            return {
                "status": "error",
                "error": "Cannot execute health check while event loop is running",
            }

        return asyncio.run(self.health_check())

    async def close(self):
        """Close all MCP connections."""
        await self.mcp_client.close()
