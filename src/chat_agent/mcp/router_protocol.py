from __future__ import annotations

from typing import Any, Protocol

from .models import MCPToolResult


class MCPRouterLike(Protocol):
    """Structural type for router implementations used by the agent.

    Some callers access an internal 'mcp_client' attribute on router implementations
    (e.g., to count configured endpoints). Expose it here as an optional attribute
    to satisfy static type checkers while remaining compatible with concrete
    router implementations.
    """

    mcp_client: Any | None = None

    async def execute_tool(self, name: str, **kwargs: Any) -> MCPToolResult: ...
    async def route_and_call(self, text: str) -> dict[str, Any]: ...
    async def list_tools(self) -> list[dict[str, Any]]: ...
    def list_tools_sync(self) -> list[dict[str, Any]]: ...
