from typing import Any, Optional

from pydantic import BaseModel, Field


class MCPRequest(BaseModel):
    """JSON-RPC 2.0 request for MCP server."""
    jsonrpc: str = Field(default="2.0")
    id: Optional[int | str] = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class MCPResponse(BaseModel):
    """JSON-RPC 2.0 response from MCP server."""
    jsonrpc: str = Field(default="2.0")
    id: Optional[int | str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None


class MCPToolCall(BaseModel):
    """Tool call in MCP format."""
    name: str
    arguments: dict[str, Any]


class MCPToolResult(BaseModel):
    """Result from tool execution."""
    result: Optional[Any] = None
    error: Optional[str] = None
    is_error: bool = False
